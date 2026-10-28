# voicer — detailed documentation

voicer is a tiny, local voice I/O server for English speech:

- **STT (speech → text):** Moonshine tiny (ONNX, ~28 MB quantized) + Silero VAD (~2 MB)
- **TTS (text → speech):** Piper voices (ONNX, ~60 MB), synthesized sentence-by-sentence
- **Interface:** one WebSocket endpoint for realtime full-duplex use, two REST
  endpoints for one-shot use, and a built-in browser demo client at `/`

Everything runs locally on CPU. No cloud, no Python, one static binary.

```
 ┌────────────┐  mic audio (i16 PCM)   ┌───────────────────────────────────┐
 │  browser / │ ─────────────────────▶ │  voicer                           │
 │  mobile /  │                        │   Silero VAD ─▶ Moonshine STT     │
 │  agent     │ ◀───────────────────── │        │                          │
 │  glue      │  JSON events + TTS PCM │        └─ partial/final events    │
 └────────────┘                        │   speak requests ─▶ sentence      │
        │                              │   splitter ─▶ Piper TTS ─▶ PCM    │
        │  your agent integration      └───────────────────────────────────┘
        ▼
   pi / codex / claude / any agent — sends text to `speak`, receives `final`
```

voicer is **pure voice I/O** — it knows nothing about agents. The caller
(your browser app, mobile app, or a thin agent extension) is responsible for:

1. streaming mic audio to voicer,
2. taking `final` transcripts and feeding them to the agent,
3. taking the agent's speakable output and sending it back as `speak` requests.

This keeps voicer stateless and trivially reusable with any agent.

## Contents

1. [Quick start](#quick-start)
2. [Repository layout](#repository-layout)
3. [Models](#models)
4. [WebSocket protocol](#websocket-protocol)
5. [REST endpoints](#rest-endpoints)
6. [Session internals](#session-internals)
7. [Speakability filter](#speakability-filter)
8. [Latency notes & tuning](#latency-notes--tuning)
9. [Build environment notes](#build-environment-notes)
10. [Extending voicer](#extending-voicer)
11. [Known limitations](#known-limitations)

## Quick start

```sh
./scripts/download-models.sh      # ~100 MB of models into ./models
cargo build --release
./target/release/voicer           # serves http://127.0.0.1:8710
```

Open `http://localhost:8710/` in a browser (works on a phone via your LAN IP)
and click "Start listening". Server flags:

```
--host 0.0.0.0            # default 127.0.0.1
--port 8710
--models-dir ./models
--voice en_US-amy-medium  # any Piper voice dropped into models/tts/
--no-filter               # disable speakability filter by default
--partial-interval-ms 800 # interim transcript cadence
--speech-end-ms 350       # silence that closes an utterance
```

## Repository layout

```
src/
  main.rs      CLI parsing, model loading, runtime start
  config.rs    Config struct + audio constants (16 kHz STT, 512-sample VAD chunk)
  server.rs    axum router: / (demo), /ws, /stt, /tts, /health
  session.rs   per-connection state machine: VAD→STT pipeline + TTS queue worker
  protocol.rs  WebSocket JSON message types (ClientMsg/ServerMsg)
  vad.rs       Silero VAD wrapper (stateful, 512-sample chunks)
  stt.rs       Moonshine tiny wrapper (encoder + autoregressive decoder, KV cache)
  tts.rs       Piper wrapper (piper-rs, Mutex-guarded)
  filter.rs    speakability filter (markdown/code/tool-output → speakable text)
  sentence.rs  streaming sentence splitter
  audio.rs     PCM/WAV codecs, resampling, format conversions
  ortutil.rs   ONNX session construction + tensor extraction helpers
demo/index.html  single-file browser client (served at /)
scripts/download-models.sh  fetches all models
docs/documentation.md  this file
usage.md    consumer-facing usage guide
```

## Models

All three models run through the same ONNX Runtime (via the `ort` crate, rc.12).

| role | model | file(s) | size | notes |
|---|---|---|---|---|
| VAD | Silero VAD v5 | `models/vad/silero_vad.onnx` | 2.3 MB | stateful; inputs `input` [1,512] f32, `state` [2,1,128] f32, `sr` int64; outputs `output` (prob), `stateN` |
| STT | Moonshine tiny (int8) | `models/moonshine/*.onnx` + `tokenizer.json` | ~28 MB | English-only; raw waveform in, BPE tokens out |
| TTS | Piper `en_US-amy-medium` | `models/tts/en_US-amy-medium.onnx(.json)` | 63 MB | 22.05 kHz mono output |

### Why these models

- **Moonshine tiny** was chosen over Whisper tiny.en: ~27M params, built for
  short clips, transcribes faster than realtime on CPU, and the ONNX export is
  clean (no mel filterbank — it consumes raw 16 kHz samples directly).
- **Piper** over Kokoro-82M: much smaller and faster; quality is fine for
  feedback speech. (Piper itself is archived, but `piper-rs` is actively
  maintained and uses the same voice models.) Kokoro is the documented upgrade
  path if voice quality matters more than size (~82M params, better prosody).
- **A single unified speech-to-speech model** was investigated and rejected:
  nothing practical exists at tiny scale for CPU (Moshi is ~7B; SpeechT5 is
  old and low quality). Two specialized small models is the right trade.

### Moonshine ONNX contract (as implemented in `stt.rs`)

```
encoder_model_quantized.onnx
  in : input_values [1, N] f32              (raw 16 kHz waveform)
  out: last_hidden_state [1, S, 288] f32

decoder_model_quantized.onnx           (first decode step, no cache)
  in : input_ids [1, 1] i64, encoder_hidden_states [1, S, 288] f32
  out: logits [1, 1, 32000] f32, present.{0..5}.{decoder,encoder}.{key,value}

decoder_with_past_model_quantized.onnx (subsequent steps)
  in : input_ids [1, 1] i64, past_key_values.{n}.{decoder,encoder}.{key,value}
  out: logits, present.*        (NOTE: no encoder_hidden_states input —
                                 encoder cross-attn KV is carried in the cache)

decode loop: start with token BOS(1), greedy argmax, stop at EOS(2),
cap 448 new tokens, then BPE-decode with tokenizer.json.
```

`stt.rs` discovers input/output names from the graph at load time (with
filters for the int8 `*_quantized` / `*_scale` / `*_zero_point` variants), so
swapping in a different Moonshine variant should just work.

## WebSocket protocol

Endpoint: `GET /ws` (WebSocket upgrade).

**Binary frames** are raw PCM, little-endian, mono:

- client → server: **16-bit signed, 16 kHz** (the browser demo converts from
  the mic's native rate; mobile clients should resample or send 16k directly)
- server → client: **16-bit signed, TTS sample rate** (announced per sentence
  in `audio_start`, 22050 Hz for the default voice)

**Text frames** are JSON.

### client → server

| message | fields | meaning |
|---|---|---|
| `config` | `speak_filter?: bool`, `partial_interval_ms?: u64` | per-connection options |
| `speak` | `id: u32`, `text: string`, `filter?: bool` | enqueue text for TTS; may be sent repeatedly (streaming deltas); sentences are spoken as they complete |
| `flush` | — | synthesize a trailing partial sentence (after streaming deltas) |
| `cancel` | — | barge-in: stop speaking and drop the queue immediately |
| `ping` | — | → `pong` |

### server → client

| message | fields | meaning |
|---|---|---|
| `ready` | `tts_sample_rate`, `stt_sample_rate` | sent once on connect |
| `speech_start` | — | VAD detected speech onset |
| `partial` | `text` | interim transcript while the user is still speaking |
| `speech_end` | — | utterance boundary reached |
| `final` | `text` | final transcript of the utterance |
| `audio_start` | `id`, `sentence`, `sample_rate` | binary PCM frames for this sentence follow |
| `audio_end` | `id`, `sentence` | sentence audio done |
| `tts_done` | `id` | queue drained for this id |
| `error` | `message` | something failed |

Binary PCM frames between `audio_start`/`audio_end` belong to that sentence.
Frames are ~8192 samples (~370 ms at 22.05 kHz); the client schedules them
sequentially.

### Message ordering guarantees

- `partial` messages for an utterance are always followed by exactly one
  `final` for the same audio; a `final` supersedes all earlier partials.
- TTS sentences for a given `id` are streamed in order. `cancel` may stop
  mid-sentence; the next `speak` starts a fresh queue.

## REST endpoints

### `POST /stt`
Body: a WAV file (any sample rate/bit depth; linear-resampled to 16 kHz).
Response: `{"text": "..."}`.

```sh
curl -s -X POST --data-binary @recording.wav http://localhost:8710/stt
```

### `POST /tts`
Body: JSON `{"text": "...", "filter": true}`. Response: `audio/wav`.

```sh
curl -s -X POST -H 'content-type: application/json' \
  -d '{"text":"Hello from **voicer** `let x=1`","filter":true}' \
  http://localhost:8710/tts -o out.wav
```

### `GET /health`
`{"ok":true,"stt_sample_rate":16000}`

## Session internals (`session.rs`)

Per WebSocket connection:

**Inbound pipeline** — binary frames accumulate into a buffer; every complete
512-sample chunk (32 ms) is scored by the VAD:

- speech starts when probability ≥ `speech_start_threshold` (0.5); the last
  240 ms of pre-speech audio is prepended so word onsets aren't clipped;
- while speaking, an interim STT run fires every `partial_interval_ms`
  (default 800 ms) on the accumulating segment (only if ≥ 0.5 s of audio and
  no STT run already in flight);
- the utterance closes after `speech_end_ms` (default 350 ms) below
  `speech_end_threshold` (0.35) — or when it exceeds 15 s, at which point it
  is force-finalized and a new segment starts (long continuous speech works);
- on close: `speech_end` is emitted, then a final STT run emits `final`.
  In-flight partials are invalidated via a generation counter so a stale
  partial can never arrive after the matching final.

All STT runs execute on tokio blocking threads; sessions are Mutex-guarded
(`ort`'s `run` needs `&mut`), which only serializes access — the CPU work is
already the bottleneck.

**Outbound TTS worker** — one task per connection consumes `speak`/`flush`
commands:

1. optional speakability filter,
2. sentence splitter emits complete sentences as they arrive,
3. each sentence is synthesized on a blocking thread (Piper, ~20–50× realtime),
4. PCM is chunked and streamed with `audio_start`/`audio_end` framing,
5. between sentences and between frames the worker checks the cancel flag, so
   barge-in latency is one frame (~370 ms worst case).

## Speakability filter

`filter.rs::make_speakable` is a conservative heuristic pass:

- drops fenced code blocks (if the entire text was code → *"Code output omitted."*)
- keeps inline code contents, drops the backticks
- drops URLs, images (`![..](..)`), keeps link text
- strips markdown markers (`#`, `*`, `_`, `~`, `>`)
- drops lines that look like tool payloads: JSON (`{`, `"key":`), table rows
  (`|...`), diff markers, lines that are <50% alphanumeric

Toggle: per connection (`config`), per request (`speak.filter`), or server
default (`--no-filter`). It is intentionally regex-simple — see
[Extending](#extending-voicer) for the upgrade path.

## Latency notes & tuning

Rough figures on 8 CPU cores (release build, int8 models):

- VAD: negligible (tiny model, 32 ms chunks)
- STT partial: ~150–400 ms for a few seconds of audio, runs every 800 ms
- final transcript: ready ~200–500 ms after you stop speaking
- TTS first audio: ~100–300 ms after a sentence is queued

Tunables (CLI or per-connection):

- lower `--speech-end-ms` → snappier turn-taking, more risk of splitting
- lower `partial_interval_ms` → more frequent (but heavier) interim transcripts
- raise `speech_start_threshold` if noise triggers false starts

## Build environment notes

The build needs only a Rust toolchain; `ort` downloads prebuilt ONNX Runtime
binaries at build time. On hosts without `pkg-config`/OpenSSL headers (which
the downloader's HTTP client needs), see the notes in `docs/build-env.md`
covering the user-local prefix workaround used during development.

## Extending voicer

- **Better TTS:** drop in Kokoro-82M ONNX (`onnx-community/Kokoro-82M-v1.0-ONNX`)
  behind `tts.rs`'s interface — needs an espeak-based phonemizer (misaki);
  everything else (sentence streaming, cancel, WS framing) is unchanged.
- **Streaming STT:** Moonshine has a dedicated streaming variant
  (`moonshine-streaming-tiny`); swap `stt.rs` for it to get true token-level
  streaming instead of re-transcription partials.
- **Smarter speak-filter:** replace the heuristics with a local LLM classifier
  (e.g. a 1–3B model) behind the same `make_speakable(text) -> String` signature.
- **Multiple voices:** the `speak` message could grow a `voice` field; the TTS
  mutex would become a per-voice map.
- **Auto start/stop:** VAD is already server-side; a client that plays audio
  through speakers should implement echo cancellation (the browser demo relies
  on `getUserMedia` echoCancellation) to avoid hearing its own TTS.

## Known limitations

- English only (Moonshine tiny + the Piper voice are English).
- Single utterance cap of 30 s of audio per STT call; segments force-finalize
  at 15 s.
- The speakability filter is heuristic — unusual agent output formats may
  produce odd spoken text; disable it if that's worse than the alternative.
- No authentication — bind to localhost or put it behind your own proxy.
- Piper voice sample rate is fixed per voice (22.05 kHz for medium voices).
