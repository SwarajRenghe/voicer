# voicer — usage guide

voicer gives your software ears and a voice: send it microphone audio, get
transcripts back; send it text, get speech back. It runs entirely on your
machine's CPU. For internal design details see `docs/documentation.md`.

## Running it

```sh
./scripts/download-models.sh   # once, ~100 MB
cargo build --release
./target/release/voicer        # http://127.0.0.1:8710
```

Useful flags: `--host 0.0.0.0 --port 8710 --voice en_US-amy-medium --no-filter`.

Try it instantly: open `http://localhost:8710/` in a browser (or on your phone
via your machine's LAN IP), click **Start listening**, and type something into
the text box to hear TTS.

## Two ways to use it

### 1. Realtime (WebSocket) — for voice conversations

Connect to `ws://localhost:8710/ws`, then:

- **Send binary frames**: raw PCM, 16-bit little-endian, mono, **16 kHz**.
  Stream continuously; the server's voice-activity detector decides when
  speech starts and ends.
- **Receive JSON events**:
  - `speech_start` / `speech_end` — user started/stopped talking
  - `partial {"text": ...}` — interim transcript (while still speaking)
  - `final {"text": ...}` — finished utterance → **feed this to your agent**
  - `audio_start` / `audio_end` — bracket binary TTS PCM frames
  - `tts_done {"id": ...}` — everything you queued has been sent
- **Send JSON commands**:
  - `speak {"id": 1, "text": "..."}` — say this text (send repeatedly for
    streaming; sentences are spoken as they arrive)
  - `cancel` — barge-in, stop talking immediately
  - `config {"speak_filter": true}` — strip markdown/code/tool-output noise
    before speaking (on by default)

Minimal JS sketch:

```js
const ws = new WebSocket("ws://localhost:8710/ws");
ws.binaryType = "arraybuffer";
ws.onmessage = (e) => {
  if (typeof e.data !== "string") { playAudio(e.data); return; } // TTS PCM
  const m = JSON.parse(e.data);
  if (m.type === "final") askAgent(m.text);          // your glue
  if (m.type === "speech_start") playing && ws.send(JSON.stringify({type:"cancel"}));
};
micStream16kMono.ondata = (pcm) => ws.send(pcm);      // i16 PCM @16 kHz
function agentReply(text) {
  ws.send(JSON.stringify({ type: "speak", id: nextId++, text }));
}
```

### 2. One-shot (REST) — for simple integrations

```sh
# speech -> text
curl -s -X POST --data-binary @clip.wav http://localhost:8710/stt
# {"text":"hello world"}

# text -> speech (returns a WAV file)
curl -s -X POST -H 'content-type: application/json' \
  -d '{"text":"Task complete. Three tests passing.","filter":true}' \
  http://localhost:8710/tts -o reply.wav
```

## Hooking up an agent (e.g. pi, or any coding agent)

voicer never talks to the agent itself — a thin adapter does:

1. On `final` → send the transcript to your agent as user input.
2. Stream the agent's answer to `speak` (one `speak` message per text chunk
   is fine; send `flush` after the last chunk to speak a trailing fragment).
3. Send tool calls / reasoning / code *without* `speak` — with
   `speak_filter: true` voicer additionally strips markdown, code blocks and
   JSON-ish noise from whatever you do send.
4. When the user starts speaking again (`speech_start`) while audio is
   playing, send `cancel` to cut the assistant off mid-sentence (barge-in).

A pi extension, a Python script, or a browser page can all play this adapter
role — the protocol above is all voicer requires.

## Checklist

- [ ] models downloaded (`scripts/download-models.sh`)
- [ ] server running (`/health` returns `{"ok":true}`)
- [ ] client sends 16 kHz mono 16-bit PCM binary frames
- [ ] client handles `final` (supersedes `partial`) and `tts_done`
- [ ] client plays TTS at the `sample_rate` given in `audio_start` (22050 Hz)
- [ ] echo cancellation on the mic if TTS plays through speakers
