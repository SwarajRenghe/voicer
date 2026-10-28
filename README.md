# voicer

Tiny local voice I/O server: speak to your agents, hear their answers.

- **STT:** Moonshine tiny (English, ~28 MB) + Silero VAD — streaming transcripts
- **TTS:** Piper voice (~63 MB) — sentence-by-sentence streaming speech
- **API:** WebSocket (`/ws`) for realtime full-duplex use, REST (`/stt`, `/tts`) for one-shot use
- **Zero cloud:** everything runs on CPU, one static binary

## Quick start

```sh
./scripts/download-models.sh   # ~100 MB of models
cargo build --release
./target/release/voicer
```

Open `http://localhost:8710/` for the built-in browser demo.

## Docs

- `usage.md` — how to use voicer from other software/agents
- `docs/documentation.md` — architecture, protocol spec, internals, extension guide
