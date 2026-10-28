//! voicer: tiny local voice I/O server.
//!
//! STT: Moonshine tiny (ONNX) + Silero VAD, TTS: Piper voices.
//! WebSocket at /ws, REST at /stt and /tts, demo client at /.

mod audio;
mod config;
mod filter;
mod protocol;
mod ortutil;
mod sentence;
mod server;
mod session;
mod stt;
mod stt_engine;
mod stt_parakeet;
mod tts;
mod vad;

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(name = "voicer", version, about = "Tiny local voice I/O server (STT + TTS)")]
struct Args {
    /// Address to bind.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// Port to bind.
    #[arg(long, default_value_t = 8710)]
    port: u16,
    /// Directory containing the downloaded models.
    #[arg(long, default_value = "models")]
    models_dir: PathBuf,
    /// Piper voice (file stem inside <models_dir>/tts).
    #[arg(long, default_value = "en_US-amy-medium")]
    voice: String,
    /// STT backend: moonshine-tiny | moonshine-base | parakeet
    #[arg(long, default_value = "moonshine-tiny")]
    stt_model: String,
    /// Disable the speakability filter by default (clients can still toggle it).
    #[arg(long, default_value_t = false)]
    no_filter: bool,
    /// Interval between interim transcripts while a user speaks (ms).
    #[arg(long, default_value_t = 800)]
    partial_interval_ms: u64,
    /// Silence (ms) that ends an utterance.
    #[arg(long, default_value_t = 350)]
    speech_end_ms: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let cfg = config::Config {
        host: args.host.clone(),
        port: args.port,
        models_dir: args.models_dir.clone(),
        voice: args.voice.clone(),
        stt_model: args.stt_model.clone(),
        default_filter: !args.no_filter,
        partial_interval_ms: args.partial_interval_ms,
        speech_start_threshold: 0.5,
        speech_end_threshold: 0.35,
        speech_end_ms: args.speech_end_ms,
        pre_roll_ms: 240,
        max_segment_secs: 15,
    };

    eprintln!("voicer v{}", env!("CARGO_PKG_VERSION"));

    let t0 = std::time::Instant::now();
    let stt = match cfg.stt_model.as_str() {
        "parakeet" => stt_engine::SttEngine::Parakeet(
            stt_parakeet::SttParakeet::load(&cfg.models_dir.join("parakeet"))
                .context("loading STT (parakeet) — run scripts/download-parakeet.sh first")?,
        ),
        "moonshine-tiny" | "moonshine" => stt_engine::SttEngine::Moonshine(
            stt::Stt::load(&cfg.models_dir.join("moonshine"))
                .context("loading STT (moonshine) — run scripts/download-models.sh first")?,
        ),
        "moonshine-base" => stt_engine::SttEngine::Moonshine(
            stt::Stt::load(&cfg.models_dir.join("moonshine-base"))
                .context("loading STT (moonshine-base) — run scripts/download-models.sh first")?,
        ),
        other => anyhow::bail!("unknown --stt-model '{other}' (options: moonshine-tiny, moonshine-base, parakeet)"),
    };
    eprintln!("  stt: {} loaded in {:?}", cfg.stt_model, t0.elapsed());

    let t1 = std::time::Instant::now();
    let tts = tts::Tts::load(&cfg.models_dir.join("tts"), &cfg.voice)
        .context("loading TTS (piper) — run scripts/download-models.sh first")?;
    eprintln!("  tts: {} loaded in {:?}", cfg.voice, t1.elapsed());

    let state = session::Shared {
        stt: Arc::new(stt),
        tts: Arc::new(tts),
        cfg: Arc::new(cfg.clone()),
    };

    eprintln!("  speech_end_ms={} partial_interval_ms={} default_filter={}",
        cfg.speech_end_ms, cfg.partial_interval_ms, cfg.default_filter);

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(server::run(cfg, state))
}
