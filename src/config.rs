//! Server configuration (CLI flags land here).

use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub models_dir: PathBuf,
    /// Piper voice file stem inside `<models_dir>/tts`.
    pub voice: String,
    /// Default state of the speakability filter for new connections.
    pub default_filter: bool,
    /// How often (ms) to emit partial transcripts during speech.
    pub partial_interval_ms: u64,
    /// VAD: probability above which speech starts.
    pub speech_start_threshold: f32,
    /// VAD: probability below which (after `speech_end_ms` of quiet) speech ends.
    pub speech_end_threshold: f32,
    /// Silence duration (ms) that closes an utterance.
    pub speech_end_ms: u64,
    /// Audio (ms) kept before speech start, so word onsets aren't clipped.
    pub pre_roll_ms: u64,
    /// Max utterance length (s) before it is force-finalized.
    pub max_segment_secs: u64,
}

/// Fixed audio constants.
pub const STT_SAMPLE_RATE: u32 = 16_000;
pub const VAD_CHUNK: usize = 512; // 32 ms @ 16 kHz
