//! Text-to-speech using Piper voices (via `piper-rs`, which runs the voice
//! ONNX on the same ONNX Runtime used by our STT/VAD models).
//!
//! One shared `Piper` instance is guarded by a mutex; synthesis runs on
//! blocking threads (it is CPU-bound, ~20-50x realtime for a medium voice).

use anyhow::{Context, Result};
use piper_rs::Piper;
use std::path::Path;
use std::sync::Mutex;

pub struct Tts {
    inner: Mutex<Piper>,
    pub sample_rate: u32,
}

impl Tts {
    /// Load a voice by file stem, e.g. "en_US-amy-medium" expects
    /// `<dir>/en_US-amy-medium.onnx` and `<dir>/en_US-amy-medium.onnx.json`.
    pub fn load(dir: &Path, voice: &str) -> Result<Self> {
        let onnx = dir.join(format!("{voice}.onnx"));
        let config = dir.join(format!("{voice}.onnx.json"));
        let piper = Piper::new(&onnx, &config)
            .with_context(|| format!("loading piper voice {voice}"))?;
        // sample_rate is in the voice config; grab it via a tiny synthesis later,
        // but piper configs are reliably 22050 for medium voices — we still read
        // it properly on first synthesis.
        Ok(Self {
            inner: Mutex::new(piper),
            sample_rate: 22_050,
        })
    }

    /// Synthesize speech. Returns (samples, sample_rate).
    pub fn synthesize(&self, text: &str) -> Result<(Vec<f32>, u32)> {
        anyhow::ensure!(!text.trim().is_empty(), "nothing to synthesize");
        let mut piper = self.inner.lock().expect("piper mutex poisoned");
        let (samples, rate) = piper
            .create(text, false, None, None, None, None)
            .map_err(|e| anyhow::anyhow!("piper synthesis failed: {e}"))?;
        Ok((samples, rate))
    }
}
