//! Selectable STT backends sharing one interface.

use anyhow::Result;

pub enum SttEngine {
    /// Moonshine tiny (default, ~28 MB) or base (~101 MB), int8.
    Moonshine(crate::stt::Stt),
    /// NVIDIA Parakeet TDT 0.6B v3 (~675 MB) — highest accuracy, multilingual.
    Parakeet(crate::stt_parakeet::SttParakeet),
}

impl SttEngine {
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        match self {
            SttEngine::Moonshine(s) => s.transcribe(samples),
            SttEngine::Parakeet(s) => s.transcribe(samples),
        }
    }
}
