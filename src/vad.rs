//! Voice activity detection using Silero VAD (ONNX), stateful, chunked.
//!
//! Expects 16 kHz mono f32 audio fed in chunks of exactly 512 samples
//! (32 ms). Returns speech probability per chunk.

use crate::ortutil::make_session;
use anyhow::{Context, Result};
use ndarray::{Array, ArrayD, Ix2, Ix3};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

const CHUNK: usize = 512;      // new samples per call (32 ms @ 16 kHz)
const CONTEXT: usize = 64;     // trailing samples prepended to each chunk

pub struct Vad {
    session: Session,
    state: ArrayD<f32>,
    context: Vec<f32>,
    audio_input: String,
    state_input: String,
    sr_input: String,
}

impl Vad {
    pub fn load(path: &Path) -> Result<Self> {
        let session = make_session(path, 1).context("loading VAD model")?;

        let names: Vec<String> = session.inputs().iter().map(|i| i.name().to_string()).collect();
        anyhow::ensure!(
            names.len() == 3,
            "unexpected VAD model inputs: {names:?} (expected Silero VAD v5: input, state, sr)"
        );
        let audio_input = names
            .iter()
            .find(|n| !n.contains("state") && !n.contains("sr"))
            .cloned()
            .context("VAD audio input not found")?;
        let state_input = names
            .iter()
            .find(|n| n.contains("state"))
            .cloned()
            .context("VAD state input not found")?;
        let sr_input = names
            .iter()
            .find(|n| n.contains("sr") || n.contains("sample_rate"))
            .cloned()
            .context("VAD sr input not found")?;
        Ok(Self {
            session,
            state: Array::<f32, Ix3>::zeros((2, 1, 128)).into_dyn(),
            context: vec![0.0; CONTEXT],
            audio_input,
            state_input,
            sr_input,
        })
    }

    pub fn reset(&mut self) {
        self.state = Array::<f32, Ix3>::zeros((2, 1, 128)).into_dyn();
        self.context = vec![0.0; CONTEXT];
    }

    /// Process exactly 512 new samples (32 ms at 16 kHz), returns speech probability.
    /// Mirrors the reference implementation: each call is fed the 64-sample
    /// trailing context followed by the new chunk ([1, 576] total).
    pub fn process(&mut self, chunk: &[f32]) -> Result<f32> {
        assert_eq!(chunk.len(), CHUNK, "VAD expects 512-sample chunks");
        let mut input = self.context.clone();
        input.extend_from_slice(chunk);
        let audio = Array::<f32, Ix2>::from_shape_vec((1, CHUNK + CONTEXT), input.clone())?;
        let sr = Array::<i64, ndarray::Ix0>::from_elem((), 16_000);

        let feed: Vec<(std::borrow::Cow<str>, ort::value::DynValue)> = vec![
            (self.audio_input.clone().into(), Tensor::from_array(audio)?.into()),
            (
                self.state_input.clone().into(),
                Tensor::from_array(self.state.clone())?.into(),
            ),
            (self.sr_input.clone().into(), Tensor::from_array(sr)?.into()),
        ];
        let outputs = self.session.run(feed)?;

        let mut iter = outputs.iter();
        let (_, prob_out) = iter.next().context("VAD produced no outputs")?;
        let (_, state_out) = iter.next().context("VAD produced only one output")?;
        let (_, prob) = prob_out.try_extract_tensor::<f32>()?;
        let (state_shape, state_data) = state_out.try_extract_tensor::<f32>()?;
        let dims: Vec<usize> = state_shape.iter().map(|&d| d as usize).collect();
        self.state = Array::from_shape_vec(dims, state_data.to_vec())?.into_dyn();
        // keep the trailing context for the next call
        self.context = input[input.len() - CONTEXT..].to_vec();
        Ok(prob[0])
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn chunk_size_is_32ms() {
        assert_eq!(512_f32 / 16000.0 * 1000.0, 32.0);
    }
}
