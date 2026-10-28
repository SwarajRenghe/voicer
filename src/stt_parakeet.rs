//! Speech recognition using NVIDIA Parakeet TDT 0.6B v3 (ONNX, int8) —
//! optional high-accuracy backend (multilingual, punctuation + capitalization).
//!
//! Contract (NeMo RNN-T/TDT export, as used by onnx-asr / sherpa-onnx):
//!
//!   nemo128.onnx (preprocessor)
//!     in : waveforms [1, N] f32, waveforms_lens [1] i64
//!     out: features [1, 128, T] f32, features_lens [1] i64
//!   encoder-model.int8.onnx
//!     in : audio_signal [1, 128, T] f32, length [1] i64
//!     out: outputs [1, T', D] f32, encoded_lengths [1] i64   (T' = T/8)
//!   decoder_joint-model.int8.onnx
//!     in : encoder_outputs [1, D, 1] f32 (one frame),
//!          targets [[prev_token_or_blank]] i64, target_length [1] i64,
//!          input_states_1/2 (LSTM states, shapes from the graph)
//!     out: outputs [1, 1, V+5] f32 (token logits + duration logits), new states
//!
//! Greedy TDT decoding: at each encoder frame t, take token = argmax(logits[..V])
//! and duration = argmax(logits[V..]); emit non-blank tokens (advancing the
//! LSTM state), jump t forward by the predicted duration (0 = re-emit or
//! advance one frame on blank, capped at 10 emissions per frame).
//!
//! Files (from ysdede/parakeet-tdt-0.6b-v3-onnx, CC-BY-4.0):
//!   nemo128.onnx, encoder-model.int8.onnx, decoder_joint-model.int8.onnx, vocab.txt

use crate::ortutil::make_session;
use anyhow::{Context, Result};
use ndarray::{Array, Array1, Array2, Array3, Ix1, Ix2, Ix3};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;

const MAX_TOKENS_PER_STEP: usize = 10;
const SUBSAMPLING: i64 = 8;
const MIN_SAMPLES: usize = 480;

pub struct SttParakeet {
    preprocessor: Mutex<Session>,
    encoder: Mutex<Session>,
    joint: Mutex<Session>,
    /// token id -> text (▁ already replaced by space)
    vocab: Vec<String>,
    blank_idx: usize,
    vocab_size: usize,
    /// LSTM state tensor shapes (rank 3), taken from the joint graph at load time
    state_shapes: Vec<(String, [usize; 3])>,
}

impl SttParakeet {
    pub fn load(dir: &Path) -> Result<Self> {
        let preprocessor = make_session(&dir.join("nemo128.onnx"), 2)?;
        let encoder = make_session(&dir.join("encoder-model.int8.onnx"), 4)?;
        let joint = make_session(&dir.join("decoder_joint-model.int8.onnx"), 2)?;

        // vocab: lines may be "token" (id = line number) or "token id"
        let raw = std::fs::read_to_string(dir.join("vocab.txt"))
            .with_context(|| format!("reading {}", dir.join("vocab.txt").display()))?;
        let mut vocab: Vec<String> = Vec::new();
        for line in raw.lines() {
            if line.is_empty() {
                continue;
            }
            let mut parts = line.splitn(2, ' ');
            let token = parts.next().unwrap_or_default();
            let id: Option<usize> = parts
                .next()
                .and_then(|s| s.trim().parse().ok());
            let idx = id.unwrap_or(vocab.len());
            if vocab.len() <= idx {
                vocab.resize(idx + 1, String::new());
            }
            vocab[idx] = token.replace('\u{2581}', " ");
        }
        let blank_idx = vocab
            .iter()
            .position(|t| t == "<blk>")
            .context("vocab has no <blk> token")?;
        let vocab_size = vocab.len();

        // LSTM state shapes come from the joint graph (e.g. [1, 1, 320])
        let state_shapes: Vec<(String, [usize; 3])> = joint
            .inputs()
            .iter()
            .filter(|i| i.name().starts_with("input_states_"))
            .map(|i| {
                let dims: [usize; 3] = match i.dtype() {
                    ort::value::ValueType::Tensor { shape, .. } => {
                        let v: Vec<usize> =
                            shape.iter().map(|&d| if d > 0 { d as usize } else { 1 }).collect();
                        [v[0], v[1], v[2]]
                    }
                    _ => panic!("LSTM state input is not a tensor"),
                };
                (i.name().to_string(), dims)
            })
            .collect();
        anyhow::ensure!(
            state_shapes.len() == 2,
            "expected 2 LSTM state inputs on decoder_joint"
        );

        Ok(Self {
            preprocessor: Mutex::new(preprocessor),
            encoder: Mutex::new(encoder),
            joint: Mutex::new(joint),
            vocab,
            blank_idx,
            vocab_size,
            state_shapes,
        })
    }

    fn zero_state(&self) -> Vec<(String, Array3<f32>)> {
        self.state_shapes
            .iter()
            .map(|(name, dims)| (name.clone(), Array3::<f32>::zeros(*dims)))
            .collect()
    }

    /// Transcribe 16 kHz mono f32 audio. Empty/short input yields "".
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        if samples.len() < MIN_SAMPLES {
            return Ok(String::new());
        }

        // 1. waveform -> log-mel features [1, 128, T]
        let waveforms = Array2::from_shape_vec((1, samples.len()), samples.to_vec())?;
        let lens = Array::<i64, Ix1>::from_shape_vec((1,), vec![samples.len() as i64])?;
        let (features, features_lens) = {
            let mut pp = self.preprocessor.lock().expect("parakeet preprocessor mutex");
            let out = pp.run(vec![
                (std::borrow::Cow::Borrowed("waveforms"), ort::value::DynValue::from(Tensor::from_array(waveforms)?)),
                (std::borrow::Cow::Borrowed("waveforms_lens"), ort::value::DynValue::from(Tensor::from_array(lens)?)),
            ]).map_err(|e| anyhow::anyhow!("parakeet preprocessor run: {e}"))?;
            let f = out["features"].try_extract_tensor::<f32>()?;
            let features_dims = (f.0[0] as usize, f.0[1] as usize, f.0[2] as usize);
            let features: Array3<f32> = Array3::from_shape_vec(features_dims, f.1.to_vec())?;
            let (lshape, ldata) = out["features_lens"].try_extract_tensor::<i64>()?;
            let _ = lshape;
            (features, Array::<i64, Ix1>::from_shape_vec((1,), vec![ldata[0]]).expect("lens"))
        };

        // 2. encode: [1, 128, T] -> [1, T', D], then transpose to [T', 1, D]
        let (encodings, enc_len): (Array2<f32>, usize) = {
            let mut enc = self.encoder.lock().expect("parakeet encoder mutex");
            let out = enc.run(vec![
                (std::borrow::Cow::Borrowed("audio_signal"), ort::value::DynValue::from(Tensor::from_array(features)?)),
                (std::borrow::Cow::Borrowed("length"), ort::value::DynValue::from(Tensor::from_array(features_lens)?)),
            ]).map_err(|e| anyhow::anyhow!("parakeet encoder run: {e}"))?;
            let o = out["outputs"].try_extract_tensor::<f32>()?;
            // NeMo export: outputs are [batch, D, T'] -> transpose to [T', D]
            let dims3 = (o.0[0] as usize, o.0[1] as usize, o.0[2] as usize);
            let raw = Array3::from_shape_vec(dims3, o.1.to_vec())?; // [1, D, T']
            let (lshape, ldata) = out["encoded_lengths"].try_extract_tensor::<i64>()?;
            let _ = lshape;
            let enc_len = (ldata[0] as usize).min(raw.shape()[2]);
            let encodings = raw
                .permuted_axes([2usize, 1, 0]) // [T', D, 1]
                .index_axis(ndarray::Axis(2), 0)
                .to_owned(); // [T', D]
            (encodings, enc_len)
        };

        // 3. greedy TDT decode over encoder frames
        let mut tokens: Vec<usize> = Vec::new();
        let mut state = self.zero_state();
        let mut t: usize = 0;
        let mut emitted: usize = 0;
        let mut guard = 0usize;
        let max_steps = enc_len * (MAX_TOKENS_PER_STEP + 1) + 16;

        let mut joint = self.joint.lock().expect("parakeet joint mutex");
        while t < enc_len {
            guard += 1;
            if guard > max_steps {
                break; // safety net; should not trigger
            }
            // encoder frame t: [D] -> [1, D, 1]
            let frame = encodings
                .row(t)
                .to_owned()
                .into_shape((1, encodings.shape()[1], 1))
                .expect("frame shape");
            let prev = if tokens.is_empty() {
                self.blank_idx
            } else {
                tokens[tokens.len() - 1]
            };
            let targets = Array::<i32, Ix2>::from_shape_vec((1, 1), vec![prev as i32])?;
            let target_length = Array::<i32, Ix1>::from_shape_vec((1,), vec![1])?;

            let mut feed: Vec<(std::borrow::Cow<str>, ort::value::DynValue)> = Vec::new();
            feed.push((
                "encoder_outputs".into(),
                ort::value::DynValue::from(Tensor::from_array(frame)?),
            ));
            feed.push((
                "targets".into(),
                ort::value::DynValue::from(Tensor::from_array(targets)?),
            ));
            feed.push((
                "target_length".into(),
                ort::value::DynValue::from(Tensor::from_array(target_length)?),
            ));
            let mut new_states: Vec<(String, Array3<f32>)> = Vec::new();
            for (name, arr) in &state {
                feed.push((
                    std::borrow::Cow::Owned(name.clone()),
                    ort::value::DynValue::from(Tensor::from_array(arr.clone())?),
                ));
            }

            let out = joint.run(feed).map_err(|e| anyhow::anyhow!("parakeet joint run: {e}"))?;
            let (o_shape, o_data) = out["outputs"].try_extract_tensor::<f32>()?;
            let _ = o_shape;
            let logits = o_data.to_vec();
            for name in ["output_states_1", "output_states_2"] {
                let (s, d) = out[name].try_extract_tensor::<f32>()?;
                new_states.push((
                    name.replacen("output_states_", "input_states_", 1),
                    Array3::from_shape_vec((s[0] as usize, s[1] as usize, s[2] as usize), d.to_vec())?,
                ));
            }

            // token logits: [..vocab_size]; duration logits: [vocab_size..]
            let mut token_idx = 0usize;
            let mut best = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().take(self.vocab_size).enumerate() {
                if v > best {
                    best = v;
                    token_idx = i;
                }
            }
            let mut dur_idx = 0usize;
            let mut dbest = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().skip(self.vocab_size).enumerate() {
                if v > dbest {
                    dbest = v;
                    dur_idx = i;
                }
            }
            let step: usize = dur_idx + 1; // duration 0..4 encoded as index

            if token_idx != self.blank_idx {
                state = new_states;
                tokens.push(token_idx);
                emitted += 1;
            }
            if step > 1 {
                // duration index 0 means "same frame again" for parakeet:
                // step>1 (durations 1..4) advance frames
                t += step - 1;
                emitted = 0;
            } else if token_idx == self.blank_idx || emitted >= MAX_TOKENS_PER_STEP {
                t += 1;
                emitted = 0;
            }
        }

        // 4. join tokens (▁ already became spaces)
        let text: String = tokens.iter().map(|&t| self.vocab[t].as_str()).collect();
        Ok(text.trim().to_string())
    }
}
