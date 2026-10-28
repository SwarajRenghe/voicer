//! Speech recognition using Moonshine tiny (ONNX), English-only.
//!
//! Moonshine is an encoder-decoder model that takes raw 16 kHz waveform
//! (no mel features) and autoregressively emits BPE tokens. We run it
//! greedily with a KV cache:
//!
//!   step 1: `decoder_model`            (input_ids, encoder_hidden_states)
//!   step n: `decoder_with_past_model`  (+ past_key_values.* from the previous step)
//!
//! Model files (from onnx-community/moonshine-tiny-ONNX):
//!   encoder_model_quantized.onnx        in:  input_values [1, N] f32
//!                                       out: last_hidden_state [1, S, 288] f32
//!   decoder_model_quantized.onnx        in:  input_ids [1, 1] i64, encoder_hidden_states
//!                                       out: logits [1, 1, 32000] f32, present.*
//!   decoder_with_past_model_quantized.onnx
//!                                       in:  + past_key_values.{n}.{decoder,encoder}.{key,value}
//!   tokenizer.json                      BPE tokenizer (BOS=1, EOS=2)

use crate::ortutil::{make_session, value_to_f32_array};
use anyhow::{Context, Result};
use ndarray::{Array, Array2, ArrayD, Ix2};
use ort::session::Session;
use ort::value::Tensor;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

const BOS: i64 = 1;
const EOS: i64 = 2;
const MAX_NEW_TOKENS: usize = 448;
const MAX_AUDIO_SAMPLES: usize = 16000 * 30; // model is trained for <= ~30 s clips

pub struct Stt {
    // ort sessions need `&mut` to run; the mutex only serializes access,
    // inference itself stays on blocking threads
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    decoder_past: Mutex<Session>,
    tokenizer: Tokenizer,
    encoder_input: String,
    encoder_output: String,
    decoder_ids_input: String,
    decoder_ctx_input: String,
    decoder_past_ids_input: String,
    /// KV-cache input names on the with-past decoder, e.g. "past_key_values.0.decoder.key"
    past_inputs: Vec<String>,
}

impl Stt {
    pub fn load(dir: &Path) -> Result<Self> {
        let encoder = make_session(&dir.join("encoder_model_quantized.onnx"), 4)?;
        let decoder = make_session(&dir.join("decoder_model_quantized.onnx"), 4)?;
        let decoder_past = make_session(&dir.join("decoder_with_past_model_quantized.onnx"), 4)?;
        let tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;

        let name_of = |s: &Session, want: &str| -> Result<String> {
            let names: Vec<String> = s.inputs().iter().map(|i| i.name().to_string()).collect();
            let plain = |n: &String| {
                !n.contains("quant") && !n.contains("scale") && !n.contains("zero_point")
            };
            names
                .iter()
                .find(|n| n.as_str() == want)
                .or_else(|| names.iter().find(|n| n.starts_with(want) && plain(n)))
                .or_else(|| names.iter().find(|n| n.contains(want) && plain(n)))
                .cloned()
                .with_context(|| format!("input `{want}` not found; have {names:?}"))
        };
        let encoder_input = encoder.inputs()[0].name().to_string();
        let encoder_output = encoder.outputs()[0].name().to_string();

        Ok(Self {
            decoder_ctx_input: name_of(&decoder, "encoder_hidden_states")?,
            decoder_ids_input: name_of(&decoder, "input_ids")?,
            decoder_past_ids_input: name_of(&decoder_past, "input_ids")?,
            past_inputs: decoder_past
                .inputs()
                .iter()
                .map(|i| i.name().to_string())
                .filter(|n| n.starts_with("past_key_values."))
                .collect(),
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            decoder_past: Mutex::new(decoder_past),
            tokenizer,
            encoder_input,
            encoder_output,
        })
    }

    /// Transcribe 16 kHz mono f32 audio. Empty/short input yields "".
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        if samples.len() < 480 {
            return Ok(String::new()); // < 30 ms: nothing to transcribe
        }
        let samples = if samples.len() > MAX_AUDIO_SAMPLES {
            &samples[..MAX_AUDIO_SAMPLES]
        } else {
            samples
        };

        // 1. encode raw waveform
        let audio = Array::<f32, Ix2>::from_shape_vec((1, samples.len()), samples.to_vec())?;
        let mut encoder = self.encoder.lock().expect("stt encoder mutex");
        let enc_out = encoder.run(vec![(
            self.encoder_input.clone(),
            Tensor::from_array(audio)?,
        )])?;
        let hidden: ArrayD<f32> =
            value_to_f32_array(&enc_out[self.encoder_output.as_str()])?;

        // 2. greedy decode with KV cache
        let mut tokens: Vec<i64> = vec![BOS];
        let mut past: HashMap<String, ArrayD<f32>> = HashMap::new();

        // extract logits + KV cache while the session borrow is alive
        let extract = |out: &ort::session::SessionOutputs| -> Result<(Vec<f32>, HashMap<String, ArrayD<f32>>)> {
            let (_, logits) = out["logits"].try_extract_tensor::<f32>()?;
            let mut present = HashMap::new();
            for (name, value) in out.iter() {
                if name.starts_with("present.") {
                    let (shape, data) = value.try_extract_tensor::<f32>()?;
                    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                    present.insert(
                        name.to_string(),
                        Array::from_shape_vec(dims, data.to_vec())?.into_dyn(),
                    );
                }
            }
            Ok((logits.to_vec(), present))
        };

        for step in 0..MAX_NEW_TOKENS {
            let last = tokens.last().copied().unwrap_or(BOS);
            let ids = Array2::from_shape_vec((1, 1), vec![last])?;

            let (logits, present) = if step == 0 {
                let mut decoder = self.decoder.lock().expect("stt decoder mutex");
                let out = decoder.run(vec![
                    (
                        std::borrow::Cow::Owned(self.decoder_ids_input.clone()),
                        ort::value::DynValue::from(Tensor::from_array(ids)?),
                    ),
                    (
                        std::borrow::Cow::Owned(self.decoder_ctx_input.clone()),
                        ort::value::DynValue::from(Tensor::from_array(hidden.clone())?),
                    ),
                ])?;
                extract(&out)?
            } else {
                let mut feed: Vec<(std::borrow::Cow<str>, ort::value::DynValue)> = Vec::new();
                feed.push((
                    std::borrow::Cow::Owned(self.decoder_past_ids_input.clone()),
                    ort::value::DynValue::from(Tensor::from_array(ids)?),
                ));
                for name in &self.past_inputs {
                    let key = name.replacen("past_key_values.", "present.", 1);
                    let arr = past.get(&key).context("missing KV cache entry")?;
                    feed.push((
                        std::borrow::Cow::Owned(name.clone()),
                        ort::value::DynValue::from(Tensor::from_array(arr.clone())?),
                    ));
                }
                let mut decoder_past = self.decoder_past.lock().expect("stt decoder_past mutex");
                let out = decoder_past.run(feed)?;
                extract(&out)?
            };

            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or(0) as i64;

            if next == EOS {
                break;
            }
            tokens.push(next);
            // merge: the with-past decoder only re-emits decoder.* KV entries;
            // encoder.* (cross-attention) entries stay valid for the whole decode
            for (k, v) in present {
                past.insert(k, v);
            }
        }

        let out_ids: Vec<u32> = tokens
            .into_iter()
            .skip(1) // drop BOS
            .filter(|&t| t > 0 && t < 32000)
            .map(|t| t as u32)
            .collect();
        let text = self
            .tokenizer
            .decode(&out_ids, true) // skip special tokens
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(text.trim().to_string())
    }
}
