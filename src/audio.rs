//! Small audio helpers: sample format conversion, naive resampling, WAV codecs.

use anyhow::{Context, Result};
use std::io::Cursor;

pub fn i16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect()
}

pub fn f32_to_i16_bytes(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Linear resampler. Good enough for 44.1k/48k -> 16k speech input.
pub fn linear_resample(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let out_len = (samples.len() as f64 / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 * ratio;
        let i0 = pos.floor() as usize;
        let i1 = (i0 + 1).min(samples.len() - 1);
        let frac = (pos - i0 as f64) as f32;
        out.push(samples[i0] * (1.0 - frac) + samples[i1] * frac);
    }
    out
}

/// Decode a WAV file (float or int PCM, any channel count) to mono f32.
pub fn wav_to_f32(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    let reader = hound::WavReader::new(Cursor::new(bytes)).context("invalid WAV")?;
    let spec = reader.spec();
    let rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let all: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
        hound::SampleFormat::Int => {
            let denom = match spec.bits_per_sample {
                8 => 128.0,
                16 => 32768.0,
                24 => 8388608.0,
                _ => 2147483648.0,
            };
            reader.into_samples::<i32>().filter_map(|s| s.ok()).map(|s| s as f32 / denom).collect()
        }
    };

    let mono: Vec<f32> = if channels <= 1 {
        all
    } else {
        all.chunks(channels)
            .map(|c| c.iter().sum::<f32>() / channels as f32)
            .collect()
    };
    Ok((mono, rate))
}

/// Encode mono f32 samples as a 16-bit PCM WAV file (44-byte header, written
/// manually — hound has no way to hand back the writer's buffer).
pub fn f32_to_wav(samples: &[f32], rate: u32) -> Result<Vec<u8>> {
    let data_len = samples.len() * 2;
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    out.extend_from_slice(&f32_to_i16_bytes(samples));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_i16() {
        let f = vec![0.0, 0.5, -0.5, 1.0];
        let b = f32_to_i16_bytes(&f);
        assert_eq!(b.len(), 8);
        let back = i16_bytes_to_f32(&b);
        assert!((back[1] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn resample_identity_and_downsample() {
        let x: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        assert_eq!(linear_resample(&x, 16000, 16000), x);
        let down = linear_resample(&x, 32000, 16000);
        assert_eq!(down.len(), 50);
    }
}
