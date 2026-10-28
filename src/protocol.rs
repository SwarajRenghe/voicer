//! Wire protocol types shared by the WebSocket endpoint.
//!
//! Text frames are JSON messages (see `ClientMsg` / `ServerMsg`).
//! Binary frames carry raw PCM:
//!   client -> server : 16-bit signed little-endian, mono, 16 kHz
//!   server -> client : 16-bit signed little-endian, mono, voice sample rate (sent in `audio_start`)

use serde::{Deserialize, Serialize};

/// Messages sent by a client to voicer.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMsg {
    /// Adjust per-connection options. Any omitted field keeps its current value.
    Config {
        /// Strip markdown / code / tool-output noise before speaking.
        speak_filter: Option<bool>,
        /// How often (ms) to emit `partial` transcripts while the user is speaking.
        partial_interval_ms: Option<u64>,
    },
    /// Speak some text. May be called repeatedly; sentences are synthesized and
    /// streamed in order. `filter` overrides the connection's speak_filter.
    Speak {
        id: u32,
        text: String,
        #[serde(default)]
        filter: Option<bool>,
    },
    /// Force synthesis of any buffered partial sentence (use after streaming
    /// several `speak` messages to flush a trailing incomplete sentence).
    Flush,
    /// Immediately stop all pending TTS audio (barge-in).
    Cancel,
    Ping,
}

/// Messages sent by voicer to a client.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMsg {
    /// Sent once after connection; announces capabilities.
    Ready {
        /// Sample rate of TTS audio voicer will stream.
        tts_sample_rate: u32,
        /// Expected sample rate of client audio.
        stt_sample_rate: u32,
    },
    /// User started speaking (VAD).
    SpeechStart,
    /// User stopped speaking.
    SpeechEnd,
    /// Interim transcript while the user is still speaking.
    Partial { text: String },
    /// Final transcript of the completed utterance.
    Final { text: String },
    /// A sentence's audio is about to stream (binary frames follow).
    AudioStart { id: u32, sentence: u32, sample_rate: u32 },
    /// The sentence's audio finished streaming.
    AudioEnd { id: u32, sentence: u32 },
    /// Everything queued for `id` has been sent.
    TtsDone { id: u32 },
    /// Server-side error.
    Error { message: String },
    Pong,
}

/// Outbound channel items: either a JSON message or a raw PCM frame.
#[derive(Debug)]
pub enum Out {
    Text(String),
    Audio(Vec<u8>),
}

impl ServerMsg {
    pub fn to_out(&self) -> Out {
        Out::Text(serde_json::to_string(self).expect("serialize ServerMsg"))
    }
}
