//! Per-connection session: wires the WebSocket to the VAD -> STT pipeline
//! (inbound) and the sentence-chunked TTS queue (outbound).

use crate::audio::{f32_to_i16_bytes, i16_bytes_to_f32};
use crate::config::{Config, STT_SAMPLE_RATE, VAD_CHUNK};
use crate::filter::make_speakable;
use crate::protocol::{ClientMsg, Out, ServerMsg};
use crate::sentence::SentenceSplitter;
use crate::stt::Stt;
use crate::tts::Tts;
use crate::vad::Vad;

use axum::extract::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// State shared across all connections (models are thread-safe).
#[derive(Clone)]
pub struct Shared {
    pub stt: Arc<Stt>,
    pub tts: Arc<Tts>,
    pub cfg: Arc<Config>,
}

const AUDIO_CHUNK: usize = 8192; // samples per outbound TTS frame

/// Commands for the TTS worker.
enum TtsCmd {
    Speak { id: u32, text: String, filter: bool },
    Flush,
}

fn send(tx: &mpsc::UnboundedSender<Out>, msg: &ServerMsg) {
    let _ = tx.send(msg.to_out());
}

pub async fn handle_socket(state: Shared, socket: WebSocket) {
    let (mut sink, mut stream) = socket.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<Out>();

    // pump: forwards outbound items to the socket
    let pump = tokio::spawn(async move {
        while let Some(out) = rx.recv().await {
            let msg = match out {
                Out::Text(t) => Message::Text(t.into()),
                Out::Audio(b) => Message::Binary(b.into()),
            };
            if sink.send(msg).await.is_err() {
                break;
            }
        }
    });

    send(&tx, &ServerMsg::Ready {
        tts_sample_rate: state.tts.sample_rate,
        stt_sample_rate: STT_SAMPLE_RATE,
    });

    // TTS worker task; `tts_cancel` is the barge-in flag
    let (tts_tx, tts_rx) = mpsc::unbounded_channel::<TtsCmd>();
    let tts_cancel = spawn_tts_worker(state.clone(), tts_rx, tx.clone());

    // inbound audio pipeline state
    let mut vad = match Vad::load(&state.cfg.models_dir.join("vad").join("silero_vad.onnx")) {
        Ok(v) => v,
        Err(e) => {
            send(&tx, &ServerMsg::Error { message: format!("VAD load failed: {e}") });
            return;
        }
    };
    let mut speak_filter = state.cfg.default_filter;
    let mut partial_interval = Duration::from_millis(state.cfg.partial_interval_ms);
    let mut buf: Vec<f32> = Vec::new(); // leftover samples not yet chunked
    let mut prebuf: Vec<f32> = Vec::new(); // rolling pre-roll before speech
    let pre_roll =
        (state.cfg.pre_roll_ms as usize * STT_SAMPLE_RATE as usize / 1000).min(VAD_CHUNK * 8);
    let end_chunks = (state.cfg.speech_end_ms / 32).max(1) as u32;
    let max_segment = state.cfg.max_segment_secs as usize * STT_SAMPLE_RATE as usize;
    let gen = Arc::new(AtomicU64::new(0)); // invalidates stale partial transcripts

    let mut l = Listening {
        speaking: false,
        silence_chunks: 0,
        segment: Vec::new(),
        partial_due: Instant::now(),
        partial_in_flight: false,
    };

    while let Some(frame) = stream.next().await {
        let Ok(msg) = frame else { break };
        match msg {
            Message::Binary(data) => {
                buf.extend(i16_bytes_to_f32(&data));
                while buf.len() >= VAD_CHUNK {
                    let chunk: Vec<f32> = buf.drain(..VAD_CHUNK).collect();
                    process_chunk(
                        &chunk, &state, &tx, &mut vad, &mut l, &mut prebuf, pre_roll, end_chunks,
                        max_segment, &partial_interval, &gen,
                    );
                }
            }
            Message::Text(text) => match serde_json::from_str::<ClientMsg>(&text) {
                Ok(ClientMsg::Config { speak_filter: sf, partial_interval_ms }) => {
                    if let Some(f) = sf {
                        speak_filter = f;
                    }
                    if let Some(ms) = partial_interval_ms {
                        partial_interval = Duration::from_millis(ms.max(100));
                    }
                }
                Ok(ClientMsg::Speak { id, text, filter }) => {
                    let use_filter = filter.unwrap_or(speak_filter);
                    let _ = tts_tx.send(TtsCmd::Speak { id, text, filter: use_filter });
                }
                Ok(ClientMsg::Flush) => {
                    let _ = tts_tx.send(TtsCmd::Flush);
                }
                Ok(ClientMsg::Cancel) => tts_cancel.store(true, Ordering::Relaxed),
                Ok(ClientMsg::Ping) => send(&tx, &ServerMsg::Pong),
                Err(e) => send(&tx, &ServerMsg::Error { message: format!("bad message: {e}") }),
            },
            _ => {} // pongs/close handled by axum
        }
    }

    drop(tx); // stops the pump
    let _ = pump.await;
}

struct Listening {
    speaking: bool,
    silence_chunks: u32,
    segment: Vec<f32>,
    partial_due: Instant,
    partial_in_flight: bool,
}

#[allow(clippy::too_many_arguments)]
fn process_chunk(
    chunk: &[f32],
    state: &Shared,
    tx: &mpsc::UnboundedSender<Out>,
    vad: &mut Vad,
    l: &mut Listening,
    prebuf: &mut Vec<f32>,
    pre_roll: usize,
    end_chunks: u32,
    max_segment: usize,
    partial_interval: &Duration,
    gen: &Arc<AtomicU64>,
) {
    let prob = match vad.process(chunk) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("vad error: {e}");
            return;
        }
    };

    if !l.speaking {
        // maintain pre-roll so word onsets aren't clipped
        prebuf.extend_from_slice(chunk);
        let excess = prebuf.len().saturating_sub(pre_roll);
        prebuf.drain(..excess);
        if prob >= state.cfg.speech_start_threshold {
            l.speaking = true;
            l.silence_chunks = 0;
            l.segment = prebuf.clone();
            l.segment.extend_from_slice(chunk);
            prebuf.clear();
            l.partial_due = Instant::now() + *partial_interval;
            send(tx, &ServerMsg::SpeechStart);
        }
        return;
    }

    // speaking
    l.segment.extend_from_slice(chunk);
    if prob < state.cfg.speech_end_threshold {
        l.silence_chunks += 1;
    } else {
        l.silence_chunks = 0;
    }

    let utterance_over = l.silence_chunks >= end_chunks;
    let too_long = l.segment.len() >= max_segment;
    if utterance_over || too_long {
        l.speaking = false;
        vad.reset();
        send(tx, &ServerMsg::SpeechEnd);
        finalize(&l.segment, state, tx, gen);
        l.segment.clear();
        l.partial_in_flight = false;
        return;
    }

    // interim transcripts while still speaking
    if !l.partial_in_flight
        && Instant::now() >= l.partial_due
        && l.segment.len() >= STT_SAMPLE_RATE as usize / 2
    {
        l.partial_due = Instant::now() + *partial_interval;
        l.partial_in_flight = true;
        spawn_transcribe(state, tx, gen, l.segment.clone(), false);
    }
}

/// Kick off a blocking STT run; sends `final` (or `partial`) when done.
fn spawn_transcribe(
    state: &Shared,
    tx: &mpsc::UnboundedSender<Out>,
    gen: &Arc<AtomicU64>,
    samples: Vec<f32>,
    is_final: bool,
) {
    let seq = gen.load(Ordering::Relaxed);
    let stt = state.stt.clone();
    let tx = tx.clone();
    let gen = gen.clone();
    tokio::task::spawn_blocking(move || {
        let text = stt.transcribe(&samples).unwrap_or_default();
        if gen.load(Ordering::Relaxed) == seq {
            let msg = if is_final {
                ServerMsg::Final { text }
            } else {
                ServerMsg::Partial { text }
            };
            send(&tx, &msg);
        }
    });
}

fn finalize(segment: &[f32], state: &Shared, tx: &mpsc::UnboundedSender<Out>, gen: &Arc<AtomicU64>) {
    gen.fetch_add(1, Ordering::Relaxed); // invalidate in-flight partials
    spawn_transcribe(state, tx, gen, segment.to_vec(), true);
}

// ---------------------------------------------------------------- TTS worker

/// Runs the TTS queue for one connection: splits text into sentences,
/// synthesizes on blocking threads and streams audio frames out.
/// Returns the barge-in cancel flag the connection can set at any time.
fn spawn_tts_worker(
    state: Shared,
    mut cmd_rx: mpsc::UnboundedReceiver<TtsCmd>,
    tx: mpsc::UnboundedSender<Out>,
) -> Arc<AtomicBool> {
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_in_worker = cancel.clone();

    tokio::spawn(async move {
        let cancel = cancel_in_worker;
        let mut splitter = SentenceSplitter::new();
        let mut current_id: Option<u32> = None;
        let mut sentence_idx: u32 = 0;

        while let Some(cmd) = cmd_rx.recv().await {
            // a new Speak implicitly resumes after a cancel
            if let TtsCmd::Speak { .. } = cmd {
                splitter = SentenceSplitter::new();
                sentence_idx = 0;
                cancel.store(false, Ordering::Relaxed);
            }

            match cmd {
                TtsCmd::Speak { id, text, filter } => {
                    current_id = Some(id);
                    let text = if filter { make_speakable(&text) } else { text };
                    for sentence in splitter.push(&text) {
                        if cancel.load(Ordering::Relaxed) {
                            break;
                        }
                        speak_sentence(&state, &tx, &cancel, id, sentence_idx, &sentence).await;
                        sentence_idx += 1;
                    }
                }
                TtsCmd::Flush => {
                    if let (Some(id), Some(sentence)) = (current_id, splitter.flush()) {
                        if !cancel.load(Ordering::Relaxed) {
                            speak_sentence(&state, &tx, &cancel, id, sentence_idx, &sentence).await;
                            sentence_idx += 1;
                        }
                    }
                }
            }

            // queue drained -> announce completion for the current id
            if cmd_rx.try_recv().is_err() {
                if let Some(id) = current_id {
                    send(&tx, &ServerMsg::TtsDone { id });
                }
            }
        }
    });

    cancel
}

async fn speak_sentence(
    state: &Shared,
    tx: &mpsc::UnboundedSender<Out>,
    cancel: &AtomicBool,
    id: u32,
    idx: u32,
    sentence: &str,
) {
    let tts = state.tts.clone();
    let text = sentence.to_string();
    let res = tokio::task::spawn_blocking(move || tts.synthesize(&text)).await;
    let Ok(Ok((samples, rate))) = res else { return };
    if cancel.load(Ordering::Relaxed) {
        return;
    }
    send(tx, &ServerMsg::AudioStart { id, sentence: idx, sample_rate: rate });
    for chunk in samples.chunks(AUDIO_CHUNK) {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        if tx.send(Out::Audio(f32_to_i16_bytes(chunk))).is_err() {
            return;
        }
    }
    send(tx, &ServerMsg::AudioEnd { id, sentence: idx });
}
