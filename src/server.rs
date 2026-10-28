//! HTTP + WebSocket server: routes and REST endpoints.

use crate::audio::{f32_to_wav, linear_resample, wav_to_f32};
use crate::config::{Config, STT_SAMPLE_RATE};
use crate::filter::make_speakable;
use crate::protocol::ServerMsg;
use crate::session::{handle_socket, Shared};

use anyhow::Result;
use axum::extract::{State, WebSocketUpgrade};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use std::sync::Arc;

const DEMO_HTML: &str = include_str!("../demo/index.html");

pub async fn run(cfg: Config, state: Shared) -> Result<()> {
    let cfg = Arc::new(cfg);
    let app = Router::new()
        .route("/", get(demo_page))
        .route("/ws", get(ws_upgrade))
        .route("/health", get(health))
        .route("/stt", post(rest_stt))
        .route("/tts", post(rest_tts))
        .with_state((state.clone(), cfg.clone()));

    let addr = format!("{}:{}", cfg.host, cfg.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("voicer listening on http://{addr} (demo: {addr}/ , ws: {addr}/ws)");
    axum::serve(listener, app).await?;
    Ok(())
}

type AppData = (Shared, Arc<Config>);

async fn demo_page() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], DEMO_HTML)
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "ok": true, "stt_sample_rate": STT_SAMPLE_RATE }))
}

async fn ws_upgrade(State((state, _)): State<AppData>, ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(move |socket| handle_socket(state, socket))
}

/// POST /stt — body: WAV file (any sample rate). Response: {"text": "..."}.
async fn rest_stt(State((state, _)): State<AppData>, body: axum::body::Bytes) -> Response {
    let res = tokio::task::spawn_blocking(move || -> Result<String> {
        let (samples, rate) = wav_to_f32(&body)?;
        let samples = linear_resample(&samples, rate, STT_SAMPLE_RATE);
        state.stt.transcribe(&samples)
    })
    .await
    .unwrap_or_else(|e| Err(anyhow::anyhow!("{e}")));

    match res {
        Ok(text) => Json(serde_json::json!({ "text": text })).into_response(),
        Err(e) => error_response(&e),
    }
}

#[derive(Deserialize)]
struct TtsRequest {
    text: String,
    #[serde(default)]
    filter: Option<bool>,
    #[serde(default)]
    speak_filter: Option<bool>, // accepted alias
}

/// POST /tts — JSON {"text": "...", "filter": bool?}. Response: audio/wav.
async fn rest_tts(State((state, cfg)): State<AppData>, Json(req): Json<TtsRequest>) -> Response {
    let filter = req.filter.or(req.speak_filter).unwrap_or_else(|| cfg.default_filter);
    let text = if filter { make_speakable(&req.text) } else { req.text.clone() };
    if text.trim().is_empty() {
        return error_response(&anyhow::anyhow!("nothing speakable in text"));
    }

    let res = {
        let state = state.clone();
        tokio::task::spawn_blocking(move || state.tts.synthesize(&text))
            .await
            .unwrap_or_else(|e| Err(anyhow::anyhow!("{e}")))
    };

    match res {
        Ok((samples, rate)) => match f32_to_wav(&samples, rate) {
            Ok(bytes) => (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "audio/wav")],
                bytes,
            )
                .into_response(),
            Err(e) => error_response(&e),
        },
        Err(e) => error_response(&e),
    }
}

fn error_response(e: &anyhow::Error) -> Response {
    (StatusCode::BAD_REQUEST, ServerMsg::Error { message: e.to_string() }.to_out_text())
        .into_response()
}

impl ServerMsg {
    fn to_out_text(&self) -> String {
        serde_json::to_string(self).expect("serialize ServerMsg")
    }
}
