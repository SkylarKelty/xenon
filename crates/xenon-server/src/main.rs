//! OpenAI-compatible inference server for xenon.
//!
//! Single engine, single model, serialized request handling (Mutex<Engine>).
//! The generate loop runs on a blocking thread (`tokio::task::spawn_blocking`)
//! because it does synchronous CUDA work; tokens stream back to the HTTP
//! handler via a tokio channel.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::{Event, KeepAlive, Sse}, IntoResponse, Json as JsonResponse},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use futures::stream::Stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio_stream::wrappers::ReceiverStream;
use xenon_engine::{Engine, GEMMA4_EOS};

#[derive(Parser, Debug)]
#[command(name = "xenon-server", about = "OpenAI-compatible inference server for xenon")]
struct Args {
    /// Path to the model snapshot directory (HF cache layout).
    #[arg(long)]
    model: PathBuf,
    /// Address to bind on.
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: String,
    /// KV-cache max sequence length (prompt + generated).
    #[arg(long, default_value_t = 4096)]
    max_len: usize,
}

#[derive(Clone)]
struct AppState {
    engine: Arc<Mutex<Engine>>,
    model_id: String,
    max_len: usize,
}

// ---- OpenAI request / response shapes ----

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatRequest {
    #[allow(dead_code)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
}

#[derive(Serialize)]
struct ChatResponseChoice {
    index: usize,
    message: ChatResponseMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct ChatResponseMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct UsageStats {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct ChatResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatResponseChoice>,
    usage: UsageStats,
}

#[derive(Serialize)]
struct ChatStreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Serialize)]
struct ChatStreamChoice {
    index: usize,
    delta: ChatStreamDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct ChatStreamChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatStreamChoice>,
}

fn tok_clone_decode(tok: &xenon_core::Tokenizer, ids: &[u32], skip_specials: bool)
    -> Result<String, xenon_core::Error>
{
    tok.decode(ids, skip_specials)
}

fn now_epoch() -> u64 {
    SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn completion_id() -> String {
    format!("chatcmpl-{}", now_epoch())
}

async fn get_models(State(st): State<AppState>) -> impl IntoResponse {
    JsonResponse(serde_json::json!({
        "object": "list",
        "data": [{
            "id": st.model_id,
            "object": "model",
            "owned_by": "xenon",
            "created": now_epoch(),
        }],
    }))
}

/// Flatten OpenAI messages into Gemma 4's chat-template string.
fn render_messages(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        let role = match m.role.as_str() {
            "assistant" => "model",
            other => other,
        };
        out.push_str(&format!("<|turn>{role}\n{}<turn|>\n", m.content));
    }
    out.push_str("<|turn>model\n");
    out
}

async fn post_chat_completions(
    State(st): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<axum::response::Response, (StatusCode, String)> {
    if req.messages.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "messages is required".into()));
    }
    let _ = (req.temperature, req.top_p); // greedy only for now
    let prompt_text = render_messages(&req.messages);
    let max_new = req.max_tokens.unwrap_or(256);
    let want_stream = req.stream.unwrap_or(false);

    let prompt_ids = {
        let eng = st.engine.lock().await;
        eng.tokenize(&prompt_text, /*add_specials*/ true)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("tokenize: {e}")))?
    };
    if prompt_ids.len() + max_new > st.max_len {
        return Err((StatusCode::BAD_REQUEST,
            format!("prompt ({}) + max_new ({}) exceeds max_len ({})",
                    prompt_ids.len(), max_new, st.max_len)));
    }

    if want_stream {
        Ok(chat_stream_response(st, prompt_ids, max_new).into_response())
    } else {
        Ok(chat_full_response(st, prompt_ids, max_new).await?.into_response())
    }
}

async fn chat_full_response(
    st: AppState,
    prompt_ids: Vec<u32>,
    max_new: usize,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let engine = st.engine.clone();
    let model_id = st.model_id.clone();
    let prompt_len = prompt_ids.len();
    let (text, completion_tokens) = tokio::task::spawn_blocking(move || -> anyhow::Result<(String, usize)> {
        let mut eng = engine.blocking_lock();
        let mut out_ids: Vec<u32> = Vec::with_capacity(max_new);
        let _stats = eng.generate(&prompt_ids, max_new, GEMMA4_EOS, |tok| {
            out_ids.push(tok);
            true
        })?;
        let decode_ids: Vec<u32> = out_ids.iter().copied()
            .filter(|id| !GEMMA4_EOS.contains(id)).collect();
        let text = eng.decode(&decode_ids, /*skip_specials*/ true)?;
        Ok((text, out_ids.len()))
    }).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("join: {e}")))?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("generate: {e}")))?;

    let resp = ChatResponse {
        id: completion_id(),
        object: "chat.completion",
        created: now_epoch(),
        model: model_id,
        choices: vec![ChatResponseChoice {
            index: 0,
            message: ChatResponseMessage { role: "assistant".into(), content: text },
            finish_reason: "stop".into(),
        }],
        usage: UsageStats {
            prompt_tokens: prompt_len,
            completion_tokens,
            total_tokens: prompt_len + completion_tokens,
        },
    };
    Ok(JsonResponse(resp))
}

fn chat_stream_response(
    st: AppState,
    prompt_ids: Vec<u32>,
    max_new: usize,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<ChatStreamChunk>(64);
    let engine = st.engine.clone();
    let model_id = st.model_id.clone();
    let id = completion_id();

    let id_send = id.clone();
    let model_send = model_id.clone();
    tokio::task::spawn_blocking(move || {
        let mut eng = engine.blocking_lock();
        // Clone the tokenizer so the per-token closure can decode without
        // reborrowing the engine that generate() holds mutably.
        let tok_dec = eng.tokenizer.clone();
        let _ = tx.blocking_send(ChatStreamChunk {
            id: id_send.clone(),
            object: "chat.completion.chunk",
            created: now_epoch(),
            model: model_send.clone(),
            choices: vec![ChatStreamChoice {
                index: 0,
                delta: ChatStreamDelta { role: Some("assistant".into()), content: None },
                finish_reason: None,
            }],
        });

        // Incremental decode: after each new token, decode the accumulated
        // ids and emit the diff so subword pieces assemble into valid UTF-8.
        let mut generated: Vec<u32> = Vec::new();
        let mut prev_decoded = String::new();
        let _ = eng.generate(&prompt_ids, max_new, GEMMA4_EOS, |tok| {
            if GEMMA4_EOS.contains(&tok) && !generated.is_empty() {
                return true; // generate() will break on the next iteration
            }
            generated.push(tok);
            if let Ok(s) = tok_clone_decode(&tok_dec, &generated, true) {
                if s.len() >= prev_decoded.len() && s.starts_with(prev_decoded.as_str()) {
                    let piece = s[prev_decoded.len()..].to_string();
                    if !piece.is_empty() {
                        let chunk = ChatStreamChunk {
                            id: id_send.clone(),
                            object: "chat.completion.chunk",
                            created: now_epoch(),
                            model: model_send.clone(),
                            choices: vec![ChatStreamChoice {
                                index: 0,
                                delta: ChatStreamDelta { role: None, content: Some(piece) },
                                finish_reason: None,
                            }],
                        };
                        if tx.blocking_send(chunk).is_err() {
                            return false; // client disconnected
                        }
                    }
                    prev_decoded = s;
                }
            }
            true
        });
        let _ = tx.blocking_send(ChatStreamChunk {
            id: id_send,
            object: "chat.completion.chunk",
            created: now_epoch(),
            model: model_send,
            choices: vec![ChatStreamChoice {
                index: 0,
                delta: ChatStreamDelta { role: None, content: None },
                finish_reason: Some("stop".into()),
            }],
        });
    });

    let base = ReceiverStream::new(rx)
        .map(|chunk| Ok::<_, std::convert::Infallible>(
            Event::default().data(serde_json::to_string(&chunk).unwrap())
        ));
    let done = futures::stream::once(async { Ok(Event::default().data("[DONE]")) });
    Sse::new(base.chain(done)).keep_alive(KeepAlive::default())
}

async fn health() -> &'static str { "ok" }

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")))
        .init();

    let args = Args::parse();

    tracing::info!(path = %args.model.display(), "loading model");
    let t0 = std::time::Instant::now();
    let engine = Engine::load(&args.model, args.max_len)?;
    tracing::info!(
        elapsed_s = t0.elapsed().as_secs_f64(),
        n_layers = engine.shape.n_layers,
        hidden = engine.shape.hidden,
        vocab = engine.shape.vocab,
        max_len = args.max_len,
        "engine ready",
    );

    let model_id = args.model.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "xenon-gemma4".into());
    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
        model_id,
        max_len: args.max_len,
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(get_models))
        .route("/v1/chat/completions", post(post_chat_completions))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    tracing::info!(addr = %args.bind, "serving");
    axum::serve(listener, app).await?;
    Ok(())
}
