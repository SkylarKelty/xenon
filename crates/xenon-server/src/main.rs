//! OpenAI-compatible inference server for xenon.
//!
//! A single dedicated OS thread owns the `Engine` and runs a batcher loop:
//! it pulls requests from a tokio mpsc channel, collects more that arrive
//! within a short window (up to `MAX_BATCH`), runs `generate_batch`, and
//! streams tokens back to each request's SSE channel.
//!
//! HTTP handlers tokenize, enqueue a `PendingRequest` carrying a
//! `token_tx` channel, then wrap the matching `token_rx` in an SSE stream
//! (or collect fully for non-streaming).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

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
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use xenon_core::Tokenizer;
use xenon_engine::{BatchRequest, Engine, GEMMA4_EOS};

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
    /// Max requests to pack into a single forward batch.
    #[arg(long, default_value_t = 4)]
    max_batch: usize,
    /// Once a request arrives, wait up to this many milliseconds to
    /// collect more before launching the batch. 0 = never wait.
    #[arg(long, default_value_t = 20)]
    batch_wait_ms: u64,
}

/// A message from the batcher to a waiting handler about one request's output.
enum ResponseChunk {
    /// Incremental text piece (for streaming).
    Delta(String),
    /// Final aggregated text and completion tokens (for non-streaming).
    /// Sent ALONGSIDE deltas, so handlers can use either.
    Final { text: String, completion_tokens: usize },
    /// Generation finished (reason: "stop" or "length").
    Done { finish_reason: &'static str },
}

struct PendingRequest {
    prompt_ids: Vec<u32>,
    max_new: usize,
    /// Batcher-side sender. Closing (drop) signals the batcher the client
    /// went away.
    tx: mpsc::Sender<ResponseChunk>,
}

#[derive(Clone)]
struct AppState {
    tx: mpsc::Sender<PendingRequest>,
    tokenizer: Arc<Tokenizer>,
    model_id: String,
    max_len: usize,
}

fn now_epoch() -> u64 {
    SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}
fn completion_id() -> String { format!("chatcmpl-{}", now_epoch()) }
fn completion_id_legacy() -> String { format!("cmpl-{}", now_epoch()) }

// ---- Batcher thread: owns the engine, drains the request queue ----

fn batcher_loop(
    mut engine: Engine,
    mut rx: mpsc::Receiver<PendingRequest>,
    max_batch: usize,
    batch_wait_ms: u64,
) {
    loop {
        // Wait for at least one request.
        let first = match rx.blocking_recv() {
            Some(r) => r,
            None => { tracing::info!("request channel closed; batcher exiting"); return; }
        };
        let mut batch: Vec<PendingRequest> = vec![first];
        // Collect more that arrive within the wait window, up to max_batch.
        let deadline = Instant::now() + Duration::from_millis(batch_wait_ms);
        while batch.len() < max_batch && Instant::now() < deadline {
            match rx.try_recv() {
                Ok(r) => batch.push(r),
                Err(mpsc::error::TryRecvError::Empty) => std::thread::sleep(Duration::from_millis(1)),
                Err(mpsc::error::TryRecvError::Disconnected) => break,
            }
        }
        run_batch(&mut engine, batch);
    }
}

fn run_batch(engine: &mut Engine, batch: Vec<PendingRequest>) {
    let n = batch.len();
    let (requests, txs): (Vec<BatchRequest>, Vec<mpsc::Sender<ResponseChunk>>) = batch
        .into_iter()
        .map(|p| (BatchRequest { prompt_ids: p.prompt_ids, max_new: p.max_new }, p.tx))
        .unzip();
    tracing::debug!(batch_size = n, "dispatching batch");

    // Per-request incremental-decode state.
    let tok_dec = engine.tokenizer.clone();
    let mut generated: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut prev_decoded: Vec<String> = vec![String::new(); n];
    let mut finished: Vec<bool> = vec![false; n];

    let gen_result = engine.generate_batch(requests, GEMMA4_EOS, |i, tok| {
        if finished[i] { return true; }  // closed early below
        if GEMMA4_EOS.contains(&tok) && !generated[i].is_empty() { return true; }
        generated[i].push(tok);
        if let Ok(s) = tok_dec.decode(&generated[i], /*skip_special*/ true) {
            if s.len() >= prev_decoded[i].len() && s.starts_with(prev_decoded[i].as_str()) {
                let piece = s[prev_decoded[i].len()..].to_string();
                if !piece.is_empty() {
                    // If the client is gone, close the tx so the batcher
                    // stops doing work for it.
                    if txs[i].blocking_send(ResponseChunk::Delta(piece)).is_err() {
                        finished[i] = true;
                    }
                }
                prev_decoded[i] = s;
            }
        }
        true
    });

    // Emit final + done per request. If generate_batch returned an error,
    // every request gets Done("error")-like behavior; the handler already
    // sent whatever deltas it managed to.
    match gen_result {
        Ok(results) => {
            for (i, res) in results.into_iter().enumerate() {
                let content = tok_dec.decode(
                    &res.generated.iter().copied()
                        .filter(|id| !GEMMA4_EOS.contains(id))
                        .collect::<Vec<_>>(),
                    true,
                ).unwrap_or_default();
                let completion_tokens = res.generated.len();
                let _ = txs[i].blocking_send(ResponseChunk::Final { text: content, completion_tokens });
                let finish = if completion_tokens >= 1 && res.generated.last().map_or(false, |t| GEMMA4_EOS.contains(t)) {
                    "stop"
                } else { "length" };
                let _ = txs[i].blocking_send(ResponseChunk::Done { finish_reason: finish });
            }
        }
        Err(e) => {
            tracing::error!("generate_batch error: {e}");
            for tx in &txs {
                let _ = tx.blocking_send(ResponseChunk::Done { finish_reason: "error" });
            }
        }
    }
}

// ---- Chat completions ----

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
struct ChatResponseChoice { index: usize, message: ChatResponseMessage, finish_reason: String }

#[derive(Serialize)]
struct ChatResponseMessage { role: String, content: String }

#[derive(Serialize)]
struct UsageStats { prompt_tokens: usize, completion_tokens: usize, total_tokens: usize }

#[derive(Serialize)]
struct ChatResponse {
    id: String, object: &'static str, created: u64, model: String,
    choices: Vec<ChatResponseChoice>, usage: UsageStats,
}

#[derive(Serialize)]
struct ChatStreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")] role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] content: Option<String>,
}

#[derive(Serialize)]
struct ChatStreamChoice {
    index: usize, delta: ChatStreamDelta,
    #[serde(skip_serializing_if = "Option::is_none")] finish_reason: Option<String>,
}

#[derive(Serialize)]
struct ChatStreamChunk {
    id: String, object: &'static str, created: u64, model: String,
    choices: Vec<ChatStreamChoice>,
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

fn render_messages(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        let role = match m.role.as_str() { "assistant" => "model", other => other };
        out.push_str(&format!("<|turn>{role}\n{}<turn|>\n", m.content));
    }
    out.push_str("<|turn>model\n");
    out
}

/// Common plumbing: enqueue + collect responses.
///
/// Returns the handler-side `mpsc::Receiver<ResponseChunk>` which yields
/// Delta / Final / Done messages from the batcher thread.
async fn enqueue(
    st: &AppState,
    prompt_ids: Vec<u32>,
    max_new: usize,
) -> Result<mpsc::Receiver<ResponseChunk>, (StatusCode, String)> {
    if prompt_ids.len() + max_new > st.max_len {
        return Err((StatusCode::BAD_REQUEST,
            format!("prompt ({}) + max_new ({}) exceeds max_len ({})",
                    prompt_ids.len(), max_new, st.max_len)));
    }
    let (tx, rx) = mpsc::channel::<ResponseChunk>(128);
    st.tx.send(PendingRequest { prompt_ids, max_new, tx }).await
        .map_err(|_| (StatusCode::SERVICE_UNAVAILABLE, "batcher channel closed".into()))?;
    Ok(rx)
}

async fn post_chat_completions(
    State(st): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<axum::response::Response, (StatusCode, String)> {
    if req.messages.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "messages is required".into()));
    }
    let _ = (req.temperature, req.top_p);
    let prompt = render_messages(&req.messages);
    let prompt_ids = st.tokenizer.encode(&prompt, /*add_specials*/ true)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("tokenize: {e}")))?;
    let prompt_len = prompt_ids.len();
    let max_new = req.max_tokens.unwrap_or(256);
    let want_stream = req.stream.unwrap_or(false);

    let rx = enqueue(&st, prompt_ids, max_new).await?;
    if want_stream {
        Ok(chat_stream_from_rx(st.model_id, rx).into_response())
    } else {
        Ok(chat_full_from_rx(st.model_id, rx, prompt_len).await.into_response())
    }
}

async fn chat_full_from_rx(
    model_id: String,
    mut rx: mpsc::Receiver<ResponseChunk>,
    prompt_len: usize,
) -> impl IntoResponse {
    let mut content = String::new();
    let mut completion_tokens = 0usize;
    let mut finish_reason = "stop".to_string();
    while let Some(chunk) = rx.recv().await {
        match chunk {
            ResponseChunk::Delta(_) => {}  // non-streaming ignores partial deltas
            ResponseChunk::Final { text, completion_tokens: ct } => {
                content = text; completion_tokens = ct;
            }
            ResponseChunk::Done { finish_reason: fr } => {
                finish_reason = fr.into();
                break;
            }
        }
    }
    JsonResponse(ChatResponse {
        id: completion_id(),
        object: "chat.completion",
        created: now_epoch(),
        model: model_id,
        choices: vec![ChatResponseChoice {
            index: 0,
            message: ChatResponseMessage { role: "assistant".into(), content },
            finish_reason,
        }],
        usage: UsageStats {
            prompt_tokens: prompt_len,
            completion_tokens,
            total_tokens: prompt_len + completion_tokens,
        },
    })
}

fn chat_stream_from_rx(
    model_id: String,
    rx: mpsc::Receiver<ResponseChunk>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let id = completion_id();
    let model_for_role = model_id.clone();
    let role_chunk = ChatStreamChunk {
        id: id.clone(),
        object: "chat.completion.chunk",
        created: now_epoch(),
        model: model_for_role,
        choices: vec![ChatStreamChoice {
            index: 0,
            delta: ChatStreamDelta { role: Some("assistant".into()), content: None },
            finish_reason: None,
        }],
    };
    let prefix = futures::stream::once(async move {
        Ok::<_, std::convert::Infallible>(
            Event::default().data(serde_json::to_string(&role_chunk).unwrap())
        )
    });

    let id_send = id.clone();
    let body = ReceiverStream::new(rx).filter_map(move |chunk| {
        let id = id_send.clone();
        let model = model_id.clone();
        async move {
            match chunk {
                ResponseChunk::Delta(piece) => Some(Ok(Event::default().data(
                    serde_json::to_string(&ChatStreamChunk {
                        id, object: "chat.completion.chunk", created: now_epoch(), model,
                        choices: vec![ChatStreamChoice {
                            index: 0,
                            delta: ChatStreamDelta { role: None, content: Some(piece) },
                            finish_reason: None,
                        }],
                    }).unwrap()
                ))),
                ResponseChunk::Final { .. } => None,
                ResponseChunk::Done { finish_reason } => Some(Ok(Event::default().data(
                    serde_json::to_string(&ChatStreamChunk {
                        id, object: "chat.completion.chunk", created: now_epoch(), model,
                        choices: vec![ChatStreamChoice {
                            index: 0,
                            delta: ChatStreamDelta { role: None, content: None },
                            finish_reason: Some(finish_reason.into()),
                        }],
                    }).unwrap()
                ))),
            }
        }
    });

    let done = futures::stream::once(async { Ok(Event::default().data("[DONE]")) });
    Sse::new(prefix.chain(body).chain(done)).keep_alive(KeepAlive::default())
}

// ---- Legacy /v1/completions ----

#[derive(Deserialize)]
struct CompletionRequest {
    #[allow(dead_code)]
    model: Option<String>,
    prompt: String,
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
struct CompletionChoice { index: usize, text: String, finish_reason: String }

#[derive(Serialize)]
struct CompletionResponse {
    id: String, object: &'static str, created: u64, model: String,
    choices: Vec<CompletionChoice>, usage: UsageStats,
}

#[derive(Serialize)]
struct CompletionStreamChoice {
    index: usize, text: String,
    #[serde(skip_serializing_if = "Option::is_none")] finish_reason: Option<String>,
}

#[derive(Serialize)]
struct CompletionStreamChunk {
    id: String, object: &'static str, created: u64, model: String,
    choices: Vec<CompletionStreamChoice>,
}

async fn post_completions(
    State(st): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<axum::response::Response, (StatusCode, String)> {
    let _ = (req.temperature, req.top_p);
    let max_new = req.max_tokens.unwrap_or(128);
    let want_stream = req.stream.unwrap_or(false);

    let prompt_ids = st.tokenizer.encode(&req.prompt, true)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("tokenize: {e}")))?;
    let prompt_len = prompt_ids.len();
    let rx = enqueue(&st, prompt_ids, max_new).await?;
    if want_stream {
        Ok(completion_stream_from_rx(st.model_id, rx).into_response())
    } else {
        Ok(completion_full_from_rx(st.model_id, rx, prompt_len).await.into_response())
    }
}

async fn completion_full_from_rx(
    model_id: String,
    mut rx: mpsc::Receiver<ResponseChunk>,
    prompt_len: usize,
) -> impl IntoResponse {
    let mut text = String::new();
    let mut completion_tokens = 0usize;
    let mut finish_reason = "stop".to_string();
    while let Some(chunk) = rx.recv().await {
        match chunk {
            ResponseChunk::Delta(_) => {}
            ResponseChunk::Final { text: t, completion_tokens: ct } => {
                text = t; completion_tokens = ct;
            }
            ResponseChunk::Done { finish_reason: fr } => { finish_reason = fr.into(); break; }
        }
    }
    JsonResponse(CompletionResponse {
        id: completion_id_legacy(),
        object: "text_completion",
        created: now_epoch(),
        model: model_id,
        choices: vec![CompletionChoice { index: 0, text, finish_reason }],
        usage: UsageStats {
            prompt_tokens: prompt_len,
            completion_tokens,
            total_tokens: prompt_len + completion_tokens,
        },
    })
}

fn completion_stream_from_rx(
    model_id: String,
    rx: mpsc::Receiver<ResponseChunk>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let id = completion_id_legacy();
    let id_send = id.clone();
    let body = ReceiverStream::new(rx).filter_map(move |chunk| {
        let id = id_send.clone();
        let model = model_id.clone();
        async move {
            match chunk {
                ResponseChunk::Delta(text) => Some(Ok(Event::default().data(
                    serde_json::to_string(&CompletionStreamChunk {
                        id, object: "text_completion.chunk", created: now_epoch(), model,
                        choices: vec![CompletionStreamChoice { index: 0, text, finish_reason: None }],
                    }).unwrap()
                ))),
                ResponseChunk::Final { .. } => None,
                ResponseChunk::Done { finish_reason } => Some(Ok(Event::default().data(
                    serde_json::to_string(&CompletionStreamChunk {
                        id, object: "text_completion.chunk", created: now_epoch(), model,
                        choices: vec![CompletionStreamChoice {
                            index: 0, text: String::new(), finish_reason: Some(finish_reason.into()),
                        }],
                    }).unwrap()
                ))),
            }
        }
    });

    let done = futures::stream::once(async { Ok(Event::default().data("[DONE]")) });
    Sse::new(body.chain(done)).keep_alive(KeepAlive::default())
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
    let t0 = Instant::now();
    let engine = Engine::load(&args.model, args.max_len)?;
    let tokenizer = Arc::new(engine.tokenizer.clone());
    tracing::info!(
        elapsed_s = t0.elapsed().as_secs_f64(),
        n_layers = engine.shape.n_layers,
        hidden = engine.shape.hidden,
        vocab = engine.shape.vocab,
        max_len = args.max_len,
        max_batch = args.max_batch,
        batch_wait_ms = args.batch_wait_ms,
        "engine ready",
    );

    let model_id = args.model.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "xenon-gemma4".into());

    // Spawn the batcher on a dedicated OS thread; it owns the engine for
    // the rest of the process lifetime.
    let (req_tx, req_rx) = mpsc::channel::<PendingRequest>(64);
    let max_batch = args.max_batch;
    let batch_wait_ms = args.batch_wait_ms;
    std::thread::spawn(move || batcher_loop(engine, req_rx, max_batch, batch_wait_ms));

    let state = AppState {
        tx: req_tx, tokenizer, model_id, max_len: args.max_len,
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(get_models))
        .route("/v1/chat/completions", post(post_chat_completions))
        .route("/v1/completions", post(post_completions))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    tracing::info!(addr = %args.bind, "serving");
    axum::serve(listener, app).await?;
    Ok(())
}
