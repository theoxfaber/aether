//! OpenAI-compatible HTTP server for private on-prem LLM deployments.
//!
//! Environment variables (override CLI defaults):
//!   AETHER_MODEL_PATH   — path to GGUF model file
//!   AETHER_API_KEY      — Bearer token required on `/v1/*`
//!   AETHER_HOST         — bind address (default 127.0.0.1)
//!   AETHER_PORT         — listen port (default 8080)
//!   AETHER_CPU_ONLY     — `1`/`true` force CPU inference (default: true)
//!   AETHER_MAX_TOKENS   — cap per request (default 512)
//!   AETHER_RATE_LIMIT   — requests/minute (default: 60, 0 = off)
//!   RUST_LOG            — tracing filter (default `aether=info,aether_server=info`)

#![allow(dead_code)]

use axum::{
    extract::{Request, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};

use clap::Parser;
use futures::StreamExt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex as TokioMutex, Semaphore};
use tracing::{error, info, warn};

use aether::inference::runner::{sample, LlamaRunner, LoadOptions};

// ── CLI ────────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "aether-server",
    about = "OpenAI-compatible private LLM server (Aether)"
)]
struct Args {
    #[arg(short, long, env = "AETHER_MODEL_PATH", default_value = "model.gguf")]
    model: String,

    #[arg(long, env = "AETHER_HOST", default_value = "127.0.0.1")]
    host: String,

    #[arg(short, long, env = "AETHER_PORT", default_value_t = 8080)]
    port: u16,

    #[arg(long, env = "AETHER_API_KEY")]
    api_key: Option<String>,

    #[arg(long, env = "AETHER_CPU_ONLY", default_value_t = true, action = clap::ArgAction::Set)]
    cpu_only: bool,

    #[arg(long, env = "AETHER_MAX_TOKENS", default_value_t = 512)]
    max_tokens: usize,

    #[arg(long, env = "AETHER_RATE_LIMIT", default_value_t = 60)]
    rate_limit: u32,

    #[arg(long, env = "AETHER_MAX_CONCURRENCY", default_value_t = 1)]
    max_concurrency: usize,
}

// ─── OpenAI-compatible types ──────────────────────────────────────────────────

#[derive(serde::Deserialize, Clone, Debug)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
    #[serde(default = "default_repetition_penalty")]
    repetition_penalty: f32,
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
struct Message {
    role: String,
    content: String,
}

#[derive(serde::Deserialize, Clone, Debug)]
struct CompletionRequest {
    model: String,
    prompt: String,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: bool,
    #[serde(default = "default_repetition_penalty")]
    repetition_penalty: f32,
}

#[derive(serde::Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: ChatUsage,
}

#[derive(serde::Serialize)]
struct ChatChoice {
    index: usize,
    message: Message,
    finish_reason: Option<String>,
}

#[derive(serde::Serialize)]
struct ChatUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(serde::Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChunkChoice>,
}

#[derive(serde::Serialize)]
struct ChatChunkChoice {
    index: usize,
    delta: MessageDelta,
    finish_reason: Option<String>,
}

#[derive(serde::Serialize)]
struct MessageDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(serde::Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: ChatUsage,
}

#[derive(serde::Serialize)]
struct CompletionChoice {
    text: String,
    index: usize,
    finish_reason: Option<String>,
}

#[derive(serde::Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    model: String,
    cpu_only: bool,
    uptime_secs: u64,
    requests_served: u64,
    ready: bool,
}

fn default_temperature() -> f32 {
    0.7
}
fn default_top_p() -> f32 {
    0.9
}
fn default_repetition_penalty() -> f32 {
    1.0
}

// ─── Shared state ─────────────────────────────────────────────────────────────

struct ServerState {
    runner: Arc<TokioMutex<LlamaRunner>>,
    concurrency: Arc<Semaphore>,
    model_name: String,
    model_path: String,
    api_key: Option<String>,
    max_tokens_cap: usize,
    cpu_only: bool,
    ready: AtomicBool,
    started_at: Instant,
    requests_served: AtomicU64,
    rate_limit_per_min: u32,
}

struct ReceiverStream<T> {
    rx: tokio::sync::mpsc::Receiver<T>,
}

impl<T> futures::Stream for ReceiverStream<T> {
    type Item = T;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

// ─── Rate limiter (global sliding window) ─────────────────────────────────────

struct RateLimiter {
    max_per_min: u32,
    history: std::sync::Mutex<Vec<Instant>>,
}

impl RateLimiter {
    fn new(max_per_min: u32) -> Self {
        Self {
            max_per_min,
            history: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn check(&self) -> bool {
        if self.max_per_min == 0 {
            return true;
        }
        let now = Instant::now();
        let window = Duration::from_secs(60);
        let mut history = self.history.lock().unwrap();
        history.retain(|t| now.duration_since(*t) < window);
        if history.len() >= self.max_per_min as usize {
            return false;
        }
        history.push(now);
        true
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("aether=info,aether_server=info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    let args = Args::parse();

    if !Path::new(&args.model).exists() {
        error!(
            "Model file not found: {} (set AETHER_MODEL_PATH or --model)",
            args.model
        );
        std::process::exit(1);
    }

    if args.api_key.is_none() && args.host != "127.0.0.1" && args.host != "localhost" {
        warn!(
            "AETHER_API_KEY is not set but server binds to {} — API is open on the network",
            args.host
        );
    }

    info!("Loading model from {}...", args.model);
    let load_opts = LoadOptions {
        cpu_only: args.cpu_only,
    };
    let runner = match LlamaRunner::from_gguf_with_options(&args.model, load_opts) {
        Ok(r) => r,
        Err(e) => {
            error!("Failed to load model: {}", e);
            std::process::exit(1);
        }
    };

    let gpu_layers = runner.layer_assignment.gpu_layers;
    let cpu_layers = runner.layer_assignment.cpu_layers;
    info!(
        "Model loaded | cpu_only={} | layers: {} GPU, {} CPU",
        args.cpu_only, gpu_layers, cpu_layers
    );

    let model_name = Path::new(&args.model)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(&args.model)
        .to_string();

    let rate_limiter = Arc::new(RateLimiter::new(args.rate_limit));

    let concurrency = Arc::new(Semaphore::new(args.max_concurrency.max(1)));
    let state = Arc::new(ServerState {
        runner: Arc::new(TokioMutex::new(runner)),
        concurrency,
        model_name: model_name.clone(),
        model_path: args.model.clone(),
        api_key: args.api_key.clone(),
        max_tokens_cap: args.max_tokens,
        cpu_only: args.cpu_only,
        ready: AtomicBool::new(true),
        started_at: Instant::now(),
        requests_served: AtomicU64::new(0),
        rate_limit_per_min: args.rate_limit,
    });

    // ── API routes (require auth) ──────────────────────────────────────────
    let api = Router::new()
        .route("/v1/models", get(handle_models))
        .route(
            "/v1/chat/completions",
            post(handle_chat_completions).options(handle_options),
        )
        .route(
            "/v1/completions",
            post(handle_completions).options(handle_options),
        )
        .route_layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            require_api_key,
        ));

    // ── Public routes (no auth) ────────────────────────────────────────────
    let app = Router::new()
        .route("/health", get(handle_health))
        .route("/ready", get(handle_ready))
        .route("/metrics", get(handle_metrics))
        .merge(api)
        .layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            access_log_middleware,
        ))
        .layer(middleware::from_fn_with_state(
            Arc::clone(&rate_limiter),
            rate_limit_middleware,
        ))
        .layer(middleware::from_fn(x_request_id_middleware))
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            error!("Failed to bind {}: {}", addr, e);
            std::process::exit(1);
        }
    };

    info!(
        "Aether server listening on http://{} | model={} | max_tokens={} | max_concurrency={}",
        addr, model_name, args.max_tokens, args.max_concurrency
    );
    if let Some(ref key) = args.api_key {
        info!("API key auth enabled on /v1/* (key length={})", key.len());
    } else {
        warn!("No API key — /v1/* routes are unauthenticated");
    }
    if args.rate_limit > 0 {
        info!("Rate limit: {} requests/minute", args.rate_limit);
    } else {
        info!("Rate limiting disabled");
    }

    let graceful = async {
        let _ = tokio::signal::ctrl_c().await;
        info!("Shutdown signal received, stopping server...");
        // Brief drain period for in-flight requests
        tokio::time::sleep(Duration::from_millis(500)).await;
    };

    if let Err(e) = axum::serve(listener, app)
        .with_graceful_shutdown(graceful)
        .await
    {
        error!("Server error: {}", e);
        std::process::exit(1);
    }
}

// ─── Middleware ───────────────────────────────────────────────────────────────

async fn x_request_id_middleware(req: Request, next: Next) -> Response {
    let id = fast_rand_id();
    let mut resp = next.run(req).await;
    resp.headers_mut().insert(
        header::HeaderName::from_static("x-request-id"),
        HeaderValue::from_str(&id).unwrap(),
    );
    resp
}

async fn rate_limit_middleware(
    State(limiter): State<Arc<RateLimiter>>,
    req: Request,
    next: Next,
) -> Response {
    if !limiter.check() {
        return api_error(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_exceeded",
            "Too many requests (global rate limit per minute exceeded)",
        );
    }
    next.run(req).await
}

async fn access_log_middleware(
    State(state): State<Arc<ServerState>>,
    req: Request,
    next: Next,
) -> Response {
    let method = req.method().clone();
    let uri = req.uri().path().to_string();
    let start = Instant::now();
    let resp = next.run(req).await;
    let status = resp.status();
    let elapsed = start.elapsed();
    state.requests_served.fetch_add(1, Ordering::Relaxed);
    info!("{} {} -> {} in {:?}", method, uri, status.as_u16(), elapsed);
    resp
}

async fn require_api_key(
    State(state): State<Arc<ServerState>>,
    req: Request,
    next: Next,
) -> Response {
    if let Some(expected) = &state.api_key {
        let authorized = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|h| {
                h.strip_prefix("Bearer ")
                    .or_else(|| h.strip_prefix("bearer "))
                    .unwrap_or(h)
                    == expected.as_str()
            });
        if !authorized {
            return api_error(
                StatusCode::UNAUTHORIZED,
                "invalid_api_key",
                "Missing or invalid Authorization header (expected Bearer token)",
            );
        }
    }
    next.run(req).await
}

// ─── Handlers ─────────────────────────────────────────────────────────────────

fn api_error(status: StatusCode, error_type: &str, message: &str) -> Response {
    let body = serde_json::json!({
        "error": { "message": message, "type": error_type, "code": null }
    });
    (status, cors_headers(), Json(body)).into_response()
}

fn cors_headers() -> HeaderMap {
    let mut h = HeaderMap::new();
    h.insert(
        header::ACCESS_CONTROL_ALLOW_ORIGIN,
        HeaderValue::from_static("*"),
    );
    h.insert(
        header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("GET, POST, OPTIONS"),
    );
    h.insert(
        header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("content-type, authorization"),
    );
    h
}

async fn handle_options() -> impl IntoResponse {
    (StatusCode::OK, cors_headers(), "")
}

async fn handle_health(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let resp = HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        model: state.model_name.clone(),
        cpu_only: state.cpu_only,
        uptime_secs: state.started_at.elapsed().as_secs(),
        requests_served: state.requests_served.load(Ordering::Relaxed),
        ready: state.ready.load(Ordering::Relaxed),
    };
    (StatusCode::OK, Json(resp))
}

async fn handle_ready(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    if state.ready.load(Ordering::Relaxed) {
        (StatusCode::OK, Json(serde_json::json!({ "ready": true })))
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "ready": false })),
        )
    }
}

async fn handle_metrics(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let uptime = state.started_at.elapsed().as_secs_f64();
    let served = state.requests_served.load(Ordering::Relaxed);
    let rate = if uptime > 0.0 {
        served as f64 / uptime
    } else {
        0.0
    };
    let body = format!(
        "\
# HELP aether_uptime_seconds Server uptime
# TYPE aether_uptime_seconds gauge
aether_uptime_seconds {uptime}
# HELP aether_requests_served_total Total requests served
# TYPE aether_requests_served_total counter
aether_requests_served_total {served}
# HELP aether_requests_per_second Request rate
# TYPE aether_requests_per_second gauge
aether_requests_per_second {rate:.4}
# HELP aether_model_info Static model metadata
# TYPE aether_model_info gauge
aether_model_info{{model=\"{name}\",cpu_only=\"{cpu}\"}} 1
",
        name = state.model_name,
        cpu = state.cpu_only,
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        body,
    )
}

async fn handle_models(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let resp = serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "created": now,
            "owned_by": "aether"
        }]
    });
    (StatusCode::OK, cors_headers(), Json(resp))
}

// ─── Chat completions ─────────────────────────────────────────────────────────

async fn handle_chat_completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    if req.messages.is_empty() {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "messages must not be empty",
        );
    }

    let max_tokens = req
        .max_tokens
        .unwrap_or(state.max_tokens_cap)
        .min(state.max_tokens_cap);
    if max_tokens == 0 {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "max_tokens must be at least 1",
        );
    }

    let prompt = format_messages(&req.messages);
    let chat_id = format!("chatcmpl-{}", fast_rand_id());
    let created_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let model_name = state.model_name.clone();

    if req.stream {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let shared_runner = Arc::clone(&state.runner);
        let shared_concurrency = Arc::clone(&state.concurrency);
        let chat_id_clone = chat_id.clone();
        let model_name_clone = model_name.clone();
        let temperature = req.temperature;
        let top_p = req.top_p;
        let repetition_penalty = req.repetition_penalty;

        tokio::spawn(async move {
            let _permit = shared_concurrency.acquire().await.unwrap();
            let mut runner = shared_runner.lock().await;
            runner.kv.reset();

            let token_ids = runner.tokenizer.encode(&prompt, true);
            let prompt_len = token_ids.len();
            let mut prev_tokens = token_ids.clone();

            let last_logits = match runner.prefill(&token_ids) {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                    return;
                }
            };

            let mut next_token = sample(
                &last_logits,
                temperature,
                top_p,
                &prev_tokens,
                repetition_penalty,
            );
            prev_tokens.push(next_token);
            let mut pos = prompt_len;

            let first_text = runner.tokenizer.decode_one(next_token);
            if tx.send(Ok(first_text)).await.is_err() {
                return;
            }

            for _step in 0..max_tokens.saturating_sub(1) {
                if next_token == runner.tokenizer.eos_id {
                    break;
                }
                if runner.kv.seq_len >= runner.kv.max_seq {
                    break;
                }

                let mut layer_tel = vec![
                    aether::inference::telemetry::LayerTelemetry::default();
                    runner.model.config.num_layers
                ];
                let logits = match runner.decode_step(next_token, pos, &mut layer_tel) {
                    Ok(l) => l,
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        return;
                    }
                };

                next_token = sample(
                    &logits,
                    temperature,
                    top_p,
                    &prev_tokens,
                    repetition_penalty,
                );
                prev_tokens.push(next_token);
                pos += 1;

                let text = runner.tokenizer.decode_one(next_token);
                if tx.send(Ok(text)).await.is_err() {
                    return;
                }
            }

            let stop_chunk = ChatCompletionChunk {
                id: chat_id_clone.clone(),
                object: "chat.completion.chunk",
                created: created_time,
                model: model_name_clone.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: MessageDelta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
            };
            let _ = tx
                .send(Ok(serde_json::to_string(&stop_chunk).unwrap_or_default()))
                .await;
            let _ = tx.send(Ok("[DONE]".to_string())).await;
        });

        let stream = ReceiverStream { rx }.map(move |res| match res {
            Ok(data) if data == "[DONE]" => {
                Ok::<Event, std::convert::Infallible>(Event::default().data("[DONE]"))
            }
            Ok(data) if data.starts_with('{') => Ok(Event::default().data(data)),
            Ok(text) => {
                let chunk = ChatCompletionChunk {
                    id: chat_id.clone(),
                    object: "chat.completion.chunk",
                    created: created_time,
                    model: model_name.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: MessageDelta {
                            role: None,
                            content: Some(text),
                        },
                        finish_reason: None,
                    }],
                };
                Ok(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()))
            }
            Err(e) => {
                let chunk = ChatCompletionChunk {
                    id: chat_id.clone(),
                    object: "chat.completion.chunk",
                    created: created_time,
                    model: model_name.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: MessageDelta {
                            role: None,
                            content: Some(format!("[Error: {}]", e)),
                        },
                        finish_reason: Some("error".to_string()),
                    }],
                };
                Ok(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()))
            }
        });

        (StatusCode::OK, cors_headers(), Sse::new(stream)).into_response()
    } else {
        let (response_content, prompt_tokens, completion_tokens) = {
            let _permit = state.concurrency.acquire().await.unwrap();
            let mut runner = state.runner.lock().await;
            runner.kv.reset();

            let token_ids = runner.tokenizer.encode(&prompt, true);
            let prompt_len = token_ids.len();

            let generated_text = match runner.generate(
                &prompt,
                max_tokens,
                req.temperature,
                req.top_p,
                req.repetition_penalty,
            ) {
                Ok(t) => t,
                Err(e) => {
                    return api_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "server_error",
                        &format!("Generation failed: {}", e),
                    );
                }
            };

            let completion_len = runner.tokenizer.encode(&generated_text, false).len();
            (generated_text, prompt_len, completion_len)
        };

        let response = ChatCompletionResponse {
            id: chat_id,
            object: "chat.completion",
            created: created_time,
            model: model_name,
            choices: vec![ChatChoice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: response_content,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: ChatUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        };

        (StatusCode::OK, cors_headers(), Json(response)).into_response()
    }
}

// ─── Text completions ─────────────────────────────────────────────────────────

async fn handle_completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<CompletionRequest>,
) -> impl IntoResponse {
    let max_tokens = req
        .max_tokens
        .unwrap_or(state.max_tokens_cap)
        .min(state.max_tokens_cap);
    if max_tokens == 0 {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "max_tokens must be at least 1",
        );
    }

    let comp_id = format!("cmpl-{}", fast_rand_id());
    let created_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let model_name = state.model_name.clone();

    if req.stream {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let shared_runner = Arc::clone(&state.runner);
        let shared_concurrency = Arc::clone(&state.concurrency);
        let temperature = req.temperature;
        let top_p = req.top_p;
        let repetition_penalty = req.repetition_penalty;

        tokio::spawn(async move {
            let _permit = shared_concurrency.acquire().await.unwrap();
            let mut runner = shared_runner.lock().await;
            runner.kv.reset();

            let token_ids = runner.tokenizer.encode(&req.prompt, true);
            let prompt_len = token_ids.len();
            let mut prev_tokens = token_ids.clone();
            let mut pos = prompt_len;

            let last_logits = match runner.prefill(&token_ids) {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                    return;
                }
            };

            let mut next_token = sample(
                &last_logits,
                temperature,
                top_p,
                &prev_tokens,
                repetition_penalty,
            );
            prev_tokens.push(next_token);

            let first_text = runner.tokenizer.decode_one(next_token);
            if tx.send(Ok(first_text)).await.is_err() {
                return;
            }

            for _step in 0..max_tokens.saturating_sub(1) {
                if next_token == runner.tokenizer.eos_id {
                    break;
                }
                if runner.kv.seq_len >= runner.kv.max_seq {
                    break;
                }

                let mut layer_tel = vec![
                    aether::inference::telemetry::LayerTelemetry::default();
                    runner.model.config.num_layers
                ];
                let logits = match runner.decode_step(next_token, pos, &mut layer_tel) {
                    Ok(l) => l,
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        return;
                    }
                };

                next_token = sample(
                    &logits,
                    temperature,
                    top_p,
                    &prev_tokens,
                    repetition_penalty,
                );
                prev_tokens.push(next_token);
                pos += 1;

                let text = runner.tokenizer.decode_one(next_token);
                if tx.send(Ok(text)).await.is_err() {
                    return;
                }
            }

            let _ = tx.send(Ok("[DONE]".to_string())).await;
        });

        let stream = ReceiverStream { rx }.map(move |res| match res {
            Ok(data) if data == "[DONE]" => {
                Ok::<Event, std::convert::Infallible>(Event::default().data("[DONE]"))
            }
            Ok(text) => Ok(Event::default().data(text)),
            Err(e) => Ok(Event::default().data(format!("[Error: {}]", e))),
        });

        (StatusCode::OK, cors_headers(), Sse::new(stream)).into_response()
    } else {
        let generated_text = {
            let _permit = state.concurrency.acquire().await.unwrap();
            let mut runner = state.runner.lock().await;
            runner.kv.reset();
            match runner.generate(
                &req.prompt,
                max_tokens,
                req.temperature,
                req.top_p,
                req.repetition_penalty,
            ) {
                Ok(t) => t,
                Err(e) => {
                    return api_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "server_error",
                        &format!("Generation failed: {}", e),
                    );
                }
            }
        };

        let prompt_tokens = generated_text.split_whitespace().count();
        let completion_tokens = max_tokens;

        let response = CompletionResponse {
            id: comp_id,
            object: "text_completion",
            created: created_time,
            model: model_name,
            choices: vec![CompletionChoice {
                text: generated_text,
                index: 0,
                finish_reason: Some("stop".to_string()),
            }],
            usage: ChatUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        };

        (StatusCode::OK, cors_headers(), Json(response)).into_response()
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn format_messages(messages: &[Message]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        match msg.role.as_str() {
            "system" => prompt.push_str(&format!("<|system|>\n{}</s>\n", msg.content)),
            "user" => prompt.push_str(&format!("<|user|>\n{}</s>\n", msg.content)),
            "assistant" => prompt.push_str(&format!("<|assistant|>\n{}</s>\n", msg.content)),
            _ => prompt.push_str(&format!("{}: {}\n", msg.role, msg.content)),
        }
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}

fn fast_rand_id() -> String {
    static STATE: AtomicU64 = AtomicU64::new(0x9ABCDEF12345678);
    let mut x = STATE.load(Ordering::Relaxed);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    STATE.store(x, Ordering::Relaxed);
    format!("{:016x}", x)
}
