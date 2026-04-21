use std::path::PathBuf;

use axum::{routing::get, Router};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "xenon-server", about = "OpenAI-compatible inference server for xenon")]
struct Args {
    /// Path to the model snapshot directory (HF cache layout).
    #[arg(long)]
    model: Option<PathBuf>,

    /// Address to bind on.
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")))
        .init();

    let args = Args::parse();

    if let Some(model) = &args.model {
        let cfg = xenon_core::GemmaConfig::from_path(&model.join("config.json"))?;
        tracing::info!(
            layers = cfg.text_config.num_hidden_layers,
            hidden = cfg.text_config.hidden_size,
            "loaded model config (skeleton: inference not yet wired up)",
        );
    }

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(|| async { "TODO" }));

    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    tracing::info!(addr = %args.bind, "serving");
    axum::serve(listener, app).await?;
    Ok(())
}
