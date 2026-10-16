//! OCR through a vision LLM behind an OpenAI-compatible `/chat/completions`
//! endpoint (Ollama, vLLM, LM Studio, OpenAI, OpenRouter, ...).
//!
//! Enabled with `LLM_OCR=true`. The enhanced OCR service tries the LLM first
//! for PDFs and images and falls back to the Tesseract/ocrmypdf pipeline when
//! the request fails or exceeds `LLM_OCR_TIMEOUT_SECONDS`.
//!
//! The OCR worker runs many documents at once (a frontend batch is one job per
//! file), so the client caps in-flight requests at
//! `LLM_OCR_MAX_CONCURRENT_REQUESTS` and only starts the timeout once a request
//! holds a slot. Otherwise time spent queued inside a self-hosted server would
//! count against the timeout and healthy batches would fall back to Tesseract.
//! When the endpoint stops responding, a circuit breaker skips it for a while
//! so the rest of the batch goes straight to Tesseract instead of each
//! document waiting out its own timeout.

use anyhow::{anyhow, Result};
use base64ct::Encoding;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tracing::{info, warn};

pub const DEFAULT_LLM_OCR_TIMEOUT_SECONDS: u64 = 120;
pub const DEFAULT_LLM_OCR_MAX_CONCURRENT_REQUESTS: usize = 2;

/// Consecutive timeouts/connection failures before the endpoint is skipped.
const UNRESPONSIVE_THRESHOLD: u32 = 2;
/// How long the endpoint is skipped once it is considered unresponsive.
const UNRESPONSIVE_COOLDOWN: Duration = Duration::from_secs(60);

/// Which uploads go to the LLM (`LLM_OCR_FILE_TYPES`). Everything else uses
/// the Tesseract/ocrmypdf pipeline as if LLM OCR were off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LlmOcrFileTypes {
    /// Images and PDFs (`all`).
    #[default]
    All,
    /// Images only; PDFs keep using pdftotext/Tesseract/ocrmypdf (`images`).
    Images,
}

impl LlmOcrFileTypes {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "all" => Some(Self::All),
            "images" => Some(Self::Images),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Images => "images",
        }
    }

    pub fn includes(&self, mime_type: &str) -> bool {
        match self {
            Self::All => mime_type == "application/pdf" || mime_type.starts_with("image/"),
            Self::Images => mime_type.starts_with("image/"),
        }
    }
}

pub const DEFAULT_LLM_OCR_PROMPT: &str = "Transcribe all of the text in this document image exactly as it appears. \
Preserve the reading order, paragraphs and line breaks. Output only the transcribed text with no commentary, \
explanations or markdown code fences. If the image contains no text, output nothing.";

#[derive(Clone)]
pub struct LlmOcrConfig {
    /// Base URL of the API (e.g. `http://localhost:11434/v1`) or the full
    /// `.../chat/completions` URL.
    pub api_url: String,
    pub model: String,
    pub api_key: Option<String>,
    /// Applied to each request (one page or image), so a hung endpoint falls
    /// back to Tesseract after at most this long.
    pub timeout_seconds: u64,
    /// Requests allowed in flight at once across all OCR jobs.
    pub max_concurrent_requests: usize,
    pub file_types: LlmOcrFileTypes,
    pub prompt: String,
}

impl std::fmt::Debug for LlmOcrConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmOcrConfig")
            .field("api_url", &self.api_url)
            .field("model", &self.model)
            .field("api_key", &self.api_key.as_ref().map(|_| "***"))
            .field("timeout_seconds", &self.timeout_seconds)
            .field("max_concurrent_requests", &self.max_concurrent_requests)
            .field("file_types", &self.file_types)
            .field("prompt", &self.prompt)
            .finish()
    }
}

impl LlmOcrConfig {
    /// Full URL of the chat completions endpoint.
    pub fn chat_completions_url(&self) -> String {
        let base = self.api_url.trim_end_matches('/');
        if base.ends_with("/chat/completions") {
            base.to_string()
        } else {
            format!("{}/chat/completions", base)
        }
    }
}

/// Shared by every clone of the client so all OCR jobs see the same state.
#[derive(Default)]
struct EndpointHealth {
    consecutive_failures: AtomicU32,
    unresponsive_until: Mutex<Option<Instant>>,
}

#[derive(Clone)]
pub struct LlmOcrClient {
    config: LlmOcrConfig,
    http: reqwest::Client,
    slots: Arc<Semaphore>,
    health: Arc<EndpointHealth>,
}

impl LlmOcrClient {
    pub fn new(config: LlmOcrConfig) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .connect_timeout(Duration::from_secs(config.timeout_seconds.min(30)))
            .build()
            .map_err(|e| anyhow!("Failed to build LLM OCR HTTP client: {}", e))?;
        let slots = Arc::new(Semaphore::new(config.max_concurrent_requests.max(1)));
        Ok(Self { config, http, slots, health: Arc::default() })
    }

    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Whether files of this type should be sent to the LLM at all.
    pub fn handles(&self, mime_type: &str) -> bool {
        self.config.file_types.includes(mime_type)
    }

    /// False while the endpoint is being skipped after repeated timeouts or
    /// connection failures.
    pub fn is_available(&self) -> bool {
        match *self.health.unresponsive_until.lock().unwrap() {
            Some(until) => Instant::now() >= until,
            None => true,
        }
    }

    fn record_responsive(&self) {
        if self.health.consecutive_failures.swap(0, Ordering::SeqCst) > 0 {
            let mut until = self.health.unresponsive_until.lock().unwrap();
            if until.take().is_some() {
                info!("LLM OCR endpoint is responding again");
            }
        }
    }

    fn record_unresponsive(&self) {
        let failures = self.health.consecutive_failures.fetch_add(1, Ordering::SeqCst) + 1;
        if failures >= UNRESPONSIVE_THRESHOLD {
            *self.health.unresponsive_until.lock().unwrap() = Some(Instant::now() + UNRESPONSIVE_COOLDOWN);
            warn!(
                "LLM OCR endpoint failed {} times in a row; using Tesseract for the next {}s",
                failures,
                UNRESPONSIVE_COOLDOWN.as_secs()
            );
        }
    }

    fn unavailable_error(&self) -> anyhow::Error {
        anyhow!("LLM OCR endpoint is unresponsive; skipping it for now")
    }

    /// Send one image to the model and return the transcribed text.
    pub async fn transcribe_image(&self, image_bytes: &[u8], mime_type: &str) -> Result<String> {
        if !self.is_available() {
            return Err(self.unavailable_error());
        }
        let _slot = self.slots.acquire().await?;
        // The endpoint may have been marked unresponsive while we waited.
        if !self.is_available() {
            return Err(self.unavailable_error());
        }

        let data_url = format!(
            "data:{};base64,{}",
            mime_type,
            base64ct::Base64::encode_string(image_bytes)
        );
        let body = json!({
            "model": self.config.model,
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": self.config.prompt },
                    { "type": "image_url", "image_url": { "url": data_url } }
                ]
            }]
        });

        let mut request = self.http.post(self.config.chat_completions_url()).json(&body);
        if let Some(key) = &self.config.api_key {
            request = request.bearer_auth(key);
        }

        // reqwest's client timeout covers the whole exchange; the outer timeout
        // is a backstop so a stalled connection can never outlive the limit.
        let timeout = Duration::from_secs(self.config.timeout_seconds);
        let response = match tokio::time::timeout(timeout, async {
            let response = request.send().await?;
            let status = response.status();
            let text = response.text().await?;
            Ok::<_, reqwest::Error>((status, text))
        })
        .await
        {
            Ok(Ok(response)) => response,
            Ok(Err(e)) if e.is_connect() => {
                self.record_unresponsive();
                return Err(anyhow!("LLM OCR endpoint unreachable: {}", e));
            }
            Ok(Err(e)) if !e.is_timeout() => {
                return Err(anyhow!("LLM OCR request failed: {}", e));
            }
            // Either our backstop or reqwest's own timeout fired.
            _ => {
                self.record_unresponsive();
                return Err(anyhow!("LLM OCR request timed out after {}s", self.config.timeout_seconds));
            }
        };
        // Any HTTP response, even an error status, means the endpoint is alive.
        self.record_responsive();

        let (status, text) = response;
        if !status.is_success() {
            let snippet: String = text.chars().take(500).collect();
            return Err(anyhow!("LLM OCR endpoint returned HTTP {}: {}", status, snippet));
        }

        let parsed: Value = serde_json::from_str(&text)
            .map_err(|e| anyhow!("LLM OCR endpoint returned invalid JSON: {}", e))?;
        parse_chat_completion_text(&parsed)
    }
}

/// Pull the assistant text out of a chat completions response. `content` is
/// usually a string, but some servers return an array of content parts.
pub fn parse_chat_completion_text(response: &Value) -> Result<String> {
    let content = response
        .pointer("/choices/0/message/content")
        .ok_or_else(|| anyhow!("LLM OCR response has no choices[0].message.content"))?;

    let text = match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        Value::Null => String::new(),
        other => return Err(anyhow!("Unexpected LLM OCR content type: {}", other)),
    };

    Ok(strip_code_fence(&text).trim().to_string())
}

/// Models sometimes wrap the transcription in a ``` block despite the prompt.
fn strip_code_fence(text: &str) -> &str {
    let trimmed = text.trim();
    let Some(rest) = trimmed.strip_prefix("```") else {
        return text;
    };
    let Some(inner) = rest.strip_suffix("```") else {
        return text;
    };
    // Drop an optional language tag on the opening fence line.
    match inner.find('\n') {
        Some(idx) => &inner[idx + 1..],
        None => inner,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(api_url: &str) -> LlmOcrConfig {
        LlmOcrConfig {
            api_url: api_url.to_string(),
            model: "m".to_string(),
            api_key: None,
            timeout_seconds: 5,
            max_concurrent_requests: DEFAULT_LLM_OCR_MAX_CONCURRENT_REQUESTS,
            file_types: LlmOcrFileTypes::All,
            prompt: DEFAULT_LLM_OCR_PROMPT.to_string(),
        }
    }

    #[test]
    fn chat_completions_url_appends_path_to_base_url() {
        assert_eq!(config("http://h:11434/v1").chat_completions_url(), "http://h:11434/v1/chat/completions");
        assert_eq!(config("http://h:11434/v1/").chat_completions_url(), "http://h:11434/v1/chat/completions");
        assert_eq!(
            config("http://h/v1/chat/completions").chat_completions_url(),
            "http://h/v1/chat/completions"
        );
    }

    #[test]
    fn parses_string_and_array_content() {
        let s = json!({"choices": [{"message": {"content": "  Hello world \n"}}]});
        assert_eq!(parse_chat_completion_text(&s).unwrap(), "Hello world");

        let a = json!({"choices": [{"message": {"content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}]}}]});
        assert_eq!(parse_chat_completion_text(&a).unwrap(), "Hello world");

        assert!(parse_chat_completion_text(&json!({"choices": []})).is_err());
    }

    #[test]
    fn strips_wrapping_code_fence() {
        let r = json!({"choices": [{"message": {"content": "```text\nLine one\nLine two\n```"}}]});
        assert_eq!(parse_chat_completion_text(&r).unwrap(), "Line one\nLine two");
    }

    #[tokio::test]
    async fn skips_endpoint_after_consecutive_failures_and_recovers_on_response() {
        let client = LlmOcrClient::new(config("http://h")).unwrap();
        assert!(client.is_available());

        client.record_unresponsive();
        assert!(client.is_available(), "a single failure must not trip the breaker");
        client.record_unresponsive();
        assert!(!client.is_available());
        assert!(client.transcribe_image(b"x", "image/png").await.is_err());

        // Clones share state, so every OCR job sees the same breaker.
        let clone = client.clone();
        assert!(!clone.is_available());
        clone.record_responsive();
        assert!(client.is_available());
    }

    #[test]
    fn file_types_select_images_or_images_and_pdfs() {
        assert_eq!(LlmOcrFileTypes::parse(" Images "), Some(LlmOcrFileTypes::Images));
        assert_eq!(LlmOcrFileTypes::parse("ALL"), Some(LlmOcrFileTypes::All));
        assert_eq!(LlmOcrFileTypes::parse("pdf"), None);

        assert!(LlmOcrFileTypes::All.includes("application/pdf"));
        assert!(LlmOcrFileTypes::All.includes("image/png"));
        assert!(!LlmOcrFileTypes::Images.includes("application/pdf"));
        assert!(LlmOcrFileTypes::Images.includes("image/jpeg"));
        assert!(!LlmOcrFileTypes::All.includes("text/plain"));
    }

    #[test]
    fn debug_masks_api_key() {
        let mut c = config("http://h");
        c.api_key = Some("secret-key".to_string());
        assert!(!format!("{:?}", c).contains("secret-key"));
    }
}
