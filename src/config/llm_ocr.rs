//! The `LLM_OCR*` settings for vision-LLM OCR. Tesseract remains the fallback.

use anyhow::Result;
use std::env;

use super::parsing;
use crate::ocr::llm::{
    LlmOcrConfig, LlmOcrFileTypes, DEFAULT_LLM_OCR_MAX_CONCURRENT_REQUESTS, DEFAULT_LLM_OCR_PROMPT,
    DEFAULT_LLM_OCR_TIMEOUT_SECONDS,
};

/// Read the `LLM_OCR*` variables through `var`. Returns `None` unless
/// `LLM_OCR` is true; when it is, `LLM_OCR_API_URL` and `LLM_OCR_MODEL` are
/// required.
pub(super) fn parse_llm_ocr_config(var: impl Fn(&str) -> Option<String>) -> Result<Option<LlmOcrConfig>> {
    let non_empty = |name: &str| var(name).map(|v| v.trim().to_string()).filter(|v| !v.is_empty());

    let enabled = match non_empty("LLM_OCR") {
        Some(v) => parsing::parse_flag(&v)
            .ok_or_else(|| anyhow::anyhow!("LLM_OCR has an unrecognized value '{}'; use true or false", v))?,
        None => false,
    };
    if !enabled {
        return Ok(None);
    }

    let api_url = non_empty("LLM_OCR_API_URL")
        .ok_or_else(|| anyhow::anyhow!("LLM_OCR is enabled but LLM_OCR_API_URL is not set"))?;
    let parsed_url = url::Url::parse(&api_url)
        .map_err(|e| anyhow::anyhow!("LLM_OCR_API_URL '{}' is not a valid URL: {}", api_url, e))?;
    if !matches!(parsed_url.scheme(), "http" | "https") {
        return Err(anyhow::anyhow!("LLM_OCR_API_URL must be an http(s) URL, got '{}'", api_url));
    }
    let model = non_empty("LLM_OCR_MODEL")
        .ok_or_else(|| anyhow::anyhow!("LLM_OCR is enabled but LLM_OCR_MODEL is not set"))?;
    let timeout_seconds = match non_empty("LLM_OCR_TIMEOUT_SECONDS") {
        Some(v) => match v.parse::<u64>() {
            Ok(t) if t > 0 => t,
            _ => return Err(anyhow::anyhow!("LLM_OCR_TIMEOUT_SECONDS must be a positive integer, got '{}'", v)),
        },
        None => DEFAULT_LLM_OCR_TIMEOUT_SECONDS,
    };
    let max_concurrent_requests = match non_empty("LLM_OCR_MAX_CONCURRENT_REQUESTS") {
        Some(v) => match v.parse::<usize>() {
            Ok(n) if n > 0 => n,
            _ => return Err(anyhow::anyhow!("LLM_OCR_MAX_CONCURRENT_REQUESTS must be a positive integer, got '{}'", v)),
        },
        None => DEFAULT_LLM_OCR_MAX_CONCURRENT_REQUESTS,
    };
    let file_types = match non_empty("LLM_OCR_FILE_TYPES") {
        Some(v) => LlmOcrFileTypes::parse(&v)
            .ok_or_else(|| anyhow::anyhow!("LLM_OCR_FILE_TYPES must be 'all' or 'images', got '{}'", v))?,
        None => LlmOcrFileTypes::default(),
    };

    Ok(Some(LlmOcrConfig {
        api_url,
        model,
        api_key: non_empty("LLM_OCR_API_KEY"),
        timeout_seconds,
        max_concurrent_requests,
        file_types,
        prompt: non_empty("LLM_OCR_PROMPT").unwrap_or_else(|| DEFAULT_LLM_OCR_PROMPT.to_string()),
    }))
}

/// Read the `LLM_OCR*` variables from the environment and log the result.
pub(super) fn from_env() -> Result<Option<LlmOcrConfig>> {
    let config = parse_llm_ocr_config(|name| env::var(name).ok())?;
    match &config {
        Some(c) => println!(
            "✅ LLM_OCR: true (file types: {}, model '{}' at {}, {}s timeout per page, {} concurrent requests, API key {})",
            c.file_types.as_str(),
            c.model,
            c.chat_completions_url(),
            c.timeout_seconds,
            c.max_concurrent_requests,
            if c.api_key.is_some() { "configured" } else { "not set" }
        ),
        None => println!("⚠️  LLM_OCR: false (using Tesseract/ocrmypdf)"),
    }
    Ok(config)
}

/// The LLM OCR line of the startup configuration summary.
pub(super) fn print_summary(config: Option<&LlmOcrConfig>) {
    match config {
        Some(llm) => println!(
            "🤖 LLM OCR: enabled for {} ({}), Tesseract fallback after {}s",
            if llm.file_types == LlmOcrFileTypes::Images { "images only" } else { "images and PDFs" },
            llm.model,
            llm.timeout_seconds
        ),
        None => println!("🤖 LLM OCR: disabled"),
    }
}
