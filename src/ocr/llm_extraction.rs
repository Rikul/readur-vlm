//! LLM OCR for `EnhancedOcrService`: transcribe PDFs and images with the
//! configured vision LLM before the Tesseract/ocrmypdf pipeline runs.
//!
//! The service calls `try_llm_ocr` from its type dispatch; `None` means use
//! Tesseract. The HTTP client, concurrency limit and circuit breaker live in
//! `super::llm`.

use tracing::{error, info};

use super::enhanced::EnhancedOcrService;
use super::llm::{LlmOcrClient, LlmOcrConfig};

impl EnhancedOcrService {
    pub fn with_llm_ocr(mut self, client: LlmOcrClient) -> Self {
        self.llm_ocr = Some(client);
        self
    }

    /// Enable LLM OCR from `config` when it is set. If the client cannot be
    /// built, log it and keep the service on Tesseract only.
    pub fn with_llm_ocr_config(self, config: Option<&LlmOcrConfig>, worker_id: &str) -> Self {
        let Some(config) = config else { return self };
        match LlmOcrClient::new(config.clone()) {
            Ok(client) => {
                info!("LLM OCR enabled for worker {} (model '{}')", worker_id, client.model());
                self.with_llm_ocr(client)
            }
            Err(e) => {
                error!("LLM OCR disabled, using Tesseract only: {}", e);
                self
            }
        }
    }
}

#[cfg(feature = "ocr")]
mod extraction {
    use anyhow::{anyhow, Result};
    use tracing::{debug, info, warn};

    use crate::ocr::enhanced::{EnhancedOcrService, FileCleanupGuard, OcrResult, ProgressCallback};
    use crate::ocr::llm::LlmOcrClient;

    /// LLMs report no confidence; this fixed value keeps their output above the
    /// quality-validation thresholds.
    const LLM_OCR_CONFIDENCE: f32 = 90.0;
    /// Render resolution for PDF pages sent to the LLM. Lower than the 300 DPI
    /// used for Tesseract to keep requests small; vision models downscale anyway.
    const LLM_OCR_PDF_DPI: u32 = 200;

    /// Whether the LLM should get `file_path`: its type is enabled in
    /// `LLM_OCR_FILE_TYPES` and the endpoint is not marked unresponsive.
    fn llm_should_handle(llm: &LlmOcrClient, file_path: &str, mime_type: &str) -> bool {
        if !llm.handles(mime_type) {
            debug!("LLM OCR not enabled for {} (LLM_OCR_FILE_TYPES), using the standard pipeline for: {}", mime_type, file_path);
            return false;
        }
        if !llm.is_available() {
            info!("LLM OCR endpoint is unresponsive, using Tesseract for: {}", file_path);
            return false;
        }
        true
    }

    /// Log an LLM attempt and keep its result only if it produced text.
    fn llm_attempt_outcome(file_path: &str, attempt: Result<OcrResult>) -> Option<OcrResult> {
        match attempt {
            Ok(result) if result.word_count > 0 => {
                info!("LLM OCR successful for '{}': {} words in {}ms", file_path, result.word_count, result.processing_time_ms);
                Some(result)
            }
            Ok(_) => {
                warn!("LLM OCR returned no text for '{}', falling back to Tesseract", file_path);
                None
            }
            Err(e) => {
                warn!("LLM OCR failed for '{}': {}, falling back to Tesseract", file_path, e);
                None
            }
        }
    }

    impl EnhancedOcrService {
        /// OCR a PDF or image with the configured LLM. `None` means LLM OCR is
        /// disabled, failed, timed out or produced no text, and the caller should
        /// run the Tesseract pipeline instead.
        pub(in crate::ocr) async fn try_llm_ocr(&self, file_path: &str, mime_type: &str, progress_callback: Option<ProgressCallback>) -> Option<OcrResult> {
            let llm = self.llm_ocr.as_ref()?;
            if !llm_should_handle(llm, file_path, mime_type) {
                return None;
            }
            let start_time = std::time::Instant::now();
            info!("Starting LLM OCR (model '{}') for: {}", llm.model(), file_path);

            let attempt = if mime_type == "application/pdf" {
                self.extract_text_from_pdf_with_llm(llm, file_path, start_time, progress_callback).await
            } else {
                self.extract_text_from_image_with_llm(llm, file_path, mime_type, start_time).await
            };

            llm_attempt_outcome(file_path, attempt)
        }

        async fn extract_text_from_image_with_llm(&self, llm: &LlmOcrClient, file_path: &str, mime_type: &str, start_time: std::time::Instant) -> Result<OcrResult> {
            let bytes = tokio::fs::read(file_path).await?;

            // Vision APIs reliably accept PNG/JPEG/WebP/GIF; convert anything else to PNG.
            let (bytes, mime) = match mime_type {
                "image/png" | "image/webp" | "image/gif" => (bytes, mime_type.to_string()),
                "image/jpeg" | "image/jpg" => (bytes, "image/jpeg".to_string()),
                _ => {
                    let png = tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
                        let img = image::load_from_memory(&bytes)?;
                        let mut out = std::io::Cursor::new(Vec::new());
                        img.write_to(&mut out, image::ImageFormat::Png)?;
                        Ok(out.into_inner())
                    }).await??;
                    (png, "image/png".to_string())
                }
            };

            let text = Self::remove_null_bytes(&llm.transcribe_image(&bytes, &mime).await?);
            let word_count = self.count_words_safely(&text);

            Ok(OcrResult {
                text,
                confidence: LLM_OCR_CONFIDENCE,
                processing_time_ms: start_time.elapsed().as_millis() as u64,
                word_count,
                preprocessing_applied: vec![format!("LLM OCR ({})", llm.model())],
                processed_image_path: None,
            })
        }

        /// Render each PDF page to PNG and transcribe it with the LLM. Any page
        /// failing fails the whole document so the fallback produces consistent text.
        async fn extract_text_from_pdf_with_llm(&self, llm: &LlmOcrClient, file_path: &str, start_time: std::time::Instant, progress_callback: Option<ProgressCallback>) -> Result<OcrResult> {
            let file_size = tokio::fs::metadata(file_path).await?.len();
            if file_size > self.max_pdf_size {
                return Err(anyhow!("PDF exceeds the configured size limit"));
            }

            let page_count = self.get_pdf_page_count(file_path).await?;
            if let Some(ref cb) = progress_callback {
                cb(0, page_count as i32);
            }

            let image_paths = self.extract_pdf_pages_as_images(file_path, page_count, LLM_OCR_PDF_DPI).await?;
            let _cleanup: Vec<FileCleanupGuard> = image_paths.iter().map(|p| FileCleanupGuard::new(p)).collect();
            if image_paths.len() != page_count {
                return Err(anyhow!("rendered {} of {} PDF pages", image_paths.len(), page_count));
            }

            let mut all_text = String::new();
            for (i, image_path) in image_paths.iter().enumerate() {
                info!("LLM OCR processing page {}/{}", i + 1, page_count);
                let bytes = tokio::fs::read(image_path).await?;
                let page_text = llm
                    .transcribe_image(&bytes, "image/png")
                    .await
                    .map_err(|e| anyhow!("page {}: {}", i + 1, e))?;

                if !all_text.is_empty() && !page_text.is_empty() {
                    all_text.push_str("\n\n--- Page Break ---\n\n");
                }
                all_text.push_str(&page_text);

                let _ = tokio::fs::remove_file(image_path).await;
                if let Some(ref cb) = progress_callback {
                    cb((i + 1) as i32, page_count as i32);
                }
            }

            let text = Self::remove_null_bytes(&all_text);
            let word_count = self.count_words_safely(&text);

            Ok(OcrResult {
                text,
                confidence: LLM_OCR_CONFIDENCE,
                processing_time_ms: start_time.elapsed().as_millis() as u64,
                word_count,
                preprocessing_applied: vec![format!("LLM OCR ({}, {} pages)", llm.model(), image_paths.len())],
                processed_image_path: None,
            })
        }
    }
}
