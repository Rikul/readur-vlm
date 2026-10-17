//! LLM OCR (`LLM_OCR=true`): documents are transcribed by the vision LLM, and
//! fall back to the Tesseract/ocrmypdf pipeline when the endpoint errors or
//! hangs past `LLM_OCR_TIMEOUT_SECONDS`. The batch tests mirror a frontend
//! upload, where every file becomes its own OCR job and the worker runs them
//! concurrently against one shared service.

#![cfg(feature = "ocr")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use readur::models::Settings;
use readur::ocr::enhanced::EnhancedOcrService;
use readur::ocr::llm::{LlmOcrClient, LlmOcrConfig, LlmOcrFileTypes, DEFAULT_LLM_OCR_PROMPT};
use readur::services::file_service::FileService;
use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

const LLM_TEXT: &str = "Faded ledger entry, 14th March 1887: received of Mr. Hargreaves twelve shillings.";

fn llm_config(base_uri: &str, timeout_seconds: u64, max_concurrent_requests: usize) -> LlmOcrConfig {
    LlmOcrConfig {
        api_url: format!("{}/v1", base_uri),
        model: "test-vision-model".to_string(),
        api_key: Some("test-key".to_string()),
        timeout_seconds,
        max_concurrent_requests,
        file_types: LlmOcrFileTypes::All,
        prompt: DEFAULT_LLM_OCR_PROMPT.to_string(),
    }
}

fn completion(text: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "choices": [{ "message": { "role": "assistant", "content": text } }]
    }))
}

fn service(temp_dir: &tempfile::TempDir, llm: Option<LlmOcrConfig>) -> EnhancedOcrService {
    let temp_path = temp_dir.path().to_string_lossy().to_string();
    #[allow(deprecated)]
    let file_service = FileService::new(temp_path.clone());
    let service = EnhancedOcrService::new(temp_path, file_service, 100, 100, 300);
    match llm {
        Some(config) => service.with_llm_ocr(LlmOcrClient::new(config).unwrap()),
        None => service,
    }
}

fn fixture(name: &str) -> String {
    format!("{}/test_files/{}", env!("CARGO_MANIFEST_DIR"), name)
}

fn used_llm(preprocessing: &[String]) -> bool {
    preprocessing.iter().any(|step| step.starts_with("LLM OCR"))
}

#[tokio::test]
async fn image_is_ocrd_by_llm_when_enabled() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_partial_json(json!({ "model": "test-vision-model" })))
        .respond_with(completion(LLM_TEXT))
        .expect(1)
        .mount(&server)
        .await;

    let temp_dir = tempfile::tempdir().unwrap();
    let ocr = service(&temp_dir, Some(llm_config(&server.uri(), 10, 2)));

    let result = ocr
        .extract_text(&fixture("test_image.jpg"), "image/jpeg", &Settings::default(), None)
        .await
        .unwrap();

    assert_eq!(result.text, LLM_TEXT);
    assert!(used_llm(&result.preprocessing_applied), "{:?}", result.preprocessing_applied);
    assert!(ocr.validate_ocr_quality(&result, &Settings::default()).is_ok());

    // The image is sent inline as a data URL.
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = requests[0].body_json().unwrap();
    let url = body.pointer("/messages/0/content/1/image_url/url").and_then(|v| v.as_str()).unwrap();
    assert!(url.starts_with("data:image/jpeg;base64,"));
}

#[tokio::test]
async fn image_falls_back_to_tesseract_when_llm_hangs_past_timeout() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(completion(LLM_TEXT).set_delay(Duration::from_secs(30)))
        .mount(&server)
        .await;

    let temp_dir = tempfile::tempdir().unwrap();
    let ocr = service(&temp_dir, Some(llm_config(&server.uri(), 1, 2)));

    let started = Instant::now();
    let result = ocr
        .extract_text(&fixture("test_image.jpg"), "image/jpeg", &Settings::default(), None)
        .await
        .unwrap();

    assert!(started.elapsed() < Duration::from_secs(20), "did not give up on the LLM at the timeout");
    assert!(!used_llm(&result.preprocessing_applied), "{:?}", result.preprocessing_applied);
    assert!(!result.text.contains("Hargreaves"));
    assert!(result.text.contains("JPEG"), "expected Tesseract text, got {:?}", result.text);
}

#[tokio::test]
async fn image_falls_back_to_tesseract_when_llm_errors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("model not loaded"))
        .mount(&server)
        .await;

    let temp_dir = tempfile::tempdir().unwrap();
    let ocr = service(&temp_dir, Some(llm_config(&server.uri(), 10, 2)));

    let result = ocr
        .extract_text(&fixture("test_image.jpg"), "image/jpeg", &Settings::default(), None)
        .await
        .unwrap();

    assert!(!used_llm(&result.preprocessing_applied));
    assert!(result.text.contains("JPEG"), "expected Tesseract text, got {:?}", result.text);
}

#[tokio::test]
async fn pdf_is_ocrd_by_llm_page_by_page() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(completion(LLM_TEXT))
        .expect(1)
        .mount(&server)
        .await;

    let temp_dir = tempfile::tempdir().unwrap();
    let ocr = service(&temp_dir, Some(llm_config(&server.uri(), 10, 2)));

    let result = ocr
        .extract_text(&fixture("single_page_v14.pdf"), "application/pdf", &Settings::default(), None)
        .await
        .unwrap();

    assert_eq!(result.text, LLM_TEXT);
    assert!(used_llm(&result.preprocessing_applied), "{:?}", result.preprocessing_applied);

    let requests: Vec<Request> = server.received_requests().await.unwrap();
    let body: serde_json::Value = requests[0].body_json().unwrap();
    let url = body.pointer("/messages/0/content/1/image_url/url").and_then(|v| v.as_str()).unwrap();
    assert!(url.starts_with("data:image/png;base64,"));

    // Rendered page images are cleaned up.
    let mut entries = tokio::fs::read_dir(temp_dir.path()).await.unwrap();
    while let Some(entry) = entries.next_entry().await.unwrap() {
        let name = entry.file_name().to_string_lossy().to_string();
        assert!(!name.ends_with(".png"), "leftover page image {}", name);
    }
}

#[tokio::test]
async fn pdf_falls_back_to_existing_pipeline_when_llm_hangs_past_timeout() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(completion(LLM_TEXT).set_delay(Duration::from_secs(30)))
        .mount(&server)
        .await;

    let temp_dir = tempfile::tempdir().unwrap();
    let ocr = service(&temp_dir, Some(llm_config(&server.uri(), 1, 2)));

    let started = Instant::now();
    let result = ocr
        .extract_text(&fixture("single_page_v14.pdf"), "application/pdf", &Settings::default(), None)
        .await
        .unwrap();

    assert!(started.elapsed() < Duration::from_secs(20), "did not give up on the LLM at the timeout");
    assert!(!used_llm(&result.preprocessing_applied), "{:?}", result.preprocessing_applied);
    assert!(result.text.contains("Single Page PDF Document"), "got {:?}", result.text);
}

#[tokio::test]
async fn llm_is_not_called_when_disabled() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(completion(LLM_TEXT))
        .expect(0)
        .mount(&server)
        .await;

    let temp_dir = tempfile::tempdir().unwrap();
    let ocr = service(&temp_dir, None);

    let result = ocr
        .extract_text(&fixture("test_image.jpg"), "image/jpeg", &Settings::default(), None)
        .await
        .unwrap();

    assert!(!used_llm(&result.preprocessing_applied));
    assert!(result.text.contains("JPEG"));
}

#[test]
fn pdftoppm_page_paths_match_its_zero_padding() {
    use readur::ocr::enhanced::pdftoppm_page_path;
    assert_eq!(pdftoppm_page_path("/t/p", 3, 3), "/t/p-3.png");
    assert_eq!(pdftoppm_page_path("/t/p", 3, 12), "/t/p-03.png");
    assert_eq!(pdftoppm_page_path("/t/p", 12, 12), "/t/p-12.png");
    assert_eq!(pdftoppm_page_path("/t/p", 7, 468), "/t/p-007.png");
}

#[tokio::test]
async fn pdf_with_ten_or_more_pages_sends_every_page_to_llm() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(NumberedTranscriptions(AtomicUsize::new(0)))
        .mount(&server)
        .await;

    // Build a 12-page PDF from the 3-page fixture so pdftoppm pads page numbers.
    let temp_dir = tempfile::tempdir().unwrap();
    let pdf_path = temp_dir.path().join("twelve_pages.pdf");
    let mut args: Vec<String> = std::iter::repeat(fixture("multipage_test.pdf")).take(4).collect();
    args.push(pdf_path.to_string_lossy().to_string());
    let status = std::process::Command::new("pdfunite").args(&args).status().expect("pdfunite (poppler-utils) is required");
    assert!(status.success());

    let ocr = service(&temp_dir, Some(llm_config(&server.uri(), 10, 2)));
    let result = ocr
        .extract_text(&pdf_path.to_string_lossy(), "application/pdf", &Settings::default(), None)
        .await
        .unwrap();

    assert!(used_llm(&result.preprocessing_applied), "{:?}", result.preprocessing_applied);
    assert_eq!(server.received_requests().await.unwrap().len(), 12);
    assert_eq!(tokens(&result.text).len(), 12, "{:?}", result.text);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn images_only_sends_images_to_llm_and_pdfs_through_the_standard_pipeline() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(completion(LLM_TEXT))
        .expect(2) // the two images; never the PDFs
        .mount(&server)
        .await;

    let temp_dir = tempfile::tempdir().unwrap();
    let mut config = llm_config(&server.uri(), 10, 2);
    config.file_types = LlmOcrFileTypes::Images;
    let ocr = Arc::new(service(&temp_dir, Some(config)));

    let files = [
        ("test_image.jpg", "image/jpeg"),
        ("single_page_v14.pdf", "application/pdf"),
        ("landscape_300x200.png", "image/png"),
        ("multipage_test.pdf", "application/pdf"),
    ];
    let results = ocr_batch(ocr, &files).await;

    for ((name, mime), result) in files.iter().zip(&results) {
        if mime.starts_with("image/") {
            assert!(used_llm(&result.preprocessing_applied), "{}: {:?}", name, result.preprocessing_applied);
            assert_eq!(result.text, LLM_TEXT);
        } else {
            assert!(!used_llm(&result.preprocessing_applied), "{}: {:?}", name, result.preprocessing_applied);
            assert!(!result.text.contains("Hargreaves"), "{}: {:?}", name, result.text);
        }
    }
    assert!(results[1].text.contains("Single Page PDF Document"), "{:?}", results[1].text);
}

// --- Batches -----------------------------------------------------------------

/// Answers each request with a distinct token so a batch can prove every page
/// got its own transcription and nothing was mixed between documents.
struct NumberedTranscriptions(AtomicUsize);

impl Respond for NumberedTranscriptions {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        let n = self.0.fetch_add(1, Ordering::SeqCst) + 1;
        completion(&format!("Archive transcription token{:03} recovered", n))
    }
}

fn tokens(text: &str) -> Vec<String> {
    text.split_whitespace().filter(|w| w.starts_with("token")).map(str::to_string).collect()
}

/// Run every file through one shared service at the same time, the way the
/// OCR worker processes a frontend batch.
async fn ocr_batch(ocr: Arc<EnhancedOcrService>, files: &[(&str, &str)]) -> Vec<readur::ocr::enhanced::OcrResult> {
    let tasks: Vec<_> = files
        .iter()
        .map(|(name, mime)| {
            let ocr = ocr.clone();
            let (path, mime) = (fixture(name), mime.to_string());
            tokio::spawn(async move { ocr.extract_text(&path, &mime, &Settings::default(), None).await })
        })
        .collect();
    let mut results = Vec::new();
    for task in tasks {
        results.push(task.await.unwrap().unwrap());
    }
    results
}

/// Stand-in for a self-hosted LLM server: it works on `slots` requests at a
/// time and queues the rest, like Ollama does.
struct FakeLlmServer {
    slots: tokio::sync::Semaphore,
    work: Duration,
    requests: AtomicUsize,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
}

struct InFlightGuard(Arc<FakeLlmServer>);

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

async fn fake_llm_handler(axum::extract::State(server): axum::extract::State<Arc<FakeLlmServer>>) -> axum::Json<serde_json::Value> {
    server.requests.fetch_add(1, Ordering::SeqCst);
    let now = server.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
    server.max_in_flight.fetch_max(now, Ordering::SeqCst);
    let _guard = InFlightGuard(server.clone());

    let _slot = server.slots.acquire().await.unwrap();
    tokio::time::sleep(server.work).await;
    axum::Json(json!({ "choices": [{ "message": { "role": "assistant", "content": LLM_TEXT } }] }))
}

async fn start_fake_llm(slots: usize, work: Duration) -> (String, Arc<FakeLlmServer>) {
    let server = Arc::new(FakeLlmServer {
        slots: tokio::sync::Semaphore::new(slots),
        work,
        requests: AtomicUsize::new(0),
        in_flight: AtomicUsize::new(0),
        max_in_flight: AtomicUsize::new(0),
    });
    let app = axum::Router::new()
        .route("/v1/chat/completions", axum::routing::post(fake_llm_handler))
        .with_state(server.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_uri = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (base_uri, server)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn batch_of_images_and_pdfs_is_ocrd_by_llm() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(NumberedTranscriptions(AtomicUsize::new(0)))
        .mount(&server)
        .await;

    let temp_dir = tempfile::tempdir().unwrap();
    let ocr = Arc::new(service(&temp_dir, Some(llm_config(&server.uri(), 10, 4))));

    let files = [
        ("test_image.jpg", "image/jpeg"),
        ("exif_orientation_1_normal.jpg", "image/jpeg"),
        ("exif_orientation_6_rotate_90_cw.jpg", "image/jpeg"),
        ("landscape_300x200.png", "image/png"),
        ("square_150x150.png", "image/png"),
        ("portrait_100x200.png", "image/png"),
        ("multipage_test.pdf", "application/pdf"),
        ("single_page_v14.pdf", "application/pdf"),
        ("multipage_test.pdf", "application/pdf"),
    ];
    let results = ocr_batch(ocr, &files).await;

    // One request per image plus one per PDF page (multipage_test.pdf has 3).
    let expected_requests = 6 + 3 + 1 + 3;
    assert_eq!(server.received_requests().await.unwrap().len(), expected_requests);

    let mut all_tokens = Vec::new();
    for ((name, _), result) in files.iter().zip(&results) {
        assert!(used_llm(&result.preprocessing_applied), "{} not OCR'd by the LLM: {:?}", name, result.preprocessing_applied);
        let doc_tokens = tokens(&result.text);
        let expected_pages = if *name == "multipage_test.pdf" { 3 } else { 1 };
        assert_eq!(doc_tokens.len(), expected_pages, "{}: {:?}", name, result.text);
        if expected_pages > 1 {
            assert_eq!(result.text.matches("--- Page Break ---").count(), expected_pages - 1);
        }
        all_tokens.extend(doc_tokens);
    }

    // Every transcription landed in exactly one document.
    all_tokens.sort();
    let expected: Vec<String> = (1..=expected_requests).map(|n| format!("token{:03}", n)).collect();
    assert_eq!(all_tokens, expected);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn batch_waits_for_a_free_slot_without_timing_out() {
    // The server handles one request at a time, 300ms each. Eight images
    // take ~2.4s end to end, well past the 1s timeout, yet each request only
    // needs 300ms once it is sent, so none should fall back to Tesseract.
    let (base_uri, server) = start_fake_llm(1, Duration::from_millis(300)).await;

    let temp_dir = tempfile::tempdir().unwrap();
    let ocr = Arc::new(service(&temp_dir, Some(llm_config(&base_uri, 1, 1))));

    let files = [("test_image.jpg", "image/jpeg"); 8];
    let results = ocr_batch(ocr, &files).await;

    for result in &results {
        assert!(used_llm(&result.preprocessing_applied), "fell back to Tesseract: {:?}", result.preprocessing_applied);
        assert_eq!(result.text, LLM_TEXT);
    }
    assert_eq!(server.requests.load(Ordering::SeqCst), 8);
    assert_eq!(server.max_in_flight.load(Ordering::SeqCst), 1, "client sent more requests than LLM_OCR_MAX_CONCURRENT_REQUESTS");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn batch_with_hung_llm_falls_back_to_tesseract_without_each_file_waiting_out_the_timeout() {
    let (base_uri, server) = start_fake_llm(1, Duration::from_secs(60)).await;

    let temp_dir = tempfile::tempdir().unwrap();
    let ocr = Arc::new(service(&temp_dir, Some(llm_config(&base_uri, 1, 1))));

    // Ten files one at a time against a 1s timeout would take 10s+ if every
    // file waited out its own timeout.
    let files = [("test_image.jpg", "image/jpeg"); 10];
    let started = Instant::now();
    let results = ocr_batch(ocr, &files).await;
    let elapsed = started.elapsed();

    for result in &results {
        assert!(!used_llm(&result.preprocessing_applied), "{:?}", result.preprocessing_applied);
        assert!(result.text.contains("JPEG"), "expected Tesseract text, got {:?}", result.text);
    }
    // Two consecutive timeouts mark the endpoint unresponsive; the rest skip it.
    assert_eq!(server.requests.load(Ordering::SeqCst), 2);
    assert!(elapsed < Duration::from_secs(8), "batch took {:?}", elapsed);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn batch_with_hung_llm_falls_back_for_pdfs_too() {
    let (base_uri, server) = start_fake_llm(1, Duration::from_secs(60)).await;

    let temp_dir = tempfile::tempdir().unwrap();
    let ocr = Arc::new(service(&temp_dir, Some(llm_config(&base_uri, 1, 1))));

    let files = [
        ("single_page_v14.pdf", "application/pdf"),
        ("test_image.jpg", "image/jpeg"),
        ("single_page_v14.pdf", "application/pdf"),
        ("test_image.jpg", "image/jpeg"),
        ("single_page_v14.pdf", "application/pdf"),
    ];
    let results = ocr_batch(ocr, &files).await;

    for ((name, mime), result) in files.iter().zip(&results) {
        assert!(!used_llm(&result.preprocessing_applied), "{}: {:?}", name, result.preprocessing_applied);
        let expected = if *mime == "application/pdf" { "Single Page PDF Document" } else { "JPEG" };
        assert!(result.text.contains(expected), "{}: got {:?}", name, result.text);
    }
    assert_eq!(server.requests.load(Ordering::SeqCst), 2);
}
