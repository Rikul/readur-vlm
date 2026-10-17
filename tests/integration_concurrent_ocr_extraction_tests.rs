//! Concurrent extraction through one shared `EnhancedOcrService`, as the OCR
//! worker does.

use std::sync::Arc;

use readur::models::Settings;
use readur::ocr::enhanced::EnhancedOcrService;
use readur::services::file_service::FileService;
use readur::storage::{factory::create_storage_backend, StorageConfig};
use tempfile::TempDir;

async fn create_test_file_service(temp_path: &str) -> FileService {
    let storage_config = StorageConfig::Local { upload_path: temp_path.to_string() };
    let storage_backend = create_storage_backend(storage_config).await.unwrap();
    FileService::with_storage(temp_path.to_string(), storage_backend)
}

/// The OCR worker extracts many documents at once. Temp files must be
/// unique per call, or one document can be saved with another's text.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_concurrent_pdf_extraction_keeps_each_documents_text() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let temp_path = temp_dir.path().to_str().unwrap().to_string();
    let file_service = create_test_file_service(&temp_path).await;
    let service = Arc::new(EnhancedOcrService::new(temp_path, file_service, 100, 100, 300));

    let fixtures = [
        ("single_page_v14.pdf", "Single Page PDF Document"),
        ("complex_content.pdf", "Document with Multiple Fonts"),
        ("multipage_test.pdf", "Page 1 of Multi-page Document"),
    ];

    let mut tasks = Vec::new();
    for _ in 0..8 {
        for (name, marker) in fixtures {
            let service = service.clone();
            let path = format!("{}/test_files/{}", env!("CARGO_MANIFEST_DIR"), name);
            tasks.push(tokio::spawn(async move {
                let result = service
                    .extract_text(&path, "application/pdf", &Settings::default(), None)
                    .await
                    .unwrap();
                (name, marker, result.text)
            }));
        }
    }

    for task in tasks {
        let (name, marker, text) = task.await.unwrap();
        assert!(text.contains(marker), "{} got another document's text: {:?}", name, text);
        for (other, other_marker) in fixtures {
            if other != name {
                assert!(!text.contains(other_marker), "{} contains text from {}: {:?}", name, other, text);
            }
        }
    }
}
