//! Helpers for the `pdftoppm` page renderer used by the PDF OCR pipeline.

/// Path of the PNG `pdftoppm -png <pdf> <prefix>` writes for `page_num`.
/// pdftoppm zero-pads page numbers to the width of the page count, so a
/// 12-page PDF produces `<prefix>-01.png` through `<prefix>-12.png`.
pub fn pdftoppm_page_path(prefix: &str, page_num: usize, page_count: usize) -> String {
    let width = page_count.to_string().len();
    format!("{}-{:0width$}.png", prefix, page_num, width = width)
}
