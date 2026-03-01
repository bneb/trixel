//! # trixel_scanner
//!
//! WebAssembly-compatible decoder for Trixel ternary matrix codes.
//! Exposes a single function that takes PNG bytes and returns the decoded string.
//!
//! Used by the PWA camera scanner to decode matrices on-device.

use image::DynamicImage;
use wasm_bindgen::prelude::*;

use trixel_core::{ErrorCorrection, MockCodec, RsEcc, TernaryCodec, TritMatrix};
use trixel_cv::{AnchorVision, VisionPipeline};
use trixel_solver::anchor::is_in_anchor_region;

// ---------------------------------------------------------------------------
// Core decode pipeline (platform-agnostic)
// ---------------------------------------------------------------------------

/// Decode a Trixel matrix from an already-loaded `DynamicImage`.
///
/// This is the platform-agnostic core: it does not touch any Web APIs.
/// Both the WASM entry point and native tests call this function.
pub fn decode_image(image: &DynamicImage, module_size: u32) -> Result<String, String> {
    // 1. Extract trit matrix via anchor-calibrated CV pipeline
    let matrix = AnchorVision::extract_matrix(image, module_size)
        .map_err(|e| format!("Vision extraction failed: {e}"))?;

    // 2. Extract payload trits from non-anchor cells
    let payload = extract_payload(&matrix);

    // 3. Reed-Solomon error correction
    let clean = RsEcc::correct_errors(&payload)
        .map_err(|e| format!("ECC correction failed: {e}"))?;

    // 4. Trits → bytes → UTF-8 string
    let bytes = MockCodec::decode_trits(&clean)
        .map_err(|e| format!("Codec decode failed: {e}"))?;

    String::from_utf8(bytes)
        .map_err(|e| format!("UTF-8 decode failed: {e}"))
}

/// Decode a Trixel matrix from raw PNG bytes.
///
/// This is convenient for both WASM (where the JS layer passes a `Uint8Array`)
/// and native code (where you can `std::fs::read` the file).
pub fn decode_png_bytes(png_bytes: &[u8], module_size: u32) -> Result<String, String> {
    let image = image::load_from_memory(png_bytes)
        .map_err(|e| format!("Failed to load image: {e}"))?;
    decode_image(&image, module_size)
}

// ---------------------------------------------------------------------------
// WASM entry point
// ---------------------------------------------------------------------------

/// Decode a Trixel PNG image from raw bytes.
///
/// Called from JavaScript:
/// ```js
/// const result = decode_png(new Uint8Array(pngBuffer), 10);
/// ```
///
/// Returns the decoded string (e.g., a URL) or throws on error.
#[wasm_bindgen]
pub fn decode_png(png_bytes: &[u8], module_size: u32) -> Result<String, JsValue> {
    decode_png_bytes(png_bytes, module_size)
        .map_err(|e| JsValue::from_str(&e))
}

/// Try multiple common module sizes and return the first successful decode.
///
/// This is the "just scan it" entry point — the user doesn't need to know
/// what module size was used during encoding.
#[wasm_bindgen]
pub fn decode_png_auto(png_bytes: &[u8]) -> Result<String, JsValue> {
    // Try common module sizes from largest to smallest
    let sizes = [10u32, 8, 6, 4, 12, 16, 5, 3];
    let mut last_err = String::new();

    for &ms in &sizes {
        match decode_png_bytes(png_bytes, ms) {
            Ok(result) => return Ok(result),
            Err(e) => last_err = e,
        }
    }

    Err(JsValue::from_str(&format!(
        "Failed to decode with any module size. Last error: {last_err}"
    )))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Extract payload trits from an anchor-bearing matrix by skipping anchor regions.
fn extract_payload(matrix: &TritMatrix) -> Vec<u8> {
    let n = matrix.width; // assumes square
    let mut payload = Vec::new();
    for y in 0..matrix.height {
        for x in 0..matrix.width {
            if !is_in_anchor_region(x, y, n) {
                payload.push(matrix.get(x, y).unwrap_or(0));
            }
        }
    }
    payload
}
