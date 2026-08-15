//! WASM bindings for PrismCode in-browser scanner and embedder.
//!
//! The native frame-scanning pipeline lives in [`scanner`] (plain Rust, no
//! wasm-bindgen); the bindings here are thin delegating wrappers.

pub mod scanner;

use wasm_bindgen::prelude::*;
use image::{RgbImage, Rgb};
use prism_render::embedder::PrismEmbedder;

#[wasm_bindgen(start)]
pub fn main_js() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Stateful scanner binding: keeps the extractor and scratch buffers alive
/// across `scan_frame` calls so same-size frames are not reallocated per call.
#[wasm_bindgen]
pub struct PrismScanner {
    scanner: scanner::FrameScanner,
}

impl Default for PrismScanner {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl PrismScanner {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            scanner: scanner::FrameScanner::new(),
        }
    }

    /// Scans a single RGBA camera video frame.
    /// Returns the decoded UTF-8 string payload (trailing NUL padding
    /// trimmed) or null if no valid frame detected.
    pub fn scan_frame(&mut self, rgba_data: &[u8], width: u32, height: u32) -> Option<String> {
        self.scanner.decode_camera_frame(rgba_data, width, height)
    }
}

/// One-shot scan of an RGBA camera frame via a thread-local shared scanner
/// (scratch buffers reused across calls). Returns the decoded UTF-8 string
/// payload (trailing NUL padding trimmed) or null on failure.
#[wasm_bindgen]
pub fn decode_camera_frame(rgba_data: &[u8], width: u32, height: u32) -> Option<String> {
    scanner::decode_camera_frame(rgba_data, width, height)
}

/// Embeds `payload` into an RGBA image and returns the embedded RGBA bytes.
#[wasm_bindgen]
pub fn prism_embed_rgba(rgba_data: &[u8], width: u32, height: u32, payload: &str) -> Result<Vec<u8>, JsValue> {
    if width < 192 || height < 128 || rgba_data.len() < (width * height * 4) as usize {
        return Err(JsValue::from_str("Invalid image dimensions"));
    }

    let mut rgb_img = RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            rgb_img.put_pixel(x, y, Rgb([rgba_data[idx], rgba_data[idx + 1], rgba_data[idx + 2]]));
        }
    }

    let embedder = PrismEmbedder::new();
    let embedded_rgb = embedder.embed(&rgb_img, payload.as_bytes())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Convert back to RGBA
    let mut out_rgba = vec![255u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let px = embedded_rgb.get_pixel(x, y);
            let idx = ((y * width + x) * 4) as usize;
            out_rgba[idx] = px[0];
            out_rgba[idx + 1] = px[1];
            out_rgba[idx + 2] = px[2];
            out_rgba[idx + 3] = 255;
        }
    }

    Ok(out_rgba)
}
