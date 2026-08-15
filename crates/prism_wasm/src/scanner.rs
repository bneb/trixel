//! Native frame-scanning pipeline (no wasm-bindgen in this module).
//!
//! `FrameScanner` wraps the CV extractor with reusable scratch buffers: the
//! RGBA -> RGB staging image is only reallocated when the frame dimensions
//! change, so repeated scans of same-size frames avoid per-call allocations
//! for the conversion. The one-shot `decode_camera_frame` entry point routes
//! through a thread-local shared scanner for the same reason.
//!
//! The decoded payload is the full 32-byte codeword (zero padding included);
//! trailing NUL trimming happens only when building the display `String`.

use image::{Rgb, RgbImage};
use prism_cv::extractor::PrismExtractor;

/// Minimum frame size accepted by the extractor (blocks: 32x24 of 6x5 px).
pub const MIN_FRAME_WIDTH: u32 = 192;
pub const MIN_FRAME_HEIGHT: u32 = 128;

/// Reusable single-frame scanner with scratch buffers kept across calls.
pub struct FrameScanner {
    extractor: PrismExtractor,
    /// RGBA -> RGB staging image, reallocated only when dimensions change.
    rgb: RgbImage,
    /// Scratch for the display-string path (full payload, NUL-trimmed read).
    scratch: Vec<u8>,
}

impl Default for FrameScanner {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameScanner {
    pub fn new() -> Self {
        Self {
            extractor: PrismExtractor::new(),
            rgb: RgbImage::new(0, 0),
            scratch: Vec::new(),
        }
    }

    /// Scan a single RGBA camera frame, writing the full decoded payload
    /// (32 bytes for the z64 codec, zero padding included) into `out`.
    ///
    /// Returns `true` on success; `false` for invalid dimensions or a buffer
    /// too small for `width * height * 4`, when the LDPC decoder does not
    /// converge, or when the decoded payload is all zero bytes. The last case
    /// is a degenerate decode: with no detectable signal every block LLR is
    /// ~0 and the zero codeword is always syndrome-valid, so an all-zero
    /// payload is treated as "no payload present" (a blank frame, or an
    /// empty-payload embed) rather than as a successful read.
    /// On failure `out` is left empty.
    pub fn scan_into(
        &mut self,
        rgba: &[u8],
        width: u32,
        height: u32,
        out: &mut Vec<u8>,
    ) -> bool {
        out.clear();
        if !valid_frame(rgba, width, height) {
            return false;
        }

        if self.rgb.width() != width || self.rgb.height() != height {
            self.rgb = RgbImage::new(width, height);
        }
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 4) as usize;
                self.rgb
                    .put_pixel(x, y, Rgb([rgba[idx], rgba[idx + 1], rgba[idx + 2]]));
            }
        }

        match self.extractor.extract(&self.rgb) {
            Ok(payload) if payload.iter().any(|&b| b != 0) => {
                out.extend_from_slice(&payload);
                true
            }
            _ => false,
        }
    }

    /// Convenience scan returning the payload as a display `String` with
    /// trailing NUL bytes (codeword zero padding) trimmed.
    ///
    /// Returns `None` for invalid frames, LDPC non-convergence, all-zero
    /// payloads (no detectable signal), or payloads that are not valid UTF-8.
    pub fn decode_camera_frame(&mut self, rgba: &[u8], width: u32, height: u32) -> Option<String> {
        // Take the scratch out of `self` so scan_into's `&mut self` borrow does
        // not conflict with the `&mut Vec` it writes into.
        let mut scratch = std::mem::take(&mut self.scratch);
        let decoded = if self.scan_into(rgba, width, height, &mut scratch) {
            let trimmed = trim_trailing_nuls(&scratch);
            std::str::from_utf8(trimmed).ok().map(String::from)
        } else {
            None
        };
        self.scratch = scratch;
        decoded
    }
}

thread_local! {
    /// Shared scanner reused across `decode_camera_frame` calls so the
    /// wasm-bindgen entry point does not rebuild scratch buffers per frame.
    static SHARED_SCANNER: std::cell::RefCell<FrameScanner> =
        std::cell::RefCell::new(FrameScanner::new());
}

/// One-shot scan of an RGBA camera frame through a thread-local shared
/// scanner; returns the NUL-trimmed payload `String` (or `None` on failure).
pub fn decode_camera_frame(rgba: &[u8], width: u32, height: u32) -> Option<String> {
    SHARED_SCANNER.with(|s| s.borrow_mut().decode_camera_frame(rgba, width, height))
}

/// True when the frame dimensions meet the extractor minimum and the buffer
/// holds at least `width * height * 4` bytes (overflow-checked).
fn valid_frame(rgba: &[u8], width: u32, height: u32) -> bool {
    if width < MIN_FRAME_WIDTH || height < MIN_FRAME_HEIGHT {
        return false;
    }
    match (width as u64) * (height as u64) * 4 {
        need if need > usize::MAX as u64 => false,
        need => rgba.len() >= need as usize,
    }
}

/// Returns the payload slice without trailing 0x00 bytes.
fn trim_trailing_nuls(payload: &[u8]) -> &[u8] {
    let end = payload
        .iter()
        .rposition(|&b| b != 0)
        .map(|i| i + 1)
        .unwrap_or(0);
    &payload[..end]
}
