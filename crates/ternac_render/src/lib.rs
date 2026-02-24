//! # ternac_render
//!
//! Rendering engine for the Ternac system.
//! Converts a `TritMatrix` into a physical PNG and provides the `FontEngine`
//! for mapping text into spatial constraints.

use image::{DynamicImage, ImageBuffer, RgbImage};
use ternac_core::TritMatrix;
use ternac_solver::ConstraintMask;
use thiserror::Error;

pub mod glyphs;
pub mod font;

pub use font::TernacFont;

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum RenderError {
    #[error("matrix is empty")]
    EmptyMatrix,
    #[error("module_pixel_size must be > 0")]
    ZeroModuleSize,
}

// ---------------------------------------------------------------------------
// Renderer Trait
// ---------------------------------------------------------------------------

/// Converts a solved `TritMatrix` into a physical image.
pub trait Renderer {
    /// Renders the matrix as a PNG-compatible image.
    /// Luminance is monotonically increasing with state value:
    /// - State 0 → Black `[0, 0, 0]`       (0–20% luminance band)
    /// - State 1 → `state_1_rgb` (accent)   (40–60% luminance band)
    /// - State 2 → White `[255, 255, 255]`  (80–100% luminance band)
    fn render_png(
        matrix: &TritMatrix,
        module_pixel_size: u32,
        state_1_rgb: [u8; 3],
    ) -> Result<DynamicImage, RenderError>;
}

/// Maps text strings into spatial constraints on the grid.
pub trait FontEngine {
    /// Converts a text string into a list of cell constraints starting at `(start_x, start_y)`.
    fn string_to_constraints(
        text: &str,
        start_x: usize,
        start_y: usize,
    ) -> Vec<ConstraintMask>;
}

// ---------------------------------------------------------------------------
// Mock Implementations
// ---------------------------------------------------------------------------

/// Mock renderer: produces a flat-color grid with no anchors or quiet zones.
pub struct MockRenderer;

impl Renderer for MockRenderer {
    fn render_png(
        matrix: &TritMatrix,
        module_pixel_size: u32,
        state_1_rgb: [u8; 3],
    ) -> Result<DynamicImage, RenderError> {
        if matrix.width == 0 || matrix.height == 0 {
            return Err(RenderError::EmptyMatrix);
        }
        if module_pixel_size == 0 {
            return Err(RenderError::ZeroModuleSize);
        }

        let img_w = matrix.width as u32 * module_pixel_size;
        let img_h = matrix.height as u32 * module_pixel_size;
        let mut img: RgbImage = ImageBuffer::new(img_w, img_h);

        for gy in 0..matrix.height {
            for gx in 0..matrix.width {
                let trit = matrix.get(gx, gy).unwrap_or(0);
                let color = match trit {
                    0 => [0u8, 0, 0],          // black  (lowest luminance)
                    1 => state_1_rgb,          // accent (mid luminance)
                    2 => [255u8, 255, 255],    // white  (highest luminance)
                    _ => [128u8, 128, 128],    // erasure → gray
                };
                let px_x = gx as u32 * module_pixel_size;
                let px_y = gy as u32 * module_pixel_size;
                for dy in 0..module_pixel_size {
                    for dx in 0..module_pixel_size {
                        img.put_pixel(px_x + dx, px_y + dy, image::Rgb(color));
                    }
                }
            }
        }

        Ok(DynamicImage::ImageRgb8(img))
    }
}

/// Mock font engine: returns an empty constraint list.
/// Real glyph rasterization comes in Phase 3.
pub struct MockFontEngine;

impl FontEngine for MockFontEngine {
    fn string_to_constraints(
        _text: &str,
        _start_x: usize,
        _start_y: usize,
    ) -> Vec<ConstraintMask> {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// Anchor-Aware Renderer
// ---------------------------------------------------------------------------

/// Production renderer: renders a `TritMatrix` that already contains L-bracket
/// anchors (placed by `AnchorSolver`). Rendering logic is identical to
/// `MockRenderer` — the anchor patterns are embedded in the matrix data.
pub struct AnchorRenderer;

impl Renderer for AnchorRenderer {
    fn render_png(
        matrix: &TritMatrix,
        module_pixel_size: u32,
        state_1_rgb: [u8; 3],
    ) -> Result<DynamicImage, RenderError> {
        if matrix.width == 0 || matrix.height == 0 {
            return Err(RenderError::EmptyMatrix);
        }
        if module_pixel_size == 0 {
            return Err(RenderError::ZeroModuleSize);
        }

        let img_w = matrix.width as u32 * module_pixel_size;
        let img_h = matrix.height as u32 * module_pixel_size;
        let mut img: RgbImage = ImageBuffer::new(img_w, img_h);

        for gy in 0..matrix.height {
            for gx in 0..matrix.width {
                let trit = matrix.get(gx, gy).unwrap_or(0);
                let color = match trit {
                    0 => [0u8, 0, 0],
                    1 => state_1_rgb,
                    2 => [255u8, 255, 255],
                    _ => [128u8, 128, 128],
                };
                let px_x = gx as u32 * module_pixel_size;
                let px_y = gy as u32 * module_pixel_size;
                for dy in 0..module_pixel_size {
                    for dx in 0..module_pixel_size {
                        img.put_pixel(px_x + dx, px_y + dy, image::Rgb(color));
                    }
                }
            }
        }

        Ok(DynamicImage::ImageRgb8(img))
    }
}
