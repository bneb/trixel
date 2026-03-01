//! # ternac_render
//!
//! Rendering engine for the Ternac system.
//! Converts a `TritMatrix` into a physical PNG and provides the `FontEngine`
//! for mapping text into spatial constraints.

use image::{DynamicImage, GenericImageView, ImageBuffer, RgbImage};
use ternac_core::TritMatrix;
use ternac_solver::ConstraintMask;
use ternac_solver::anchor;
use thiserror::Error;
use palette::{Srgb, Hsl, IntoColor};

pub mod glyphs;
pub mod font;
pub mod halftone;

pub use font::TernacFont;
pub use halftone::HalftoneEngine;

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

impl AnchorRenderer {
    /// Context-aware halftone renderer with Typography Immunity.
    ///
    /// Takes the original resized image, the solved `TritMatrix`, and the raw
    /// `FontEngine` constraint mask. For every module:
    ///
    /// **Priority 1 (Font Immunity):** If `font_mask[y][x]` is `Some(state)`,
    /// the cell is painted with a hardcoded high-contrast color, bypassing HSL:
    ///   - `Some(0)` → pure black  `[0, 0, 0]`
    ///   - `Some(1)` → pure white  `[255, 255, 255]`  (stroke)
    ///   - `Some(2)` → pure black  `[0, 0, 0]`        (halo/quiet zone)
    ///
    /// **Priority 2 (Anchor Immunity):** Anchor regions use strict B/W.
    ///
    /// **Priority 3 (HSL Art):** All other cells fetch the original pixel's
    /// Hue and Saturation, then override Lightness per trit state.
    pub fn render_halftone_png(
        matrix: &TritMatrix,
        module_pixel_size: u32,
        original_image: &DynamicImage,
        font_mask: &[Vec<Option<u8>>],
    ) -> Result<DynamicImage, RenderError> {
        if matrix.width == 0 || matrix.height == 0 {
            return Err(RenderError::EmptyMatrix);
        }
        if module_pixel_size == 0 {
            return Err(RenderError::ZeroModuleSize);
        }

        let n = matrix.width; // assumes square

        // Resize the original image to match the matrix grid
        let scaled = original_image.resize_exact(
            n as u32,
            n as u32,
            image::imageops::FilterType::Lanczos3,
        );

        let img_w = n as u32 * module_pixel_size;
        let img_h = matrix.height as u32 * module_pixel_size;
        let mut img: RgbImage = ImageBuffer::new(img_w, img_h);

        for gy in 0..matrix.height {
            for gx in 0..n {
                let trit = matrix.get(gx, gy).unwrap_or(0);

                // Check font mask first (Typography Immunity)
                let font_state = if gy < font_mask.len() && gx < font_mask[gy].len() {
                    font_mask[gy][gx]
                } else {
                    None
                };

                let color = if let Some(fs) = font_state {
                    match fs {
                        // STROKE: pure opaque black for crisp letter lines.
                        0 => [0u8, 0, 0],
                        // FROSTED GLASS HALO: inherit the original pixel's
                        // hue and saturation, but force L=0.90.  This keeps
                        // the cell firmly in the State 2 (Light) CV band
                        // while letting the illustration bleed through.
                        2 => {
                            let orig_px = scaled.get_pixel(gx as u32, gy as u32);
                            let srgb = Srgb::new(
                                orig_px[0] as f32 / 255.0,
                                orig_px[1] as f32 / 255.0,
                                orig_px[2] as f32 / 255.0,
                            );
                            let hsl: Hsl = srgb.into_color();
                            let glass = Hsl::new(hsl.hue, hsl.saturation, 0.90);
                            let rgb: Srgb = glass.into_color();
                            [
                                (rgb.red * 255.0).round() as u8,
                                (rgb.green * 255.0).round() as u8,
                                (rgb.blue * 255.0).round() as u8,
                            ]
                        }
                        _ => [128u8, 128, 128], // fallback: mid-gray
                    }
                } else if anchor::is_in_anchor_region(gx, gy, n) {
                    // ANCHOR IMMUNITY: strict B/W for CV baseline.
                    match trit {
                        0 => [0u8, 0, 0],
                        1 => [128u8, 128, 128],
                        2 => [255u8, 255, 255],
                        _ => [128u8, 128, 128],
                    }
                } else {
                    // HSL ART: preserve original hue/saturation
                    let orig_px = scaled.get_pixel(gx as u32, gy as u32);
                    let srgb = Srgb::new(
                        orig_px[0] as f32 / 255.0,
                        orig_px[1] as f32 / 255.0,
                        orig_px[2] as f32 / 255.0,
                    );
                    let hsl: Hsl = srgb.into_color();

                    let target_l = match trit {
                        0 => 0.10,
                        1 => 0.50,
                        2 => 0.90,
                        _ => 0.50,
                    };

                    let modified_hsl = Hsl::new(hsl.hue, hsl.saturation, target_l);
                    let rgb: Srgb = modified_hsl.into_color();

                    [
                        (rgb.red * 255.0).round() as u8,
                        (rgb.green * 255.0).round() as u8,
                        (rgb.blue * 255.0).round() as u8,
                    ]
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
