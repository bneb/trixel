//! # trixel_render
//!
//! Rendering engine for the Trixel system.
//! Converts a `TritMatrix` into a physical PNG and provides the `FontEngine`
//! for mapping text into spatial constraints.

use image::{DynamicImage, GenericImageView, ImageBuffer, RgbImage};
use trixel_core::TritMatrix;
use trixel_solver::ConstraintMask;
use trixel_solver::anchor;
use thiserror::Error;
use palette::{Srgb, Hsl, IntoColor};

pub mod glyphs;
pub mod font;
pub mod halftone;
pub mod tri_render;
pub mod tri_diffusion;

pub use font::TrixelFont;
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
    /// Context-aware halftone renderer with Triangle Quadrant system.
    ///
    /// Each HSL art cell is divided into 4 triangles meeting at the cell center.
    /// Each triangle samples an independent hue/saturation from a 2× source image,
    /// but all 4 share the same luminosity (trit value). This gives 4× visual
    /// fidelity with zero impact on data capacity or scanner reliability.
    ///
    /// **Priority 1 (Font Immunity):** Font mask cells are flat-fill (crisp edges).
    /// **Priority 2 (Anchor Immunity):** Anchor regions use strict B/W flat-fill.
    /// **Priority 3 (Triangle HSL Art):** 4 triangles per cell, independent hue.
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

        // Resize the original image to 2× the matrix grid for sub-pixel sampling.
        // Each grid cell maps to a 2×2 block of source pixels.
        let scaled = original_image.resize_exact(
            (n * 2) as u32,
            (matrix.height * 2) as u32,
            image::imageops::FilterType::Lanczos3,
        );

        let img_w = n as u32 * module_pixel_size;
        let img_h = matrix.height as u32 * module_pixel_size;
        let mut img: RgbImage = ImageBuffer::new(img_w, img_h);

        let half = module_pixel_size as f32 / 2.0;

        for gy in 0..matrix.height {
            for gx in 0..n {
                let trit = matrix.get(gx, gy).unwrap_or(0);

                // Check font mask first (Typography Immunity)
                let font_state = if gy < font_mask.len() && gx < font_mask[gy].len() {
                    font_mask[gy][gx]
                } else {
                    None
                };

                let px_x = gx as u32 * module_pixel_size;
                let px_y = gy as u32 * module_pixel_size;

                if let Some(fs) = font_state {
                    // FONT IMMUNITY: flat fill for crisp typography
                    let color = match fs {
                        0 => [0u8, 0, 0],       // stroke: pure black
                        2 => {
                            // Frosted glass halo: sample center pixel
                            let orig_px = scaled.get_pixel((gx * 2) as u32, (gy * 2) as u32);
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
                        _ => [128u8, 128, 128],
                    };
                    for dy in 0..module_pixel_size {
                        for dx in 0..module_pixel_size {
                            img.put_pixel(px_x + dx, px_y + dy, image::Rgb(color));
                        }
                    }
                } else if anchor::is_in_anchor_region(gx, gy, n) {
                    // ANCHOR IMMUNITY: strict B/W flat fill for CV baseline
                    let color = match trit {
                        0 => [0u8, 0, 0],
                        1 => [128u8, 128, 128],
                        2 => [255u8, 255, 255],
                        _ => [128u8, 128, 128],
                    };
                    for dy in 0..module_pixel_size {
                        for dx in 0..module_pixel_size {
                            img.put_pixel(px_x + dx, px_y + dy, image::Rgb(color));
                        }
                    }
                } else {
                    // TRIANGLE HSL ART: 4 triangles per cell, independent hue.
                    //
                    // Sub-pixel layout in the 2× source image:
                    //   (2*gx, 2*gy)     = top-left     → Top triangle
                    //   (2*gx+1, 2*gy)   = top-right    → Right triangle
                    //   (2*gx+1, 2*gy+1) = bottom-right → Bottom triangle
                    //   (2*gx, 2*gy+1)   = bottom-left  → Left triangle

                    let target_l: f32 = match trit {
                        0 => 0.10,
                        1 => 0.50,
                        2 => 0.90,
                        _ => 0.50,
                    };

                    // Pre-compute the 4 triangle colors from 4 sub-pixels.
                    let sub_coords: [(u32, u32); 4] = [
                        ((gx * 2) as u32, (gy * 2) as u32),         // Top
                        ((gx * 2 + 1) as u32, (gy * 2) as u32),     // Right
                        ((gx * 2 + 1) as u32, (gy * 2 + 1) as u32), // Bottom
                        ((gx * 2) as u32, (gy * 2 + 1) as u32),     // Left
                    ];

                    let mut tri_colors = [[0u8; 3]; 4];
                    for (i, &(sx, sy)) in sub_coords.iter().enumerate() {
                        let orig_px = scaled.get_pixel(sx, sy);
                        let srgb = Srgb::new(
                            orig_px[0] as f32 / 255.0,
                            orig_px[1] as f32 / 255.0,
                            orig_px[2] as f32 / 255.0,
                        );
                        let hsl: Hsl = srgb.into_color();
                        let modified = Hsl::new(hsl.hue, hsl.saturation, target_l);
                        let rgb: Srgb = modified.into_color();
                        tri_colors[i] = [
                            (rgb.red * 255.0).round() as u8,
                            (rgb.green * 255.0).round() as u8,
                            (rgb.blue * 255.0).round() as u8,
                        ];
                    }

                    // Rasterize: for each pixel, determine which triangle it
                    // belongs to using a diagonal quadrant test.
                    //
                    //  TL ──────── TR     Triangle membership:
                    //  │ \  TOP  / │      rx = dx - center_x
                    //  │  \    /   │      ry = dy - center_y
                    //  │ L  \/  R  │
                    //  │   /  \    │      if |ry| > |rx|: TOP (ry<0) or BOTTOM (ry>0)
                    //  │  / BOT \  │      else:           LEFT (rx<0) or RIGHT (rx>0)
                    //  BL ──────── BR
                    for dy in 0..module_pixel_size {
                        for dx in 0..module_pixel_size {
                            let rx = dx as f32 - half + 0.5;
                            let ry = dy as f32 - half + 0.5;

                            let tri_idx = if ry.abs() > rx.abs() {
                                if ry < 0.0 { 0 } else { 2 } // Top or Bottom
                            } else {
                                if rx > 0.0 { 1 } else { 3 } // Right or Left
                            };

                            img.put_pixel(
                                px_x + dx,
                                px_y + dy,
                                image::Rgb(tri_colors[tri_idx]),
                            );
                        }
                    }
                }
            }
        }

        Ok(DynamicImage::ImageRgb8(img))
    }
}
