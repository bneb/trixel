//! # Triangular Grid Renderer
//!
//! Renders a `TriGrid` into a physical PNG image using **Asymmetric Perceptual
//! Encoding**. Colors are shifted in CIELAB space to satisfy scanner Rec.601
//! Luma thresholds while minimizing perceptual ΔE.
//!
//! Each cell is an actual triangle — up-pointing `▲` or down-pointing `▽`.

use image::{DynamicImage, GenericImageView, ImageBuffer, RgbImage};
use trixel_core::trigrid::TriGrid;
use trixel_solver::tri_anchor;
use palette::{Srgb, Lab, IntoColor};

use crate::RenderError;

// ---------------------------------------------------------------------------
// Rec.601 Luma & CIELAB Color Shifting
// ---------------------------------------------------------------------------

/// Rec.601 Luma from linear RGB (0-255 scale).
/// Y = 0.299*R + 0.587*G + 0.114*B
#[inline]
fn rec601_luma(r: u8, g: u8, b: u8) -> f32 {
    0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32
}

/// Scanner Luma thresholds (Rec.601 Y values).
/// State 0 (Dark):  Y ≤ 89
/// State 1 (Mid):   89 < Y ≤ 165
/// State 2 (Light): Y > 165
const LUMA_THRESHOLD_01: f32 = 89.0;   // boundary between State 0 and State 1
const LUMA_THRESHOLD_12: f32 = 165.0;  // boundary between State 1 and State 2

/// Target Luma ranges for each trit state (midpoints of scanner bands).
/// These are the ideal Y values the encoder aims for.
fn target_luma_range(trit: u8) -> (f32, f32) {
    match trit {
        0 => (0.0, LUMA_THRESHOLD_01),          // Y ∈ [0, 89]
        1 => (LUMA_THRESHOLD_01 + 1.0, LUMA_THRESHOLD_12), // Y ∈ [90, 165]
        2 => (LUMA_THRESHOLD_12 + 1.0, 255.0),  // Y ∈ [166, 255]
        _ => (90.0, 165.0),
    }
}

/// Shift a source CIELAB color to satisfy a Rec.601 Luma constraint.
///
/// Preserves the chromaticity (a*, b*) of the source color and adjusts
/// only L* (CIE Lightness) via bisection search to find the L* value
/// whose resulting RGB satisfies the target Luma range.
///
/// This is the core of Asymmetric Perceptual Encoding: minimal ΔE for
/// maximum scanner discriminability.
fn shift_lab_for_luma(source_lab: Lab, trit: u8, correction: f32) -> [u8; 3] {
    let (y_min, y_max) = target_luma_range(trit);
    let y_target = ((y_min + y_max) / 2.0 + correction * 255.0).clamp(y_min, y_max);

    // Bisection search: find L* such that the resulting RGB has Y ≈ y_target
    let mut lo: f32 = 0.0;
    let mut hi: f32 = 100.0;
    let mut best_rgb = [0u8; 3];
    let mut best_err = f32::MAX;

    for _ in 0..20 {
        let mid = (lo + hi) / 2.0;
        let candidate = Lab::new(mid, source_lab.a, source_lab.b);
        let srgb: Srgb = candidate.into_color();

        // Clamp to valid sRGB gamut
        let r = (srgb.red.clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = (srgb.green.clamp(0.0, 1.0) * 255.0).round() as u8;
        let b = (srgb.blue.clamp(0.0, 1.0) * 255.0).round() as u8;

        let y = rec601_luma(r, g, b);
        let err = (y - y_target).abs();

        if err < best_err {
            best_err = err;
            best_rgb = [r, g, b];
        }

        if y < y_target {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // Final verification: ensure we're within the scanner band
    let y_final = rec601_luma(best_rgb[0], best_rgb[1], best_rgb[2]);
    if y_final < y_min || y_final > y_max {
        // Fallback: grayscale at the target luma
        let gray = y_target.clamp(0.0, 255.0).round() as u8;
        return [gray, gray, gray];
    }

    best_rgb
}

// ---------------------------------------------------------------------------
// Geometry Helpers
// ---------------------------------------------------------------------------

/// Compute the pixel bounding box for a triangle at `(col, row)`.
///
/// Each row has height `cell_h` pixels. Each pair of columns (▲▽)
/// occupies `cell_w` pixels width total, so each triangle has base `cell_w`
/// at one edge and apex at the opposite.
///
/// Returns `(px_x, px_y)` — the top-left of the bounding rectangle.
#[inline]
fn cell_pixel_origin(col: usize, row: usize, cell_w: u32, cell_h: u32) -> (u32, u32) {
    // Each column has width cell_w/2 (half-cell stagger)
    let px_x = col as u32 * cell_w / 2;
    let px_y = row as u32 * cell_h;
    (px_x, px_y)
}

/// Check if pixel `(px, py)` relative to the cell bounding box is inside
/// the triangle. For an up-triangle (▲), the apex is at the top center and
/// the base is at the bottom. For a down-triangle (▽), apex at bottom.
///
/// `cell_w` is the full width of one triangle's bounding box (`total_w / (cols/2)`).
/// `cell_h` is the height of one row.
#[inline]
fn pixel_in_triangle(dx: u32, dy: u32, cell_w: u32, cell_h: u32, is_up: bool) -> bool {
    // Normalized coordinates: fx ∈ [0, 1], fy ∈ [0, 1]
    let fx = dx as f32 / cell_w as f32;
    let fy = dy as f32 / cell_h as f32;

    if is_up {
        // ▲: apex at (0.5, 0), base from (0, 1) to (1, 1)
        // Left edge: x = 0.5 - 0.5*y → x > 0.5 - 0.5*y
        // Right edge: x = 0.5 + 0.5*y → x < 0.5 + 0.5*y
        let half_width_at_y = 0.5 * fy;
        fx >= 0.5 - half_width_at_y && fx <= 0.5 + half_width_at_y
    } else {
        // ▽: apex at (0.5, 1), base from (0, 0) to (1, 0)
        // Left edge: x = 0.5*y → x > 0.5*y - 0.5
        // Right edge: x = 1 - 0.5*y → x < 1 - 0.5*y + 0.5
        let half_width_at_y = 0.5 * (1.0 - fy);
        fx >= 0.5 - half_width_at_y && fx <= 0.5 + half_width_at_y
    }
}

// ---------------------------------------------------------------------------
// TriAnchorRenderer
// ---------------------------------------------------------------------------

/// Production renderer for triangular grids.
pub struct TriAnchorRenderer;

impl TriAnchorRenderer {
    /// Render a `TriGrid` as a flat-color image (no halftone).
    pub fn render_trigrid(
        grid: &TriGrid,
        cell_h: u32,
        state_1_rgb: [u8; 3],
    ) -> Result<DynamicImage, RenderError> {
        if grid.rows == 0 || grid.cols == 0 {
            return Err(RenderError::EmptyMatrix);
        }
        if cell_h == 0 {
            return Err(RenderError::ZeroModuleSize);
        }

        let cell_w = cell_h; // equilateral-ish: width = height
        let img_w = (grid.cols as u32 * cell_w) / 2 + cell_w / 2;
        let img_h = grid.rows as u32 * cell_h;
        let mut img: RgbImage = ImageBuffer::from_pixel(img_w, img_h, image::Rgb([255, 255, 255]));

        for row in 0..grid.rows {
            for col in 0..grid.cols {
                let trit = grid.get(col, row).unwrap_or(0);
                let is_up = TriGrid::is_up(col, row);

                let color = match trit {
                    0 => [0u8, 0, 0],
                    1 => state_1_rgb,
                    2 => [255u8, 255, 255],
                    _ => [128u8, 128, 128],
                };

                let (px_x, px_y) = cell_pixel_origin(col, row, cell_w, cell_h);

                for dy in 0..cell_h {
                    for dx in 0..cell_w {
                        if px_x + dx < img_w && px_y + dy < img_h
                            && pixel_in_triangle(dx, dy, cell_w, cell_h, is_up)
                        {
                            img.put_pixel(px_x + dx, px_y + dy, image::Rgb(color));
                        }
                    }
                }
            }
        }

        Ok(DynamicImage::ImageRgb8(img))
    }

    /// Halftone renderer for triangular grids.
    /// Preserves hue/saturation from the source image, overrides lightness per trit.
    pub fn render_halftone_trigrid(
        grid: &TriGrid,
        cell_h: u32,
        original_image: &DynamicImage,
        font_mask: &[Vec<Option<u8>>],
    ) -> Result<DynamicImage, RenderError> {
        if grid.rows == 0 || grid.cols == 0 {
            return Err(RenderError::EmptyMatrix);
        }
        if cell_h == 0 {
            return Err(RenderError::ZeroModuleSize);
        }

        let cell_w = cell_h;
        let img_w = (grid.cols as u32 * cell_w) / 2 + cell_w / 2;
        let img_h = grid.rows as u32 * cell_h;
        let mut img: RgbImage = ImageBuffer::from_pixel(img_w, img_h, image::Rgb([255, 255, 255]));

        // Flatten alpha over white BEFORE resizing to prevent transparent
        // pixels (0,0,0,0) from becoming solid black (0,0,0) in RGB space.
        let flattened = {
            let rgba = original_image.to_rgba8();
            let (w, h) = (rgba.width(), rgba.height());
            let mut rgb = image::RgbImage::new(w, h);
            for (x, y, px) in rgba.enumerate_pixels() {
                let a = px[3] as f32 / 255.0;
                let r = (px[0] as f32 * a + 255.0 * (1.0 - a)).round() as u8;
                let g = (px[1] as f32 * a + 255.0 * (1.0 - a)).round() as u8;
                let b = (px[2] as f32 * a + 255.0 * (1.0 - a)).round() as u8;
                rgb.put_pixel(x, y, image::Rgb([r, g, b]));
            }
            DynamicImage::ImageRgb8(rgb)
        };

        // Resize flattened source to match grid dimensions (1 pixel per cell)
        let scaled = flattened.resize_exact(
            grid.cols as u32,
            grid.rows as u32,
            image::imageops::FilterType::Lanczos3,
        );

        // Compute Floyd-Steinberg error diffusion corrections
        let scaled_gray = scaled.to_luma8();
        let corrections = crate::tri_diffusion::diffuse_lightness(
            grid, &scaled_gray, grid.rows, grid.cols,
        );

        for row in 0..grid.rows {
            for col in 0..grid.cols {
                let trit = grid.get(col, row).unwrap_or(0);
                let is_up = TriGrid::is_up(col, row);

                // Check font mask
                let font_state = if row < font_mask.len() && col < font_mask[row].len() {
                    font_mask[row][col]
                } else {
                    None
                };

                let color = if let Some(fs) = font_state {
                    match fs {
                        0 => [0u8, 0, 0],
                        2 => {
                            // Font glass: CIELAB shift to State 2
                            let orig_px = scaled.get_pixel(col as u32, row as u32);
                            let srgb = Srgb::new(
                                orig_px[0] as f32 / 255.0,
                                orig_px[1] as f32 / 255.0,
                                orig_px[2] as f32 / 255.0,
                            );
                            let lab: Lab = srgb.into_color();
                            shift_lab_for_luma(lab, 2, 0.0)
                        }
                        _ => [128u8, 128, 128],
                    }
                } else if tri_anchor::is_in_tri_anchor_region(col, row, grid.rows, grid.cols) {
                    // Anchor immunity: strict Luma-correct B/W/Mid
                    match trit {
                        0 => [0u8, 0, 0],         // Y = 0
                        1 => [128u8, 128, 128],   // Y = 128
                        2 => [255u8, 255, 255],   // Y = 255
                        _ => [128u8, 128, 128],
                    }
                } else {
                    // CIELAB Perceptual Encoding: preserve chromaticity (a*, b*),
                    // shift L* to satisfy scanner Rec.601 Luma thresholds.
                    let orig_px = scaled.get_pixel(col as u32, row as u32);
                    let srgb = Srgb::new(
                        orig_px[0] as f32 / 255.0,
                        orig_px[1] as f32 / 255.0,
                        orig_px[2] as f32 / 255.0,
                    );
                    let lab: Lab = srgb.into_color();
                    let correction = corrections[row][col];
                    shift_lab_for_luma(lab, trit, correction)
                };

                let (px_x, px_y) = cell_pixel_origin(col, row, cell_w, cell_h);

                for dy in 0..cell_h {
                    for dx in 0..cell_w {
                        if px_x + dx < img_w && px_y + dy < img_h
                            && pixel_in_triangle(dx, dy, cell_w, cell_h, is_up)
                        {
                            img.put_pixel(px_x + dx, px_y + dy, image::Rgb(color));
                        }
                    }
                }
            }
        }

        Ok(DynamicImage::ImageRgb8(img))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;
    use trixel_core::trigrid::TriGrid;

    // -------------------------------------------------------------------
    // Pixel-in-Triangle
    // -------------------------------------------------------------------

    #[test]
    fn pixel_in_up_triangle_apex() {
        // Apex of ▲ is at top-center
        let w = 20;
        let h = 20;
        // Near apex (10, 0) — should be inside
        assert!(pixel_in_triangle(10, 1, w, h, true),
            "Pixel near up-triangle apex should be inside");
    }

    #[test]
    fn pixel_in_up_triangle_base() {
        let w = 20;
        let h = 20;
        // Bottom center — clearly inside
        assert!(pixel_in_triangle(10, 19, w, h, true));
        // Near bottom, slightly inside the left edge
        assert!(pixel_in_triangle(3, 16, w, h, true));
        // Exact bottom-left corner (0,19): fx=0.0, fy=0.95 →
        // half_w = 0.475, check: 0.0 >= 0.025? No → outside
        assert!(!pixel_in_triangle(0, 19, w, h, true));
    }

    #[test]
    fn pixel_outside_up_triangle() {
        let w = 20;
        let h = 20;
        // Top-left corner — outside ▲
        assert!(!pixel_in_triangle(0, 0, w, h, true));
        // Top-right corner — outside ▲
        assert!(!pixel_in_triangle(19, 0, w, h, true));
    }

    #[test]
    fn pixel_in_down_triangle() {
        let w = 20;
        let h = 20;
        // Top center — should be inside ▽ (base at top)
        assert!(pixel_in_triangle(10, 0, w, h, false));
        // Near bottom apex — should be inside
        assert!(pixel_in_triangle(10, 19, w, h, false));
    }

    // -------------------------------------------------------------------
    // Render Dimensions
    // -------------------------------------------------------------------

    #[test]
    fn tri_render_correct_dimensions() {
        let mut grid = TriGrid::zeros(10, 20);
        grid.set(0, 0, 1);

        let img = TriAnchorRenderer::render_trigrid(&grid, 12, [128, 128, 128]).unwrap();
        let rgb = img.to_rgb8();

        assert_eq!(rgb.height(), 10 * 12);
        // Width: (20 * 12) / 2 + 12/2 = 120 + 6 = 126
        assert_eq!(rgb.width(), (20 * 12) / 2 + 12 / 2);
    }

    // -------------------------------------------------------------------
    // Color Correctness
    // -------------------------------------------------------------------

    #[test]
    fn tri_render_state0_is_black() {
        let mut grid = TriGrid::zeros(8, 12);
        // Set a non-anchor cell to state 0
        grid.set(6, 4, 0);

        let img = TriAnchorRenderer::render_trigrid(&grid, 20, [128, 128, 128]).unwrap();
        let rgb = img.to_rgb8();

        // Sample inside the triangle at (6, 4)
        let (px_x, px_y) = cell_pixel_origin(6, 4, 20, 20);
        let is_up = TriGrid::is_up(6, 4);

        // Find a pixel that's inside the triangle
        let mut found_black = false;
        for dy in 0..20 {
            for dx in 0..20 {
                if pixel_in_triangle(dx, dy, 20, 20, is_up) && px_x + dx < rgb.width() && px_y + dy < rgb.height() {
                    let px = rgb.get_pixel(px_x + dx, px_y + dy).0;
                    assert_eq!(px, [0, 0, 0], "State 0 triangle pixel should be black, got {:?}", px);
                    found_black = true;
                    break;
                }
            }
            if found_black { break; }
        }
        assert!(found_black, "Should find at least one pixel inside the triangle");
    }

    #[test]
    fn tri_render_state2_is_white() {
        let mut grid = TriGrid::zeros(8, 12);
        // Use a cell well inside the grid and not in an anchor
        grid.set(6, 4, 2);

        let img = TriAnchorRenderer::render_trigrid(&grid, 20, [128, 128, 128]).unwrap();
        let rgb = img.to_rgb8();

        let (px_x, px_y) = cell_pixel_origin(6, 4, 20, 20);
        let is_up = TriGrid::is_up(6, 4);

        // Sample near the center of the triangle — guaranteed inside
        let center_dx = 10u32;
        let center_dy = if is_up { 14 } else { 6 }; // bias toward base
        if px_x + center_dx < rgb.width() && px_y + center_dy < rgb.height() {
            assert!(pixel_in_triangle(center_dx, center_dy, 20, 20, is_up),
                "Center sample must be inside triangle");
            let px = rgb.get_pixel(px_x + center_dx, px_y + center_dy).0;
            assert_eq!(px, [255, 255, 255], "State 2 triangle should be white, got {:?}", px);
        }
    }

    // -------------------------------------------------------------------
    // Empty / Error Cases
    // -------------------------------------------------------------------

    #[test]
    fn tri_render_empty_grid_errors() {
        let grid = TriGrid::zeros(0, 0);
        assert!(TriAnchorRenderer::render_trigrid(&grid, 10, [128, 128, 128]).is_err());
    }

    #[test]
    fn tri_render_zero_cell_size_errors() {
        let grid = TriGrid::zeros(10, 20);
        assert!(TriAnchorRenderer::render_trigrid(&grid, 0, [128, 128, 128]).is_err());
    }
}
