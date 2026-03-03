//! # Triangular CV Extraction Pipeline
//!
//! Extracts a `TriGrid` from a camera frame or rendered image of a
//! triangular trixel code.
//!
//! ## Pipeline
//!
//! 1. Grayscale → adaptive threshold → binary mask
//! 2. Contour detection → Douglas-Peucker → filter for triangles
//! 3. Identify 3 anchor triangles (TL, TR, BL) by position
//! 4. Compute affine warp from detected anchors to ideal grid
//! 5. Sample each triangle centroid through the affine → quantize → TriGrid

use image::{DynamicImage, GrayImage, Luma};
use imageproc::contours::find_contours;
use trixel_core::trigrid::TriGrid;
use trixel_solver::tri_anchor;

use crate::geometry::{
    Point, douglas_peucker, triangle_area, is_valid_triangle,
    classify_tri_corners, affine_from_triangles, affine_transform, centroid,
};
use crate::{LuminanceBands, VisionError};

// ---------------------------------------------------------------------------
// Binary Mask (Otsu Threshold)
// ---------------------------------------------------------------------------

/// Compute the Otsu threshold for a grayscale image.
pub fn otsu_threshold(gray: &GrayImage) -> u8 {
    let mut histogram = [0u64; 256];
    for px in gray.pixels() {
        histogram[px.0[0] as usize] += 1;
    }

    let total = gray.width() as f64 * gray.height() as f64;
    let mut sum_total = 0.0f64;
    for (i, &count) in histogram.iter().enumerate() {
        sum_total += i as f64 * count as f64;
    }

    let mut weight_bg = 0.0f64;
    let mut sum_bg = 0.0f64;
    let mut max_variance = 0.0f64;
    let mut threshold = 0u8;

    for (t, &count) in histogram.iter().enumerate() {
        weight_bg += count as f64;
        if weight_bg == 0.0 {
            continue;
        }

        let weight_fg = total - weight_bg;
        if weight_fg == 0.0 {
            break;
        }

        sum_bg += t as f64 * count as f64;
        let mean_bg = sum_bg / weight_bg;
        let mean_fg = (sum_total - sum_bg) / weight_fg;

        let between_variance = weight_bg * weight_fg * (mean_bg - mean_fg).powi(2);
        if between_variance > max_variance {
            max_variance = between_variance;
            threshold = t as u8;
        }
    }

    threshold
}

/// Convert a grayscale image to a binary mask using Otsu's method.
pub fn to_binary_mask(gray: &GrayImage) -> GrayImage {
    let thresh = otsu_threshold(gray);
    let (w, h) = gray.dimensions();
    let mut binary = GrayImage::new(w, h);

    for (x, y, px) in gray.enumerate_pixels() {
        let val = if px.0[0] > thresh { 255 } else { 0 };
        binary.put_pixel(x, y, Luma([val]));
    }

    binary
}

// ---------------------------------------------------------------------------
// Contour Detection & Triangle Filtering
// ---------------------------------------------------------------------------

/// Find all triangle contours in a binary image via Douglas-Peucker decimation.
///
/// Returns triangles as arrays of 3 `Point`s.
pub fn find_triangle_contours(
    binary: &GrayImage,
    epsilon: f64,
    min_area: f64,
    max_area: f64,
) -> Vec<[Point; 3]> {
    let contours = find_contours::<u32>(binary);
    let mut triangles = Vec::new();

    for contour in &contours {
        // Convert contour points to our Point type
        let pts: Vec<Point> = contour.points.iter()
            .map(|p| Point::new(p.x as f64, p.y as f64))
            .collect();

        if pts.len() < 3 {
            continue;
        }

        // Douglas-Peucker simplification
        let simplified = douglas_peucker(&pts, epsilon);

        if simplified.len() == 3 || simplified.len() == 4 {
            // 4-vertex closed polygon where first == last is also a triangle
            let tri = if simplified.len() == 4 {
                // Check if it's a closed triangle (first ≈ last)
                let d = ((simplified[0].x - simplified[3].x).powi(2)
                    + (simplified[0].y - simplified[3].y).powi(2))
                    .sqrt();
                if d < epsilon * 2.0 {
                    [simplified[0], simplified[1], simplified[2]]
                } else {
                    continue;
                }
            } else {
                [simplified[0], simplified[1], simplified[2]]
            };

            let poly = vec![tri[0], tri[1], tri[2]];
            if is_valid_triangle(&poly, min_area, max_area) {
                triangles.push(tri);
            }
        }
    }

    triangles
}

// ---------------------------------------------------------------------------
// Anchor Identification
// ---------------------------------------------------------------------------

/// From a set of candidate triangles, identify the 3 that form the
/// TriGrid anchor pattern (TL, TR, BL).
///
/// Returns the centroids of the 3 anchor triangles as `[TL, TR, BL]`.
pub fn identify_anchors(
    triangles: &[[Point; 3]],
    _image_dims: (u32, u32),
) -> Result<[Point; 3], VisionError> {
    if triangles.len() < 3 {
        return Err(VisionError::NoAnchors);
    }

    // Compute centroids for all triangles
    let centroids: Vec<Point> = triangles.iter()
        .map(|tri| centroid(tri.as_slice()))
        .collect();

    // If exactly 3, classify directly
    if triangles.len() == 3 {
        let [tl, tr, bl] = classify_tri_corners(&centroids)
            .ok_or(VisionError::NoAnchors)?;
        return Ok([centroids[tl], centroids[tr], centroids[bl]]);
    }

    // If more than 3: find the 3 with the largest total bounding area
    // (anchor triangles form the corners of the grid)
    let mut best_area = 0.0f64;
    let mut best_triple = [0usize; 3];

    let n = centroids.len().min(50); // cap for performance
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let area = triangle_area(centroids[i], centroids[j], centroids[k]);
                if area > best_area {
                    best_area = area;
                    best_triple = [i, j, k];
                }
            }
        }
    }

    let top3: Vec<Point> = best_triple.iter().map(|&i| centroids[i]).collect();
    let [tl, tr, bl] = classify_tri_corners(&top3)
        .ok_or(VisionError::NoAnchors)?;

    Ok([top3[tl], top3[tr], top3[bl]])
}

// ---------------------------------------------------------------------------
// TriCvPipeline
// ---------------------------------------------------------------------------

/// Triangular CV extraction pipeline.
pub struct TriCvPipeline;

impl TriCvPipeline {
    /// Extract a `TriGrid` from a rendered triangular trixel image.
    ///
    /// For digitally-rendered images (known cell_h), this uses direct
    /// grid sampling without contour detection.
    pub fn extract_trigrid_digital(
        image: &DynamicImage,
        rows: usize,
        cols: usize,
        cell_h: u32,
    ) -> Result<TriGrid, VisionError> {
        let gray = image.to_luma8();
        let (img_w, img_h) = gray.dimensions();

        if img_w == 0 || img_h == 0 {
            return Err(VisionError::EmptyImage);
        }

        let cell_w = cell_h;
        let mut grid = TriGrid::zeros(rows, cols);

        // Calibrate from anchor cells
        let bands = calibrate_from_tri_anchors(&gray, rows, cols, cell_w, cell_h);

        for row in 0..rows {
            for col in 0..cols {
                let is_up = TriGrid::is_up(col, row);

                // Triangle centroid pixel position
                let px_x = col as u32 * cell_w / 2;
                let px_y = row as u32 * cell_h;

                // Centroid is at:
                // Up triangle ▲: (px_x + cell_w/2, px_y + 2*cell_h/3)
                // Down triangle ▽: (px_x + cell_w/2, px_y + cell_h/3)
                let cx = px_x + cell_w / 2;
                let cy = if is_up {
                    px_y + 2 * cell_h / 3
                } else {
                    px_y + cell_h / 3
                };

                if cx < img_w && cy < img_h {
                    let lum = gray.get_pixel(cx, cy).0[0];
                    grid.set(col, row, bands.quantize(lum));
                }
            }
        }

        Ok(grid)
    }

    /// Extract a `TriGrid` from a camera frame using full CV pipeline.
    ///
    /// 1. Threshold → binary mask
    /// 2. Contour detection → Douglas-Peucker → filter triangles
    /// 3. Identify 3 anchors (TL, TR, BL)
    /// 4. Affine warp to ideal grid
    /// 5. Sample each cell centroid → quantize
    pub fn extract_trigrid_camera(
        image: &DynamicImage,
        rows: usize,
        cols: usize,
        epsilon: f64,
    ) -> Result<TriGrid, VisionError> {
        let gray = image.to_luma8();
        let (img_w, img_h) = gray.dimensions();

        if img_w == 0 || img_h == 0 {
            return Err(VisionError::EmptyImage);
        }

        // Step 1: Binary mask
        let binary = to_binary_mask(&gray);

        // Step 2: Find triangle contours
        let img_area = img_w as f64 * img_h as f64;
        let min_area = img_area * 0.0001; // at least 0.01% of image
        let max_area = img_area * 0.1;     // at most 10% of image
        let triangles = find_triangle_contours(&binary, epsilon, min_area, max_area);

        // Step 3: Identify anchors
        let anchors = identify_anchors(&triangles, (img_w, img_h))?;

        // Step 4: Compute affine from detected → ideal positions
        // Ideal anchor centroids for a grid of `rows × cols` with unit cell:
        let ideal_tl = Point::new(
            tri_anchor::TRI_ANCHOR_COLS as f64 / 2.0,
            tri_anchor::TRI_ANCHOR_ROWS as f64 / 2.0,
        );
        let ideal_tr = Point::new(
            (cols - tri_anchor::TRI_ANCHOR_COLS / 2) as f64,
            tri_anchor::TRI_ANCHOR_ROWS as f64 / 2.0,
        );
        let ideal_bl = Point::new(
            tri_anchor::TRI_ANCHOR_COLS as f64 / 2.0,
            (rows - tri_anchor::TRI_ANCHOR_ROWS / 2) as f64,
        );

        let detected = [anchors[0], anchors[1], anchors[2]];
        let ideal = [ideal_tl, ideal_tr, ideal_bl];
        let affine = affine_from_triangles(detected, ideal)
            .ok_or(VisionError::NoAnchors)?;

        // Step 5: Sample each cell through the affine
        let bands = LuminanceBands::default();
        let mut grid = TriGrid::zeros(rows, cols);

        for row in 0..rows {
            for col in 0..cols {
                let is_up = TriGrid::is_up(col, row);

                // Ideal centroid in grid coordinates
                let gc_x = col as f64 + 0.5;
                let gc_y = if is_up {
                    row as f64 + 0.667
                } else {
                    row as f64 + 0.333
                };

                // Map grid coords → pixel coords via inverse of the affine
                // (We computed detected→ideal, so apply to ideal coords? No —
                //  we need ideal→detected. Compute the reverse affine.)
                let inv_affine = affine_from_triangles(ideal, detected)
                    .ok_or(VisionError::NoAnchors)?;
                let px = affine_transform(&inv_affine, Point::new(gc_x, gc_y));

                let px_x = px.x.round() as u32;
                let px_y = px.y.round() as u32;

                if px_x < img_w && px_y < img_h {
                    let lum = gray.get_pixel(px_x, px_y).0[0];
                    grid.set(col, row, bands.quantize(lum));
                }
            }
        }

        Ok(grid)
    }
}

// ---------------------------------------------------------------------------
// Anchor Calibration
// ---------------------------------------------------------------------------

/// Calibrate luminance bands from the known tri-anchor positions.
fn calibrate_from_tri_anchors(
    gray: &GrayImage,
    rows: usize,
    cols: usize,
    cell_w: u32,
    cell_h: u32,
) -> LuminanceBands {
    let mut s0_samples = Vec::new();
    let mut s1_samples = Vec::new();
    let mut s2_samples = Vec::new();

    for &(ac, ar, pi) in &tri_anchor::tri_corner_positions(rows, cols) {
        let pattern = &tri_anchor::TRI_ANCHOR_PATTERNS[pi];
        for dr in 0..tri_anchor::TRI_ANCHOR_ROWS {
            for dc in 0..tri_anchor::TRI_ANCHOR_COLS {
                let col = ac + dc;
                let row = ar + dr;

                // Sample centroid of this triangle cell
                let px_x = col as u32 * cell_w / 2 + cell_w / 2;
                let py = row as u32 * cell_h;
                let is_up = TriGrid::is_up(col, row);
                let px_y = if is_up { py + 2 * cell_h / 3 } else { py + cell_h / 3 };

                if px_x < gray.width() && px_y < gray.height() {
                    let lum = gray.get_pixel(px_x, px_y).0[0];
                    match pattern[dr][dc] {
                        0 => s0_samples.push(lum),
                        1 => s1_samples.push(lum),
                        2 => s2_samples.push(lum),
                        _ => {}
                    }
                }
            }
        }
    }

    if s0_samples.is_empty() || s1_samples.is_empty() || s2_samples.is_empty() {
        return LuminanceBands::default();
    }

    s0_samples.sort_unstable();
    s1_samples.sort_unstable();
    s2_samples.sort_unstable();

    let s0 = s0_samples[s0_samples.len() / 2];
    let s1 = s1_samples[s1_samples.len() / 2];
    let s2 = s2_samples[s2_samples.len() / 2];

    LuminanceBands::calibrate(s0, s1, s2)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------
    // Otsu Threshold
    // -------------------------------------------------------------------

    #[test]
    fn otsu_binary_mask_produces_only_0_and_255() {
        // Create a synthetic grayscale image with bimodal distribution
        let mut gray = GrayImage::new(100, 100);
        for y in 0..50 {
            for x in 0..100 {
                gray.put_pixel(x, y, Luma([20]));  // dark
            }
        }
        for y in 50..100 {
            for x in 0..100 {
                gray.put_pixel(x, y, Luma([230])); // light
            }
        }

        let binary = to_binary_mask(&gray);
        for px in binary.pixels() {
            assert!(px.0[0] == 0 || px.0[0] == 255,
                "Binary mask must contain only 0 or 255, got {}", px.0[0]);
        }
    }

    #[test]
    fn otsu_separates_bimodal_image() {
        let mut gray = GrayImage::new(100, 100);
        for y in 0..50 {
            for x in 0..100 {
                gray.put_pixel(x, y, Luma([20]));
            }
        }
        for y in 50..100 {
            for x in 0..100 {
                gray.put_pixel(x, y, Luma([230]));
            }
        }

        let thresh = otsu_threshold(&gray);
        assert!(thresh >= 20 && thresh <= 230,
            "Otsu threshold should be between 20 and 230, got {thresh}");
    }

    // -------------------------------------------------------------------
    // Triangle Detection from Synthetic Image
    // -------------------------------------------------------------------

    #[test]
    fn find_triangles_detects_drawn_triangle() {
        // Draw a single white triangle on a black background
        let mut img = GrayImage::new(200, 200);
        // Fill triangle by scanline: apex at (100,20), base from (40,180) to (160,180)
        for y in 20..=180 {
            let t = (y - 20) as f64 / 160.0;
            let left = 100.0 - 60.0 * t;
            let right = 100.0 + 60.0 * t;
            for x in left as u32..=right as u32 {
                if x < 200 {
                    img.put_pixel(x, y, Luma([255]));
                }
            }
        }

        let triangles = find_triangle_contours(&img, 5.0, 100.0, 40000.0);
        assert!(!triangles.is_empty(),
            "Should detect at least 1 triangle, found {}", triangles.len());
    }

    #[test]
    fn find_triangles_empty_image_returns_none() {
        let img = GrayImage::new(100, 100);
        let triangles = find_triangle_contours(&img, 3.0, 10.0, 100000.0);
        assert!(triangles.is_empty());
    }

    // -------------------------------------------------------------------
    // Anchor Identification
    // -------------------------------------------------------------------

    #[test]
    fn identify_anchors_from_three_triangles() {
        let triangles = vec![
            // TL
            [Point::new(10.0, 10.0), Point::new(30.0, 10.0), Point::new(20.0, 30.0)],
            // TR
            [Point::new(170.0, 10.0), Point::new(190.0, 10.0), Point::new(180.0, 30.0)],
            // BL
            [Point::new(10.0, 170.0), Point::new(30.0, 170.0), Point::new(20.0, 190.0)],
        ];

        let anchors = identify_anchors(&triangles, (200, 200)).unwrap();
        // TL should be near (20, ~17)
        assert!(anchors[0].x < 100.0 && anchors[0].y < 100.0, "TL wrong: {:?}", anchors[0]);
        // TR should be near (180, ~17)
        assert!(anchors[1].x > 100.0 && anchors[1].y < 100.0, "TR wrong: {:?}", anchors[1]);
        // BL should be near (20, ~177)
        assert!(anchors[2].x < 100.0 && anchors[2].y > 100.0, "BL wrong: {:?}", anchors[2]);
    }

    #[test]
    fn identify_anchors_too_few_returns_error() {
        let triangles = vec![
            [Point::new(10.0, 10.0), Point::new(30.0, 10.0), Point::new(20.0, 30.0)],
        ];
        assert!(identify_anchors(&triangles, (200, 200)).is_err());
    }

    // -------------------------------------------------------------------
    // Digital Extraction Roundtrip
    // -------------------------------------------------------------------

    #[test]
    fn extract_trigrid_digital_roundtrip() {
        use trixel_core::trigrid::TriGrid;
        use trixel_solver::tri_gauss_solver::TriGaussSolver;
        use trixel_core::{MockCodec, TernaryCodec};

        let data = b"hi";
        let trits = MockCodec::encode_bytes(data).unwrap();
        let rows = 12;
        let cols = 16;

        let grid = TriGaussSolver::resolve_trigrid(&trits, rows, cols, &[]).unwrap();

        // Render to image
        use trixel_render::tri_render::TriAnchorRenderer;
        let cell_h = 20u32;
        let img = TriAnchorRenderer::render_trigrid(&grid, cell_h, [128, 128, 128]).unwrap();

        // Extract back
        let extracted = TriCvPipeline::extract_trigrid_digital(&img, rows, cols, cell_h).unwrap();

        // Compare anchor cells — they should match exactly
        for &(ac, ar, pi) in &tri_anchor::tri_corner_positions(rows, cols) {
            let pattern = &tri_anchor::TRI_ANCHOR_PATTERNS[pi];
            for dr in 0..tri_anchor::TRI_ANCHOR_ROWS {
                for dc in 0..tri_anchor::TRI_ANCHOR_COLS {
                    let expected = pattern[dr][dc];
                    let got = extracted.get(ac + dc, ar + dr).unwrap_or(255);
                    assert_eq!(expected, got,
                        "Anchor mismatch at ({},{}) pattern[{}][{}]: expected {}, got {}",
                        ac + dc, ar + dr, dr, dc, expected, got
                    );
                }
            }
        }
    }
}
