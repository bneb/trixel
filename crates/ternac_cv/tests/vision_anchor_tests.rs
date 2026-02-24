use ternac_cv::{AnchorVision, LuminanceBands, VisionPipeline};
use ternac_cv::geometry::{self, Point};
use ternac_render::{AnchorRenderer, Renderer};
use ternac_solver::anchor::{ANCHOR_PATTERNS, ANCHOR_SIZE, corner_positions};
use ternac_solver::{AnchorSolver, MatrixSolver};

// ---------------------------------------------------------------------------
// Douglas-Peucker Tests
// ---------------------------------------------------------------------------

#[test]
fn douglas_peucker_reduces_line_to_endpoints() {
    let points = vec![
        Point::new(0.0, 0.0),
        Point::new(1.0, 0.1),
        Point::new(2.0, -0.1),
        Point::new(3.0, 0.05),
        Point::new(4.0, 0.0),
    ];
    let simplified = geometry::douglas_peucker(&points, 0.5);
    assert_eq!(simplified.len(), 2, "nearly-collinear points → 2 endpoints");
}

#[test]
fn douglas_peucker_preserves_sharp_corner() {
    let points = vec![
        Point::new(0.0, 0.0),
        Point::new(5.0, 0.0),
        Point::new(5.0, 5.0),
    ];
    let simplified = geometry::douglas_peucker(&points, 0.1);
    assert_eq!(simplified.len(), 3, "sharp corner must be preserved");
}

#[test]
fn douglas_peucker_l_shape_to_6_vertices() {
    // L-shape contour: walk the outline of a 3×3 L with some jitter
    let points = vec![
        Point::new(0.0, 0.0),
        Point::new(1.0, 0.05),
        Point::new(2.0, -0.02),
        Point::new(3.0, 0.0),   // corner
        Point::new(3.0, 1.0),   // corner
        Point::new(1.0, 1.0),   // corner
        Point::new(1.0, 2.0),
        Point::new(1.0, 3.0),   // corner
        Point::new(0.0, 3.0),   // corner
        Point::new(0.0, 1.0),
        Point::new(0.0, 0.0),   // back to start
    ];
    let simplified = geometry::douglas_peucker(&points, 0.2);
    // Should simplify to approximately 6-7 vertices
    assert!(
        simplified.len() <= 8,
        "L-shape should simplify to ~6-7 vertices, got {}",
        simplified.len()
    );
}

// ---------------------------------------------------------------------------
// L-Shape Classifier Tests
// ---------------------------------------------------------------------------

#[test]
fn is_l_shape_accepts_perfect_l() {
    let polygon = vec![
        Point::new(0.0, 0.0),
        Point::new(3.0, 0.0),
        Point::new(3.0, 1.0),
        Point::new(1.0, 1.0),
        Point::new(1.0, 3.0),
        Point::new(0.0, 3.0),
    ];
    assert!(geometry::is_l_shape(&polygon), "6-vertex L should be accepted");
}

#[test]
fn is_l_shape_rejects_rectangle() {
    let polygon = vec![
        Point::new(0.0, 0.0),
        Point::new(3.0, 0.0),
        Point::new(3.0, 3.0),
        Point::new(0.0, 3.0),
    ];
    assert!(!geometry::is_l_shape(&polygon), "4-vertex rectangle should be rejected");
}

// ---------------------------------------------------------------------------
// Corner Classification Tests
// ---------------------------------------------------------------------------

#[test]
fn classify_corners_assigns_correctly() {
    let centroids = vec![
        Point::new(1.0, 1.0),   // TL
        Point::new(9.0, 1.0),   // TR
        Point::new(1.0, 9.0),   // BL
        Point::new(9.0, 9.0),   // BR
    ];
    let result = geometry::classify_corners(&centroids).unwrap();
    assert_eq!(result, [0, 1, 2, 3]);
}

#[test]
fn classify_corners_handles_shuffled_input() {
    let centroids = vec![
        Point::new(9.0, 9.0),   // BR (index 0)
        Point::new(1.0, 1.0),   // TL (index 1)
        Point::new(9.0, 1.0),   // TR (index 2)
        Point::new(1.0, 9.0),   // BL (index 3)
    ];
    let result = geometry::classify_corners(&centroids).unwrap();
    assert_eq!(result, [1, 2, 3, 0]); // TL=1, TR=2, BL=3, BR=0
}

// ---------------------------------------------------------------------------
// Luminance Calibration Tests
// ---------------------------------------------------------------------------

#[test]
fn calibrated_bands_quantize_correctly() {
    // Simulate: State 0 = lum 10, State 1 = lum 128, State 2 = lum 240
    let bands = LuminanceBands::calibrate(10, 128, 240);

    assert_eq!(bands.quantize(10), 0, "measured state 0 luminance");
    assert_eq!(bands.quantize(128), 1, "measured state 1 luminance");
    assert_eq!(bands.quantize(240), 2, "measured state 2 luminance");
    assert_eq!(bands.quantize(0), 0, "very dark → state 0");
    assert_eq!(bands.quantize(255), 2, "very bright → state 2");
}

// ---------------------------------------------------------------------------
// Full Pipeline: AnchorRenderer → AnchorVision Round-Trip
// ---------------------------------------------------------------------------

#[test]
fn anchor_vision_round_trip_with_renderer() {
    let n = 10;
    let module_size = 5u32;
    let state_1_rgb = [128u8, 128, 128]; // luminance 128, safely in mid band

    // 1. Pack data with AnchorSolver
    let free_cells = n * n - 4 * ANCHOR_SIZE * ANCHOR_SIZE;
    let payload: Vec<u8> = (0..free_cells).map(|i| (i % 3) as u8).collect();
    let matrix = AnchorSolver::resolve_matrix(&payload, n, &[]).unwrap();

    // 2. Render with AnchorRenderer
    let img = AnchorRenderer::render_png(&matrix, module_size, state_1_rgb).unwrap();

    // 3. Extract with AnchorVision
    let extracted = AnchorVision::extract_matrix(&img, module_size).unwrap();

    // 4. Compare: all cells should match
    for y in 0..n {
        for x in 0..n {
            let expected = matrix.get(x, y).unwrap();
            let actual = extracted.get(x, y).unwrap();
            assert_eq!(
                actual, expected,
                "cell ({x},{y}) mismatch: expected {expected}, got {actual}"
            );
        }
    }
}

#[test]
fn anchor_vision_calibrates_with_unusual_colors() {
    let n = 10;
    let module_size = 4u32;
    // Use a state_1 color that maps to luminance ~105 (would fail with default bands)
    let state_1_rgb = [90u8, 110, 115]; // grayscale ≈ 0.2126*90 + 0.7152*110 + 0.0722*115 ≈ 106

    let payload: Vec<u8> = vec![0; n * n - 4 * ANCHOR_SIZE * ANCHOR_SIZE];
    let matrix = AnchorSolver::resolve_matrix(&payload, n, &[]).unwrap();
    let img = AnchorRenderer::render_png(&matrix, module_size, state_1_rgb).unwrap();

    // AnchorVision should calibrate correctly even with unusual colors
    let extracted = AnchorVision::extract_matrix(&img, module_size).unwrap();

    // At minimum, anchor cells should be correctly identified
    for &(cx, cy, pi) in &corner_positions(n) {
        let pattern = &ANCHOR_PATTERNS[pi];
        for dy in 0..ANCHOR_SIZE {
            for dx in 0..ANCHOR_SIZE {
                let expected = pattern[dy][dx];
                let actual = extracted.get(cx + dx, cy + dy).unwrap();
                assert_eq!(
                    actual, expected,
                    "anchor cell ({},{}) should be {expected}, got {actual}",
                    cx + dx, cy + dy
                );
            }
        }
    }
}
