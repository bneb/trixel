use ternac_render::{AnchorRenderer, Renderer};
use ternac_solver::anchor::{ANCHOR_PATTERNS, ANCHOR_SIZE, corner_positions};
use ternac_core::TritMatrix;

// ---------------------------------------------------------------------------
// Anchor Renderer Tests
// ---------------------------------------------------------------------------

#[test]
fn anchor_renderer_correct_dimensions() {
    let mut matrix = TritMatrix::zeros(10, 10);
    // Put anchors
    for &(cx, cy, pi) in &corner_positions(10) {
        let pattern = &ANCHOR_PATTERNS[pi];
        for dy in 0..ANCHOR_SIZE {
            for dx in 0..ANCHOR_SIZE {
                matrix.set(cx + dx, cy + dy, pattern[dy][dx]);
            }
        }
    }

    let img = AnchorRenderer::render_png(&matrix, 5, [128, 128, 128]).unwrap();
    let (w, h) = (img.width(), img.height());
    assert_eq!(w, 50, "image width = 10 modules × 5 px");
    assert_eq!(h, 50, "image height = 10 modules × 5 px");
}

#[test]
fn anchor_renderer_corner_pixels_match_anchors() {
    let n = 10;
    let module_size = 4u32;
    let state_1_rgb = [128u8, 128, 128];

    let mut matrix = TritMatrix::zeros(n, n);
    for &(cx, cy, pi) in &corner_positions(n) {
        let pattern = &ANCHOR_PATTERNS[pi];
        for dy in 0..ANCHOR_SIZE {
            for dx in 0..ANCHOR_SIZE {
                matrix.set(cx + dx, cy + dy, pattern[dy][dx]);
            }
        }
    }

    let img = AnchorRenderer::render_png(&matrix, module_size, state_1_rgb).unwrap();
    let rgb = img.to_rgb8();

    // Check TL anchor: (0,0) should be state 0 → black
    let px = rgb.get_pixel(0, 0).0;
    assert_eq!(px, [0, 0, 0], "TL corner (0,0) should be black (state 0)");

    // Check TL anchor crook: pattern[2][2] = 1 → state_1_rgb
    let crook_px = rgb.get_pixel(2 * module_size, 2 * module_size).0;
    assert_eq!(crook_px, state_1_rgb, "TL crook should be state 1 color");

    // Check TL anchor quiet zone: pattern[1][1] = 2 → white
    let quiet_px = rgb.get_pixel(1 * module_size, 1 * module_size).0;
    assert_eq!(quiet_px, [255, 255, 255], "TL quiet zone should be white");
}

#[test]
fn anchor_renderer_empty_matrix_fails() {
    let matrix = TritMatrix::zeros(0, 0);
    let result = AnchorRenderer::render_png(&matrix, 5, [128, 128, 128]);
    assert!(result.is_err());
}

#[test]
fn anchor_renderer_zero_module_size_fails() {
    let matrix = TritMatrix::zeros(10, 10);
    let result = AnchorRenderer::render_png(&matrix, 0, [128, 128, 128]);
    assert!(result.is_err());
}
