use image::{DynamicImage, RgbImage, Rgb};
use trixel_core::TritMatrix;
use trixel_solver::anchor::{ANCHOR_PATTERNS, ANCHOR_SIZE, corner_positions};
use trixel_render::AnchorRenderer;

fn build_test_matrix(n: usize) -> TritMatrix {
    let mut matrix = TritMatrix::zeros(n, n);
    for &(cx, cy, pi) in &corner_positions(n) {
        let pattern = &ANCHOR_PATTERNS[pi];
        for dy in 0..ANCHOR_SIZE {
            for dx in 0..ANCHOR_SIZE {
                matrix.set(cx + dx, cy + dy, pattern[dy][dx]);
            }
        }
    }
    matrix
}

fn build_font_mask(n: usize) -> Vec<Vec<Option<u8>>> {
    vec![vec![None; n]; n]
}

// -----------------------------------------------------------------------
// Font Immunity: Stroke=Some(0) → pure black (opaque)
// -----------------------------------------------------------------------

/// Font stroke (Some(0)) must render as pure black, ignoring red background
#[test]
fn font_stroke_renders_black() {
    let n = 20;
    let module_size = 4u32;

    let mut src = RgbImage::new(n as u32, n as u32);
    for y in 0..n as u32 {
        for x in 0..n as u32 {
            src.put_pixel(x, y, Rgb([255, 0, 0]));
        }
    }
    let dyn_img = DynamicImage::ImageRgb8(src);

    let mut matrix = build_test_matrix(n);
    matrix.set(10, 10, 0);

    let mut font_mask = build_font_mask(n);
    font_mask[10][10] = Some(0); // stroke

    let img = AnchorRenderer::render_halftone_png(
        &matrix, module_size, &dyn_img, &font_mask
    ).unwrap();
    let rgb = img.to_rgb8();

    let px = rgb.get_pixel(10 * module_size, 10 * module_size).0;
    assert_eq!(px, [0, 0, 0], "Font stroke must be pure black, got {:?}", px);
}

// -----------------------------------------------------------------------
// Frosted Glass: Halo=Some(2) → HSL lightened (L=0.90), preserving hue
// -----------------------------------------------------------------------

/// Font halo over a red background should produce a very light red (frosted glass),
/// NOT pure white. The hue of the original pixel must be preserved.
#[test]
fn font_halo_frosted_glass_preserves_hue() {
    let n = 20;
    let module_size = 4u32;

    // Pure red background
    let mut src = RgbImage::new(n as u32, n as u32);
    for y in 0..n as u32 {
        for x in 0..n as u32 {
            src.put_pixel(x, y, Rgb([255, 0, 0]));
        }
    }
    let dyn_img = DynamicImage::ImageRgb8(src);

    let mut matrix = build_test_matrix(n);
    matrix.set(10, 10, 2);

    let mut font_mask = build_font_mask(n);
    font_mask[10][10] = Some(2); // halo

    let img = AnchorRenderer::render_halftone_png(
        &matrix, module_size, &dyn_img, &font_mask
    ).unwrap();
    let rgb = img.to_rgb8();

    let px = rgb.get_pixel(10 * module_size, 10 * module_size).0;
    // L=0.90 over pure red should give a very light pink/red, not pure white.
    // Red channel should be very high (>220), but green+blue should also be elevated
    // (since high lightness desaturates slightly). The key assertion: it must NOT
    // be pure white [255, 255, 255].
    assert!(px[0] > 220, "Red channel should be very high (light red): got {}", px[0]);
    assert_ne!(px, [255, 255, 255], "Frosted glass halo must NOT be pure white");
    // Red channel must dominate (hue preserved)
    assert!(px[0] > px[1], "Red must dominate over green: R={} G={}", px[0], px[1]);
    assert!(px[0] > px[2], "Red must dominate over blue: R={} B={}", px[0], px[2]);
}

/// Font halo over a teal background should produce a very light teal
#[test]
fn font_halo_frosted_glass_teal() {
    let n = 20;
    let module_size = 4u32;

    // Teal background (#75B3B8)
    let mut src = RgbImage::new(n as u32, n as u32);
    for y in 0..n as u32 {
        for x in 0..n as u32 {
            src.put_pixel(x, y, Rgb([0x75, 0xB3, 0xB8]));
        }
    }
    let dyn_img = DynamicImage::ImageRgb8(src);

    let mut matrix = build_test_matrix(n);
    matrix.set(10, 10, 2);

    let mut font_mask = build_font_mask(n);
    font_mask[10][10] = Some(2); // halo

    let img = AnchorRenderer::render_halftone_png(
        &matrix, module_size, &dyn_img, &font_mask
    ).unwrap();
    let rgb = img.to_rgb8();

    let px = rgb.get_pixel(10 * module_size, 10 * module_size).0;
    // Teal at L=0.90: should be very light, but blue/green channels must dominate red
    assert!(px[0] > 200, "All channels high at L=0.90: R={}", px[0]);
    assert!(px[1] > px[0], "Green should be >= Red for teal hue: G={} R={}", px[1], px[0]);
    assert!(px[2] > px[0], "Blue should be >= Red for teal hue: B={} R={}", px[2], px[0]);
    assert_ne!(px, [255, 255, 255], "Must NOT be pure white");
}

/// Font halo lightness must be firmly in State 2 band (L > 0.65)
/// so the CV scanner reads it correctly
#[test]
fn font_halo_lightness_in_state2_band() {
    let n = 20;
    let module_size = 4u32;

    // Dark navy background — worst case for lightness
    let mut src = RgbImage::new(n as u32, n as u32);
    for y in 0..n as u32 {
        for x in 0..n as u32 {
            src.put_pixel(x, y, Rgb([10, 20, 50]));
        }
    }
    let dyn_img = DynamicImage::ImageRgb8(src);

    let mut matrix = build_test_matrix(n);
    matrix.set(10, 10, 2);

    let mut font_mask = build_font_mask(n);
    font_mask[10][10] = Some(2); // halo

    let img = AnchorRenderer::render_halftone_png(
        &matrix, module_size, &dyn_img, &font_mask
    ).unwrap();
    let rgb = img.to_rgb8();

    let px = rgb.get_pixel(10 * module_size, 10 * module_size).0;
    // Even over a dark background, the frosted glass L=0.90 shift
    // must produce a pixel light enough to stay in the State 2 band.
    // Luminance approximation: 0.299*R + 0.587*G + 0.114*B
    let luma = 0.299 * px[0] as f64 + 0.587 * px[1] as f64 + 0.114 * px[2] as f64;
    assert!(luma > 200.0,
        "Frosted glass pixel must be very light (L=0.90), got luma={:.1} from {:?}", luma, px);
}

// -----------------------------------------------------------------------
// Non-font cells and anchors
// -----------------------------------------------------------------------

/// Non-font cells (None in mask) still use HSL logic: red stays red
#[test]
fn non_font_cell_preserves_hue() {
    let n = 20;
    let module_size = 4u32;

    let mut src = RgbImage::new(n as u32, n as u32);
    for y in 0..n as u32 {
        for x in 0..n as u32 {
            src.put_pixel(x, y, Rgb([255, 0, 0]));
        }
    }
    let dyn_img = DynamicImage::ImageRgb8(src);

    let mut matrix = build_test_matrix(n);
    matrix.set(10, 10, 1);

    let font_mask = build_font_mask(n); // all None

    let img = AnchorRenderer::render_halftone_png(
        &matrix, module_size, &dyn_img, &font_mask
    ).unwrap();
    let rgb = img.to_rgb8();

    let px = rgb.get_pixel(10 * module_size, 10 * module_size).0;
    assert!(px[0] > 200, "Red channel should be high: {}", px[0]);
    assert!(px[1] < 10, "Green channel should be low: {}", px[1]);
    assert!(px[2] < 10, "Blue channel should be low: {}", px[2]);
}

/// Anchor cells remain strictly black/white regardless of image or font
#[test]
fn anchors_remain_strict_bw() {
    let n = 20;
    let module_size = 4u32;

    let mut src = RgbImage::new(n as u32, n as u32);
    for y in 0..n as u32 {
        for x in 0..n as u32 {
            src.put_pixel(x, y, Rgb([0x75, 0xB3, 0xB8]));
        }
    }
    let dyn_img = DynamicImage::ImageRgb8(src);

    let matrix = build_test_matrix(n);
    let font_mask = build_font_mask(n);

    let img = AnchorRenderer::render_halftone_png(
        &matrix, module_size, &dyn_img, &font_mask
    ).unwrap();
    let rgb = img.to_rgb8();

    let tl = rgb.get_pixel(0, 0).0;
    assert_eq!(tl, [0, 0, 0], "TL anchor corner must be pure black");

    let ql = rgb.get_pixel(1 * module_size, 1 * module_size).0;
    assert_eq!(ql, [255, 255, 255], "TL anchor quiet zone must be pure white");
}

// -----------------------------------------------------------------------
// Triangle Quadrant Rendering
// -----------------------------------------------------------------------

/// Each HSL art cell must contain exactly 4 triangle regions with distinct colors
/// when the source image has 4 different colored quadrants.
#[test]
fn triangle_quadrant_produces_four_distinct_hues() {
    let n = 20;
    let module_size = 8u32;

    // Build a source image with 4 colored quadrants:
    // Top-left=red, Top-right=green, Bottom-left=blue, Bottom-right=yellow
    let mut src = RgbImage::new(n as u32, n as u32);
    for y in 0..n as u32 {
        for x in 0..n as u32 {
            let color = match (x < n as u32 / 2, y < n as u32 / 2) {
                (true, true) => Rgb([255, 0, 0]),
                (false, true) => Rgb([0, 255, 0]),
                (true, false) => Rgb([0, 0, 255]),
                (false, false) => Rgb([255, 255, 0]),
            };
            src.put_pixel(x, y, color);
        }
    }
    let dyn_img = DynamicImage::ImageRgb8(src);

    let mut matrix = build_test_matrix(n);
    matrix.set(10, 10, 1);

    let font_mask = build_font_mask(n);

    let img = AnchorRenderer::render_halftone_png(
        &matrix, module_size, &dyn_img, &font_mask
    ).unwrap();
    let rgb = img.to_rgb8();

    let px_x = 10 * module_size;
    let px_y = 10 * module_size;
    let half = module_size / 2;

    // Sample pixels from each triangle region
    let top = rgb.get_pixel(px_x + half, px_y + 0).0;
    let bottom = rgb.get_pixel(px_x + half, px_y + module_size - 1).0;
    let left = rgb.get_pixel(px_x + 0, px_y + half).0;
    let right = rgb.get_pixel(px_x + module_size - 1, px_y + half).0;

    // All 4 should be different colors
    assert_ne!(top, bottom, "Top and bottom triangles must have different hues");
    assert_ne!(left, right, "Left and right triangles must have different hues");
    assert_ne!(top, left, "Top and left triangles must have different hues");
}

/// All 4 triangles in a cell must produce dark pixels for trit=0 (L=0.10).
#[test]
fn triangle_quadrant_shared_luminosity_dark() {
    let n = 20;
    let module_size = 8u32;

    let mut src = RgbImage::new(n as u32, n as u32);
    for y in 0..n as u32 {
        for x in 0..n as u32 {
            src.put_pixel(x, y, Rgb([255, 128, 0]));
        }
    }
    let dyn_img = DynamicImage::ImageRgb8(src);

    let mut matrix = build_test_matrix(n);
    matrix.set(10, 10, 0); // dark trit

    let font_mask = build_font_mask(n);

    let img = AnchorRenderer::render_halftone_png(
        &matrix, module_size, &dyn_img, &font_mask
    ).unwrap();
    let rgb = img.to_rgb8();

    let px_x = 10 * module_size;
    let px_y = 10 * module_size;
    let half = module_size / 2;

    for &(dx, dy, name) in &[
        (half, 0u32, "top"),
        (half, module_size - 1, "bottom"),
        (0u32, half, "left"),
        (module_size - 1, half, "right"),
    ] {
        let px = rgb.get_pixel(px_x + dx, px_y + dy).0;
        let luma = 0.299 * px[0] as f64 + 0.587 * px[1] as f64 + 0.114 * px[2] as f64;
        assert!(luma < 80.0,
            "{} triangle must be dark (L=0.10), got luma={:.1} from {:?}", name, luma, px);
    }
}

/// All 4 triangles must produce light pixels for trit=2 (L=0.90).
#[test]
fn triangle_quadrant_shared_luminosity_light() {
    let n = 20;
    let module_size = 8u32;

    let mut src = RgbImage::new(n as u32, n as u32);
    for y in 0..n as u32 {
        for x in 0..n as u32 {
            src.put_pixel(x, y, Rgb([0, 100, 200]));
        }
    }
    let dyn_img = DynamicImage::ImageRgb8(src);

    let mut matrix = build_test_matrix(n);
    matrix.set(10, 10, 2); // light trit

    let font_mask = build_font_mask(n);

    let img = AnchorRenderer::render_halftone_png(
        &matrix, module_size, &dyn_img, &font_mask
    ).unwrap();
    let rgb = img.to_rgb8();

    let px_x = 10 * module_size;
    let px_y = 10 * module_size;
    let half = module_size / 2;

    for &(dx, dy, name) in &[
        (half, 0u32, "top"),
        (half, module_size - 1, "bottom"),
        (0u32, half, "left"),
        (module_size - 1, half, "right"),
    ] {
        let px = rgb.get_pixel(px_x + dx, px_y + dy).0;
        let luma = 0.299 * px[0] as f64 + 0.587 * px[1] as f64 + 0.114 * px[2] as f64;
        assert!(luma > 200.0,
            "{} triangle must be light (L=0.90), got luma={:.1} from {:?}", name, luma, px);
    }
}

/// Font mask cells must be flat-fill (no triangles), even with multicolor source.
#[test]
fn font_cells_remain_flat_fill_no_triangles() {
    let n = 20;
    let module_size = 8u32;

    let mut src = RgbImage::new(n as u32, n as u32);
    for y in 0..n as u32 {
        for x in 0..n as u32 {
            let r = ((x * 255) / n as u32) as u8;
            let g = ((y * 255) / n as u32) as u8;
            src.put_pixel(x, y, Rgb([r, g, 128]));
        }
    }
    let dyn_img = DynamicImage::ImageRgb8(src);

    let mut matrix = build_test_matrix(n);
    matrix.set(10, 10, 0);

    let mut font_mask = build_font_mask(n);
    font_mask[10][10] = Some(0);

    let img = AnchorRenderer::render_halftone_png(
        &matrix, module_size, &dyn_img, &font_mask
    ).unwrap();
    let rgb = img.to_rgb8();

    let px_x = 10 * module_size;
    let px_y = 10 * module_size;

    let reference = rgb.get_pixel(px_x, px_y).0;
    for dy in 0..module_size {
        for dx in 0..module_size {
            let px = rgb.get_pixel(px_x + dx, px_y + dy).0;
            assert_eq!(px, reference,
                "Font cell must be flat fill, pixel ({},{}) differs: {:?} vs {:?}",
                dx, dy, px, reference);
        }
    }
    assert_eq!(reference, [0, 0, 0], "Font stroke must be pure black");
}

/// Anchor cells must be flat-fill (no triangles).
#[test]
fn anchor_cells_remain_flat_fill_no_triangles() {
    let n = 20;
    let module_size = 8u32;

    let mut src = RgbImage::new(n as u32, n as u32);
    for y in 0..n as u32 {
        for x in 0..n as u32 {
            src.put_pixel(x, y, Rgb([255, 128, 0]));
        }
    }
    let dyn_img = DynamicImage::ImageRgb8(src);

    let matrix = build_test_matrix(n);
    let font_mask = build_font_mask(n);

    let img = AnchorRenderer::render_halftone_png(
        &matrix, module_size, &dyn_img, &font_mask
    ).unwrap();
    let rgb = img.to_rgb8();

    let reference = rgb.get_pixel(0, 0).0;
    for dy in 0..module_size {
        for dx in 0..module_size {
            let px = rgb.get_pixel(dx, dy).0;
            assert_eq!(px, reference,
                "Anchor cell must be flat fill, pixel ({},{}) differs", dx, dy);
        }
    }
}

/// Every pixel in an HSL art cell must be filled (no gaps in triangle rasterization).
#[test]
fn triangle_rasterization_complete_coverage() {
    let n = 20;
    let module_size = 7u32; // odd size to stress-test center alignment

    let mut src = RgbImage::new(n as u32, n as u32);
    for y in 0..n as u32 {
        for x in 0..n as u32 {
            src.put_pixel(x, y, Rgb([200, 100, 50]));
        }
    }
    let dyn_img = DynamicImage::ImageRgb8(src);

    let mut matrix = build_test_matrix(n);
    matrix.set(10, 10, 1);

    let font_mask = build_font_mask(n);

    let img = AnchorRenderer::render_halftone_png(
        &matrix, module_size, &dyn_img, &font_mask
    ).unwrap();
    let rgb = img.to_rgb8();

    let px_x = 10 * module_size;
    let px_y = 10 * module_size;

    for dy in 0..module_size {
        for dx in 0..module_size {
            let px = rgb.get_pixel(px_x + dx, px_y + dy).0;
            let sum: u32 = px[0] as u32 + px[1] as u32 + px[2] as u32;
            assert!(sum > 0,
                "Pixel ({},{}) is black — triangle rasterization gap!", dx, dy);
        }
    }
}

/// Uniform source must produce a uniform cell (all triangles same color).
#[test]
fn uniform_source_produces_uniform_cell() {
    let n = 20;
    let module_size = 8u32;

    let mut src = RgbImage::new(n as u32, n as u32);
    for y in 0..n as u32 {
        for x in 0..n as u32 {
            src.put_pixel(x, y, Rgb([0, 200, 200]));
        }
    }
    let dyn_img = DynamicImage::ImageRgb8(src);

    let mut matrix = build_test_matrix(n);
    matrix.set(10, 10, 1);

    let font_mask = build_font_mask(n);

    let img = AnchorRenderer::render_halftone_png(
        &matrix, module_size, &dyn_img, &font_mask
    ).unwrap();
    let rgb = img.to_rgb8();

    let px_x = 10 * module_size;
    let px_y = 10 * module_size;

    let reference = rgb.get_pixel(px_x, px_y).0;
    for dy in 0..module_size {
        for dx in 0..module_size {
            let px = rgb.get_pixel(px_x + dx, px_y + dy).0;
            for c in 0..3 {
                let diff = (px[c] as i16 - reference[c] as i16).unsigned_abs();
                assert!(diff <= 1,
                    "Uniform source: pixel ({},{}) ch {} differs by {}", dx, dy, c, diff);
            }
        }
    }
}

/// Module size 2 must not panic and produce a valid image.
#[test]
fn small_module_size_no_panic() {
    let n = 20;
    let module_size = 2u32;

    let mut src = RgbImage::new(n as u32, n as u32);
    for y in 0..n as u32 {
        for x in 0..n as u32 {
            src.put_pixel(x, y, Rgb([100, 200, 50]));
        }
    }
    let dyn_img = DynamicImage::ImageRgb8(src);

    let mut matrix = build_test_matrix(n);
    matrix.set(10, 10, 1);

    let font_mask = build_font_mask(n);

    let result = AnchorRenderer::render_halftone_png(
        &matrix, module_size, &dyn_img, &font_mask
    );
    assert!(result.is_ok(), "Module size 2 must not panic");
    let img = result.unwrap();
    assert_eq!(img.to_rgb8().width(), n as u32 * module_size);
}

/// Module size 1 must not panic.
#[test]
fn module_size_one_no_panic() {
    let n = 20;
    let module_size = 1u32;

    let mut src = RgbImage::new(n as u32, n as u32);
    for y in 0..n as u32 {
        for x in 0..n as u32 {
            src.put_pixel(x, y, Rgb([100, 200, 50]));
        }
    }
    let dyn_img = DynamicImage::ImageRgb8(src);

    let mut matrix = build_test_matrix(n);
    matrix.set(10, 10, 1);

    let font_mask = build_font_mask(n);

    let result = AnchorRenderer::render_halftone_png(
        &matrix, module_size, &dyn_img, &font_mask
    );
    assert!(result.is_ok(), "Module size 1 must not panic");
}
