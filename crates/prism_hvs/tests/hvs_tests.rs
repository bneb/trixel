//! Phase 2 red tests: 2D DCT-II & Yang-Bovik JND engine, CIELAB invariants.
//!
//! Plan targets (implementation_plan.md, Phase 2 — `prism_hvs`):
//! - DCT/IDCT roundtrip error < 1e-5 (max abs and RMS) with Parseval energy
//!   preservation within 1e-4 relative.
//! - JND map allocates ~zero energy to flat white/black areas and maximum energy
//!   to high-variance textures.
//! - CIELAB roundtrip within 2/255 per channel on in-gamut colors.

use prism_hvs::color::{lab_to_srgb, srgb_to_lab};
use prism_hvs::dct::{dct_8x8, idct_8x8};
use prism_hvs::jnd::compute_spatial_jnd;

const DCT_MAX_ABS_TOL: f32 = 1e-5;
const DCT_RMS_TOL: f32 = 1e-5;
const PARSEVAL_REL_TOL: f32 = 1e-4;
const JND_MIN: f32 = 1.5;
const JND_MAX: f32 = 32.0;

/// Deterministic 24-bit LCG so DCT failures are reproducible.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0xD1B54A32D192ED03))
    }

    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as u32 as f32) / ((1u32 << 24) as f32)
    }
}

fn fill_random(block: &mut [f32; 64], rng: &mut Lcg, scale: f32) {
    for v in block.iter_mut() {
        *v = rng.next_f32() * scale;
    }
}

/// Forward DCT then inverse, returning (max abs error, RMS error).
fn roundtrip_max_rms_err(input: &[f32; 64]) -> (f32, f32) {
    let mut coeff = [0.0f32; 64];
    let mut out = [0.0f32; 64];
    dct_8x8(input, &mut coeff);
    idct_8x8(&coeff, &mut out);

    let mut max_abs = 0.0f32;
    let mut sq_sum = 0.0f32;
    for i in 0..64 {
        let e = (input[i] - out[i]).abs();
        max_abs = max_abs.max(e);
        sq_sum += e * e;
    }
    (max_abs, (sq_sum / 64.0).sqrt())
}

/// Relative Parseval energy drift |E(x) - E(DCT(x))| / E(x).
fn energy_relative_drift(input: &[f32; 64]) -> f32 {
    let mut coeff = [0.0f32; 64];
    dct_8x8(input, &mut coeff);
    let e_in: f32 = input.iter().map(|v| v * v).sum();
    let e_co: f32 = coeff.iter().map(|v| v * v).sum();
    (e_in - e_co).abs() / e_in
}

// ---------------------------------------------------------------------------
// DCT / IDCT roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_dct8x8_orthonormal_roundtrip() {
    let mut input = [0.0f32; 64];
    for y in 0..8 {
        for x in 0..8 {
            input[y * 8 + x] = (x * 13 + y * 7) as f32 / 10.0 + ((x + y) as f32).sin();
        }
    }

    let mut coeff = [0.0f32; 64];
    dct_8x8(&input, &mut coeff);

    let mut output = [0.0f32; 64];
    idct_8x8(&coeff, &mut output);

    for i in 0..64 {
        let err = (input[i] - output[i]).abs();
        assert!(err < 1e-5, "DCT roundtrip error at {}: expected {}, got {} (err: {})", i, input[i], output[i], err);
    }
}

#[test]
fn test_dct_roundtrip_random_blocks_normalized() {
    // Plan target: roundtrip error < 1e-5. f32 has ~1.2e-7 relative precision, so
    // the target is met at unit-magnitude inputs (the natural scale for testing an
    // f32 unitary transform); full-range 0-255 magnitude is covered separately.
    let mut rng = Lcg::new(0x5EED_1234);
    let mut worst_max = 0.0f32;
    let mut worst_rms = 0.0f32;
    for block_idx in 0..8 {
        let mut input = [0.0f32; 64];
        fill_random(&mut input, &mut rng, 1.0);
        let (max_abs, rms) = roundtrip_max_rms_err(&input);
        worst_max = worst_max.max(max_abs);
        worst_rms = worst_rms.max(rms);
        assert!(
            max_abs < DCT_MAX_ABS_TOL,
            "block {block_idx}: max abs error {max_abs:.3e} >= 1e-5"
        );
        assert!(
            rms < DCT_RMS_TOL,
            "block {block_idx}: RMS error {rms:.3e} >= 1e-5"
        );
    }
    eprintln!("random [0,1] blocks (8): worst max abs {worst_max:.3e}, worst RMS {worst_rms:.3e}");
}

#[test]
fn test_dct_roundtrip_constant_gradient_checkerboard_normalized() {
    let const_block = [0.5f32; 64];

    let mut grad_block = [0.0f32; 64];
    for i in 0..64 {
        grad_block[i] = i as f32 / 63.0;
    }

    let mut checker_block = [0.0f32; 64];
    for y in 0..8 {
        for x in 0..8 {
            checker_block[y * 8 + x] = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
        }
    }

    for (name, block) in [
        ("constant 0.5", &const_block),
        ("gradient 0..1", &grad_block),
        ("checkerboard", &checker_block),
    ] {
        let (max_abs, rms) = roundtrip_max_rms_err(block);
        assert!(
            max_abs < DCT_MAX_ABS_TOL,
            "{name}: max abs error {max_abs:.3e} >= 1e-5"
        );
        assert!(
            rms < DCT_RMS_TOL,
            "{name}: RMS error {rms:.3e} >= 1e-5"
        );
    }
}

#[test]
fn test_dct_roundtrip_full_range_255_blocks() {
    // Full 0-255 magnitude, the scale the embedder actually feeds the JND engine.
    // f32 ulp at 255 is ~1.5e-5, so the 1e-5 plan target is not reachable here;
    // assert the measured f32 floor (see report for exact numbers).
    let const_block = [255.0f32; 64];

    let mut grad_block = [0.0f32; 64];
    for i in 0..64 {
        grad_block[i] = (i as f32) * 255.0 / 63.0;
    }

    let mut checker_block = [0.0f32; 64];
    for y in 0..8 {
        for x in 0..8 {
            checker_block[y * 8 + x] = if (x + y) % 2 == 0 { 255.0 } else { 0.0 };
        }
    }

    let mut rng = Lcg::new(0xC0FFEE);
    let mut rand_block = [0.0f32; 64];
    fill_random(&mut rand_block, &mut rng, 255.0);

    for (name, block) in [
        ("constant 255", &const_block),
        ("gradient 0..255", &grad_block),
        ("checkerboard 0/255", &checker_block),
        ("random 0..255", &rand_block),
    ] {
        let (max_abs, rms) = roundtrip_max_rms_err(block);
        eprintln!("{name}: max abs {max_abs:.3e}, RMS {rms:.3e}");
        assert!(
            max_abs < 1e-3,
            "{name}: max abs error {max_abs:.3e} exceeds f32 full-range floor"
        );
        assert!(
            rms < 1e-3,
            "{name}: RMS error {rms:.3e} exceeds f32 full-range floor"
        );
    }
}

#[test]
fn test_dct_constant_block_dc_only() {
    // A constant block must produce only a DC coefficient.
    let input = [0.25f32; 64];
    let mut coeff = [0.0f32; 64];
    dct_8x8(&input, &mut coeff);
    assert!((coeff[0] - 2.0).abs() < 1e-5, "DC coeff {} != 2.0", coeff[0]);
    for i in 1..64 {
        assert!(
            coeff[i].abs() < 1e-5,
            "AC coeff {i} = {} for constant block",
            coeff[i]
        );
    }
}

#[test]
fn test_dct_parseval_energy_preservation() {
    let const_block = [0.5f32; 64];

    let mut grad_block = [0.0f32; 64];
    for i in 0..64 {
        grad_block[i] = i as f32 / 63.0;
    }

    let mut rng = Lcg::new(0xABBA_CAFE);
    let mut rand_block = [0.0f32; 64];
    fill_random(&mut rand_block, &mut rng, 1.0);
    let mut rand_255 = [0.0f32; 64];
    fill_random(&mut rand_255, &mut rng, 255.0);

    for (name, block) in [
        ("constant 0.5", &const_block),
        ("gradient", &grad_block),
        ("random [0,1]", &rand_block),
        ("random [0,255]", &rand_255),
    ] {
        let drift = energy_relative_drift(block);
        eprintln!("{name}: relative energy drift {drift:.3e}");
        assert!(
            drift < PARSEVAL_REL_TOL,
            "{name}: relative energy drift {drift:.3e} >= 1e-4"
        );
    }
}

// ---------------------------------------------------------------------------
// Yang-Bovik JND
// ---------------------------------------------------------------------------

fn flat_image(val: f32, width: usize, height: usize) -> Vec<f32> {
    vec![val; width * height]
}

#[test]
fn test_jnd_output_length_and_bounds() {
    let (w, h) = (37, 23);
    let mut rng = Lcg::new(0xBADF00D);
    let img: Vec<f32> = (0..w * h).map(|_| rng.next_f32() * 255.0).collect();
    let map = compute_spatial_jnd(&img, w, h);

    assert_eq!(map.len(), w * h, "JND map must match input dimensions");
    for (i, &v) in map.iter().enumerate() {
        assert!(v.is_finite(), "JND value {v} at {i} is not finite");
        assert!(
            v >= JND_MIN && v <= JND_MAX,
            "JND value {v} at {i} outside clamp bounds [{JND_MIN}, {JND_MAX}]"
        );
    }
}

#[test]
fn test_jnd_flat_extremes_at_minimum() {
    // Plan: JND allocates ~zero energy to flat white/black areas, so the threshold
    // must sit at/near the minimum (clamp floor 1.5), not ~6 as with the classic
    // Yang-Bovik T_L at bg=255.
    for (name, luma) in [
        ("white", 250.0f32),
        ("pure white", 255.0f32),
        ("black", 5.0f32),
        ("pure black", 0.0f32),
    ] {
        let map = compute_spatial_jnd(&flat_image(luma, 32, 32), 32, 32);
        let center = map[16 * 32 + 16];
        eprintln!("flat {name} (bg={luma}): center jnd = {center}");
        assert!(
            center <= 3.0,
            "flat {name} jnd {center} should be at/near the minimum threshold"
        );
    }
}

#[test]
fn test_jnd_flat_midgray_low_threshold() {
    let map = compute_spatial_jnd(&flat_image(128.0, 32, 32), 32, 32);
    let center = map[16 * 32 + 16];
    eprintln!("flat mid-gray: center jnd = {center}");
    assert!(
        center < 10.0,
        "flat mid-gray jnd {center} should stay low (max is 32)"
    );
    assert!(
        center > JND_MIN + 1e-3,
        "flat mid-gray jnd {center} should sit above the extreme floor"
    );
}

#[test]
fn test_jnd_checkerboard_reaches_maximum() {
    let mut img = vec![0.0f32; 32 * 32];
    for y in 0..32 {
        for x in 0..32 {
            img[y * 32 + x] = if (x + y) % 2 == 0 { 255.0 } else { 0.0 };
        }
    }
    let map = compute_spatial_jnd(&img, 32, 32);
    let center = map[16 * 32 + 16];
    eprintln!("checkerboard: center jnd = {center}");
    assert!(
        center >= 30.0,
        "high-variance checkerboard jnd {center} should saturate near the max 32"
    );
}

#[test]
fn test_jnd_masks_smooth_vs_textured_regions() {
    let width = 32;
    let height = 32;
    let flat_image = vec![128.0f32; width * height];
    let mut textured_image = vec![128.0f32; width * height];

    // Add high-frequency checkerboard texture to textured_image
    for y in 0..height {
        for x in 0..width {
            if (x + y) % 2 == 0 {
                textured_image[y * width + x] = 220.0;
            } else {
                textured_image[y * width + x] = 40.0;
            }
        }
    }

    let jnd_flat = compute_spatial_jnd(&flat_image, width, height);
    let jnd_textured = compute_spatial_jnd(&textured_image, width, height);

    // Center pixel of textured region must have much higher masking threshold than flat region
    let center_idx = (height / 2) * width + (width / 2);
    eprintln!("Flat JND at center: {}, Textured JND: {}", jnd_flat[center_idx], jnd_textured[center_idx]);

    assert!(jnd_textured[center_idx] > jnd_flat[center_idx] * 2.0, "Texture must afford at least 2x higher JND threshold than flat areas");
}

// ---------------------------------------------------------------------------
// CIELAB
// ---------------------------------------------------------------------------

#[test]
fn test_cielab_srgb_roundtrip() {
    let test_colors = [
        [0u8, 0, 0],
        [255, 255, 255],
        [128, 128, 128],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [240, 180, 140], // skin tone
        [34, 139, 34],   // forest green
        [255, 165, 0],   // orange
    ];

    for &rgb in &test_colors {
        let lab = srgb_to_lab(rgb[0], rgb[1], rgb[2]);
        let recovered = lab_to_srgb(lab.l, lab.a, lab.b);

        for c in 0..3 {
            let diff = (rgb[c] as i32 - recovered[c] as i32).abs();
            assert!(diff <= 1, "Color {:?} recovered as {:?} (diff: {})", rgb, recovered, diff);
        }
    }
}

#[test]
fn test_cielab_roundtrip_in_gamut_grid() {
    // Plan target: srgb -> lab -> srgb roundtrip within 2/255 per channel across a
    // grid of in-gamut colors (6 values per channel = 216 colors).
    let vals = [0u8, 51, 102, 153, 204, 255];
    let mut max_err: u32 = 0;
    for &r in &vals {
        for &g in &vals {
            for &b in &vals {
                let lab = srgb_to_lab(r, g, b);
                let out = lab_to_srgb(lab.l, lab.a, lab.b);
                for (c, &orig) in [r, g, b].iter().enumerate() {
                    let diff = (orig as i32 - out[c] as i32).unsigned_abs();
                    max_err = max_err.max(diff);
                    assert!(
                        diff <= 2,
                        "rgb({r},{g},{b}) recovered as ({},{},{}) channel {c} diff {diff} > 2",
                        out[0], out[1], out[2]
                    );
                }
                assert!(
                    (0.0..=100.0).contains(&lab.l),
                    "L = {} outside [0,100] for rgb({r},{g},{b})",
                    lab.l
                );
            }
        }
    }
    eprintln!("CIELAB grid roundtrip max channel error: {max_err}");
}

#[test]
fn test_cielab_axis_invariants() {
    let white = srgb_to_lab(255, 255, 255);
    assert!(white.l > 99.5 && white.l <= 100.0, "white L = {} != ~100", white.l);
    assert!(white.a.abs() < 0.5 && white.b.abs() < 0.5, "white a,b = {},{} != ~0", white.a, white.b);

    let black = srgb_to_lab(0, 0, 0);
    assert!(black.l >= 0.0 && black.l < 0.5, "black L = {} != ~0", black.l);

    let gray = srgb_to_lab(128, 128, 128);
    assert!(
        gray.a.abs() <= 1.0 && gray.b.abs() <= 1.0,
        "gray a,b = {},{} should be ~0",
        gray.a,
        gray.b
    );
    assert!(gray.l > 40.0 && gray.l < 70.0, "gray L = {} out of mid range", gray.l);
}
