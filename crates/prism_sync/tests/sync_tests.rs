use prism_sync::pilot::{generate_pilot_grid, PilotConfig};
use prism_sync::fft::{fft2d, ifft2d};
use prism_sync::detector::{detect_pilot_peaks, estimate_affine, estimate_rotation_and_scale};
use prism_sync::{apply_homography, invert_homography};

#[test]
fn test_pilot_generation_bounded() {
    let width = 64;
    let height = 64;
    let config = PilotConfig {
        ku: 16.0,
        kv: 16.0,
        amplitude: 1.5,
    };

    let grid = generate_pilot_grid(width, height, &config);
    assert_eq!(grid.len(), width * height);

    for &v in &grid {
        assert!(v.abs() <= 2.0 * config.amplitude + 1e-4, "Pilot value {} exceeded bound", v);
    }
}

#[test]
fn test_fft2d_roundtrip() {
    let size = 32;
    let mut real = vec![0.0f32; size * size];
    let mut imag = vec![0.0f32; size * size];

    for y in 0..size {
        for x in 0..size {
            real[y * size + x] = (x * 5 + y * 9) as f32 + ((x * y) as f32).sin();
        }
    }

    let orig_real = real.clone();

    fft2d(&mut real, &mut imag, size, size);
    ifft2d(&mut real, &mut imag, size, size);

    for i in 0..size * size {
        let err = (real[i] - orig_real[i]).abs();
        assert!(err < 1e-3, "FFT 2D roundtrip error at {}: expected {}, got {}", i, orig_real[i], real[i]);
    }
}

#[test]
fn test_pilot_peak_detection_canonical() {
    let size = 64;
    let config = PilotConfig {
        ku: 16.0,
        kv: 16.0,
        amplitude: 2.0,
    };

    let mut image = vec![128.0f32; size * size];
    let pilot = generate_pilot_grid(size, size, &config);
    for i in 0..size * size {
        image[i] += pilot[i];
    }

    let peaks = detect_pilot_peaks(&image, size, size, &config)
        .expect("Peaks must be detected in canonical grid");

    assert_eq!(peaks.len(), 4, "Must detect 4 conjugate peaks");
    eprintln!("Detected peaks: {:?}", peaks);

    let (angle, scale) = estimate_rotation_and_scale(&peaks, size, size, &config)
        .expect("Rotation and scale must be estimated");

    assert!(angle.abs() < 0.05, "Canonical angle must be near 0, got {}", angle);
    assert!((scale - 1.0).abs() < 0.05, "Canonical scale must be near 1.0, got {}", scale);
}

/// Bilinear sample of a float image at fractional coordinates.
fn bilinear_sample(img: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let a = img[y0 * w + x0];
    let b = img[y0 * w + x1];
    let c = img[y1 * w + x0];
    let d = img[y1 * w + x1];
    a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy
}

/// Rotates an image by `theta_deg` via inverse bilinear sampling. Pixels whose
/// source falls outside the image are filled with `fill`.
fn rotate_image(img: &[f32], w: usize, h: usize, theta_deg: f32, fill: f32) -> Vec<f32> {
    let (s, c) = theta_deg.to_radians().sin_cos();
    let cx = (w as f32 - 1.0) * 0.5;
    let cy = (h as f32 - 1.0) * 0.5;
    let mut out = vec![fill; w * h];
    for y in 0..h {
        let dy = y as f32 - cy;
        for x in 0..w {
            let dx = x as f32 - cx;
            // Inverse map: src = R(-theta) * (dst - c) + c
            let sx = c * dx + s * dy + cx;
            let sy = -s * dx + c * dy + cy;
            if sx >= 0.0 && sx <= (w - 1) as f32 && sy >= 0.0 && sy <= (h - 1) as f32 {
                out[y * w + x] = bilinear_sample(img, w, h, sx, sy);
            }
        }
    }
    out
}

/// Warps an image with a centered homography (linear part + perspective terms
/// around the image center) via inverse bilinear sampling.
fn warp_homography(img: &[f32], w: usize, h: usize, h_mat: &[[f32; 3]; 3], fill: f32) -> Vec<f32> {
    let inv = invert_homography(h_mat).expect("warp homography must be invertible");
    let cx = (w as f32 - 1.0) * 0.5;
    let cy = (h as f32 - 1.0) * 0.5;
    let mut out = vec![fill; w * h];
    for y in 0..h {
        let dy = y as f32 - cy;
        for x in 0..w {
            let dx = x as f32 - cx;
            let (sxc, syc) = apply_homography(&inv, dx, dy);
            let sx = sxc + cx;
            let sy = syc + cy;
            if sx >= 0.0 && sx <= (w - 1) as f32 && sy >= 0.0 && sy <= (h - 1) as f32 {
                out[y * w + x] = bilinear_sample(img, w, h, sx, sy);
            }
        }
    }
    out
}

/// Builds the pilot-on-DC test image: 128.0 DC plus amplitude-2.0 pilot.
fn pilot_image(size: usize, config: &PilotConfig) -> Vec<f32> {
    let pilot = generate_pilot_grid(size, size, config);
    let mut image = vec![128.0f32; size * size];
    for i in 0..image.len() {
        image[i] += pilot[i];
    }
    image
}

#[test]
fn test_rotation_35_deg_signed_recovery() {
    let size = 256;
    let config = PilotConfig {
        ku: 16.0,
        kv: 16.0,
        amplitude: 2.0,
    };
    let image = pilot_image(size, &config);

    let mut max_theta_err = 0.0f32;
    let mut max_scale_err = 0.0f32;
    for &theta_deg in &[35.0f32, -35.0] {
        let rotated = rotate_image(&image, size, size, theta_deg, 128.0);
        let peaks = detect_pilot_peaks(&rotated, size, size, &config)
            .expect("Peaks must be detected under rotation");
        let (theta, sx, sy) = estimate_affine(&peaks, &config).expect("Affine must be estimated");

        let theta_err = (theta.to_degrees() - theta_deg).abs();
        let scale_err = ((sx - 1.0).abs()).max((sy - 1.0).abs());
        max_theta_err = max_theta_err.max(theta_err);
        max_scale_err = max_scale_err.max(scale_err);

        eprintln!(
            "theta={:+.0} deg: est {:+.6} deg (err {:.6}), sx={:.6}, sy={:.6}, scale err {:.6}",
            theta_deg,
            theta.to_degrees(),
            theta_err,
            sx,
            sy,
            scale_err
        );
        assert!(theta_err < 0.1, "theta error {theta_err} deg >= 0.1 deg at {theta_deg}");
        assert!(
            scale_err < 0.005,
            "scale error {scale_err} >= 0.005 at {theta_deg}"
        );
    }
    eprintln!(
        "MAX over +/-35 deg: theta err {:.6} deg, scale err {:.6}",
        max_theta_err, max_scale_err
    );
}

#[test]
fn test_perspective_skew_affine_recovery() {
    let size = 256;
    let config = PilotConfig {
        ku: 16.0,
        kv: 16.0,
        amplitude: 2.0,
    };
    let image = pilot_image(size, &config);

    // Centered tilt: rotation, anisotropic scale, mild perspective terms.
    // The spectrum undergoes the inverse-transpose of the image's linear map.
    let theta_deg = 12.0f32;
    let sx_img = 1.05f32;
    let sy_img = 0.97f32;
    let (s, c) = theta_deg.to_radians().sin_cos();
    let h = [
        [c * sx_img, -s * sy_img, 0.0],
        [s * sx_img, c * sy_img, 0.0],
        [0.0004, 0.0003, 1.0],
    ];

    let warped = warp_homography(&image, size, size, &h, 128.0);
    let peaks = detect_pilot_peaks(&warped, size, size, &config)
        .expect("Peaks must be detected under perspective skew");
    let (theta_r, sx_r, sy_r) = estimate_affine(&peaks, &config)
        .expect("Affine must be estimated under perspective skew");

    // Expected spike transform A = (J^-1)^T, J = linear part of H at the center.
    let j = [[c * sx_img, -s * sy_img], [s * sx_img, c * sy_img]];
    let det_j = j[0][0] * j[1][1] - j[0][1] * j[1][0];
    let j_inv = [
        [j[1][1] / det_j, -j[0][1] / det_j],
        [-j[1][0] / det_j, j[0][0] / det_j],
    ];
    let a = [
        [j_inv[0][0], j_inv[1][0]],
        [j_inv[0][1], j_inv[1][1]],
    ];

    // Polar decomposition of A: R(phi) with phi = atan2(A10 - A01, A00 + A11),
    // then scales from S = R^T A.
    let phi = (a[1][0] - a[0][1]).atan2(a[0][0] + a[1][1]);
    let (se, ce) = phi.sin_cos();
    let s_mat = [
        [ce * a[0][0] + se * a[1][0], ce * a[0][1] + se * a[1][1]],
        [-se * a[0][0] + ce * a[1][0], -se * a[0][1] + ce * a[1][1]],
    ];
    let sx_e = s_mat[0][0];
    let sy_e = s_mat[1][1];

    let theta_err = (theta_r - phi).abs().to_degrees();
    eprintln!(
        "perspective: recovered theta {:+.6} deg, expected {:+.6} deg (err {:.6})",
        theta_r.to_degrees(),
        phi.to_degrees(),
        theta_err
    );
    eprintln!(
        "perspective: recovered sx={:.6}, sy={:.6}; expected sx={:.6}, sy={:.6}",
        sx_r, sy_r, sx_e, sy_e
    );

    assert!(theta_err < 2.0, "theta error {theta_err} deg under perspective skew");
    assert!((sx_r - sx_e).abs() < 0.08, "sx error under perspective skew");
    assert!((sy_r - sy_e).abs() < 0.08, "sy error under perspective skew");
}
