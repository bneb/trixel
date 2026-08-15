use prism_sync::homography::{apply_homography, invert_homography, solve_dlt};

fn rotation_homography(theta_deg: f32, tx: f32, ty: f32) -> [[f32; 3]; 3] {
    let (s, c) = theta_deg.to_radians().sin_cos();
    [[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]]
}

/// Deterministic set of 8 source points in general position.
fn source_points() -> [(f32, f32); 8] {
    [
        (12.0, 34.0),
        (67.0, 8.0),
        (91.0, 52.0),
        (23.0, 88.0),
        (45.0, 45.0),
        (80.0, 95.0),
        (5.0, 62.0),
        (58.0, 21.0),
    ]
}

fn max_reprojection_error(
    h: &[[f32; 3]; 3],
    h_est: &[[f32; 3]; 3],
    pts: &[(f32, f32)],
) -> f32 {
    pts.iter()
        .map(|&(x, y)| {
            let (dx, dy) = apply_homography(h, x, y);
            let (ex, ey) = apply_homography(h_est, x, y);
            ((dx - ex).powi(2) + (dy - ey).powi(2)).sqrt()
        })
        .fold(0.0f32, f32::max)
}

#[test]
fn test_dlt_rotation_reprojection() {
    let h = rotation_homography(23.0, 10.0, -7.0);
    let pts = source_points();
    let correspondences: Vec<(f32, f32, f32, f32)> = pts
        .iter()
        .map(|&(x, y)| {
            let (dx, dy) = apply_homography(&h, x, y);
            (x, y, dx, dy)
        })
        .collect();

    let h_est = solve_dlt(&correspondences).expect("DLT must solve for rotation");
    let err = max_reprojection_error(&h, &h_est, &pts);
    eprintln!("rotation: max reprojection error = {:.6} px", err);
    assert!(err < 0.5, "reprojection error {err} px exceeds 0.5 px");

    // Matrix recovery is close to ground truth
    for i in 0..3 {
        for j in 0..3 {
            let d = (h[i][j] - h_est[i][j]).abs();
            assert!(d < 0.01, "H[{i}][{j}] error {d}");
        }
    }
}

#[test]
fn test_dlt_scale_and_tilt_reprojection() {
    // Scale, shear and perspective tilt
    let h = [[1.2, 0.1, 5.0], [0.05, 0.9, -3.0], [0.001, 0.0005, 1.0]];
    let pts = source_points();
    let correspondences: Vec<(f32, f32, f32, f32)> = pts
        .iter()
        .map(|&(x, y)| {
            let (dx, dy) = apply_homography(&h, x, y);
            (x, y, dx, dy)
        })
        .collect();

    let h_est = solve_dlt(&correspondences).expect("DLT must solve for scale+tilt");
    let err = max_reprojection_error(&h, &h_est, &pts);
    eprintln!("scale+tilt: max reprojection error = {:.6} px", err);
    assert!(err < 0.5, "reprojection error {err} px exceeds 0.5 px");

    // Homographies are defined up to scale: normalize both to h33 = 1.
    let hn = {
        let mut m = h_est;
        let s = m[2][2];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] /= s;
            }
        }
        m
    };
    for i in 0..3 {
        for j in 0..3 {
            let d = (h[i][j] - hn[i][j]).abs();
            assert!(d < 0.01, "H[{i}][{j}] error {d}");
        }
    }
}

#[test]
fn test_dlt_requires_four_pairs() {
    let h = rotation_homography(10.0, 0.0, 0.0);
    let pts = &source_points()[..3];
    let correspondences: Vec<(f32, f32, f32, f32)> = pts
        .iter()
        .map(|&(x, y)| {
            let (dx, dy) = apply_homography(&h, x, y);
            (x, y, dx, dy)
        })
        .collect();
    assert!(solve_dlt(&correspondences).is_none(), "3 pairs must be rejected");
    assert!(solve_dlt(&[]).is_none(), "empty input must be rejected");
}

#[test]
fn test_dlt_collinear_degenerate() {
    let h = rotation_homography(10.0, 0.0, 0.0);
    // Collinear source points: infinitely many homographies agree on them
    let pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)];
    let correspondences: Vec<(f32, f32, f32, f32)> = pts
        .iter()
        .map(|&(x, y)| {
            let (dx, dy) = apply_homography(&h, x, y);
            (x, y, dx, dy)
        })
        .collect();
    assert!(
        solve_dlt(&correspondences).is_none(),
        "collinear source points must be rejected"
    );
}

#[test]
fn test_dlt_identical_points_degenerate() {
    let h = rotation_homography(10.0, 0.0, 0.0);
    let correspondences: Vec<(f32, f32, f32, f32)> = (0..4)
        .map(|_| {
            let (dx, dy) = apply_homography(&h, 5.0, 5.0);
            (5.0, 5.0, dx, dy)
        })
        .collect();
    assert!(
        solve_dlt(&correspondences).is_none(),
        "coincident source points must be rejected"
    );
}

#[test]
fn test_invert_homography_roundtrip() {
    let h = [[1.3, -0.4, 12.0], [0.2, 0.9, -5.0], [0.0007, -0.0003, 1.0]];
    let inv = invert_homography(&h).expect("invertible homography");

    // H * H^-1 ~= identity (row-major multiply)
    let prod = {
        let mut p = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                p[i][j] = h[i][0] * inv[0][j] + h[i][1] * inv[1][j] + h[i][2] * inv[2][j];
            }
        }
        p
    };
    let mut max_dev = 0.0f32;
    for i in 0..3 {
        for j in 0..3 {
            let want = if i == j { 1.0 } else { 0.0 };
            max_dev = max_dev.max((prod[i][j] - want).abs());
        }
    }
    eprintln!("inverse roundtrip max deviation = {:.8}", max_dev);
    assert!(max_dev < 1e-4, "H*H^-1 deviates {max_dev} from identity");
}

#[test]
fn test_invert_homography_singular() {
    let singular = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    assert!(
        invert_homography(&singular).is_none(),
        "singular homography must not invert"
    );
}

#[test]
fn test_apply_homography_known_points() {
    // Translation
    let t = [[1.0, 0.0, 5.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]];
    assert_eq!(apply_homography(&t, 1.0, 1.0), (6.0, -1.0));

    // Perspective: x' = x / (1 + 0.5x), y' = y / (1 + 0.5x)
    let p = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 1.0]];
    let (x, y) = apply_homography(&p, 2.0, 4.0);
    assert!((x - 1.0).abs() < 1e-5, "x = {x}");
    assert!((y - 2.0).abs() < 1e-5, "y = {y}");
}
