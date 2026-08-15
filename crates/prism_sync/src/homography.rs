//! Direct Linear Transformation (DLT) homography estimation and application.
//!
//! Recovers the 3x3 projective transform mapping source to destination points
//! from point correspondences, with Hartley normalization for numerical
//! stability. Used by the sync pipeline to recover scale, rotation and
//! perspective warp from Fourier pilot peaks.

/// Solves for the 3x3 homography mapping source points to destination points.
///
/// Each correspondence is `(src_x, src_y, dst_x, dst_y)`. At least 4
/// correspondences are required (the 8-DOF homography needs 8 equations).
/// Coordinates are Hartley-normalized (zero mean, RMS distance sqrt(2)) before
/// building the linear system; the 8x8 least-squares system is solved with
/// Gaussian elimination and partial pivoting, and the solution is
/// denormalized back to the original coordinate frame.
///
/// Returns `None` for degenerate input: fewer than 4 correspondences, collinear
/// or coincident source points, or a numerically singular system.
pub fn solve_dlt(correspondences: &[(f32, f32, f32, f32)]) -> Option<[[f32; 3]; 3]> {
    if correspondences.len() < 4 {
        return None;
    }

    let (ts, _) = normalizing_transform(correspondences.iter().map(|c| (c.0, c.1)))?;
    let (td, inv_td) = normalizing_transform(correspondences.iter().map(|c| (c.2, c.3)))?;

    // Normal equations: M^T M h8 = M^T b, where the DLT rows are
    //   [-x, -y, -1, 0, 0, 0, x'x, x'y | x']  and
    //   [ 0,  0,  0, -x, -y, -1, y'x, y'y | y']
    // with h9 = 1 moved to the right-hand side.
    let mut ata = [[0.0f64; 8]; 8];
    let mut atb = [0.0f64; 8];

    for &(x, y, xp, yp) in correspondences {
        let (xn, yn) = apply_h_f64(&ts, x as f64, y as f64);
        let (xpn, ypn) = apply_h_f64(&td, xp as f64, yp as f64);

        let rows: [[f64; 9]; 2] = [
            [-xn, -yn, -1.0, 0.0, 0.0, 0.0, xpn * xn, xpn * yn, xpn],
            [0.0, 0.0, 0.0, -xn, -yn, -1.0, ypn * xn, ypn * yn, ypn],
        ];
        for row in rows {
            let mut a_row = [0.0f64; 8];
            for j in 0..8 {
                a_row[j] = row[j];
            }
            let b_val = -row[8];
            for i in 0..8 {
                for j in 0..8 {
                    ata[i][j] += a_row[i] * a_row[j];
                }
                atb[i] += a_row[i] * b_val;
            }
        }
    }

    let h8 = gauss_solve(&mut ata, &mut atb)?;

    let mut h_norm = [[0.0f64; 3]; 3];
    h_norm[0] = [h8[0], h8[1], h8[2]];
    h_norm[1] = [h8[3], h8[4], h8[5]];
    h_norm[2] = [h8[6], h8[7], 1.0];

    // Denormalize: H = Td^-1 * H_norm * Ts
    let h = matmul3(&matmul3(&inv_td, &h_norm), &ts);

    let mut out = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            if !h[i][j].is_finite() {
                return None;
            }
            out[i][j] = h[i][j] as f32;
        }
    }
    Some(out)
}

/// Applies homography `h` to point `(x, y)`, returning the normalized `(x', y')`.
pub fn apply_homography(h: &[[f32; 3]; 3], x: f32, y: f32) -> (f32, f32) {
    let (xp, yp) = apply_h_f64(
        &[
            [h[0][0] as f64, h[0][1] as f64, h[0][2] as f64],
            [h[1][0] as f64, h[1][1] as f64, h[1][2] as f64],
            [h[2][0] as f64, h[2][1] as f64, h[2][2] as f64],
        ],
        x as f64,
        y as f64,
    );
    (xp as f32, yp as f32)
}

/// Inverts a 3x3 homography; returns `None` if the matrix is singular.
pub fn invert_homography(h: &[[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let m = [
        [h[0][0] as f64, h[0][1] as f64, h[0][2] as f64],
        [h[1][0] as f64, h[1][1] as f64, h[1][2] as f64],
        [h[2][0] as f64, h[2][1] as f64, h[2][2] as f64],
    ];
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }

    // Cofactor inverse (transposed adjugate / det)
    let mut inv = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let rows = [0usize, 1, 2]
                .into_iter()
                .filter(|&r| r != i)
                .collect::<Vec<_>>();
            let cols = [0usize, 1, 2]
                .into_iter()
                .filter(|&c| c != j)
                .collect::<Vec<_>>();
            let minor = m[rows[0]][cols[0]] * m[rows[1]][cols[1]]
                - m[rows[0]][cols[1]] * m[rows[1]][cols[0]];
            inv[j][i] = if (i + j) % 2 == 0 { minor } else { -minor } / det;
        }
    }

    let mut out = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = inv[i][j] as f32;
        }
    }
    Some(out)
}

/// Hartley-normalizing transform: zero mean, RMS distance sqrt(2).
fn normalizing_transform<I>(pts: I) -> Option<([[f64; 3]; 3], [[f64; 3]; 3])>
where
    I: Iterator<Item = (f32, f32)>,
{
    let pts: Vec<(f64, f64)> = pts.map(|(x, y)| (x as f64, y as f64)).collect();
    if pts.len() < 4 {
        return None;
    }
    let n = pts.len() as f64;
    let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let mean_dist = pts
        .iter()
        .map(|p| ((p.0 - mx).powi(2) + (p.1 - my).powi(2)).sqrt())
        .sum::<f64>()
        / n;
    if mean_dist < 1e-12 {
        return None;
    }
    let s = (2.0f64).sqrt() / mean_dist;
    let t = [[s, 0.0, -s * mx], [0.0, s, -s * my], [0.0, 0.0, 1.0]];
    let inv_t = [[1.0 / s, 0.0, mx], [0.0, 1.0 / s, my], [0.0, 0.0, 1.0]];
    Some((t, inv_t))
}

/// Solves the 8x8 linear system `a * x = b` with Gaussian elimination and
/// partial pivoting. Returns `None` if the matrix is singular.
fn gauss_solve(a: &mut [[f64; 8]; 8], b: &mut [f64; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        let mut piv = col;
        let mut best = a[col][col].abs();
        for r in col + 1..8 {
            if a[r][col].abs() > best {
                best = a[r][col].abs();
                piv = r;
            }
        }
        if best < 1e-9 {
            return None;
        }
        if piv != col {
            a.swap(col, piv);
            b.swap(col, piv);
        }
        for r in col + 1..8 {
            let f = a[r][col] / a[col][col];
            if f == 0.0 {
                continue;
            }
            for c in col..8 {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }

    let mut x = [0.0f64; 8];
    for i in (0..8).rev() {
        let mut s = b[i];
        for j in i + 1..8 {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

fn matmul3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

fn apply_h_f64(h: &[[f64; 3]; 3], x: f64, y: f64) -> (f64, f64) {
    let w = h[2][0] * x + h[2][1] * y + h[2][2];
    let xp = (h[0][0] * x + h[0][1] * y + h[0][2]) / w;
    let yp = (h[1][0] * x + h[1][1] * y + h[1][2]) / w;
    (xp, yp)
}
