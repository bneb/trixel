//! # GF(3) Gaussian Elimination Solver
//!
//! Solves the system `A·x = b (mod 3)` for the free trit variables in a
//! Reed-Solomon parity-check equation. This replaces the Z3 SMT solver with
//! a deterministic O(n³) linear algebra engine — no external C++ dependency.
//!
//! ## Representation
//!
//! All values are in GF(3) = {0, 1, 2}. Arithmetic:
//! - Addition: `(a + b) % 3`
//! - Negation: `(3 - a) % 3` (i.e. 0→0, 1→2, 2→1)
//! - Multiplication: `(a * b) % 3`
//! - Inverse: `inv(1) = 1`, `inv(2) = 2` (both are self-inverse in GF(3))

/// A dense matrix over GF(3) stored in row-major order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gf3Matrix {
    pub rows: usize,
    pub cols: usize,
    /// Flat row-major storage. Each element is 0, 1, or 2.
    pub data: Vec<u8>,
}

impl Gf3Matrix {
    /// Create a zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0; rows * cols],
        }
    }

    /// Get element at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> u8 {
        self.data[row * self.cols + col]
    }

    /// Set element at (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: u8) {
        self.data[row * self.cols + col] = val % 3;
    }

    /// Swap two rows.
    pub fn swap_rows(&mut self, r1: usize, r2: usize) {
        if r1 == r2 {
            return;
        }
        let start1 = r1 * self.cols;
        let start2 = r2 * self.cols;
        for c in 0..self.cols {
            self.data.swap(start1 + c, start2 + c);
        }
    }
}

/// GF(3) arithmetic helpers.
#[inline]
fn gf3_add(a: u8, b: u8) -> u8 {
    (a + b) % 3
}

#[inline]
fn gf3_sub(a: u8, b: u8) -> u8 {
    (a + 3 - b) % 3
}

#[inline]
fn gf3_mul(a: u8, b: u8) -> u8 {
    (a * b) % 3
}

#[inline]
fn gf3_inv(a: u8) -> u8 {
    // In GF(3): inv(0) is undefined, inv(1)=1, inv(2)=2.
    debug_assert!(a == 1 || a == 2, "Cannot invert 0 in GF(3)");
    a // Both 1 and 2 are self-inverse in GF(3)
}

#[inline]
fn gf3_neg(a: u8) -> u8 {
    (3 - a) % 3
}

/// Solve `A·x = b` over GF(3) using Gaussian elimination with partial pivoting.
///
/// - `a`: An `m × n` coefficient matrix (m equations, n unknowns).
/// - `b`: A length-`m` right-hand-side vector.
///
/// Returns `Some(x)` where `x` has length `n`, or `None` if the system is
/// inconsistent (a pivot row has all-zero coefficients but nonzero b).
///
/// For under-determined systems (rank < n), free variables are set to 0.
pub fn solve_gf3(a: &Gf3Matrix, b: &[u8]) -> Option<Vec<u8>> {
    solve_gf3_with_default(a, b, 0)
}

/// Like `solve_gf3`, but free (unpivoted) variables are set to `default_free`
/// instead of 0. Since free variables are unconstrained by any equation,
/// any GF(3) value is valid — the pivot variables adjust through
/// back-substitution to maintain A·x = b.
pub fn solve_gf3_with_default(a: &Gf3Matrix, b: &[u8], default_free: u8) -> Option<Vec<u8>> {
    assert_eq!(a.rows, b.len(), "A.rows must equal b.len()");

    let m = a.rows;
    let n = a.cols;

    // Build augmented matrix [A | b] of size m × (n+1)
    let mut aug = Gf3Matrix::zeros(m, n + 1);
    for r in 0..m {
        for c in 0..n {
            aug.set(r, c, a.get(r, c));
        }
        aug.set(r, n, b[r]);
    }

    // Forward elimination with partial pivoting
    let mut pivot_col = 0;
    let mut pivot_row = 0;

    // Track which column each pivot row corresponds to
    let mut pivot_cols: Vec<Option<usize>> = vec![None; m];

    while pivot_row < m && pivot_col < n {
        // Find a non-zero entry in column pivot_col at or below pivot_row
        let mut found = None;
        for r in pivot_row..m {
            if aug.get(r, pivot_col) != 0 {
                found = Some(r);
                break;
            }
        }

        let Some(swap_row) = found else {
            // No pivot in this column — it's a free variable
            pivot_col += 1;
            continue;
        };

        // Swap into pivot position
        aug.swap_rows(pivot_row, swap_row);

        // Scale pivot row so the pivot element becomes 1
        let inv = gf3_inv(aug.get(pivot_row, pivot_col));
        for c in pivot_col..=n {
            let v = gf3_mul(aug.get(pivot_row, c), inv);
            aug.set(pivot_row, c, v);
        }

        // Eliminate all other rows in this column
        for r in 0..m {
            if r == pivot_row {
                continue;
            }
            let factor = aug.get(r, pivot_col);
            if factor != 0 {
                for c in pivot_col..=n {
                    let v = gf3_sub(aug.get(r, c), gf3_mul(factor, aug.get(pivot_row, c)));
                    aug.set(r, c, v);
                }
            }
        }

        pivot_cols[pivot_row] = Some(pivot_col);
        pivot_row += 1;
        pivot_col += 1;
    }

    // Check for inconsistency: any row with all-zero coefficients but nonzero b
    for r in 0..m {
        let all_zero = (0..n).all(|c| aug.get(r, c) == 0);
        if all_zero && aug.get(r, n) != 0 {
            return None; // Inconsistent system
        }
    }

    // Back-substitute: build solution vector
    let mut x = vec![default_free % 3; n];

    // Process pivot rows in reverse order
    for r in (0..m).rev() {
        let Some(pc) = pivot_cols[r] else { continue };

        let mut val = aug.get(r, n);
        for c in (pc + 1)..n {
            val = gf3_sub(val, gf3_mul(aug.get(r, c), x[c]));
        }
        x[pc] = val;
    }

    Some(x)
}

/// Solve `A·x = b` over GF(3) with per-variable target values.
///
/// Like `solve_gf3_with_default`, but each free variable gets its own
/// target value from `targets[i]`. Pivot variables are computed by
/// back-substitution to satisfy `A·x = b`.
///
/// `targets` must have length `n` (number of unknowns). For pivot
/// variables, the target is ignored (the equation determines them).
///
/// This is the core of Zhang's image-guided solver: each free variable
/// is set to the trit value closest to the source image's luminance at
/// that triangle's centroid, producing an aesthetically optimal codeword
/// that still satisfies all Reed-Solomon parity constraints.
pub fn solve_gf3_with_targets(a: &Gf3Matrix, b: &[u8], targets: &[u8]) -> Option<Vec<u8>> {
    assert_eq!(a.rows, b.len(), "A.rows must equal b.len()");
    let n = a.cols;
    assert_eq!(targets.len(), n, "targets.len() must equal A.cols");

    let m = a.rows;

    // Build augmented matrix [A | b] of size m × (n+1)
    let mut aug = Gf3Matrix::zeros(m, n + 1);
    for r in 0..m {
        for c in 0..n {
            aug.set(r, c, a.get(r, c));
        }
        aug.set(r, n, b[r]);
    }

    // Forward elimination with partial pivoting
    let mut pivot_col = 0;
    let mut pivot_row = 0;
    let mut pivot_cols: Vec<Option<usize>> = vec![None; m];

    while pivot_row < m && pivot_col < n {
        let mut found = None;
        for r in pivot_row..m {
            if aug.get(r, pivot_col) != 0 {
                found = Some(r);
                break;
            }
        }

        let Some(swap_row) = found else {
            pivot_col += 1;
            continue;
        };

        aug.swap_rows(pivot_row, swap_row);

        let inv = gf3_inv(aug.get(pivot_row, pivot_col));
        for c in pivot_col..=n {
            let v = gf3_mul(aug.get(pivot_row, c), inv);
            aug.set(pivot_row, c, v);
        }

        for r in 0..m {
            if r == pivot_row {
                continue;
            }
            let factor = aug.get(r, pivot_col);
            if factor != 0 {
                for c in pivot_col..=n {
                    let v = gf3_sub(aug.get(r, c), gf3_mul(factor, aug.get(pivot_row, c)));
                    aug.set(r, c, v);
                }
            }
        }

        pivot_cols[pivot_row] = Some(pivot_col);
        pivot_row += 1;
        pivot_col += 1;
    }

    // Check for inconsistency
    for r in 0..m {
        let all_zero = (0..n).all(|c| aug.get(r, c) == 0);
        if all_zero && aug.get(r, n) != 0 {
            return None;
        }
    }

    // Initialize solution with per-variable targets (free vars keep theirs)
    let mut x = Vec::with_capacity(n);
    for i in 0..n {
        x.push(targets[i] % 3);
    }

    // Back-substitute: pivot variables are recomputed to satisfy A·x = b
    for r in (0..m).rev() {
        let Some(pc) = pivot_cols[r] else { continue };

        let mut val = aug.get(r, n);
        for c in (pc + 1)..n {
            val = gf3_sub(val, gf3_mul(aug.get(r, c), x[c]));
        }
        x[pc] = val;
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // GF(3) Arithmetic
    // -----------------------------------------------------------------------

    #[test]
    fn gf3_add_table() {
        assert_eq!(gf3_add(0, 0), 0);
        assert_eq!(gf3_add(0, 1), 1);
        assert_eq!(gf3_add(0, 2), 2);
        assert_eq!(gf3_add(1, 1), 2);
        assert_eq!(gf3_add(1, 2), 0);
        assert_eq!(gf3_add(2, 2), 1);
    }

    #[test]
    fn gf3_sub_table() {
        assert_eq!(gf3_sub(0, 0), 0);
        assert_eq!(gf3_sub(1, 1), 0);
        assert_eq!(gf3_sub(2, 1), 1);
        assert_eq!(gf3_sub(0, 1), 2);
        assert_eq!(gf3_sub(1, 2), 2);
    }

    #[test]
    fn gf3_mul_table() {
        assert_eq!(gf3_mul(0, 0), 0);
        assert_eq!(gf3_mul(0, 1), 0);
        assert_eq!(gf3_mul(1, 1), 1);
        assert_eq!(gf3_mul(1, 2), 2);
        assert_eq!(gf3_mul(2, 2), 1);
    }

    #[test]
    fn gf3_neg_table() {
        assert_eq!(gf3_neg(0), 0);
        assert_eq!(gf3_neg(1), 2);
        assert_eq!(gf3_neg(2), 1);
    }

    #[test]
    fn gf3_inv_self_inverse() {
        assert_eq!(gf3_inv(1), 1);
        assert_eq!(gf3_inv(2), 2);
    }

    // -----------------------------------------------------------------------
    // Gaussian Elimination
    // -----------------------------------------------------------------------

    #[test]
    fn solve_identity_system() {
        // I · x = [1, 2, 0]  →  x = [1, 2, 0]
        let a = Gf3Matrix {
            rows: 3,
            cols: 3,
            data: vec![
                1, 0, 0,
                0, 1, 0,
                0, 0, 1,
            ],
        };
        let b = vec![1, 2, 0];
        let x = solve_gf3(&a, &b).unwrap();
        assert_eq!(x, vec![1, 2, 0]);
    }

    #[test]
    fn solve_simple_2x2() {
        // | 1 1 | · x = | 0 |
        // | 1 2 |       | 1 |
        //
        // Row 2 - Row 1: 0·x0 + 1·x1 = 1 → x1 = 1
        // Row 1: x0 + 1 = 0 (mod 3) → x0 = 2
        let a = Gf3Matrix {
            rows: 2,
            cols: 2,
            data: vec![1, 1, 1, 2],
        };
        let b = vec![0, 1];
        let x = solve_gf3(&a, &b).unwrap();
        assert_eq!(x, vec![2, 1]);

        // Verify: A·x = b
        for r in 0..2 {
            let mut sum = 0u8;
            for c in 0..2 {
                sum = gf3_add(sum, gf3_mul(a.get(r, c), x[c]));
            }
            assert_eq!(sum, b[r], "row {r}");
        }
    }

    #[test]
    fn solve_3x3_with_elimination() {
        // | 2 1 0 |       | 1 |
        // | 1 0 2 | · x = | 2 |
        // | 0 2 1 |       | 0 |
        let a = Gf3Matrix {
            rows: 3,
            cols: 3,
            data: vec![
                2, 1, 0,
                1, 0, 2,
                0, 2, 1,
            ],
        };
        let b = vec![1, 2, 0];
        let x = solve_gf3(&a, &b).unwrap();

        // Verify: A·x = b (mod 3)
        for r in 0..3 {
            let mut sum = 0u8;
            for c in 0..3 {
                sum = gf3_add(sum, gf3_mul(a.get(r, c), x[c]));
            }
            assert_eq!(sum, b[r], "row {r}: expected {}, got {sum}", b[r]);
        }
    }

    #[test]
    fn solve_underdetermined_system() {
        // 2 equations, 4 unknowns — under-determined
        // | 1 0 1 2 |       | 1 |
        // | 0 1 2 0 | · x = | 2 |
        //
        // Free variables x2, x3 default to 0 → x0=1, x1=2
        let a = Gf3Matrix {
            rows: 2,
            cols: 4,
            data: vec![
                1, 0, 1, 2,
                0, 1, 2, 0,
            ],
        };
        let b = vec![1, 2];
        let x = solve_gf3(&a, &b).unwrap();
        assert_eq!(x.len(), 4);

        // Verify: A·x = b
        for r in 0..2 {
            let mut sum = 0u8;
            for c in 0..4 {
                sum = gf3_add(sum, gf3_mul(a.get(r, c), x[c]));
            }
            assert_eq!(sum, b[r], "row {r}");
        }
    }

    #[test]
    fn solve_inconsistent_returns_none() {
        // | 1 0 |       | 1 |
        // | 0 0 | · x = | 2 |  ← inconsistent (0 = 2 is impossible)
        let a = Gf3Matrix {
            rows: 2,
            cols: 2,
            data: vec![1, 0, 0, 0],
        };
        let b = vec![1, 2];
        assert!(solve_gf3(&a, &b).is_none());
    }

    #[test]
    fn solve_requires_pivot_swap() {
        // | 0 1 |       | 2 |
        // | 1 0 | · x = | 1 |
        //
        // Needs row swap to find pivot in column 0
        // After swap: x0=1, x1=2
        let a = Gf3Matrix {
            rows: 2,
            cols: 2,
            data: vec![0, 1, 1, 0],
        };
        let b = vec![2, 1];
        let x = solve_gf3(&a, &b).unwrap();
        assert_eq!(x, vec![1, 2]);
    }

    #[test]
    fn solve_larger_system_6x6() {
        // Build a random-ish 6x6 system, verify A·x = b
        let a = Gf3Matrix {
            rows: 6,
            cols: 6,
            data: vec![
                1, 2, 0, 1, 0, 2,
                2, 1, 1, 0, 2, 0,
                0, 0, 2, 1, 1, 1,
                1, 0, 0, 2, 0, 1,
                0, 1, 0, 0, 1, 2,
                2, 0, 1, 1, 0, 0,
            ],
        };
        let b = vec![1, 0, 2, 1, 2, 0];
        let result = solve_gf3(&a, &b);

        if let Some(x) = result {
            for r in 0..6 {
                let mut sum = 0u8;
                for c in 0..6 {
                    sum = gf3_add(sum, gf3_mul(a.get(r, c), x[c]));
                }
                assert_eq!(sum, b[r], "row {r}");
            }
        }
        // If None, the system was inconsistent — that's also valid
    }

    #[test]
    fn solve_heavily_underdetermined() {
        // 2 equations, 100 unknowns — massive freedom
        // This should solve instantly
        let mut a = Gf3Matrix::zeros(2, 100);
        a.set(0, 0, 1);
        a.set(0, 50, 2);
        a.set(1, 1, 1);
        a.set(1, 99, 1);

        let b = vec![1, 2];
        let x = solve_gf3(&a, &b).unwrap();
        assert_eq!(x.len(), 100);

        // Verify
        for r in 0..2 {
            let mut sum = 0u8;
            for c in 0..100 {
                sum = gf3_add(sum, gf3_mul(a.get(r, c), x[c]));
            }
            assert_eq!(sum, b[r], "row {r}");
        }
    }

    // -----------------------------------------------------------------------
    // Per-Variable Targeted Solver
    // -----------------------------------------------------------------------

    #[test]
    fn solve_with_targets_satisfies_parity() {
        // 2 equations, 4 unknowns — under-determined
        let a = Gf3Matrix {
            rows: 2,
            cols: 4,
            data: vec![
                1, 0, 1, 2,
                0, 1, 2, 0,
            ],
        };
        let b = vec![1, 2];
        let targets = vec![2, 1, 0, 1]; // arbitrary per-variable targets

        let x = solve_gf3_with_targets(&a, &b, &targets).unwrap();
        assert_eq!(x.len(), 4);

        // Verify: A·x = b
        for r in 0..2 {
            let mut sum = 0u8;
            for c in 0..4 {
                sum = gf3_add(sum, gf3_mul(a.get(r, c), x[c]));
            }
            assert_eq!(sum, b[r], "row {r}: parity violated");
        }
    }

    #[test]
    fn solve_with_targets_free_vars_match_targets() {
        // Identity-like: 2 equations, 4 unknowns
        // Columns 0 and 1 have pivots; columns 2 and 3 are free
        let a = Gf3Matrix {
            rows: 2,
            cols: 4,
            data: vec![
                1, 0, 0, 0,  // x0 = b[0] - 0*x2 - 0*x3 = b[0]
                0, 1, 0, 0,  // x1 = b[1] - 0*x2 - 0*x3 = b[1]
            ],
        };
        let b = vec![1, 2];
        let targets = vec![0, 0, 2, 1]; // targets for free vars: x2=2, x3=1

        let x = solve_gf3_with_targets(&a, &b, &targets).unwrap();

        // x0 = 1 (pivot, determined by equation)
        // x1 = 2 (pivot, determined by equation)
        // x2 = 2 (free, matches target)
        // x3 = 1 (free, matches target)
        assert_eq!(x[0], 1, "pivot x0");
        assert_eq!(x[1], 2, "pivot x1");
        assert_eq!(x[2], 2, "free x2 should match target");
        assert_eq!(x[3], 1, "free x3 should match target");
    }

    #[test]
    fn solve_with_targets_large_underdetermined() {
        // 2 equations, 100 unknowns — massive freedom
        let mut a = Gf3Matrix::zeros(2, 100);
        a.set(0, 0, 1);
        a.set(0, 50, 2);
        a.set(1, 1, 1);
        a.set(1, 99, 1);

        let b = vec![1, 2];

        // Set all targets to 2 except positions 0, 1 (pivots), 50, 99 (coupled)
        let mut targets = vec![2u8; 100];
        targets[0] = 0;  // will be overridden by pivot
        targets[1] = 0;  // will be overridden by pivot

        let x = solve_gf3_with_targets(&a, &b, &targets).unwrap();
        assert_eq!(x.len(), 100);

        // Verify parity
        for r in 0..2 {
            let mut sum = 0u8;
            for c in 0..100 {
                sum = gf3_add(sum, gf3_mul(a.get(r, c), x[c]));
            }
            assert_eq!(sum, b[r], "row {r}");
        }

        // Count how many free variables actually got their target (2)
        // (excluding the pivots at 0, 1 and coupled vars at 50, 99)
        let target_matches: usize = (2..100)
            .filter(|&i| i != 50 && i != 99)
            .filter(|&i| x[i] == 2)
            .count();
        // Should be the vast majority
        assert!(target_matches >= 90,
            "Expected most free vars to match target=2, got {target_matches}/96");
    }
}
