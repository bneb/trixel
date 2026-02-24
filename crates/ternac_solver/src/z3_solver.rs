//! # Z3 SMT Parity Solver
//!
//! Solves for free trit values such that the entire matrix forms a valid
//! Reed-Solomon codeword with zero syndrome: H·c^T = 0.
//!
//! ## Linear Transformation Breakthrough
//!
//! GF(3⁶) multiplication by a known constant k is a linear transformation.
//! We pre-compute a 6×6 base-3 matrix M(k) for each parity-check constant,
//! reducing Z3 assertions to pure Linear Integer Arithmetic mod 3.

use ternac_core::gf3::{self, GF3, TRITS_PER_SYMBOL};
use ternac_core::TritMatrix;
use crate::{ConstraintMask, SolverError, anchor};
use z3::ast::{Ast, Int};
use z3::{Config, Context, SatResult, Solver};

// ---------------------------------------------------------------------------
// Linear Transform
// ---------------------------------------------------------------------------

/// Compute the 6×6 base-3 transformation matrix for multiplying by constant k
/// in GF(3⁶). M[row][col] gives the coefficient (0, 1, or 2).
///
/// For a symbol b with trits [t₀..t₅], the product k·b has trits:
///   result_trit[row] = Σ(M[row][col] * b_trit[col]) mod 3
pub fn compute_transform_matrix(gf: &GF3, k: u16) -> [[u64; 6]; 6] {
    let mut m = [[0u64; 6]; 6];
    // Basis vectors: 3⁰=1, 3¹=3, 3²=9, 3³=27, 3⁴=81, 3⁵=243
    let basis: [u16; 6] = [1, 3, 9, 27, 81, 243];

    for col in 0..6 {
        let product = gf.mul(k, basis[col]);
        let trits = gf3::symbol_to_trits(product);
        for row in 0..6 {
            m[row][col] = trits[row] as u64;
        }
    }
    m
}

// ---------------------------------------------------------------------------
// Parity-Check Matrix
// ---------------------------------------------------------------------------

/// Build the parity-check matrix H for an (n, n-2t) RS code.
///
/// H[i][j] = α^{(i+1)·j} for i ∈ [0, 2t), j ∈ [0, n).
///
/// A valid codeword c satisfies: ∀i: Σⱼ H[i][j]·c[j] = 0
pub fn build_parity_check_matrix(gf: &GF3, codeword_len: usize, parity_count: usize) -> Vec<Vec<u16>> {
    let mut h = Vec::with_capacity(parity_count);
    for i in 0..parity_count {
        let mut row = Vec::with_capacity(codeword_len);
        for j in 0..codeword_len {
            // α^{(i+1)*j}
            let exp = ((i + 1) * j) % gf3::FIELD_ORDER;
            row.push(gf.exp(exp));
        }
        h.push(row);
    }
    h
}

// ---------------------------------------------------------------------------
// Z3 SMT Solver
// ---------------------------------------------------------------------------

/// Z3-based matrix solver that enforces RS parity-check constraints.
pub struct Z3Solver;

impl Z3Solver {
    /// Fraction of symbols used for parity.
    const PARITY_FRACTION: f32 = 0.3;
}

impl super::MatrixSolver for Z3Solver {
    fn resolve_matrix(
        payload_trits: &[u8],
        matrix_size: usize,
        constraints: &[ConstraintMask],
    ) -> Result<TritMatrix, SolverError> {
        let n = matrix_size;
        if n < anchor::ANCHOR_SIZE * 2 {
            return Err(SolverError::MatrixTooSmall { size: n, trits: payload_trits.len() });
        }

        // Validate: no constraint overlaps anchor
        for c in constraints {
            if anchor::is_in_anchor_region(c.x, c.y, n) {
                return Err(SolverError::Conflict { x: c.x, y: c.y });
            }
        }

        let gf = GF3::new();

        // Build the mapping: (x,y) → flat index for non-anchor cells
        let mut cell_coords: Vec<(usize, usize)> = Vec::new();
        for y in 0..n {
            for x in 0..n {
                if !anchor::is_in_anchor_region(x, y, n) {
                    cell_coords.push((x, y));
                }
            }
        }
        let total_trits = cell_coords.len();

        // Round total_trits down to multiple of 6 for symbol grouping
        let usable_trits = (total_trits / TRITS_PER_SYMBOL) * TRITS_PER_SYMBOL;
        let num_symbols = usable_trits / TRITS_PER_SYMBOL;

        // Parity budget
        let parity_count = (num_symbols as f32 * Self::PARITY_FRACTION).ceil() as usize;
        let data_symbols = num_symbols - parity_count;
        let data_trits_capacity = data_symbols * TRITS_PER_SYMBOL;

        if payload_trits.len() > data_trits_capacity {
            return Err(SolverError::MatrixTooSmall { size: n, trits: payload_trits.len() });
        }

        // Build constraint lookup
        let constraint_map: std::collections::HashMap<(usize, usize), u8> = constraints
            .iter()
            .map(|c| ((c.x, c.y), c.required_state))
            .collect();

        // Classify each trit position:
        // - Fixed by payload data
        // - Fixed by font constraint
        // - Free (Z3 decides)
        let mut trit_values: Vec<Option<u8>> = vec![None; usable_trits];

        // Place payload in the data region (after parity symbols)
        let parity_trits = parity_count * TRITS_PER_SYMBOL;
        for (i, &t) in payload_trits.iter().enumerate() {
            trit_values[parity_trits + i] = Some(t);
        }
        // Zero-fill remaining data region
        for i in (parity_trits + payload_trits.len())..usable_trits {
            trit_values[i] = Some(0);
        }

        // Override with font constraints — map (x,y) to flat index
        for (flat_idx, &(x, y)) in cell_coords.iter().enumerate().take(usable_trits) {
            if let Some(&state) = constraint_map.get(&(x, y)) {
                trit_values[flat_idx] = Some(state);
            }
        }

        // --- Z3 Solve ---
        let cfg = Config::new();
        let ctx = Context::new(&cfg);
        let solver = Solver::new(&ctx);

        let zero = Int::from_u64(&ctx, 0);
        let two = Int::from_u64(&ctx, 2);
        let three = Int::from_u64(&ctx, 3);

        // Create Z3 variables for each trit
        let trit_vars: Vec<Int> = (0..usable_trits)
            .map(|i| Int::new_const(&ctx, format!("t{i}")))
            .collect();

        // Bound each trit: 0 <= t <= 2
        for tv in &trit_vars {
            solver.assert(&tv.ge(&zero));
            solver.assert(&tv.le(&two));
        }

        // Fix known trits
        for (i, val) in trit_values.iter().enumerate() {
            if let Some(v) = val {
                solver.assert(&trit_vars[i]._eq(&Int::from_u64(&ctx, *v as u64)));
            }
        }

        // Build the parity-check matrix H
        let h = build_parity_check_matrix(&gf, num_symbols, parity_count);

        // Pre-compute transform matrices for all H constants
        let transforms: Vec<Vec<[[u64; 6]; 6]>> = h.iter().map(|row| {
            row.iter().map(|&h_ij| compute_transform_matrix(&gf, h_ij)).collect()
        }).collect();

        // Assert parity-check: for each row of H, for each trit position,
        // Σ(M[trit_idx][col] * symbol_trit[col]) % 3 == 0
        for (i, row_transforms) in transforms.iter().enumerate() {
            for trit_idx in 0..6 {
                let mut sum_terms: Vec<Int> = Vec::new();

                for (j, transform) in row_transforms.iter().enumerate() {
                    let sym_offset = j * TRITS_PER_SYMBOL;

                    for col in 0..6 {
                        let coeff = transform[trit_idx][col];
                        if coeff == 0 {
                            continue;
                        }
                        let trit_var = &trit_vars[sym_offset + col];
                        if coeff == 1 {
                            sum_terms.push(trit_var.clone());
                        } else {
                            let coeff_ast = Int::from_u64(&ctx, coeff);
                            sum_terms.push(Int::mul(&ctx, &[&coeff_ast, trit_var]));
                        }
                    }
                }

                if sum_terms.is_empty() {
                    continue;
                }

                let refs: Vec<&Int> = sum_terms.iter().collect();
                let sum = Int::add(&ctx, &refs);
                let rem = sum.rem(&three);
                solver.assert(&rem._eq(&zero));
            }
        }

        // Solve
        match solver.check() {
            SatResult::Sat => {
                let model = solver.get_model().unwrap();
                let mut matrix = TritMatrix::zeros(n, n);

                // Place anchors
                for &(cx, cy, pi) in &anchor::corner_positions(n) {
                    let pattern = &anchor::ANCHOR_PATTERNS[pi];
                    for dy in 0..anchor::ANCHOR_SIZE {
                        for dx in 0..anchor::ANCHOR_SIZE {
                            matrix.set(cx + dx, cy + dy, pattern[dy][dx]);
                        }
                    }
                }

                // Extract solved trit values
                for (flat_idx, &(x, y)) in cell_coords.iter().enumerate().take(usable_trits) {
                    let val = model.eval(&trit_vars[flat_idx], true).unwrap();
                    let v = val.as_u64().unwrap() as u8;
                    matrix.set(x, y, v);
                }

                // Fill any remaining non-symbol trits with 0
                for &(x, y) in cell_coords.iter().skip(usable_trits) {
                    matrix.set(x, y, 0);
                }

                Ok(matrix)
            }
            _ => Err(SolverError::Unsatisfiable),
        }
    }
}
