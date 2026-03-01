//! # Gaussian Parity Solver
//!
//! Replaces the Z3 SMT solver with deterministic O(n³) Gaussian elimination
//! over GF(3). The entire matrix (anchors + font + data + parity) forms one
//! valid RS codeword with zero syndrome.
//!
//! ## Algorithm
//!
//! 1. Map grid → flat trit stream (same as Z3Solver)
//! 2. Partition trits into **fixed** (message, font, length prefix) and **free**
//! 3. Build the trit-level parity-check system from the GF(3⁶) transform matrices
//! 4. Formulate `A·x = b` where:
//!    - `A` = columns of the parity system for free trits
//!    - `b` = −(columns for fixed trits · fixed values) mod 3
//! 5. Solve via Gaussian elimination
//! 6. Reconstruct the full codeword and render into the TritMatrix

use ternac_core::gf3::{self, GF3, TRITS_PER_SYMBOL, FIELD_ORDER};
use ternac_core::{LENGTH_PREFIX_TRITS, RS_HEADER_SYMBOLS, encode_length, TritMatrix};
use crate::{ConstraintMask, SolverError, anchor};
use crate::gauss::{Gf3Matrix, solve_gf3};

// ---------------------------------------------------------------------------
// Deterministic 2D→1D Mapping
// ---------------------------------------------------------------------------

/// Deterministic row-major mapping from grid to flat trit index.
///
/// Enumerates non-anchor cells in row-major order (y outer, x inner).
/// Returns `(x, y)` for each flat index. This mapping MUST be identical
/// between the encoder (solver) and the decoder (vision pipeline).
pub fn grid_to_flat_coords(matrix_size: usize) -> Vec<(usize, usize)> {
    let mut coords = Vec::new();
    for y in 0..matrix_size {
        for x in 0..matrix_size {
            if !anchor::is_in_anchor_region(x, y, matrix_size) {
                coords.push((x, y));
            }
        }
    }
    coords
}

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

/// Build the parity-check matrix H for an RS code.
///
/// H[i][j] = α^{(i+1)·j} for i ∈ [0, 2t), j ∈ [0, n).
/// A valid codeword c satisfies: ∀i: Σⱼ H[i][j]·c[j] = 0
pub fn build_parity_check_matrix(gf: &GF3, codeword_len: usize, parity_count: usize) -> Vec<Vec<u16>> {
    let mut h = Vec::with_capacity(parity_count);
    for i in 0..parity_count {
        let mut row = Vec::with_capacity(codeword_len);
        for j in 0..codeword_len {
            let exp = ((i + 1) * j) % FIELD_ORDER;
            row.push(gf.exp(exp));
        }
        h.push(row);
    }
    h
}

/// Gaussian elimination solver — pure Rust, no Z3, no external C++ deps.
pub struct GaussSolver;

impl GaussSolver {
    /// Fraction of symbols used for parity.
    const PARITY_FRACTION: f32 = 0.3;
}

impl super::MatrixSolver for GaussSolver {
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

        // --- Step 1: Map grid coordinates to flat trit indices ---
        let cell_coords = grid_to_flat_coords(n);

        // Compute layout sizes
        let trits_for_codeword = cell_coords.len().saturating_sub(LENGTH_PREFIX_TRITS);
        let num_symbols = trits_for_codeword / TRITS_PER_SYMBOL;
        let msg_symbols = RS_HEADER_SYMBOLS + ((payload_trits.len() + TRITS_PER_SYMBOL - 1) / TRITS_PER_SYMBOL);

        // Build constraint lookup for font
        let constraint_map: std::collections::HashMap<(usize, usize), u8> = constraints
            .iter()
            .map(|c| ((c.x, c.y), c.required_state))
            .collect();

        // --- Step 2: Find layout offset that avoids font conflicts with message ---
        let mut parity_count = (num_symbols as f32 * Self::PARITY_FRACTION).ceil() as usize;
        if parity_count % 2 != 0 {
            parity_count += 1;
        }
        let parity_count = parity_count.min(728).max(2);
        let parity_trits = parity_count * TRITS_PER_SYMBOL;
        let max_offset = num_symbols.saturating_sub(parity_count + msg_symbols);

        let mut best_offset = None;
        let mut final_font_conflicts = Vec::new();

        for offset in 0..=max_offset {
            let offset_trits = offset * TRITS_PER_SYMBOL;
            let locked_msg_start = parity_trits + offset_trits;
            let locked_msg_end = locked_msg_start + (msg_symbols * TRITS_PER_SYMBOL);

            let mut conflict = false;
            let mut potential_conflicts = Vec::new();

            for (flat_idx, &(x, y)) in cell_coords.iter().enumerate().take(LENGTH_PREFIX_TRITS + num_symbols * TRITS_PER_SYMBOL) {
                if let Some(&state) = constraint_map.get(&(x, y)) {
                    if flat_idx < LENGTH_PREFIX_TRITS {
                        conflict = true;
                        break;
                    }
                    let cw_idx = flat_idx - LENGTH_PREFIX_TRITS;
                    if cw_idx >= locked_msg_start && cw_idx < locked_msg_end {
                        conflict = true;
                        break;
                    }
                    potential_conflicts.push((cw_idx, state));
                }
            }

            if !conflict {
                best_offset = Some(offset);
                final_font_conflicts = potential_conflicts;
                break;
            }
        }

        let message_offset = best_offset.ok_or(SolverError::Unsatisfiable)?;
        let locked_msg_start = parity_trits + message_offset * TRITS_PER_SYMBOL;
        let locked_msg_end = locked_msg_start + msg_symbols * TRITS_PER_SYMBOL;
        let codeword_symbols = num_symbols;
        let codeword_trits = codeword_symbols * TRITS_PER_SYMBOL;

        // --- Step 3: Build the message symbols ---
        let original_len = payload_trits.len();
        let mut message = Vec::with_capacity(msg_symbols);
        message.push((original_len % 729) as u16);
        message.push((original_len / 729) as u16);
        message.push(parity_count as u16);

        let padded_len = ((original_len + TRITS_PER_SYMBOL - 1) / TRITS_PER_SYMBOL) * TRITS_PER_SYMBOL;
        let mut padded_payload = payload_trits.to_vec();
        padded_payload.resize(padded_len, 0);

        let data_symbols: Vec<u16> = padded_payload
            .chunks(TRITS_PER_SYMBOL)
            .map(|chunk| gf3::trits_to_symbol(chunk))
            .collect();
        message.extend_from_slice(&data_symbols);

        let mut locked_trits_values: Vec<u8> = Vec::with_capacity(msg_symbols * TRITS_PER_SYMBOL);
        for &sym in &message {
            locked_trits_values.extend_from_slice(&gf3::symbol_to_trits(sym));
        }

        let len_prefix = encode_length(codeword_trits);
        let font_conflicts = final_font_conflicts;

        // --- Step 4: Build the trit-level parity system ---
        // Each parity row × 6 trit positions = one scalar equation over GF(3).
        // Total equations: parity_count × 6.
        let h = build_parity_check_matrix(&gf, codeword_symbols, parity_count);
        let transforms: Vec<Vec<[[u64; 6]; 6]>> = h.iter().map(|row| {
            row.iter().map(|&h_ij| compute_transform_matrix(&gf, h_ij)).collect()
        }).collect();

        let num_equations = parity_count * 6;

        // Classify each codeword trit as fixed or free
        let mut is_fixed = vec![false; codeword_trits];
        let mut fixed_val = vec![0u8; codeword_trits];

        for idx in 0..codeword_trits {
            // Locked user message region
            if idx >= locked_msg_start && idx < locked_msg_end {
                is_fixed[idx] = true;
                fixed_val[idx] = locked_trits_values[idx - locked_msg_start];
            }
            // Locked by font constraint
            else if let Some(&(_cw_idx, state)) = font_conflicts.iter().find(|&&(cw_idx, _)| cw_idx == idx) {
                is_fixed[idx] = true;
                fixed_val[idx] = state;
            }
        }

        // Map free indices
        let free_indices: Vec<usize> = (0..codeword_trits).filter(|&i| !is_fixed[i]).collect();
        let num_free = free_indices.len();

        // Build the full parity coefficient matrix (num_equations × codeword_trits)
        // and simultaneously partition into A (free) and compute b (from fixed).
        let mut a_mat = Gf3Matrix::zeros(num_equations, num_free);
        let mut b_vec = vec![0u8; num_equations];

        let mut eq_idx = 0;
        for (_i, row_transforms) in transforms.iter().enumerate() {
            for trit_idx in 0..6 {
                // For this equation, accumulate coefficients
                let mut fixed_sum = 0u8;

                for (j, transform) in row_transforms.iter().enumerate() {
                    let sym_offset = j * TRITS_PER_SYMBOL;

                    for col in 0..6 {
                        let coeff = transform[trit_idx][col] as u8;
                        if coeff == 0 {
                            continue;
                        }
                        let flat_idx = sym_offset + col;

                        if is_fixed[flat_idx] {
                            // Accumulate into b: b += coeff * fixed_val
                            fixed_sum = (fixed_sum + coeff * fixed_val[flat_idx]) % 3;
                        } else {
                            // Find position of flat_idx in free_indices
                            // (binary search since free_indices is sorted)
                            let free_pos = free_indices.binary_search(&flat_idx).unwrap();
                            a_mat.set(eq_idx, free_pos, coeff);
                        }
                    }
                }

                // b = -fixed_sum (mod 3) = (3 - fixed_sum) % 3
                b_vec[eq_idx] = (3 - fixed_sum) % 3;
                eq_idx += 1;
            }
        }

        // --- Step 5: Solve A·x = b over GF(3) ---
        let solution = solve_gf3(&a_mat, &b_vec).ok_or(SolverError::Unsatisfiable)?;

        // --- Step 6: Reconstruct the full codeword ---
        let mut full_trits = vec![0u8; codeword_trits];

        // Place fixed values
        for idx in 0..codeword_trits {
            if is_fixed[idx] {
                full_trits[idx] = fixed_val[idx];
            }
        }

        // Place solved free values
        for (free_pos, &trit_idx) in free_indices.iter().enumerate() {
            full_trits[trit_idx] = solution[free_pos];
        }

        // --- Step 7: Build output matrix ---
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

        // Place length prefix
        for (i, &(x, y)) in cell_coords.iter().enumerate().take(LENGTH_PREFIX_TRITS) {
            matrix.set(x, y, len_prefix[i]);
        }

        // Place codeword trits
        for (i, &(x, y)) in cell_coords.iter().enumerate()
            .skip(LENGTH_PREFIX_TRITS)
            .take(codeword_trits)
        {
            let cw_idx = i - LENGTH_PREFIX_TRITS;
            matrix.set(x, y, full_trits[cw_idx]);
        }

        // Fill remaining cells with State 2 (white/light) to avoid black voids
        for &(x, y) in cell_coords.iter().skip(LENGTH_PREFIX_TRITS + codeword_trits) {
            matrix.set(x, y, 2);
        }

        Ok(matrix)
    }
}
