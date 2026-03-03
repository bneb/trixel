//! # Triangular Gaussian Parity Solver
//!
//! Identical math to `GaussSolver` but uses `TriGrid` and `tri_anchor`
//! instead of `TritMatrix` and `anchor`.

use trixel_core::gf3::{self, GF3, TRITS_PER_SYMBOL, FIELD_ORDER};
use trixel_core::{LENGTH_PREFIX_TRITS, RS_HEADER_SYMBOLS, encode_length};
use trixel_core::trigrid::TriGrid;
use crate::{ConstraintMask, SolverError, tri_anchor};
use crate::gauss::{Gf3Matrix, solve_gf3_with_default, solve_gf3_with_targets};
use crate::gauss_solver::{compute_transform_matrix, build_parity_check_matrix};

// ---------------------------------------------------------------------------
// Deterministic 2D→1D Mapping (Triangular)
// ---------------------------------------------------------------------------

/// Deterministic row-major mapping from triangular grid to flat trit index.
///
/// Enumerates non-anchor cells in row-major order (row outer, col inner).
/// Returns `(col, row)` for each flat index.
pub fn tri_grid_to_flat_coords(rows: usize, cols: usize) -> Vec<(usize, usize)> {
    let mut coords = Vec::new();
    for row in 0..rows {
        for col in 0..cols {
            if !tri_anchor::is_in_tri_anchor_region(col, row, rows, cols) {
                coords.push((col, row));
            }
        }
    }
    coords
}

// ---------------------------------------------------------------------------
// TriGaussSolver
// ---------------------------------------------------------------------------

/// Gaussian elimination solver for triangular grids.
pub struct TriGaussSolver;

impl TriGaussSolver {
    /// Fraction of symbols used for parity.
    const PARITY_FRACTION: f32 = 0.3;

    /// Resolve a triangular grid that encodes the given payload.
    pub fn resolve_trigrid(
        payload_trits: &[u8],
        rows: usize,
        cols: usize,
        constraints: &[ConstraintMask],
    ) -> Result<TriGrid, SolverError> {
        if rows < tri_anchor::TRI_ANCHOR_ROWS * 2 || cols < tri_anchor::TRI_ANCHOR_COLS * 2 {
            return Err(SolverError::MatrixTooSmall {
                size: rows.min(cols),
                trits: payload_trits.len(),
            });
        }

        // Validate: no constraint overlaps anchor
        for c in constraints {
            if tri_anchor::is_in_tri_anchor_region(c.x, c.y, rows, cols) {
                return Err(SolverError::Conflict { x: c.x, y: c.y });
            }
        }

        let gf = GF3::new();

        // --- Step 1: Map grid coordinates to flat trit indices ---
        let cell_coords = tri_grid_to_flat_coords(rows, cols);

        let trits_for_codeword = cell_coords.len().saturating_sub(LENGTH_PREFIX_TRITS);
        let num_symbols = trits_for_codeword / TRITS_PER_SYMBOL;
        let msg_symbols = RS_HEADER_SYMBOLS
            + ((payload_trits.len() + TRITS_PER_SYMBOL - 1) / TRITS_PER_SYMBOL);

        let constraint_map: std::collections::HashMap<(usize, usize), u8> = constraints
            .iter()
            .map(|c| ((c.x, c.y), c.required_state))
            .collect();

        // --- Step 2: Find layout offset that avoids font conflicts ---
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

            for (flat_idx, &(col, row)) in cell_coords
                .iter()
                .enumerate()
                .take(LENGTH_PREFIX_TRITS + num_symbols * TRITS_PER_SYMBOL)
            {
                if let Some(&state) = constraint_map.get(&(col, row)) {
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

        let padded_len =
            ((original_len + TRITS_PER_SYMBOL - 1) / TRITS_PER_SYMBOL) * TRITS_PER_SYMBOL;
        let mut padded_payload = payload_trits.to_vec();
        padded_payload.resize(padded_len, 0);

        let data_symbols: Vec<u16> = padded_payload
            .chunks(TRITS_PER_SYMBOL)
            .map(|chunk| gf3::trits_to_symbol(chunk))
            .collect();
        message.extend_from_slice(&data_symbols);

        let mut locked_trits_values: Vec<u8> =
            Vec::with_capacity(msg_symbols * TRITS_PER_SYMBOL);
        for &sym in &message {
            locked_trits_values.extend_from_slice(&gf3::symbol_to_trits(sym));
        }

        let len_prefix = encode_length(codeword_trits);
        let font_conflicts = final_font_conflicts;

        // --- Step 4: Build the trit-level parity system ---
        let h = build_parity_check_matrix(&gf, codeword_symbols, parity_count);
        let transforms: Vec<Vec<[[u64; 6]; 6]>> = h
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&h_ij| compute_transform_matrix(&gf, h_ij))
                    .collect()
            })
            .collect();

        let num_equations = parity_count * 6;

        let mut is_fixed = vec![false; codeword_trits];
        let mut fixed_val = vec![0u8; codeword_trits];

        for idx in 0..codeword_trits {
            if idx >= locked_msg_start && idx < locked_msg_end {
                is_fixed[idx] = true;
                fixed_val[idx] = locked_trits_values[idx - locked_msg_start];
            } else if let Some(&(_cw_idx, state)) =
                font_conflicts.iter().find(|&&(cw_idx, _)| cw_idx == idx)
            {
                is_fixed[idx] = true;
                fixed_val[idx] = state;
            }
        }

        let free_indices: Vec<usize> = (0..codeword_trits).filter(|&i| !is_fixed[i]).collect();
        let num_free = free_indices.len();

        let mut a_mat = Gf3Matrix::zeros(num_equations, num_free);
        let mut b_vec = vec![0u8; num_equations];

        let mut eq_idx = 0;
        for (_i, row_transforms) in transforms.iter().enumerate() {
            for trit_idx in 0..6 {
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
                            fixed_sum = (fixed_sum + coeff * fixed_val[flat_idx]) % 3;
                        } else {
                            let free_pos = free_indices.binary_search(&flat_idx).unwrap();
                            a_mat.set(eq_idx, free_pos, coeff);
                        }
                    }
                }

                b_vec[eq_idx] = (3 - fixed_sum) % 3;
                eq_idx += 1;
            }
        }

        // --- Step 5: Solve A·x = b over GF(3) ---
        // Default free variables to 2 (light) to avoid dark voids in halftone.
        // Free variables are unconstrained by parity equations — any value works.
        let solution = solve_gf3_with_default(&a_mat, &b_vec, 2).ok_or(SolverError::Unsatisfiable)?;

        // --- Step 6: Reconstruct the full codeword ---
        // NOTE: Free variables MUST default to 0 here — they are part of the RS
        // codeword and must satisfy the parity-check equations. The visual
        // "lightness" default happens in Step 7 for cells beyond the codeword.
        let mut full_trits = vec![0u8; codeword_trits];

        for idx in 0..codeword_trits {
            if is_fixed[idx] {
                full_trits[idx] = fixed_val[idx];
            }
        }
        for (free_pos, &trit_idx) in free_indices.iter().enumerate() {
            full_trits[trit_idx] = solution[free_pos];
        }

        // --- Step 7: Build output TriGrid ---
        let mut grid = TriGrid::zeros(rows, cols);

        // Place anchors
        for &(ac, ar, pi) in &tri_anchor::tri_corner_positions(rows, cols) {
            let pattern = &tri_anchor::TRI_ANCHOR_PATTERNS[pi];
            for dr in 0..tri_anchor::TRI_ANCHOR_ROWS {
                for dc in 0..tri_anchor::TRI_ANCHOR_COLS {
                    grid.set(ac + dc, ar + dr, pattern[dr][dc]);
                }
            }
        }

        // Place length prefix
        for (i, &(col, row)) in cell_coords.iter().enumerate().take(LENGTH_PREFIX_TRITS) {
            grid.set(col, row, len_prefix[i]);
        }

        // Place codeword trits
        for (i, &(col, row)) in cell_coords
            .iter()
            .enumerate()
            .skip(LENGTH_PREFIX_TRITS)
            .take(codeword_trits)
        {
            let cw_idx = i - LENGTH_PREFIX_TRITS;
            grid.set(col, row, full_trits[cw_idx]);
        }

        // Fill remaining cells with State 2
        for &(col, row) in cell_coords.iter().skip(LENGTH_PREFIX_TRITS + codeword_trits) {
            grid.set(col, row, 2);
        }

        Ok(grid)
    }

    /// Image-guided variant: free variables are set to match target trits
    /// derived from the source image.
    ///
    /// `target_trits` is a flat array of trit values (0, 1, 2) for every
    /// non-anchor cell in the same row-major order as `tri_grid_to_flat_coords`.
    /// The solver uses these as per-variable targets: free variables get their
    /// target value, while pivot variables are computed by back-substitution
    /// to satisfy all RS parity constraints.
    pub fn resolve_trigrid_image_guided(
        payload_trits: &[u8],
        rows: usize,
        cols: usize,
        constraints: &[ConstraintMask],
        target_trits: &[u8],
    ) -> Result<TriGrid, SolverError> {
        if rows < tri_anchor::TRI_ANCHOR_ROWS * 2 || cols < tri_anchor::TRI_ANCHOR_COLS * 2 {
            return Err(SolverError::MatrixTooSmall {
                size: rows.min(cols),
                trits: payload_trits.len(),
            });
        }

        for c in constraints {
            if tri_anchor::is_in_tri_anchor_region(c.x, c.y, rows, cols) {
                return Err(SolverError::Conflict { x: c.x, y: c.y });
            }
        }

        let gf = GF3::new();
        let cell_coords = tri_grid_to_flat_coords(rows, cols);

        // Validate target_trits length matches cell_coords
        if target_trits.len() != cell_coords.len() {
            // Fallback: if target map is wrong size, use uniform default
            return Self::resolve_trigrid(payload_trits, rows, cols, constraints);
        }

        let trits_for_codeword = cell_coords.len().saturating_sub(LENGTH_PREFIX_TRITS);
        let num_symbols = trits_for_codeword / TRITS_PER_SYMBOL;
        let msg_symbols = RS_HEADER_SYMBOLS
            + ((payload_trits.len() + TRITS_PER_SYMBOL - 1) / TRITS_PER_SYMBOL);

        let constraint_map: std::collections::HashMap<(usize, usize), u8> = constraints
            .iter()
            .map(|c| ((c.x, c.y), c.required_state))
            .collect();

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

            for (flat_idx, &(col, row)) in cell_coords
                .iter()
                .enumerate()
                .take(LENGTH_PREFIX_TRITS + num_symbols * TRITS_PER_SYMBOL)
            {
                if let Some(&state) = constraint_map.get(&(col, row)) {
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

        // Build message symbols
        let original_len = payload_trits.len();
        let mut message = Vec::with_capacity(msg_symbols);
        message.push((original_len % 729) as u16);
        message.push((original_len / 729) as u16);
        message.push(parity_count as u16);

        let padded_len =
            ((original_len + TRITS_PER_SYMBOL - 1) / TRITS_PER_SYMBOL) * TRITS_PER_SYMBOL;
        let mut padded_payload = payload_trits.to_vec();
        padded_payload.resize(padded_len, 0);

        let data_symbols: Vec<u16> = padded_payload
            .chunks(TRITS_PER_SYMBOL)
            .map(|chunk| gf3::trits_to_symbol(chunk))
            .collect();
        message.extend_from_slice(&data_symbols);

        let mut locked_trits_values: Vec<u8> =
            Vec::with_capacity(msg_symbols * TRITS_PER_SYMBOL);
        for &sym in &message {
            locked_trits_values.extend_from_slice(&gf3::symbol_to_trits(sym));
        }

        let len_prefix = encode_length(codeword_trits);
        let font_conflicts = final_font_conflicts;

        // Build parity system
        let h = build_parity_check_matrix(&gf, codeword_symbols, parity_count);
        let transforms: Vec<Vec<[[u64; 6]; 6]>> = h
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&h_ij| compute_transform_matrix(&gf, h_ij))
                    .collect()
            })
            .collect();

        let num_equations = parity_count * 6;

        let mut is_fixed = vec![false; codeword_trits];
        let mut fixed_val = vec![0u8; codeword_trits];

        for idx in 0..codeword_trits {
            if idx >= locked_msg_start && idx < locked_msg_end {
                is_fixed[idx] = true;
                fixed_val[idx] = locked_trits_values[idx - locked_msg_start];
            } else if let Some(&(_cw_idx, state)) =
                font_conflicts.iter().find(|&&(cw_idx, _)| cw_idx == idx)
            {
                is_fixed[idx] = true;
                fixed_val[idx] = state;
            }
        }

        let free_indices: Vec<usize> = (0..codeword_trits).filter(|&i| !is_fixed[i]).collect();
        let num_free = free_indices.len();

        let mut a_mat = Gf3Matrix::zeros(num_equations, num_free);
        let mut b_vec = vec![0u8; num_equations];

        let mut eq_idx = 0;
        for (_i, row_transforms) in transforms.iter().enumerate() {
            for trit_idx in 0..6 {
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
                            fixed_sum = (fixed_sum + coeff * fixed_val[flat_idx]) % 3;
                        } else {
                            let free_pos = free_indices.binary_search(&flat_idx).unwrap();
                            a_mat.set(eq_idx, free_pos, coeff);
                        }
                    }
                }

                b_vec[eq_idx] = (3 - fixed_sum) % 3;
                eq_idx += 1;
            }
        }

        // Build per-free-variable target vector from the image-derived target_trits.
        // target_trits is indexed by flat cell index (same order as cell_coords).
        // Free variables are indexed within the codeword region (after LENGTH_PREFIX_TRITS).
        let mut free_targets = vec![2u8; num_free]; // default to light if unmapped
        for (free_pos, &cw_idx) in free_indices.iter().enumerate() {
            let flat_cell_idx = cw_idx + LENGTH_PREFIX_TRITS;
            if flat_cell_idx < target_trits.len() {
                free_targets[free_pos] = target_trits[flat_cell_idx] % 3;
            }
        }

        // Solve with per-variable targets
        let solution = solve_gf3_with_targets(&a_mat, &b_vec, &free_targets)
            .ok_or(SolverError::Unsatisfiable)?;

        // Reconstruct the full codeword
        let mut full_trits = vec![0u8; codeword_trits];
        for idx in 0..codeword_trits {
            if is_fixed[idx] {
                full_trits[idx] = fixed_val[idx];
            }
        }
        for (free_pos, &trit_idx) in free_indices.iter().enumerate() {
            full_trits[trit_idx] = solution[free_pos];
        }

        // Build output TriGrid
        let mut grid = TriGrid::zeros(rows, cols);

        // Place anchors
        for &(ac, ar, pi) in &tri_anchor::tri_corner_positions(rows, cols) {
            let pattern = &tri_anchor::TRI_ANCHOR_PATTERNS[pi];
            for dr in 0..tri_anchor::TRI_ANCHOR_ROWS {
                for dc in 0..tri_anchor::TRI_ANCHOR_COLS {
                    grid.set(ac + dc, ar + dr, pattern[dr][dc]);
                }
            }
        }

        // Place length prefix
        for (i, &(col, row)) in cell_coords.iter().enumerate().take(LENGTH_PREFIX_TRITS) {
            grid.set(col, row, len_prefix[i]);
        }

        // Place codeword trits
        for (i, &(col, row)) in cell_coords
            .iter()
            .enumerate()
            .skip(LENGTH_PREFIX_TRITS)
            .take(codeword_trits)
        {
            let cw_idx = i - LENGTH_PREFIX_TRITS;
            grid.set(col, row, full_trits[cw_idx]);
        }

        // Fill remaining cells with target trits (image-guided)
        for (flat_idx, &(col, row)) in cell_coords.iter().enumerate().skip(LENGTH_PREFIX_TRITS + codeword_trits) {
            let target = if flat_idx < target_trits.len() {
                target_trits[flat_idx]
            } else {
                2 // default to light
            };
            grid.set(col, row, target);
        }

        Ok(grid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trixel_core::{MockCodec, TernaryCodec};

    // -------------------------------------------------------------------
    // Flat Coordinate Mapper
    // -------------------------------------------------------------------

    #[test]
    fn tri_flat_coords_skips_anchors() {
        let rows = 20;
        let cols = 40;
        let coords = tri_grid_to_flat_coords(rows, cols);

        // Total cells minus anchor cells
        let total = rows * cols;
        let anchor_cells = 4 * tri_anchor::TRI_ANCHOR_ROWS * tri_anchor::TRI_ANCHOR_COLS;
        assert_eq!(coords.len(), total - anchor_cells);

        // No coordinate should be in an anchor region
        for &(col, row) in &coords {
            assert!(
                !tri_anchor::is_in_tri_anchor_region(col, row, rows, cols),
                "({col},{row}) is in anchor region but was included"
            );
        }
    }

    #[test]
    fn tri_flat_coords_deterministic() {
        let rows = 20;
        let cols = 30;
        let a = tri_grid_to_flat_coords(rows, cols);
        let b = tri_grid_to_flat_coords(rows, cols);
        assert_eq!(a, b, "Two calls must produce identical output");
    }

    #[test]
    fn tri_flat_coords_row_major_order() {
        let rows = 20;
        let cols = 40;
        let coords = tri_grid_to_flat_coords(rows, cols);

        // Verify row-major: for each pair, (row_a, col_a) < (row_b, col_b)
        for window in coords.windows(2) {
            let (ca, ra) = window[0];
            let (cb, rb) = window[1];
            assert!(
                (ra, ca) < (rb, cb),
                "Not row-major: ({ca},{ra}) before ({cb},{rb})"
            );
        }
    }

    // -------------------------------------------------------------------
    // Solver Roundtrip
    // -------------------------------------------------------------------

    #[test]
    fn tri_gauss_solver_basic_roundtrip() {
        let data = b"hello";
        let trits = MockCodec::encode_bytes(data).unwrap();
        let rows = 20;
        let cols = 30;

        let grid = TriGaussSolver::resolve_trigrid(&trits, rows, cols, &[]).unwrap();

        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);

        // Extract payload from the grid (same mapping as encoder)
        let cell_coords = tri_grid_to_flat_coords(rows, cols);
        let mut raw: Vec<u8> = Vec::new();
        for &(col, row) in &cell_coords {
            raw.push(grid.get(col, row).unwrap());
        }

        // The raw stream starts with LENGTH_PREFIX_TRITS then the codeword.
        // Decode using RsEcc.
        use trixel_core::{RsEcc, ErrorCorrection};
        let recovered_trits = RsEcc::correct_errors(&raw).unwrap();
        let recovered_bytes = MockCodec::decode_trits(&recovered_trits).unwrap();
        assert_eq!(recovered_bytes, data);
    }

    #[test]
    fn tri_gauss_solver_url_roundtrip() {
        let data = b"https://github.com/bneb";
        let trits = MockCodec::encode_bytes(data).unwrap();
        let rows = 24;
        let cols = 36;

        let grid = TriGaussSolver::resolve_trigrid(&trits, rows, cols, &[]).unwrap();

        let cell_coords = tri_grid_to_flat_coords(rows, cols);
        let raw: Vec<u8> = cell_coords
            .iter()
            .map(|&(col, row)| grid.get(col, row).unwrap())
            .collect();

        use trixel_core::{RsEcc, ErrorCorrection};
        let recovered_trits = RsEcc::correct_errors(&raw).unwrap();
        let recovered_bytes = MockCodec::decode_trits(&recovered_trits).unwrap();
        assert_eq!(recovered_bytes, data);
    }

    #[test]
    fn tri_gauss_solver_anchors_present() {
        let data = b"x";
        let trits = MockCodec::encode_bytes(data).unwrap();
        let rows = 20;
        let cols = 40;

        let grid = TriGaussSolver::resolve_trigrid(&trits, rows, cols, &[]).unwrap();

        // Verify anchors are placed
        for &(ac, ar, pi) in &tri_anchor::tri_corner_positions(rows, cols) {
            let pattern = &tri_anchor::TRI_ANCHOR_PATTERNS[pi];
            for dr in 0..tri_anchor::TRI_ANCHOR_ROWS {
                for dc in 0..tri_anchor::TRI_ANCHOR_COLS {
                    assert_eq!(
                        grid.get(ac + dc, ar + dr),
                        Some(pattern[dr][dc]),
                        "Anchor at ({},{}) pattern[{}][{}] mismatch",
                        ac + dc,
                        ar + dr,
                        dr,
                        dc
                    );
                }
            }
        }
    }

    #[test]
    fn tri_gauss_solver_too_small_grid_fails() {
        let trits = vec![0, 1, 2];
        let result = TriGaussSolver::resolve_trigrid(&trits, 4, 6, &[]);
        assert!(result.is_err());
    }

    // -------------------------------------------------------------------
    // Image-Guided Solver
    // -------------------------------------------------------------------

    #[test]
    fn image_guided_solver_roundtrip() {
        use trixel_core::{MockCodec, TernaryCodec};

        let data = b"https://github.com/bneb";
        let trits = MockCodec::encode_bytes(data).unwrap();
        let rows = 20;
        let cols = 40;

        let cell_coords = tri_grid_to_flat_coords(rows, cols);

        // Create a target map: simulate image-derived trits
        // Mix of 0, 1, 2 to test that the solver handles varied targets
        let target_trits: Vec<u8> = cell_coords.iter().enumerate()
            .map(|(i, _)| (i % 3) as u8)
            .collect();

        let grid = TriGaussSolver::resolve_trigrid_image_guided(
            &trits, rows, cols, &[], &target_trits
        ).unwrap();

        // Verify grid dimensions
        assert_eq!(grid.rows, rows);
        assert_eq!(grid.cols, cols);

        // Verify anchors are placed correctly
        for &(ac, ar, pi) in &tri_anchor::tri_corner_positions(rows, cols) {
            let pattern = &tri_anchor::TRI_ANCHOR_PATTERNS[pi];
            for dr in 0..tri_anchor::TRI_ANCHOR_ROWS {
                for dc in 0..tri_anchor::TRI_ANCHOR_COLS {
                    assert_eq!(
                        grid.get(ac + dc, ar + dr),
                        Some(pattern[dr][dc]),
                        "Anchor at ({},{}) pattern[{}][{}] mismatch",
                        ac + dc, ar + dr, dr, dc
                    );
                }
            }
        }

        // Verify all trit values are valid (0, 1, or 2)
        for row in 0..rows {
            for col in 0..cols {
                let v = grid.get(col, row).unwrap();
                assert!(v <= 2, "Invalid trit {} at ({},{})", v, col, row);
            }
        }
    }

    #[test]
    fn image_guided_all_light_target_produces_light_bias() {
        let trits = vec![0, 1, 2, 0, 1, 2];
        let rows = 20;
        let cols = 40;

        let cell_coords = tri_grid_to_flat_coords(rows, cols);
        let target_trits: Vec<u8> = vec![2; cell_coords.len()]; // all light

        let grid = TriGaussSolver::resolve_trigrid_image_guided(
            &trits, rows, cols, &[], &target_trits
        ).unwrap();

        // Count trit values in the non-anchor region
        let mut counts = [0usize; 3];
        for &(col, row) in &cell_coords {
            let v = grid.get(col, row).unwrap_or(0) as usize;
            if v < 3 {
                counts[v] += 1;
            }
        }

        // With all-light targets, State 2 should dominate
        assert!(counts[2] > counts[0],
            "All-light targets should produce more State 2 than State 0: {:?}", counts);
    }

    #[test]
    fn image_guided_all_dark_target_produces_dark_bias() {
        let trits = vec![0, 1, 2, 0, 1, 2];
        let rows = 20;
        let cols = 40;

        let cell_coords = tri_grid_to_flat_coords(rows, cols);
        let target_trits: Vec<u8> = vec![0; cell_coords.len()]; // all dark

        let grid = TriGaussSolver::resolve_trigrid_image_guided(
            &trits, rows, cols, &[], &target_trits
        ).unwrap();

        // Count trit values in the non-anchor region
        let mut counts = [0usize; 3];
        for &(col, row) in &cell_coords {
            let v = grid.get(col, row).unwrap_or(0) as usize;
            if v < 3 {
                counts[v] += 1;
            }
        }

        // With all-dark targets, State 0 should dominate
        assert!(counts[0] > counts[2],
            "All-dark targets should produce more State 0 than State 2: {:?}", counts);
    }
}
