//! # trixel_solver
//!
//! Constraint-based matrix generation for the Trixel system.
//! Provides `MockSolver` (plain packing), `AnchorSolver` (L-bracket aware),
//! and `Z3Solver` (SMT parity-check solver for typography integration).

use trixel_core::TritMatrix;
use thiserror::Error;

pub mod anchor;
pub mod gauss;
pub mod gauss_solver;

pub use gauss_solver::GaussSolver;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A spatial constraint locking a specific cell to a required trit value.
/// Used by the `FontEngine` to pin character glyphs into the grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstraintMask {
    pub x: usize,
    pub y: usize,
    /// Required trit state: 0, 1, or 2.
    pub required_state: u8,
}

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum SolverError {
    #[error("constraint at ({x}, {y}) conflicts with payload")]
    Conflict { x: usize, y: usize },
    #[error("matrix size {size}x{size} too small for {trits} payload trits")]
    MatrixTooSmall { size: usize, trits: usize },
    #[error("unsatisfiable: visual constraints are too dense for the payload")]
    Unsatisfiable,
    #[error("false anchor detected after {attempts} re-pack attempts")]
    FalseAnchor { attempts: usize },
}

// ---------------------------------------------------------------------------
// MatrixSolver Trait
// ---------------------------------------------------------------------------

/// Generates a valid trit matrix that satisfies both the ECC parity equations
/// and the user's visual/font constraints.
pub trait MatrixSolver {
    /// Finds a grid layout that encodes `payload_trits` into a `matrix_size × matrix_size`
    /// grid while respecting the given spatial constraints.
    fn resolve_matrix(
        payload_trits: &[u8],
        matrix_size: usize,
        constraints: &[ConstraintMask],
    ) -> Result<TritMatrix, SolverError>;
}

// ---------------------------------------------------------------------------
// Mock Implementation
// ---------------------------------------------------------------------------

/// Mock solver: packs payload into a square matrix row-by-row,
/// zero-fills remaining cells, ignores constraints entirely.
pub struct MockSolver;

impl MatrixSolver for MockSolver {
    fn resolve_matrix(
        payload_trits: &[u8],
        matrix_size: usize,
        _constraints: &[ConstraintMask],
    ) -> Result<TritMatrix, SolverError> {
        let capacity = matrix_size * matrix_size;
        if payload_trits.len() > capacity {
            return Err(SolverError::MatrixTooSmall {
                size: matrix_size,
                trits: payload_trits.len(),
            });
        }

        let mut matrix = TritMatrix::zeros(matrix_size, matrix_size);
        for (i, &t) in payload_trits.iter().enumerate() {
            let x = i % matrix_size;
            let y = i / matrix_size;
            matrix.set(x, y, t);
        }
        Ok(matrix)
    }
}

// ---------------------------------------------------------------------------
// Anchor-Aware Solver
// ---------------------------------------------------------------------------

/// Production solver: places L-bracket anchors in the 4 corners, packs payload
/// into the remaining free cells, and validates that no false anchor patterns
/// appear in the data region.
pub struct AnchorSolver;

impl AnchorSolver {
    /// Maximum re-pack attempts when false anchors are detected.
    const MAX_RETRIES: usize = 3;
}

impl MatrixSolver for AnchorSolver {
    fn resolve_matrix(
        payload_trits: &[u8],
        matrix_size: usize,
        constraints: &[ConstraintMask],
    ) -> Result<TritMatrix, SolverError> {
        let n = matrix_size;
        if n < anchor::ANCHOR_SIZE * 2 {
            return Err(SolverError::MatrixTooSmall {
                size: n,
                trits: payload_trits.len(),
            });
        }

        // Validate: no constraint may overlap an anchor region
        for c in constraints {
            if anchor::is_in_anchor_region(c.x, c.y, n) {
                return Err(SolverError::Conflict { x: c.x, y: c.y });
            }
        }

        // Build constraint lookup set
        let constraint_map: std::collections::HashMap<(usize, usize), u8> = constraints
            .iter()
            .map(|c| ((c.x, c.y), c.required_state))
            .collect();

        // Calculate free capacity (total cells minus anchor cells minus constraints)
        let anchor_cells = 4 * anchor::ANCHOR_SIZE * anchor::ANCHOR_SIZE;
        let total_cells = n * n;
        let free_capacity = total_cells
            .saturating_sub(anchor_cells)
            .saturating_sub(constraint_map.len());

        if payload_trits.len() > free_capacity {
            return Err(SolverError::MatrixTooSmall {
                size: n,
                trits: payload_trits.len(),
            });
        }

        // Try packing, with retries if false anchors appear
        for attempt in 0..=Self::MAX_RETRIES {
            let mut matrix = TritMatrix::zeros(n, n);

            // 1. Place anchors in corners
            for &(cx, cy, pi) in &anchor::corner_positions(n) {
                let pattern = &anchor::ANCHOR_PATTERNS[pi];
                for dy in 0..anchor::ANCHOR_SIZE {
                    for dx in 0..anchor::ANCHOR_SIZE {
                        matrix.set(cx + dx, cy + dy, pattern[dy][dx]);
                    }
                }
            }

            // 2. Place constraint cells
            for c in constraints {
                matrix.set(c.x, c.y, c.required_state);
            }

            // 3. Pack payload into free cells (skip anchors AND constraints)
            let mut trit_idx = 0;
            for y in 0..n {
                for x in 0..n {
                    if anchor::is_in_anchor_region(x, y, n) {
                        continue;
                    }
                    if constraint_map.contains_key(&(x, y)) {
                        continue;
                    }
                    if trit_idx < payload_trits.len() {
                        let mut val = payload_trits[trit_idx];
                        // On retry, offset the value to break false patterns
                        if attempt > 0 && trit_idx == 0 {
                            val = (val + attempt as u8) % 3;
                        }
                        matrix.set(x, y, val);
                    }
                    // else: stays 0 (zero-fill)
                    trit_idx += 1;
                }
            }

            // 4. Validate: no false anchors
            let hits = anchor::scan_for_false_anchors(&matrix.data, n, n);
            if hits.is_empty() {
                return Ok(matrix);
            }
        }

        Err(SolverError::FalseAnchor {
            attempts: Self::MAX_RETRIES + 1,
        })
    }
}
