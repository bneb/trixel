//! # TriGrid — Triangular Tessellation Data Structure
//!
//! A 2D grid of alternating up-pointing (▲) and down-pointing (▽) triangles.
//! Each triangle stores exactly one trit value (0, 1, 2). Value 3 = erasure.
//!
//! ## Coordinate System
//! ```text
//! Row 0:  ▲₀ ▽₁ ▲₂ ▽₃ ▲₄ ▽₅
//! Row 1:  ▽₀ ▲₁ ▽₂ ▲₃ ▽₄ ▲₅
//! Row 2:  ▲₀ ▽₁ ▲₂ ▽₃ ▲₄ ▽₅
//! ```
//!
//! A triangle at `(col, row)` points **up** if `(col + row) % 2 == 0`,
//! **down** if `(col + row) % 2 == 1`.

/// A 2D grid of trit values stored in alternating triangles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriGrid {
    /// Number of triangle rows.
    pub rows: usize,
    /// Number of triangles per row (should be even for clean tessellation).
    pub cols: usize,
    /// Row-major storage: `data[row * cols + col]`.
    pub data: Vec<u8>,
}

impl TriGrid {
    /// Create a new grid filled with zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0; rows * cols],
        }
    }

    /// Get the trit value at `(col, row)`. Returns `None` if out of bounds.
    pub fn get(&self, col: usize, row: usize) -> Option<u8> {
        if col < self.cols && row < self.rows {
            Some(self.data[row * self.cols + col])
        } else {
            None
        }
    }

    /// Set the trit value at `(col, row)`. Panics if out of bounds.
    pub fn set(&mut self, col: usize, row: usize, val: u8) {
        assert!(col < self.cols && row < self.rows, "TriGrid::set out of bounds");
        self.data[row * self.cols + col] = val;
    }

    /// Returns `true` if the triangle at `(col, row)` points up (▲).
    #[inline]
    pub fn is_up(col: usize, row: usize) -> bool {
        (col + row) % 2 == 0
    }

    /// Total number of cells in the grid.
    #[inline]
    pub fn total_cells(&self) -> usize {
        self.rows * self.cols
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------
    // TriGrid Construction
    // -------------------------------------------------------------------

    #[test]
    fn trigrid_zeros_creates_correct_size() {
        let g = TriGrid::zeros(5, 10);
        assert_eq!(g.rows, 5);
        assert_eq!(g.cols, 10);
        assert_eq!(g.data.len(), 50);
        assert_eq!(g.total_cells(), 50);
    }

    #[test]
    fn trigrid_zeros_all_zero() {
        let g = TriGrid::zeros(4, 8);
        for &val in &g.data {
            assert_eq!(val, 0);
        }
    }

    // -------------------------------------------------------------------
    // Get / Set Roundtrip
    // -------------------------------------------------------------------

    #[test]
    fn trigrid_set_get_roundtrip() {
        let mut g = TriGrid::zeros(10, 12);
        g.set(3, 5, 2);
        g.set(7, 2, 1);
        g.set(0, 0, 0);
        assert_eq!(g.get(3, 5), Some(2));
        assert_eq!(g.get(7, 2), Some(1));
        assert_eq!(g.get(0, 0), Some(0));
    }

    #[test]
    fn trigrid_set_overwrites() {
        let mut g = TriGrid::zeros(4, 4);
        g.set(1, 1, 2);
        assert_eq!(g.get(1, 1), Some(2));
        g.set(1, 1, 0);
        assert_eq!(g.get(1, 1), Some(0));
    }

    #[test]
    fn trigrid_get_out_of_bounds_returns_none() {
        let g = TriGrid::zeros(5, 10);
        assert_eq!(g.get(10, 0), None);  // col out of bounds
        assert_eq!(g.get(0, 5), None);   // row out of bounds
        assert_eq!(g.get(10, 5), None);  // both out of bounds
        assert_eq!(g.get(100, 100), None);
    }

    // -------------------------------------------------------------------
    // Triangle Orientation
    // -------------------------------------------------------------------

    #[test]
    fn trigrid_is_up_even_sum() {
        // (col + row) % 2 == 0 → up
        assert!(TriGrid::is_up(0, 0));  // 0+0=0
        assert!(TriGrid::is_up(2, 0));  // 2+0=2
        assert!(TriGrid::is_up(1, 1));  // 1+1=2
        assert!(TriGrid::is_up(0, 2));  // 0+2=2
        assert!(TriGrid::is_up(3, 3));  // 3+3=6
    }

    #[test]
    fn trigrid_is_down_odd_sum() {
        // (col + row) % 2 == 1 → down (not up)
        assert!(!TriGrid::is_up(1, 0));  // 1+0=1
        assert!(!TriGrid::is_up(0, 1));  // 0+1=1
        assert!(!TriGrid::is_up(2, 1));  // 2+1=3
        assert!(!TriGrid::is_up(3, 2));  // 3+2=5
    }

    // -------------------------------------------------------------------
    // Data Isolation
    // -------------------------------------------------------------------

    #[test]
    fn trigrid_cells_are_independent() {
        let mut g = TriGrid::zeros(4, 6);
        // Set every cell to a unique value pattern
        for row in 0..4 {
            for col in 0..6 {
                g.set(col, row, ((col + row * 6) % 3) as u8);
            }
        }
        // Verify each cell independently
        for row in 0..4 {
            for col in 0..6 {
                let expected = ((col + row * 6) % 3) as u8;
                assert_eq!(g.get(col, row), Some(expected),
                    "Cell ({},{}) expected {} got {:?}", col, row, expected, g.get(col, row));
            }
        }
    }

    #[test]
    fn trigrid_clone_is_independent() {
        let mut g = TriGrid::zeros(3, 4);
        g.set(1, 1, 2);
        let g2 = g.clone();
        assert_eq!(g, g2);
        g.set(1, 1, 0);
        assert_ne!(g, g2);
        assert_eq!(g2.get(1, 1), Some(2)); // clone unaffected
    }
}
