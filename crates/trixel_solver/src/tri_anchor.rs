//! # Triangular Anchor System
//!
//! Defines anchor patterns for the triangular trixel grid.
//! Instead of 3×3 L-brackets (square grid), each anchor is a small
//! rectangular block of triangles in a corner region.
//!
//! ## Anchor Design
//!
//! Each anchor occupies a `ANCHOR_ROWS × ANCHOR_COLS` block of triangles.
//! The pattern uses all 3 states for reliable detection:
//! - State 0 (black): outer border
//! - State 2 (white): quiet zone
//! - State 1 (gray): sync dot (unique to each corner rotation)
//!
//! ## Layout (4 rows × 6 cols = 24 triangles per anchor)
//!
//! ```text
//!  TL corner:               TR corner:
//!  0 0 0 0 0 0              0 0 0 0 0 0
//!  0 0 2 2 0 0              0 0 2 2 0 0
//!  0 2 2 1 0 0              0 0 1 2 2 0
//!  0 0 0 0 0 0              0 0 0 0 0 0
//! ```

/// Anchor block dimensions in triangular grid cells.
pub const TRI_ANCHOR_ROWS: usize = 4;
pub const TRI_ANCHOR_COLS: usize = 6;

/// The 4 anchor rotation patterns, stored as `[row][col]`.
/// Index: 0=TL, 1=TR, 2=BL, 3=BR.
///
/// Pattern invariant: outer ring = State 0, inner = State 2 quiet zone,
/// single State 1 sync dot positioned toward the grid center.
pub const TRI_ANCHOR_PATTERNS: [[[u8; TRI_ANCHOR_COLS]; TRI_ANCHOR_ROWS]; 4] = [
    // TL — sync dot at (3, 2), pointing toward center
    [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 0, 0],
        [0, 2, 2, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
    // TR — sync dot at (2, 2), pointing toward center
    [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 0, 0],
        [0, 0, 1, 2, 2, 0],
        [0, 0, 0, 0, 0, 0],
    ],
    // BL — sync dot at (3, 1), pointing toward center
    [
        [0, 0, 0, 0, 0, 0],
        [0, 2, 2, 1, 0, 0],
        [0, 0, 2, 2, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
    // BR — sync dot at (2, 1), pointing toward center
    [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 2, 0],
        [0, 0, 2, 2, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
];

/// Corner positions for anchors in a triangular grid of `rows × cols`.
/// Returns `(col, row, pattern_index)` for each of the 4 corners.
pub fn tri_corner_positions(rows: usize, cols: usize) -> [(usize, usize, usize); 4] {
    [
        (0, 0, 0),                                              // TL
        (cols - TRI_ANCHOR_COLS, 0, 1),                         // TR
        (0, rows - TRI_ANCHOR_ROWS, 2),                         // BL
        (cols - TRI_ANCHOR_COLS, rows - TRI_ANCHOR_ROWS, 3),    // BR
    ]
}

/// Check if position `(col, row)` is inside any of the 4 anchor corners.
pub fn is_in_tri_anchor_region(col: usize, row: usize, grid_rows: usize, grid_cols: usize) -> bool {
    for &(ac, ar, _) in &tri_corner_positions(grid_rows, grid_cols) {
        if col >= ac && col < ac + TRI_ANCHOR_COLS && row >= ar && row < ar + TRI_ANCHOR_ROWS {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------
    // Corner Positions
    // -------------------------------------------------------------------

    #[test]
    fn tri_corner_positions_small_grid() {
        // Minimum viable grid: must fit 4 non-overlapping anchors
        let rows = 12;
        let cols = 16;
        let corners = tri_corner_positions(rows, cols);

        // TL at (0, 0)
        assert_eq!(corners[0], (0, 0, 0));
        // TR at (cols - 6, 0)
        assert_eq!(corners[1], (10, 0, 1));
        // BL at (0, rows - 4)
        assert_eq!(corners[2], (0, 8, 2));
        // BR at (cols - 6, rows - 4)
        assert_eq!(corners[3], (10, 8, 3));
    }

    #[test]
    fn tri_corner_positions_large_grid() {
        let rows = 60;
        let cols = 120;
        let corners = tri_corner_positions(rows, cols);

        assert_eq!(corners[0], (0, 0, 0));
        assert_eq!(corners[1], (114, 0, 1));
        assert_eq!(corners[2], (0, 56, 2));
        assert_eq!(corners[3], (114, 56, 3));
    }

    // -------------------------------------------------------------------
    // Anchor Region Detection
    // -------------------------------------------------------------------

    #[test]
    fn tri_anchor_region_corners_are_inside() {
        let rows = 20;
        let cols = 24;

        // TL block: (0..6, 0..4)
        assert!(is_in_tri_anchor_region(0, 0, rows, cols));
        assert!(is_in_tri_anchor_region(5, 3, rows, cols));
        assert!(is_in_tri_anchor_region(3, 2, rows, cols));

        // TR block
        assert!(is_in_tri_anchor_region(cols - 1, 0, rows, cols));
        assert!(is_in_tri_anchor_region(cols - 6, 0, rows, cols));

        // BL block
        assert!(is_in_tri_anchor_region(0, rows - 1, rows, cols));
        assert!(is_in_tri_anchor_region(0, rows - 4, rows, cols));

        // BR block
        assert!(is_in_tri_anchor_region(cols - 1, rows - 1, rows, cols));
    }

    #[test]
    fn tri_anchor_region_center_is_outside() {
        let rows = 20;
        let cols = 24;

        // Center of the grid should never be in an anchor
        assert!(!is_in_tri_anchor_region(12, 10, rows, cols));
        assert!(!is_in_tri_anchor_region(10, 8, rows, cols));
    }

    #[test]
    fn tri_anchor_region_edge_of_anchor_is_inside() {
        let rows = 20;
        let cols = 24;

        // Exact boundary of TL anchor
        assert!(is_in_tri_anchor_region(5, 0, rows, cols));   // right edge
        assert!(is_in_tri_anchor_region(0, 3, rows, cols));   // bottom edge
        // Just outside TL
        assert!(!is_in_tri_anchor_region(6, 0, rows, cols));
        assert!(!is_in_tri_anchor_region(0, 4, rows, cols));
    }

    // -------------------------------------------------------------------
    // Pattern Invariants
    // -------------------------------------------------------------------

    #[test]
    fn tri_anchor_patterns_outer_ring_is_black() {
        // All patterns must have State 0 on the top and bottom rows
        for (pi, pattern) in TRI_ANCHOR_PATTERNS.iter().enumerate() {
            for col in 0..TRI_ANCHOR_COLS {
                assert_eq!(pattern[0][col], 0,
                    "Pattern {pi} top row col {col} must be 0");
                assert_eq!(pattern[TRI_ANCHOR_ROWS - 1][col], 0,
                    "Pattern {pi} bottom row col {col} must be 0");
            }
            // Left and right columns must be black
            for row in 0..TRI_ANCHOR_ROWS {
                assert_eq!(pattern[row][0], 0,
                    "Pattern {pi} row {row} left col must be 0");
                assert_eq!(pattern[row][TRI_ANCHOR_COLS - 1], 0,
                    "Pattern {pi} row {row} right col must be 0");
            }
        }
    }

    #[test]
    fn tri_anchor_patterns_each_has_exactly_one_sync_dot() {
        for (pi, pattern) in TRI_ANCHOR_PATTERNS.iter().enumerate() {
            let count: usize = pattern.iter()
                .flat_map(|row| row.iter())
                .filter(|&&v| v == 1)
                .count();
            assert_eq!(count, 1,
                "Pattern {pi} must have exactly 1 sync dot (State 1), got {count}");
        }
    }

    #[test]
    fn tri_anchor_patterns_all_four_unique() {
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert_ne!(TRI_ANCHOR_PATTERNS[i], TRI_ANCHOR_PATTERNS[j],
                    "Patterns {i} and {j} must be distinct");
            }
        }
    }

    // -------------------------------------------------------------------
    // Non-Overlap
    // -------------------------------------------------------------------

    #[test]
    fn tri_anchors_do_not_overlap_in_minimum_grid() {
        // At minimum viable size, the 4 anchors must not overlap
        let rows = TRI_ANCHOR_ROWS * 2;  // 8
        let cols = TRI_ANCHOR_COLS * 2;  // 12
        let corners = tri_corner_positions(rows, cols);

        // Check that no cell belongs to more than one anchor
        for row in 0..rows {
            for col in 0..cols {
                let count: usize = corners.iter()
                    .filter(|&&(ac, ar, _)| {
                        col >= ac && col < ac + TRI_ANCHOR_COLS
                        && row >= ar && row < ar + TRI_ANCHOR_ROWS
                    })
                    .count();
                assert!(count <= 1,
                    "Cell ({col},{row}) belongs to {count} anchors — overlap detected");
            }
        }
    }
}
