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
///
/// Expanded from 4×6 to 5×8 to accommodate embedded metadata:
/// - Outer black ring (border): State 0
/// - White quiet zone: State 2
/// - Corner ID: 2 trits encoding corner position
/// - CRC-3: 3 trits checksum for self-verification
/// - Sync dot: 1 cell State 1 (unique position per corner)
pub const TRI_ANCHOR_ROWS: usize = 5;
pub const TRI_ANCHOR_COLS: usize = 8;

/// Corner ID trit pairs (base-3 encoding).
/// TL=0 → (0,0), TR=1 → (0,1), BL=2 → (0,2), BR=3 → (1,0)
pub const CORNER_IDS: [[u8; 2]; 4] = [
    [0, 0], // TL
    [0, 1], // TR
    [0, 2], // BL
    [1, 0], // BR
];

/// The 4 anchor patterns, stored as `[row][col]`.
/// Index: 0=TL, 1=TR, 2=BL, 3=BR.
///
/// Layout (5×8 = 40 cells):
/// ```text
///  Row 0: 0 0 0 0 0 0 0 0    (full black border)
///  Row 1: 0 2 2 C C K K 0    (quiet zone + corner ID(C) + CRC(K))
///  Row 2: 0 2 2 2 1 2 2 0    (quiet zone + sync dot)
///  Row 3: 0 2 2 K C C 2 0    (CRC(K) + corner ID(C) mirrored)
///  Row 4: 0 0 0 0 0 0 0 0    (full black border)
/// ```
///
/// Corner ID positions: (3,1), (4,1) and (4,3), (3,3) — mirrored
/// CRC positions: (5,1), (6,1) and (3,3) — varies
/// Sync dot: (4,2) — for TL; (3,2) for TR; (4,2) for BL; (3,2) for BR
///
/// CRC-3 = sum of corner_id trits mod 3, replicated 3 times.
pub const TRI_ANCHOR_PATTERNS: [[[u8; TRI_ANCHOR_COLS]; TRI_ANCHOR_ROWS]; 4] = [
    // TL — Corner ID = (0,0), CRC = 0, sync dot at (4,2) toward center
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 0, 0, 0, 0],
        [0, 2, 2, 2, 1, 2, 2, 0],
        [0, 2, 2, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    // TR — Corner ID = (0,1), CRC = 1, sync dot at (3,2) toward center
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 2, 0],
        [0, 2, 2, 1, 2, 2, 2, 0],
        [0, 2, 1, 1, 0, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    // BL — Corner ID = (0,2), CRC = 2, sync dot at (4,2) toward center
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 0, 2, 2, 2, 0],
        [0, 2, 2, 2, 1, 2, 2, 0],
        [0, 2, 2, 2, 0, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    // BR — Corner ID = (1,0), CRC = 1, sync dot at (3,2) toward center
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 1, 1, 1, 0, 2, 0],
        [0, 2, 2, 1, 2, 2, 2, 0],
        [0, 2, 2, 1, 0, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
];

/// Compute the CRC-3 for a corner ID.
/// CRC = (id_trit_0 + id_trit_1) mod 3
pub fn compute_anchor_crc(corner_index: usize) -> u8 {
    let id = &CORNER_IDS[corner_index];
    (id[0] + id[1]) % 3
}

/// Verify that a detected anchor pattern's CRC matches the corner ID.
pub fn verify_anchor_crc(pattern: &[[u8; TRI_ANCHOR_COLS]; TRI_ANCHOR_ROWS], corner_index: usize) -> bool {
    pattern == &TRI_ANCHOR_PATTERNS[corner_index]
}

/// Detect which corner a pattern represents.
/// Returns `Some(corner_index)` if the pattern matches any known anchor,
/// or `None` if no match.
pub fn detect_corner_id(pattern: &[[u8; TRI_ANCHOR_COLS]; TRI_ANCHOR_ROWS]) -> Option<usize> {
    TRI_ANCHOR_PATTERNS
        .iter()
        .position(|p| p == pattern)
}

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
        // Minimum viable grid: must fit 4 non-overlapping 5×8 anchors
        let rows = 12;
        let cols = 20;
        let corners = tri_corner_positions(rows, cols);

        // TL at (0, 0)
        assert_eq!(corners[0], (0, 0, 0));
        // TR at (cols - 8, 0)
        assert_eq!(corners[1], (12, 0, 1));
        // BL at (0, rows - 5)
        assert_eq!(corners[2], (0, 7, 2));
        // BR at (cols - 8, rows - 5)
        assert_eq!(corners[3], (12, 7, 3));
    }

    #[test]
    fn tri_corner_positions_large_grid() {
        let rows = 60;
        let cols = 120;
        let corners = tri_corner_positions(rows, cols);

        assert_eq!(corners[0], (0, 0, 0));
        assert_eq!(corners[1], (112, 0, 1));
        assert_eq!(corners[2], (0, 55, 2));
        assert_eq!(corners[3], (112, 55, 3));
    }

    // -------------------------------------------------------------------
    // Anchor Region Detection
    // -------------------------------------------------------------------

    #[test]
    fn tri_anchor_region_corners_are_inside() {
        let rows = 20;
        let cols = 32;

        // TL block: (0..8, 0..5)
        assert!(is_in_tri_anchor_region(0, 0, rows, cols));
        assert!(is_in_tri_anchor_region(7, 4, rows, cols));
        assert!(is_in_tri_anchor_region(3, 2, rows, cols));

        // TR block
        assert!(is_in_tri_anchor_region(cols - 1, 0, rows, cols));
        assert!(is_in_tri_anchor_region(cols - 8, 0, rows, cols));

        // BL block
        assert!(is_in_tri_anchor_region(0, rows - 1, rows, cols));
        assert!(is_in_tri_anchor_region(0, rows - 5, rows, cols));

        // BR block
        assert!(is_in_tri_anchor_region(cols - 1, rows - 1, rows, cols));
    }

    #[test]
    fn tri_anchor_region_center_is_outside() {
        let rows = 20;
        let cols = 32;

        // Center of the grid should never be in an anchor
        assert!(!is_in_tri_anchor_region(16, 10, rows, cols));
        assert!(!is_in_tri_anchor_region(10, 8, rows, cols));
    }

    #[test]
    fn tri_anchor_region_edge_of_anchor_is_inside() {
        let rows = 20;
        let cols = 32;

        // Exact boundary of TL anchor (5×8)
        assert!(is_in_tri_anchor_region(7, 0, rows, cols));   // right edge
        assert!(is_in_tri_anchor_region(0, 4, rows, cols));   // bottom edge
        // Just outside TL
        assert!(!is_in_tri_anchor_region(8, 0, rows, cols));
        assert!(!is_in_tri_anchor_region(0, 5, rows, cols));
    }

    // -------------------------------------------------------------------
    // Pattern Invariants
    // -------------------------------------------------------------------

    #[test]
    fn tri_anchor_patterns_outer_ring_is_black() {
        for (pi, pattern) in TRI_ANCHOR_PATTERNS.iter().enumerate() {
            for col in 0..TRI_ANCHOR_COLS {
                assert_eq!(pattern[0][col], 0,
                    "Pattern {pi} top row col {col} must be 0");
                assert_eq!(pattern[TRI_ANCHOR_ROWS - 1][col], 0,
                    "Pattern {pi} bottom row col {col} must be 0");
            }
            for row in 0..TRI_ANCHOR_ROWS {
                assert_eq!(pattern[row][0], 0,
                    "Pattern {pi} row {row} left col must be 0");
                assert_eq!(pattern[row][TRI_ANCHOR_COLS - 1], 0,
                    "Pattern {pi} row {row} right col must be 0");
            }
        }
    }

    #[test]
    fn tri_anchor_patterns_each_has_sync_dot() {
        for (pi, pattern) in TRI_ANCHOR_PATTERNS.iter().enumerate() {
            let count: usize = pattern.iter()
                .flat_map(|row| row.iter())
                .filter(|&&v| v == 1)
                .count();
            assert!(count >= 1,
                "Pattern {pi} must have at least 1 sync dot (State 1), got {count}");
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
    // CRC & Corner ID
    // -------------------------------------------------------------------

    #[test]
    fn anchor_crc_values_correct() {
        // TL: (0+0)%3 = 0
        assert_eq!(compute_anchor_crc(0), 0);
        // TR: (0+1)%3 = 1
        assert_eq!(compute_anchor_crc(1), 1);
        // BL: (0+2)%3 = 2
        assert_eq!(compute_anchor_crc(2), 2);
        // BR: (1+0)%3 = 1
        assert_eq!(compute_anchor_crc(3), 1);
    }

    #[test]
    fn anchor_verify_crc_all_pass() {
        for i in 0..4 {
            assert!(verify_anchor_crc(&TRI_ANCHOR_PATTERNS[i], i),
                "Pattern {i} should pass CRC verification");
        }
    }

    #[test]
    fn anchor_crc_rejects_wrong_pattern() {
        // TL pattern should not verify as TR
        assert!(!verify_anchor_crc(&TRI_ANCHOR_PATTERNS[0], 1));
        assert!(!verify_anchor_crc(&TRI_ANCHOR_PATTERNS[0], 2));
        assert!(!verify_anchor_crc(&TRI_ANCHOR_PATTERNS[0], 3));
    }

    #[test]
    fn anchor_detect_corner_id_all() {
        for i in 0..4 {
            assert_eq!(detect_corner_id(&TRI_ANCHOR_PATTERNS[i]), Some(i),
                "Should detect corner {i}");
        }
    }

    #[test]
    fn anchor_detect_corner_id_unknown_returns_none() {
        let unknown = [[0u8; TRI_ANCHOR_COLS]; TRI_ANCHOR_ROWS];
        assert_eq!(detect_corner_id(&unknown), None);
    }

    // -------------------------------------------------------------------
    // Non-Overlap
    // -------------------------------------------------------------------

    #[test]
    fn tri_anchors_do_not_overlap_in_minimum_grid() {
        let rows = TRI_ANCHOR_ROWS * 2;  // 10
        let cols = TRI_ANCHOR_COLS * 2;  // 16
        let corners = tri_corner_positions(rows, cols);

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
