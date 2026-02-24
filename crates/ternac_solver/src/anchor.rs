//! # L-Bracket Anchor System
//!
//! Defines the 3×3 L-bracket anchor patterns used in the four corners of a
//! Ternac matrix, and provides a sliding-window scanner that detects false
//! anchor occurrences in the data payload region.
//!
//! ## Anchor Layout
//! ```text
//!  TL: [0 0 0]   TR: [0 0 0]   BL: [0 2 1]   BR: [1 2 0]
//!      [0 2 2]       [2 2 0]       [0 2 2]       [2 2 0]
//!      [0 2 1]       [1 2 0]       [0 0 0]       [0 0 0]
//! ```
//! The outer L is always State 0 (black), quiet zone is State 2 (white),
//! and the color-sync dot is State 1, sitting in the inner crook.

/// Anchor size in modules (3×3).
pub const ANCHOR_SIZE: usize = 3;

/// The 4 anchor rotation patterns, stored as `[row][col]`.
/// Index: 0=TL, 1=TR, 2=BL, 3=BR.
pub const ANCHOR_PATTERNS: [[[u8; ANCHOR_SIZE]; ANCHOR_SIZE]; 4] = [
    // TL — outer corner at (0,0), crook points toward center
    [
        [0, 0, 0],
        [0, 2, 2],
        [0, 2, 1],
    ],
    // TR — outer corner at (N-1, 0), crook points toward center
    [
        [0, 0, 0],
        [2, 2, 0],
        [1, 2, 0],
    ],
    // BL — outer corner at (0, N-1), crook points toward center
    [
        [0, 2, 1],
        [0, 2, 2],
        [0, 0, 0],
    ],
    // BR — outer corner at (N-1, N-1), crook points toward center
    [
        [1, 2, 0],
        [2, 2, 0],
        [0, 0, 0],
    ],
];

/// Corner positions for anchors in a grid of size `n × n`.
/// Returns (x, y, pattern_index) for each of the 4 corners.
pub fn corner_positions(n: usize) -> [(usize, usize, usize); 4] {
    [
        (0, 0, 0),                             // TL
        (n - ANCHOR_SIZE, 0, 1),               // TR
        (0, n - ANCHOR_SIZE, 2),               // BL
        (n - ANCHOR_SIZE, n - ANCHOR_SIZE, 3), // BR
    ]
}

/// Check if position (x, y) is inside any of the 4 anchor corners.
pub fn is_in_anchor_region(x: usize, y: usize, n: usize) -> bool {
    for &(cx, cy, _) in &corner_positions(n) {
        if x >= cx && x < cx + ANCHOR_SIZE && y >= cy && y < cy + ANCHOR_SIZE {
            return true;
        }
    }
    false
}

/// Scan the matrix for false L-bracket occurrences outside the anchor corners.
///
/// Returns the (x, y) positions of any 3×3 block that matches ANY of the
/// 4 anchor rotations and is NOT in an actual corner.
pub fn scan_for_false_anchors(
    data: &[u8],
    width: usize,
    height: usize,
) -> Vec<(usize, usize)> {
    let mut hits = Vec::new();
    if width < ANCHOR_SIZE || height < ANCHOR_SIZE {
        return hits;
    }

    for y in 0..=(height - ANCHOR_SIZE) {
        for x in 0..=(width - ANCHOR_SIZE) {
            // Skip actual corners
            if is_actual_corner(x, y, width) {
                continue;
            }
            // Check all 4 rotations
            for pattern in &ANCHOR_PATTERNS {
                if matches_pattern(data, width, x, y, pattern) {
                    hits.push((x, y));
                    break; // One match is enough for this position
                }
            }
        }
    }
    hits
}

/// Check if (x, y) is one of the 4 actual anchor corner positions.
fn is_actual_corner(x: usize, y: usize, n: usize) -> bool {
    let corners = corner_positions(n);
    corners.iter().any(|&(cx, cy, _)| cx == x && cy == y)
}

/// Check if the 3×3 block at (x, y) matches the given anchor pattern.
fn matches_pattern(
    data: &[u8],
    width: usize,
    x: usize,
    y: usize,
    pattern: &[[u8; ANCHOR_SIZE]; ANCHOR_SIZE],
) -> bool {
    for dy in 0..ANCHOR_SIZE {
        for dx in 0..ANCHOR_SIZE {
            let idx = (y + dy) * width + (x + dx);
            if data[idx] != pattern[dy][dx] {
                return false;
            }
        }
    }
    true
}
