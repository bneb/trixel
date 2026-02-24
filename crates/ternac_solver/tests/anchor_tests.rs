use ternac_solver::anchor::{
    ANCHOR_PATTERNS, ANCHOR_SIZE, corner_positions, is_in_anchor_region,
    scan_for_false_anchors,
};
use ternac_solver::{AnchorSolver, MatrixSolver};

// ---------------------------------------------------------------------------
// Anchor Pattern Tests
// ---------------------------------------------------------------------------

#[test]
fn anchor_patterns_are_3x3() {
    for (i, pattern) in ANCHOR_PATTERNS.iter().enumerate() {
        assert_eq!(pattern.len(), 3, "pattern {i} height");
        for row in pattern {
            assert_eq!(row.len(), 3, "pattern {i} row width");
        }
    }
}

#[test]
fn anchor_patterns_contain_only_valid_trits() {
    for pattern in &ANCHOR_PATTERNS {
        for row in pattern {
            for &cell in row {
                assert!(cell <= 2, "anchor cell must be 0, 1, or 2, got {cell}");
            }
        }
    }
}

#[test]
fn anchor_patterns_each_have_color_sync_dot() {
    // Each pattern should have exactly one State 1 (color-sync dot)
    for (i, pattern) in ANCHOR_PATTERNS.iter().enumerate() {
        let ones: usize = pattern
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&c| c == 1)
            .count();
        assert_eq!(ones, 1, "pattern {i} should have exactly 1 color-sync dot");
    }
}

#[test]
fn anchor_patterns_have_five_black_cells() {
    // L-shape: 5 cells of State 0 (the L)
    for (i, pattern) in ANCHOR_PATTERNS.iter().enumerate() {
        let zeros: usize = pattern
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&c| c == 0)
            .count();
        assert_eq!(zeros, 5, "pattern {i} should have 5 black cells (the L shape)");
    }
}

// ---------------------------------------------------------------------------
// Corner Position Tests
// ---------------------------------------------------------------------------

#[test]
fn corner_positions_for_10x10_grid() {
    let corners = corner_positions(10);
    assert_eq!(corners[0], (0, 0, 0));   // TL
    assert_eq!(corners[1], (7, 0, 1));   // TR
    assert_eq!(corners[2], (0, 7, 2));   // BL
    assert_eq!(corners[3], (7, 7, 3));   // BR
}

#[test]
fn is_in_anchor_region_correct() {
    let n = 10;
    // (0,0) is in TL anchor
    assert!(is_in_anchor_region(0, 0, n));
    assert!(is_in_anchor_region(2, 2, n));
    // (3,3) is NOT in any anchor
    assert!(!is_in_anchor_region(3, 3, n));
    // (9,9) is in BR anchor (7..10, 7..10)
    assert!(is_in_anchor_region(9, 9, n));
    // (5,5) is not in any anchor
    assert!(!is_in_anchor_region(5, 5, n));
}

// ---------------------------------------------------------------------------
// False Anchor Scanner Tests
// ---------------------------------------------------------------------------

#[test]
fn scanner_detects_false_l_bracket() {
    // Create a 10×10 grid of all-zeros, then plant a TL-pattern at (4, 4)
    let n = 10;
    let mut data = vec![0u8; n * n];

    // Plant TL pattern at (4, 4) — this is NOT a real corner
    let pattern = &ANCHOR_PATTERNS[0]; // TL
    for dy in 0..ANCHOR_SIZE {
        for dx in 0..ANCHOR_SIZE {
            data[(4 + dy) * n + (4 + dx)] = pattern[dy][dx];
        }
    }

    let hits = scan_for_false_anchors(&data, n, n);
    assert!(
        hits.contains(&(4, 4)),
        "scanner should detect false L-bracket at (4,4)"
    );
}

#[test]
fn scanner_ignores_actual_corners() {
    // 10×10 grid with proper anchors in all 4 corners
    let n = 10;
    let mut data = vec![2u8; n * n]; // all white (no accidental matches)

    for &(cx, cy, pi) in &corner_positions(n) {
        let pattern = &ANCHOR_PATTERNS[pi];
        for dy in 0..ANCHOR_SIZE {
            for dx in 0..ANCHOR_SIZE {
                data[(cy + dy) * n + (cx + dx)] = pattern[dy][dx];
            }
        }
    }

    let hits = scan_for_false_anchors(&data, n, n);
    assert!(hits.is_empty(), "scanner should not flag real corners: {hits:?}");
}

#[test]
fn scanner_returns_empty_for_clean_grid() {
    let n = 10;
    let data = vec![2u8; n * n]; // all State 2 — no L-brackets anywhere
    let hits = scan_for_false_anchors(&data, n, n);
    assert!(hits.is_empty());
}

// ---------------------------------------------------------------------------
// AnchorSolver Tests
// ---------------------------------------------------------------------------

#[test]
fn anchor_solver_places_anchors_in_corners() {
    let payload = vec![1u8; 30]; // some trits
    let n = 10;
    let matrix = AnchorSolver::resolve_matrix(&payload, n, &[]).unwrap();

    // Verify corner anchors
    for &(cx, cy, pi) in &corner_positions(n) {
        let pattern = &ANCHOR_PATTERNS[pi];
        for dy in 0..ANCHOR_SIZE {
            for dx in 0..ANCHOR_SIZE {
                let actual = matrix.get(cx + dx, cy + dy).unwrap();
                let expected = pattern[dy][dx];
                assert_eq!(
                    actual, expected,
                    "corner ({},{}) at offset ({},{}) should be {} got {}",
                    cx, cy, dx, dy, expected, actual
                );
            }
        }
    }
}

#[test]
fn anchor_solver_produces_no_false_anchors() {
    // Use varying data that could create accidental patterns
    let n = 15;
    let data_len = n * n - 4 * ANCHOR_SIZE * ANCHOR_SIZE;
    let payload: Vec<u8> = (0..data_len).map(|i| (i % 3) as u8).collect();
    let matrix = AnchorSolver::resolve_matrix(&payload, n, &[]).unwrap();

    let hits = scan_for_false_anchors(&matrix.data, matrix.width, matrix.height);
    assert!(
        hits.is_empty(),
        "anchor solver must not produce false anchors: {hits:?}"
    );
}

#[test]
fn anchor_solver_preserves_payload() {
    let n = 10;
    let free_cells = n * n - 4 * ANCHOR_SIZE * ANCHOR_SIZE;
    let payload: Vec<u8> = (0..free_cells).map(|i| (i % 3) as u8).collect();

    let matrix = AnchorSolver::resolve_matrix(&payload, n, &[]).unwrap();

    // Extract non-anchor cells in row-major order
    let mut extracted = Vec::new();
    for y in 0..n {
        for x in 0..n {
            if !is_in_anchor_region(x, y, n) {
                extracted.push(matrix.get(x, y).unwrap());
            }
        }
    }

    assert_eq!(
        extracted[..payload.len()],
        payload[..],
        "payload must be preserved in free cells"
    );
}

#[test]
fn anchor_solver_rejects_too_large_payload() {
    let n = 6; // 36 cells - 36 anchor cells = 0 free cells
    let payload = vec![1u8; 1]; // Even 1 trit won't fit
    let result = AnchorSolver::resolve_matrix(&payload, n, &[]);
    assert!(result.is_err(), "should reject when payload exceeds free capacity");
}
