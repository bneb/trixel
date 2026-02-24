use ternac_solver::{AnchorSolver, ConstraintMask, MatrixSolver};
use ternac_solver::anchor::{ANCHOR_SIZE, corner_positions, ANCHOR_PATTERNS, is_in_anchor_region};
use ternac_render::{FontEngine, TernacFont};

// ---------------------------------------------------------------------------
// Basic Constraint Tests
// ---------------------------------------------------------------------------

#[test]
fn solver_respects_single_constraint() {
    let n = 10;
    let payload = vec![1u8; 20]; // some trits
    let constraints = vec![ConstraintMask { x: 4, y: 4, required_state: 2 }];

    let matrix = AnchorSolver::resolve_matrix(&payload, n, &constraints).unwrap();
    assert_eq!(
        matrix.get(4, 4).unwrap(), 2,
        "constrained cell (4,4) must be State 2"
    );
}

#[test]
fn solver_respects_multiple_constraints() {
    let n = 10;
    let payload = vec![0u8; 20];
    let constraints = vec![
        ConstraintMask { x: 4, y: 4, required_state: 2 },
        ConstraintMask { x: 5, y: 4, required_state: 0 },
        ConstraintMask { x: 4, y: 5, required_state: 1 },
    ];

    let matrix = AnchorSolver::resolve_matrix(&payload, n, &constraints).unwrap();
    assert_eq!(matrix.get(4, 4).unwrap(), 2);
    assert_eq!(matrix.get(5, 4).unwrap(), 0);
    assert_eq!(matrix.get(4, 5).unwrap(), 1);
}

#[test]
fn solver_rejects_constraint_in_anchor() {
    let n = 10;
    let payload = vec![1u8; 10];
    // (0,0) is inside TL anchor
    let constraints = vec![ConstraintMask { x: 0, y: 0, required_state: 1 }];

    let result = AnchorSolver::resolve_matrix(&payload, n, &constraints);
    assert!(result.is_err(), "constraint inside anchor region should be rejected");
}

#[test]
fn solver_payload_fills_around_constraints() {
    let n = 10;
    let anchor_cells = 4 * ANCHOR_SIZE * ANCHOR_SIZE;
    let num_constraints = 5;
    let free_cells = n * n - anchor_cells - num_constraints;
    let payload: Vec<u8> = (0..free_cells).map(|i| (i % 3) as u8).collect();

    let constraints: Vec<ConstraintMask> = (0..num_constraints)
        .map(|i| ConstraintMask { x: 4 + i, y: 4, required_state: 2 })
        .collect();

    let matrix = AnchorSolver::resolve_matrix(&payload, n, &constraints).unwrap();

    // Constraints must be honored
    for c in &constraints {
        assert_eq!(matrix.get(c.x, c.y).unwrap(), c.required_state);
    }

    // Extract non-anchor, non-constraint cells and verify payload
    let mut extracted = Vec::new();
    let constraint_set: std::collections::HashSet<(usize, usize)> =
        constraints.iter().map(|c| (c.x, c.y)).collect();

    for y in 0..n {
        for x in 0..n {
            if !is_in_anchor_region(x, y, n) && !constraint_set.contains(&(x, y)) {
                extracted.push(matrix.get(x, y).unwrap());
            }
        }
    }
    assert_eq!(
        extracted[..payload.len()], payload[..],
        "payload must fill free cells"
    );
}

// ---------------------------------------------------------------------------
// Font + Solver Integration
// ---------------------------------------------------------------------------

#[test]
fn solver_with_font_constraints() {
    let n = 20; // Large enough to fit text + anchors + payload
    let text = "HI";
    let constraints = TernacFont::string_to_constraints(text, 4, 4);

    // Small payload to leave room for text
    let anchor_cells = 4 * ANCHOR_SIZE * ANCHOR_SIZE;
    let free_minus_text = n * n - anchor_cells - constraints.len();
    let payload: Vec<u8> = (0..free_minus_text.min(50)).map(|i| (i % 3) as u8).collect();

    let matrix = AnchorSolver::resolve_matrix(&payload, n, &constraints).unwrap();

    // Verify all constraints are honored
    for c in &constraints {
        let actual = matrix.get(c.x, c.y).unwrap();
        assert_eq!(
            actual, c.required_state,
            "font constraint at ({},{}) should be {} got {}",
            c.x, c.y, c.required_state, actual
        );
    }

    // Verify anchors are intact
    for &(cx, cy, pi) in &corner_positions(n) {
        let pattern = &ANCHOR_PATTERNS[pi];
        for dy in 0..ANCHOR_SIZE {
            for dx in 0..ANCHOR_SIZE {
                let actual = matrix.get(cx + dx, cy + dy).unwrap();
                assert_eq!(actual, pattern[dy][dx], "anchor corrupted at ({},{})", cx + dx, cy + dy);
            }
        }
    }
}
