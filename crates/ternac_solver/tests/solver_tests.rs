use ternac_solver::{MockSolver, MatrixSolver};

#[test]
fn solver_packs_payload() {
    let payload = vec![0, 1, 2, 1, 0, 2, 2, 1, 0];
    let matrix = MockSolver::resolve_matrix(&payload, 3, &[]).unwrap();
    assert_eq!(matrix.width, 3);
    assert_eq!(matrix.height, 3);
    for (i, &t) in payload.iter().enumerate() {
        let x = i % 3;
        let y = i / 3;
        assert_eq!(matrix.get(x, y), Some(t));
    }
}

#[test]
fn solver_zero_fills_remainder() {
    let payload = vec![1, 2];
    let matrix = MockSolver::resolve_matrix(&payload, 3, &[]).unwrap();
    assert_eq!(matrix.get(2, 0), Some(0));
    assert_eq!(matrix.get(0, 2), Some(0));
}

#[test]
fn solver_matrix_too_small() {
    let payload = vec![0; 10];
    assert!(MockSolver::resolve_matrix(&payload, 3, &[]).is_err());
}
