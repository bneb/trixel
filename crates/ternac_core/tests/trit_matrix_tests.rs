use ternac_core::TritMatrix;

#[test]
fn trit_matrix_get_set() {
    let mut m = TritMatrix::zeros(3, 3);
    m.set(1, 2, 2);
    assert_eq!(m.get(1, 2), Some(2));
    assert_eq!(m.get(0, 0), Some(0));
    assert_eq!(m.get(5, 5), None);
}

#[test]
fn trit_matrix_zeros() {
    let m = TritMatrix::zeros(4, 3);
    assert_eq!(m.width, 4);
    assert_eq!(m.height, 3);
    assert_eq!(m.data.len(), 12);
    assert!(m.data.iter().all(|&t| t == 0));
}
