use ternac_core::{ErrorCorrection, MockEcc};

#[test]
fn ecc_roundtrip() {
    let payload: Vec<u8> = vec![0, 1, 2, 1, 0, 2, 2, 1];
    let with_parity = MockEcc::apply_parity(&payload, 0.3).unwrap();
    let recovered = MockEcc::correct_errors(&with_parity).unwrap();
    assert_eq!(recovered, payload);
}

#[test]
fn ecc_invalid_capacity() {
    assert!(MockEcc::apply_parity(&[1], 1.5).is_err());
    assert!(MockEcc::apply_parity(&[1], -0.1).is_err());
}

#[test]
fn ecc_empty_payload() {
    assert!(MockEcc::apply_parity(&[], 0.3).is_err());
}
