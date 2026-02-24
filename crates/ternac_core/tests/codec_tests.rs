use ternac_core::TernaryCodec;

#[test]
fn codec_roundtrip() {
    let data = b"Hello, Ternac!";
    let trits = ternac_core::MockCodec::encode_bytes(data).unwrap();
    let decoded = ternac_core::MockCodec::decode_trits(&trits).unwrap();
    assert_eq!(decoded, data);
}

#[test]
fn codec_all_byte_values() {
    use ternac_core::TernaryCodec;
    let data: Vec<u8> = (0..=255).collect();
    let trits = ternac_core::MockCodec::encode_bytes(&data).unwrap();
    let decoded = ternac_core::MockCodec::decode_trits(&trits).unwrap();
    assert_eq!(decoded, data);
}

#[test]
fn codec_invalid_trit() {
    let bad = vec![0, 1, 5, 0, 0, 0]; // 5 is invalid
    assert!(ternac_core::MockCodec::decode_trits(&bad).is_err());
}

#[test]
fn codec_bad_length() {
    let bad = vec![0, 1, 2]; // not a multiple of 6
    assert!(ternac_core::MockCodec::decode_trits(&bad).is_err());
}
