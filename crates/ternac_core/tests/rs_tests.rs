//! Tests for Reed-Solomon encoder/decoder over GF(3^6).
//!
//! TDD red phase: define the exact contract the RS engine must satisfy.

use ternac_core::gf3::GF3;
use ternac_core::rs::ReedSolomon;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a default RS codec with the given number of parity symbols.
fn make_rs(parity_symbols: usize) -> (GF3, ReedSolomon) {
    let gf = GF3::new();
    let rs = ReedSolomon::new(&gf, parity_symbols);
    (gf, rs)
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

#[test]
fn encode_appends_correct_number_of_parity_symbols() {
    let (gf, rs) = make_rs(10);
    let data: Vec<u16> = vec![1, 2, 3, 4, 5];
    let codeword = rs.encode(&gf, &data);
    assert_eq!(
        codeword.len(),
        data.len() + 10,
        "codeword must be data + parity symbols"
    );
    // Data portion must be in the high-degree positions (after parity)
    assert_eq!(&codeword[10..], &data[..]);
}

#[test]
fn encode_deterministic() {
    let (gf, rs) = make_rs(6);
    let data: Vec<u16> = vec![100, 200, 300, 400];
    let cw1 = rs.encode(&gf, &data);
    let cw2 = rs.encode(&gf, &data);
    assert_eq!(cw1, cw2, "encoding must be deterministic");
}

// ---------------------------------------------------------------------------
// Decode: No Errors
// ---------------------------------------------------------------------------

#[test]
fn decode_clean_codeword() {
    let (gf, rs) = make_rs(10);
    let data: Vec<u16> = vec![42, 100, 500, 728, 0, 1];
    let codeword = rs.encode(&gf, &data);
    let decoded = rs.decode(&gf, &codeword, &[]).unwrap();
    assert_eq!(decoded, data, "clean codeword must decode to original data");
}

// ---------------------------------------------------------------------------
// Error Correction
// ---------------------------------------------------------------------------

#[test]
fn correct_single_error() {
    let (gf, rs) = make_rs(10); // t = 5 corrections
    let data: Vec<u16> = vec![10, 20, 30, 40, 50, 60, 70, 80];
    let mut codeword = rs.encode(&gf, &data);

    // Corrupt one symbol
    codeword[3] = (codeword[3] + 1) % 729;

    let decoded = rs.decode(&gf, &codeword, &[]).unwrap();
    assert_eq!(decoded, data, "single error must be correctable");
}

#[test]
fn correct_max_errors() {
    let parity = 10; // t = 5 error corrections
    let (gf, rs) = make_rs(parity);
    let data: Vec<u16> = (0..20).map(|i| (i * 37) % 729).collect();
    let mut codeword = rs.encode(&gf, &data);

    // Corrupt exactly t = 5 symbols
    let error_positions = [0, 5, 10, 15, 25];
    for &pos in &error_positions {
        codeword[pos] = (codeword[pos] + 1) % 729;
    }

    let decoded = rs.decode(&gf, &codeword, &[]).unwrap();
    assert_eq!(decoded, data, "t={} errors must be correctable", parity / 2);
}

#[test]
fn too_many_errors_fails() {
    let parity = 10; // t = 5
    let (gf, rs) = make_rs(parity);
    let data: Vec<u16> = (0..20).map(|i| (i * 37) % 729).collect();
    let mut codeword = rs.encode(&gf, &data);

    // Corrupt t+1 = 6 symbols
    let error_positions = [0, 3, 7, 11, 15, 25];
    for &pos in &error_positions {
        codeword[pos] = (codeword[pos] + 1) % 729;
    }

    let result = rs.decode(&gf, &codeword, &[]);
    assert!(result.is_err(), "t+1 errors must be unrecoverable");
}

// ---------------------------------------------------------------------------
// Erasure Correction
// ---------------------------------------------------------------------------

#[test]
fn correct_erasures() {
    let parity = 10; // Can correct up to 2t = 10 erasures
    let (gf, rs) = make_rs(parity);
    let data: Vec<u16> = (0..20).map(|i| (i * 53) % 729).collect();
    let mut codeword = rs.encode(&gf, &data);

    // Erase 10 positions (set to 0, provide erasure locations)
    let erasure_positions: Vec<usize> = vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18];
    for &pos in &erasure_positions {
        codeword[pos] = 0; // value doesn't matter since position is known
    }

    let decoded = rs.decode(&gf, &codeword, &erasure_positions).unwrap();
    assert_eq!(decoded, data, "2t erasures must be recoverable");
}

#[test]
fn too_many_erasures_fails() {
    let parity = 10;
    let (gf, rs) = make_rs(parity);
    let data: Vec<u16> = (0..20).map(|i| (i * 53) % 729).collect();
    let mut codeword = rs.encode(&gf, &data);

    // Erase 2t + 1 = 11 positions
    let erasure_positions: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    for &pos in &erasure_positions {
        codeword[pos] = 0;
    }

    let result = rs.decode(&gf, &codeword, &erasure_positions);
    assert!(result.is_err(), "2t+1 erasures must be unrecoverable");
}

// ---------------------------------------------------------------------------
// Mixed Errors + Erasures: 2e + E ≤ 2t
// ---------------------------------------------------------------------------

#[test]
fn correct_mixed_errors_and_erasures() {
    let parity = 10; // 2t = 10
    let (gf, rs) = make_rs(parity);
    let data: Vec<u16> = (0..20).map(|i| (i * 71) % 729).collect();
    let mut codeword = rs.encode(&gf, &data);

    // 3 erasures + 3 errors = 2*3 + 3 = 9 ≤ 10
    let erasure_positions: Vec<usize> = vec![0, 5, 10];
    for &pos in &erasure_positions {
        codeword[pos] = 0;
    }
    let error_positions = [15, 20, 25]; // unknown error positions
    for &pos in &error_positions {
        codeword[pos] = (codeword[pos] + 2) % 729;
    }

    let decoded = rs.decode(&gf, &codeword, &erasure_positions).unwrap();
    assert_eq!(
        decoded, data,
        "2e + E = {} ≤ 2t = {} must be recoverable",
        2 * error_positions.len() + erasure_positions.len(),
        parity
    );
}

// ---------------------------------------------------------------------------
// Edge Cases
// ---------------------------------------------------------------------------

#[test]
fn encode_empty_data() {
    let (gf, rs) = make_rs(4);
    let data: Vec<u16> = vec![];
    let codeword = rs.encode(&gf, &data);
    assert_eq!(codeword.len(), 4, "empty data → only parity symbols");
    // Should decode back to empty
    let decoded = rs.decode(&gf, &codeword, &[]).unwrap();
    assert!(decoded.is_empty());
}

#[test]
fn encode_single_symbol() {
    let (gf, rs) = make_rs(4);
    let data: Vec<u16> = vec![42];
    let codeword = rs.encode(&gf, &data);
    assert_eq!(codeword.len(), 5);
    let decoded = rs.decode(&gf, &codeword, &[]).unwrap();
    assert_eq!(decoded, data);
}

#[test]
fn all_zero_data() {
    let (gf, rs) = make_rs(6);
    let data: Vec<u16> = vec![0, 0, 0, 0, 0];
    let codeword = rs.encode(&gf, &data);
    // All-zero message should produce all-zero parity
    assert!(codeword.iter().all(|&s| s == 0), "all-zero message → all-zero codeword");
    let decoded = rs.decode(&gf, &codeword, &[]).unwrap();
    assert_eq!(decoded, data);
}

#[test]
fn large_block_roundtrip() {
    let parity = 20;
    let (gf, rs) = make_rs(parity);
    let data: Vec<u16> = (0..700).map(|i| (i * 17 + 3) % 729).collect();
    let codeword = rs.encode(&gf, &data);
    assert_eq!(codeword.len(), 720);
    let decoded = rs.decode(&gf, &codeword, &[]).unwrap();
    assert_eq!(decoded, data);
}
