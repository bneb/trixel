use ternac_core::gf3::{self, GF3, TRITS_PER_SYMBOL};
use ternac_solver::z3_solver::{compute_transform_matrix, build_parity_check_matrix};
use ternac_solver::{Z3Solver, MatrixSolver, ConstraintMask};
use ternac_solver::anchor::is_in_anchor_region;
use ternac_render::{FontEngine, TernacFont};

// ---------------------------------------------------------------------------
// Transform Matrix Tests
// ---------------------------------------------------------------------------

#[test]
fn transform_matrix_identity() {
    let gf = GF3::new();
    // M for k=1 should be identity (multiply by 1 = no-op)
    let m = compute_transform_matrix(&gf, 1);
    for row in 0..6 {
        for col in 0..6 {
            let expected = if row == col { 1 } else { 0 };
            assert_eq!(m[row][col], expected,
                "identity matrix M[{row}][{col}] should be {expected}, got {}", m[row][col]);
        }
    }
}

#[test]
fn transform_matrix_zero() {
    let gf = GF3::new();
    // M for k=0 should be all zeros
    let m = compute_transform_matrix(&gf, 0);
    for row in 0..6 {
        for col in 0..6 {
            assert_eq!(m[row][col], 0, "zero matrix M[{row}][{col}]");
        }
    }
}

#[test]
fn transform_matrix_mul_consistent() {
    let gf = GF3::new();
    // For various constants k, verify: M(k) · trits(b) == trits(k*b)
    let test_constants = [2, 3, 42, 100, 728];
    let test_values = [1, 5, 27, 100, 500, 728];

    for &k in &test_constants {
        let m = compute_transform_matrix(&gf, k);
        for &b in &test_values {
            // Expected: GF3 multiply
            let expected_product = gf.mul(k, b);
            let expected_trits = gf3::symbol_to_trits(expected_product);

            // Actual: matrix multiply on trits of b
            let b_trits = gf3::symbol_to_trits(b);
            let mut actual_trits = [0u64; 6];
            for row in 0..6 {
                let mut sum = 0u64;
                for col in 0..6 {
                    sum += m[row][col] * b_trits[col] as u64;
                }
                actual_trits[row] = sum % 3;
            }

            for i in 0..6 {
                assert_eq!(actual_trits[i], expected_trits[i] as u64,
                    "M({k}) · trits({b})[{i}]: expected {}, got {}",
                    expected_trits[i], actual_trits[i]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Parity-Check Matrix Tests
// ---------------------------------------------------------------------------

#[test]
fn parity_check_matrix_dimensions() {
    let gf = GF3::new();
    let parity_count = 4;
    let codeword_len = 10;
    let h = build_parity_check_matrix(&gf, codeword_len, parity_count);
    assert_eq!(h.len(), parity_count, "H should have {parity_count} rows");
    for (i, row) in h.iter().enumerate() {
        assert_eq!(row.len(), codeword_len, "H row {i} should have {codeword_len} columns");
    }
}

#[test]
fn parity_check_validates_known_codeword() {
    let gf = GF3::new();
    let parity_count = 4;
    let data = vec![1u16, 5, 27, 100, 500, 42];
    let rs = ternac_core::rs::ReedSolomon::new(&gf, parity_count);
    let codeword = rs.encode(&gf, &data);
    let n = codeword.len();

    let h = build_parity_check_matrix(&gf, n, parity_count);

    // H · c^T should be all zeros for a valid codeword
    for (i, row) in h.iter().enumerate() {
        let mut syndrome = 0u16;
        for (j, &h_ij) in row.iter().enumerate() {
            syndrome = gf.add(syndrome, gf.mul(h_ij, codeword[j]));
        }
        assert_eq!(syndrome, 0, "syndrome {i} should be 0 for valid codeword");
    }
}

// ---------------------------------------------------------------------------
// Z3 Solver Tests
// ---------------------------------------------------------------------------

#[test]
fn z3_solver_produces_valid_codeword() {
    let gf = GF3::new();
    let n = 10;
    let data_trits = vec![1u8, 2, 0, 1, 1, 0, 2, 1, 0, 2, 1, 0];
    let matrix = Z3Solver::resolve_matrix(&data_trits, n, &[]).unwrap();

    // Extract all trits in row-major order (skip anchors)
    let mut flat: Vec<u8> = Vec::new();
    for y in 0..n {
        for x in 0..n {
            if !is_in_anchor_region(x, y, n) {
                flat.push(matrix.get(x, y).unwrap());
            }
        }
    }

    // Group into symbols and verify syndromes
    let symbols: Vec<u16> = flat
        .chunks(TRITS_PER_SYMBOL)
        .map(|chunk| gf3::trits_to_symbol(chunk))
        .collect();

    let parity_count = (symbols.len() as f32 * 0.3).ceil() as usize;
    let h = build_parity_check_matrix(&gf, symbols.len(), parity_count);

    for (i, row) in h.iter().enumerate() {
        let mut syndrome = 0u16;
        for (j, &h_ij) in row.iter().enumerate() {
            syndrome = gf.add(syndrome, gf.mul(h_ij, symbols[j]));
        }
        assert_eq!(syndrome, 0, "syndrome {i} should be 0 for Z3-solved codeword");
    }
}

#[test]
fn z3_solver_respects_font_constraints() {
    let n = 20;
    let constraints = TernacFont::string_to_constraints("HI", 4, 4);
    let data_trits = vec![1u8; 30];

    let matrix = Z3Solver::resolve_matrix(&data_trits, n, &constraints).unwrap();

    // Verify all font constraints are honored
    for c in &constraints {
        let actual = matrix.get(c.x, c.y).unwrap();
        assert_eq!(actual, c.required_state,
            "font constraint at ({},{}) should be {} got {}",
            c.x, c.y, c.required_state, actual);
    }
}
