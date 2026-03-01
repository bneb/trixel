use ternac_core::gf3::{self, GF3, TRITS_PER_SYMBOL};
use ternac_core::{RsEcc, ErrorCorrection, LENGTH_PREFIX_TRITS};
use ternac_solver::gauss_solver::{compute_transform_matrix, build_parity_check_matrix, grid_to_flat_coords};
use ternac_solver::{GaussSolver, MatrixSolver};
use ternac_solver::anchor::{self, is_in_anchor_region};
use ternac_render::{FontEngine, TernacFont};

// ---------------------------------------------------------------------------
// Core: Valid Codeword with Zero Syndromes
// ---------------------------------------------------------------------------

#[test]
fn gauss_solver_produces_valid_codeword() {
    let gf = GF3::new();
    let n = 20;
    let data_trits = vec![1u8, 2, 0, 1, 1, 0, 2, 1, 0, 2, 1, 0];

    let matrix = GaussSolver::resolve_matrix(&data_trits, n, &[]).unwrap();

    // Extract all non-anchor trits using the SAME mapping the solver uses
    let coords = grid_to_flat_coords(n);
    let flat: Vec<u8> = coords.iter().map(|&(x, y)| matrix.get(x, y).unwrap()).collect();

    // Verify length prefix
    let codeword_trits_len = ternac_core::decode_length(&flat[..LENGTH_PREFIX_TRITS]);
    assert!(codeword_trits_len > 0, "length prefix should be non-zero");
    assert!(codeword_trits_len <= flat.len() - LENGTH_PREFIX_TRITS,
        "codeword length {} exceeds available space {}", codeword_trits_len, flat.len() - LENGTH_PREFIX_TRITS);

    // Extract codeword, convert to symbols, verify syndromes
    let codeword_start = LENGTH_PREFIX_TRITS;
    let codeword_end = codeword_start + codeword_trits_len;
    let codeword_flat = &flat[codeword_start..codeword_end];

    let symbols: Vec<u16> = codeword_flat
        .chunks(TRITS_PER_SYMBOL)
        .map(|chunk| gf3::trits_to_symbol(chunk))
        .collect();

    // Find parity count from header
    let mut found_pc = None;
    for pc in (2..=symbols.len() / 2).step_by(2) {
        if pc + 2 < symbols.len() && symbols[pc + 2] as usize == pc {
            found_pc = Some(pc);
            break;
        }
    }
    let pc = found_pc.expect("should find valid parity count in header");

    let h = build_parity_check_matrix(&gf, symbols.len(), pc);
    for (i, row) in h.iter().enumerate() {
        let mut syndrome = 0u16;
        for (j, &h_ij) in row.iter().enumerate() {
            syndrome = gf.add(syndrome, gf.mul(h_ij, symbols[j]));
        }
        assert_eq!(syndrome, 0, "syndrome {i} should be 0 for Gauss-solved codeword");
    }
}

// ---------------------------------------------------------------------------
// Round-Trip Through RS Decoder
// ---------------------------------------------------------------------------

#[test]
fn gauss_solver_round_trips_through_rs_decoder() {
    let n = 20;
    let data_trits = vec![1u8, 2, 0, 1, 1, 0];

    let matrix = GaussSolver::resolve_matrix(&data_trits, n, &[]).unwrap();

    let coords = grid_to_flat_coords(n);
    let flat: Vec<u8> = coords.iter().map(|&(x, y)| matrix.get(x, y).unwrap()).collect();

    let recovered = RsEcc::correct_errors(&flat).expect("RS decoder should recover data");
    assert_eq!(&recovered[..data_trits.len()], &data_trits[..],
        "round-trip through RS decoder should recover original data");
}

// ---------------------------------------------------------------------------
// Font Constraints Are Respected
// ---------------------------------------------------------------------------

#[test]
fn gauss_solver_respects_font_constraints() {
    let n = 20;
    let constraints = TernacFont::string_to_constraints("HI", 4, 4);
    let data_trits = vec![1u8; 30];

    let matrix = GaussSolver::resolve_matrix(&data_trits, n, &constraints).unwrap();

    for c in &constraints {
        let actual = matrix.get(c.x, c.y).unwrap();
        assert_eq!(actual, c.required_state,
            "font constraint at ({},{}) should be {} got {}",
            c.x, c.y, c.required_state, actual);
    }
}

// ---------------------------------------------------------------------------
// Font + Decode Round-Trip
// ---------------------------------------------------------------------------

#[test]
fn gauss_solver_with_font_still_decodes() {
    let n = 20;
    let constraints = TernacFont::string_to_constraints("HI", 4, 4);

    // Use a realistic payload with enough trits to avoid RS header collisions.
    // "https://test.io" = 16 bytes = 96 trits — enough to be collision-resistant
    // in a 20×20 grid (400 total cells, ~340 non-anchor).
    use ternac_core::{TernaryCodec, MockCodec};
    let payload = b"https://test.io";
    let data_trits = MockCodec::encode_bytes(payload).unwrap();

    let matrix = GaussSolver::resolve_matrix(&data_trits, n, &constraints).unwrap();

    // Verify font constraints are respected in the solved matrix
    for c in &constraints {
        let actual = matrix.get(c.x, c.y).unwrap();
        assert_eq!(actual, c.required_state,
            "font constraint at ({},{}) should be {} got {}",
            c.x, c.y, c.required_state, actual);
    }

    // Extract flat trit stream (same path as CLI decoder)
    let coords = grid_to_flat_coords(n);
    let flat: Vec<u8> = coords.iter().map(|&(x, y)| matrix.get(x, y).unwrap()).collect();

    // RS decode must recover the exact original data trits
    let recovered = RsEcc::correct_errors(&flat)
        .expect("RS should decode despite font constraints");
    assert_eq!(&recovered[..data_trits.len()], &data_trits[..],
        "RS must recover exact data trits");

    // Full codec round-trip: trits → bytes must equal original payload
    let decoded = MockCodec::decode_trits(&recovered)
        .expect("decoded trits should be valid bytes");
    assert_eq!(&decoded[..payload.len()], payload,
        "full codec round-trip must recover original payload");
}

// ---------------------------------------------------------------------------
// Performance: Must Complete Instantly
// ---------------------------------------------------------------------------

#[test]
fn gauss_solver_is_fast() {
    let n = 20;
    let data_trits = vec![1u8, 0, 2, 1, 0, 2];

    let start = std::time::Instant::now();
    let _matrix = GaussSolver::resolve_matrix(&data_trits, n, &[]).unwrap();
    let elapsed = start.elapsed();

    // Must complete in under 1 second (it should be < 10ms)
    assert!(elapsed.as_secs() < 1,
        "Gauss solver took {:?} — should be under 1 second", elapsed);
    eprintln!("Gauss solver completed in {:?}", elapsed);
}
