use ternac_core::gf3::{self, GF3, TRITS_PER_SYMBOL};
use ternac_core::{MockCodec, RsEcc, TernaryCodec, ErrorCorrection, LENGTH_PREFIX_TRITS};
use ternac_solver::{GaussSolver, MatrixSolver};
use ternac_solver::gauss_solver::{build_parity_check_matrix, grid_to_flat_coords};
use ternac_solver::anchor::{ANCHOR_SIZE, is_in_anchor_region};
use ternac_render::{AnchorRenderer, FontEngine, Renderer, TernacFont};
use ternac_cv::{AnchorVision, VisionPipeline};

const PAYLOAD: &[u8] = b"https://miserable.work";
const OVERLAY_TEXT: &str = "SIGH";
const ACCENT: [u8; 3] = [0x75, 0xB3, 0xB8]; // #75B3B8 teal
const MODULE_SIZE: u32 = 8;

fn min_square_side(n: usize) -> usize {
    let anchor_cells = 4 * ANCHOR_SIZE * ANCHOR_SIZE;
    let total = n * 3 + anchor_cells;
    let s = (total as f64).sqrt().ceil() as usize;
    let min = ANCHOR_SIZE * 2;
    let side = if s * s >= total { s } else { s + 1 };
    side.max(min)
}

// ---------------------------------------------------------------------------
// E2E: Gaussian Solver forges a valid RS codeword with "SIGH" constraints
// ---------------------------------------------------------------------------

#[test]
fn miserable_work_encodes_with_sigh_constraints() {
    let trits = MockCodec::encode_bytes(PAYLOAD).unwrap();
    let constraints = TernacFont::string_to_constraints(OVERLAY_TEXT, 5, 5);

    eprintln!("Payload: {} bytes = {} trits", PAYLOAD.len(), trits.len());
    eprintln!("Font constraints: {} cells for '{}'", constraints.len(), OVERLAY_TEXT);

    let mut side = min_square_side(trits.len() + constraints.len());
    loop {
        let any_conflict = constraints.iter().any(|c| is_in_anchor_region(c.x, c.y, side));
        if !any_conflict { break; }
        side += 1;
    }
    side = side.max(20);

    let start = std::time::Instant::now();
    let matrix = GaussSolver::resolve_matrix(&trits, side, &constraints).unwrap();
    let elapsed = start.elapsed();

    eprintln!("Matrix: {}×{}", matrix.width, matrix.height);
    eprintln!("Gauss solve time: {:?}", elapsed);
    assert!(elapsed.as_secs() < 2, "solve took {:?}", elapsed);

    // Verify every font constraint is honored
    for c in &constraints {
        let actual = matrix.get(c.x, c.y).unwrap();
        assert_eq!(actual, c.required_state,
            "SIGH constraint at ({},{}) should be {} got {}",
            c.x, c.y, c.required_state, actual);
    }
}

// ---------------------------------------------------------------------------
// E2E: Zero-syndrome verification on the RS codeword
// ---------------------------------------------------------------------------

#[test]
fn miserable_work_codeword_has_zero_syndromes() {
    let gf = GF3::new();
    let trits = MockCodec::encode_bytes(PAYLOAD).unwrap();
    let constraints = TernacFont::string_to_constraints(OVERLAY_TEXT, 5, 5);

    let mut side = min_square_side(trits.len() + constraints.len());
    loop {
        let any_conflict = constraints.iter().any(|c| is_in_anchor_region(c.x, c.y, side));
        if !any_conflict { break; }
        side += 1;
    }
    side = side.max(20);

    let matrix = GaussSolver::resolve_matrix(&trits, side, &constraints).unwrap();

    // Extract flat trits
    let coords = grid_to_flat_coords(side);
    let flat: Vec<u8> = coords.iter().map(|&(x, y)| matrix.get(x, y).unwrap()).collect();

    // Read length prefix, extract codeword symbols
    let cw_len = ternac_core::decode_length(&flat[..LENGTH_PREFIX_TRITS]);
    let cw_trits = &flat[LENGTH_PREFIX_TRITS..LENGTH_PREFIX_TRITS + cw_len];
    let symbols: Vec<u16> = cw_trits.chunks(TRITS_PER_SYMBOL)
        .map(|c| gf3::trits_to_symbol(c))
        .collect();

    // The parity count is stored in the third header symbol (index 2) of the
    // message region. The codeword layout is: [parity symbols...][offset...][len_lo, len_hi, pc, data...]
    // We can recover it from RsEcc::correct_errors which strips the header.
    // For syndrome verification, we just need to check that H·c = 0.
    // Use a range of plausible parity counts and find the one that gives zero syndromes.
    let mut pc = 0;
    for candidate in (2..=symbols.len() / 2).step_by(2) {
        let h = build_parity_check_matrix(&gf, symbols.len(), candidate);
        let mut all_zero = true;
        for row in &h {
            let mut s = 0u16;
            for (j, &h_ij) in row.iter().enumerate() {
                s = gf.add(s, gf.mul(h_ij, symbols[j]));
            }
            if s != 0 { all_zero = false; break; }
        }
        if all_zero {
            pc = candidate;
            break;
        }
    };
    assert!(pc > 0, "no valid parity count found");
    eprintln!("Codeword: {} symbols, parity_count = {}", symbols.len(), pc);

    // Verify ALL syndromes are zero
    let h = build_parity_check_matrix(&gf, symbols.len(), pc);
    for (i, row) in h.iter().enumerate() {
        let mut syndrome = 0u16;
        for (j, &h_ij) in row.iter().enumerate() {
            syndrome = gf.add(syndrome, gf.mul(h_ij, symbols[j]));
        }
        assert_eq!(syndrome, 0, "syndrome {} is nonzero for miserable.work codeword", i);
    }
    eprintln!("All {} syndromes verified zero ✓", h.len());
}

// ---------------------------------------------------------------------------
// E2E: Full digital round-trip (encode → render → extract → decode)
// ---------------------------------------------------------------------------

#[test]
fn miserable_work_digital_roundtrip() {
    let trits = MockCodec::encode_bytes(PAYLOAD).unwrap();
    let constraints = TernacFont::string_to_constraints(OVERLAY_TEXT, 5, 5);

    let mut side = min_square_side(trits.len() + constraints.len());
    loop {
        let any_conflict = constraints.iter().any(|c| is_in_anchor_region(c.x, c.y, side));
        if !any_conflict { break; }
        side += 1;
    }
    side = side.max(20);

    // Encode + Render
    let matrix = GaussSolver::resolve_matrix(&trits, side, &constraints).unwrap();
    let img = AnchorRenderer::render_png(&matrix, MODULE_SIZE, ACCENT).unwrap();

    // Save for visual inspection
    img.save("/tmp/miserable_matrix_test.png").ok();

    // Extract + Decode (full vision pipeline)
    let extracted = AnchorVision::extract_matrix(&img, MODULE_SIZE).unwrap();
    let mut payload = Vec::new();
    for y in 0..extracted.height {
        for x in 0..extracted.width {
            if !is_in_anchor_region(x, y, extracted.width) {
                payload.push(extracted.get(x, y).unwrap_or(0));
            }
        }
    }

    let clean = RsEcc::correct_errors(&payload).expect("RS decoder must recover miserable.work");
    let decoded = MockCodec::decode_trits(&clean).unwrap();
    let text = String::from_utf8_lossy(&decoded);

    eprintln!("Decoded: '{}'", text);
    assert_eq!(decoded, PAYLOAD, "digital round-trip must recover https://miserable.work");
}
