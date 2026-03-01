use ternac_core::{MockCodec, RsEcc, TernaryCodec, ErrorCorrection};
use ternac_solver::{GaussSolver, MatrixSolver};
use ternac_solver::anchor::{ANCHOR_SIZE, is_in_anchor_region};
use ternac_render::{AnchorRenderer, FontEngine, Renderer, TernacFont};
use ternac_cv::{AnchorVision, VisionPipeline};

fn min_square_side(n: usize) -> usize {
    let anchor_cells = 4 * ANCHOR_SIZE * ANCHOR_SIZE;
    let total = n + anchor_cells;
    let s = (total as f64).sqrt().ceil() as usize;
    let min = ANCHOR_SIZE * 2;
    let side = if s * s >= total { s } else { s + 1 };
    side.max(min)
}

fn extract_payload(matrix: &ternac_core::TritMatrix) -> Vec<u8> {
    let n = matrix.width;
    let mut payload = Vec::new();
    for y in 0..matrix.height {
        for x in 0..matrix.width {
            if !is_in_anchor_region(x, y, n) {
                payload.push(matrix.get(x, y).unwrap_or(0));
            }
        }
    }
    payload
}

// ---------------------------------------------------------------------------
// GaussSolver: Encode → Decode Round-Trip (No Font)
// ---------------------------------------------------------------------------

#[test]
fn gauss_cli_roundtrip_no_font() {
    let data = b"HELLO";
    let module_size: u32 = 5;
    let accent = [128u8, 128, 128];

    // Encode: bytes → trits → GaussSolver → render
    let trits = MockCodec::encode_bytes(data).unwrap();
    let side = min_square_side(trits.len() * 3); // extra room for RS parity
    let matrix = GaussSolver::resolve_matrix(&trits, side, &[]).unwrap();
    let img = AnchorRenderer::render_png(&matrix, module_size, accent).unwrap();

    // Decode: image → extract → RS decode → trits → bytes
    let extracted = AnchorVision::extract_matrix(&img, module_size).unwrap();
    let payload = extract_payload(&extracted);
    let clean = RsEcc::correct_errors(&payload).unwrap();
    let decoded = MockCodec::decode_trits(&clean).unwrap();
    assert_eq!(decoded, data, "Gauss round-trip failed for {:?}", std::str::from_utf8(data));
}

// ---------------------------------------------------------------------------
// GaussSolver: Encode → Decode Round-Trip WITH Font
// ---------------------------------------------------------------------------

#[test]
fn gauss_cli_roundtrip_with_font() {
    let data = b"TEST";
    let module_size: u32 = 5;
    let accent = [128u8, 128, 128];

    let trits = MockCodec::encode_bytes(data).unwrap();
    let side = 20; // Large enough for font constraints to avoid anchors

    // Generate font constraints for "HI" at position (4,4)
    let constraints = TernacFont::string_to_constraints("HI", 4, 4);

    let matrix = GaussSolver::resolve_matrix(&trits, side, &constraints).unwrap();

    // Verify font constraints are honored in the matrix
    for c in &constraints {
        let actual = matrix.get(c.x, c.y).unwrap();
        assert_eq!(actual, c.required_state,
            "font constraint at ({},{}) should be {} got {}",
            c.x, c.y, c.required_state, actual);
    }

    let img = AnchorRenderer::render_png(&matrix, module_size, accent).unwrap();

    // Decode
    let extracted = AnchorVision::extract_matrix(&img, module_size).unwrap();
    let payload = extract_payload(&extracted);
    let clean = RsEcc::correct_errors(&payload).unwrap();
    let decoded = MockCodec::decode_trits(&clean).unwrap();
    assert_eq!(decoded, data, "Gauss+font round-trip failed");
}

// ---------------------------------------------------------------------------
// Performance: Full encode+decode pipeline under 2 seconds
// ---------------------------------------------------------------------------

#[test]
fn gauss_cli_performance() {
    let data = b"Performance test data 12345";

    let start = std::time::Instant::now();
    let trits = MockCodec::encode_bytes(data).unwrap();
    let side = min_square_side(trits.len() * 3);
    let matrix = GaussSolver::resolve_matrix(&trits, side, &[]).unwrap();
    let elapsed = start.elapsed();

    eprintln!("Gauss CLI encode completed in {:?}", elapsed);
    assert!(elapsed.as_secs() < 2, "pipeline took {:?}", elapsed);

    // Verify it's a valid codeword
    let payload = extract_payload(&matrix);
    let clean = RsEcc::correct_errors(&payload).unwrap();
    let decoded = MockCodec::decode_trits(&clean).unwrap();
    assert_eq!(decoded, data);
}
