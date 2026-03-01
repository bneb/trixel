use trixel_core::{MockCodec, RsEcc, TernaryCodec, ErrorCorrection};
use trixel_solver::{AnchorSolver, MatrixSolver};
use trixel_solver::anchor::{ANCHOR_SIZE, is_in_anchor_region};
use trixel_render::{AnchorRenderer, Renderer};
use trixel_cv::{AnchorVision, VisionPipeline};

fn min_square_side(n: usize) -> usize {
    let anchor_cells = 4 * ANCHOR_SIZE * ANCHOR_SIZE;
    let total = n + anchor_cells;
    let s = (total as f64).sqrt().ceil() as usize;
    let min = ANCHOR_SIZE * 2;
    let side = if s * s >= total { s } else { s + 1 };
    side.max(min)
}

#[test]
fn anchor_pipeline_roundtrip() {
    let data = b"Anchor pipeline!";
    let module_size: u32 = 5;
    let accent = [128u8, 128, 128];

    // Encode
    let trits = MockCodec::encode_bytes(data).unwrap();
    let with_parity = RsEcc::apply_parity(&trits, 0.3).unwrap();
    let side = min_square_side(with_parity.len());
    let matrix = AnchorSolver::resolve_matrix(&with_parity, side, &[]).unwrap();
    let img = AnchorRenderer::render_png(&matrix, module_size, accent).unwrap();

    // Decode
    let extracted = AnchorVision::extract_matrix(&img, module_size).unwrap();

    // Extract payload (skip anchors)
    let n = extracted.width;
    let mut payload = Vec::new();
    for y in 0..extracted.height {
        for x in 0..extracted.width {
            if !is_in_anchor_region(x, y, n) {
                payload.push(extracted.get(x, y).unwrap_or(0));
            }
        }
    }

    let clean = RsEcc::correct_errors(&payload).unwrap();
    let decoded = MockCodec::decode_trits(&clean).unwrap();
    assert_eq!(decoded, data, "anchor pipeline round-trip failed");
}

#[test]
fn anchor_pipeline_with_vivid_color() {
    let data = b"FF00FF test";
    let module_size: u32 = 4;
    let accent = [255u8, 0, 255]; // Magenta — luma ~73, normally an erasure

    let trits = MockCodec::encode_bytes(data).unwrap();
    let with_parity = RsEcc::apply_parity(&trits, 0.3).unwrap();
    let side = min_square_side(with_parity.len());
    let matrix = AnchorSolver::resolve_matrix(&with_parity, side, &[]).unwrap();
    let img = AnchorRenderer::render_png(&matrix, module_size, accent).unwrap();

    let extracted = AnchorVision::extract_matrix(&img, module_size).unwrap();
    let n = extracted.width;
    let mut payload = Vec::new();
    for y in 0..extracted.height {
        for x in 0..extracted.width {
            if !is_in_anchor_region(x, y, n) {
                payload.push(extracted.get(x, y).unwrap_or(0));
            }
        }
    }

    let clean = RsEcc::correct_errors(&payload).unwrap();
    let decoded = MockCodec::decode_trits(&clean).unwrap();
    assert_eq!(decoded, data, "magenta color pipeline round-trip failed");
}
