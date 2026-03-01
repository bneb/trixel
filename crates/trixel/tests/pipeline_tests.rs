use trixel_core::{MockCodec, MockEcc, TernaryCodec, ErrorCorrection};
use trixel_solver::{MockSolver, MatrixSolver};
use trixel_render::{MockRenderer, Renderer};
use trixel_cv::{MockVision, VisionPipeline};

fn min_square_side(n: usize) -> usize {
    let s = (n as f64).sqrt().ceil() as usize;
    if s * s >= n { s } else { s + 1 }
}

#[test]
fn full_pipeline_roundtrip() {
    let data = b"Phase 1 works!";
    let module_size: u32 = 10;
    let accent = [128u8, 128, 128]; // mid-gray → luma 128, inside State 1 band

    // Encode pipeline
    let trits = MockCodec::encode_bytes(data).unwrap();
    eprintln!("trits.len() = {}", trits.len());

    let with_parity = MockEcc::apply_parity(&trits, 0.3).unwrap();
    eprintln!("with_parity.len() = {}", with_parity.len());

    let side = min_square_side(with_parity.len());
    eprintln!("matrix side = {side}");

    let matrix = MockSolver::resolve_matrix(&with_parity, side, &[]).unwrap();
    eprintln!("matrix = {}x{}", matrix.width, matrix.height);

    let img = MockRenderer::render_png(&matrix, module_size, accent).unwrap();
    eprintln!("image = {}x{}", img.width(), img.height());

    // Decode pipeline
    let extracted = MockVision::extract_matrix(&img, module_size).unwrap();
    eprintln!("extracted = {}x{}, data len = {}", extracted.width, extracted.height, extracted.data.len());

    let clean = MockEcc::correct_errors(&extracted.data).unwrap();
    eprintln!("clean.len() = {}", clean.len());

    let decoded = MockCodec::decode_trits(&clean).unwrap();
    let text = String::from_utf8_lossy(&decoded);
    eprintln!("decoded = '{text}'");

    assert_eq!(decoded, data, "round-trip failed");
}
