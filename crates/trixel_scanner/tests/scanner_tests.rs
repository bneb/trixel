//! Integration tests for trixel_scanner.
//!
//! These tests use the full encode pipeline (trixel_render + trixel_solver)
//! to generate a PNG, then decode it via `decode_png_bytes` to verify
//! the complete round-trip.

use trixel_core::{MockCodec, TernaryCodec};
use trixel_render::{AnchorRenderer, Renderer};
use trixel_solver::{GaussSolver, MatrixSolver};
use trixel_scanner::decode_png_bytes;

/// Encode "https://test.io" → PNG → decode via scanner → assert exact match.
#[test]
fn scanner_round_trip_plain() {
    let url = "https://test.io";

    // Encode
    let data_trits = MockCodec::encode_bytes(url.as_bytes()).unwrap();
    let side = 20;
    let module_size = 10u32;
    let matrix = GaussSolver::resolve_matrix(&data_trits, side, &[]).unwrap();
    let color_rgb = [128u8, 128, 128];
    let img = AnchorRenderer::render_png(&matrix, module_size, color_rgb).unwrap();

    // Serialize to PNG bytes in memory
    let mut png_bytes: Vec<u8> = Vec::new();
    img.write_to(
        &mut std::io::Cursor::new(&mut png_bytes),
        image::ImageFormat::Png,
    )
    .unwrap();

    // Decode via scanner
    let decoded = decode_png_bytes(&png_bytes, module_size)
        .expect("scanner should decode the PNG");

    assert_eq!(decoded, url, "round-trip must recover the original URL");
}

/// Encode with font constraints → decode via scanner → assert URL matches.
#[test]
fn scanner_round_trip_with_text() {
    use trixel_render::{FontEngine, TrixelFont};

    let url = "https://miserable.work";
    let text = "WORK";

    let data_trits = MockCodec::encode_bytes(url.as_bytes()).unwrap();
    let side = 29;
    let module_size = 10u32;

    let text_constraints = TrixelFont::string_to_constraints(text, 4, 4);
    let matrix = GaussSolver::resolve_matrix(&data_trits, side, &text_constraints).unwrap();

    let color_rgb = [0x75u8, 0xB3, 0xB8];
    let img = AnchorRenderer::render_png(&matrix, module_size, color_rgb).unwrap();

    let mut png_bytes: Vec<u8> = Vec::new();
    img.write_to(
        &mut std::io::Cursor::new(&mut png_bytes),
        image::ImageFormat::Png,
    )
    .unwrap();

    let decoded = decode_png_bytes(&png_bytes, module_size)
        .expect("scanner should decode PNG with embedded text");

    assert_eq!(decoded, url, "round-trip must recover URL despite font constraints");
}

/// decode_png_bytes with garbage data should return an error, not panic.
#[test]
fn scanner_rejects_garbage() {
    let garbage = vec![0u8, 1, 2, 3, 4, 5];
    let result = decode_png_bytes(&garbage, 10);
    assert!(result.is_err(), "garbage bytes should produce an error");
}
