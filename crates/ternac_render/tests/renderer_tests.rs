use ternac_core::TritMatrix;
use ternac_render::{MockRenderer, Renderer};

#[test]
fn render_dimensions() {
    let matrix = TritMatrix::zeros(5, 4);
    let img = MockRenderer::render_png(&matrix, 10, [255, 0, 255]).unwrap();
    assert_eq!(img.width(), 50);
    assert_eq!(img.height(), 40);
}

#[test]
fn render_colors() {
    let mut matrix = TritMatrix::zeros(3, 1);
    matrix.set(0, 0, 0); // black (low luminance)
    matrix.set(1, 0, 1); // accent (mid luminance)
    matrix.set(2, 0, 2); // white (high luminance)

    let accent = [255u8, 0, 128];
    let img = MockRenderer::render_png(&matrix, 1, accent).unwrap();
    let rgb = img.to_rgb8();
    assert_eq!(rgb.get_pixel(0, 0).0, [0, 0, 0]);
    assert_eq!(rgb.get_pixel(1, 0).0, accent);
    assert_eq!(rgb.get_pixel(2, 0).0, [255, 255, 255]);
}

#[test]
fn render_empty_matrix() {
    let matrix = TritMatrix::zeros(0, 0);
    assert!(MockRenderer::render_png(&matrix, 10, [0, 0, 0]).is_err());
}
