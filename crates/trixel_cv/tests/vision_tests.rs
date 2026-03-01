use image::{DynamicImage, ImageBuffer, RgbImage};
use trixel_core::TritMatrix;
use trixel_cv::{MockVision, VisionPipeline};

/// Build a synthetic RGB image from a trit matrix using known luminance values
/// that land squarely inside the default `LuminanceBands`.
fn synth_image(matrix: &TritMatrix, module_size: u32) -> DynamicImage {
    let w = matrix.width as u32 * module_size;
    let h = matrix.height as u32 * module_size;
    let mut img: RgbImage = ImageBuffer::new(w, h);
    for gy in 0..matrix.height {
        for gx in 0..matrix.width {
            let trit = matrix.get(gx, gy).unwrap();
            let lum: u8 = match trit {
                0 => 20,   // inside [0, 51] → State 0
                1 => 128,  // inside [102, 152] → State 1
                2 => 230,  // inside [204, 255] → State 2
                _ => 128,
            };
            let px_x = gx as u32 * module_size;
            let px_y = gy as u32 * module_size;
            for dy in 0..module_size {
                for dx in 0..module_size {
                    img.put_pixel(px_x + dx, px_y + dy, image::Rgb([lum, lum, lum]));
                }
            }
        }
    }
    DynamicImage::ImageRgb8(img)
}

#[test]
fn vision_roundtrip() {
    let mut original = TritMatrix::zeros(4, 4);
    original.set(0, 0, 0);
    original.set(1, 0, 1);
    original.set(2, 0, 2);
    original.set(3, 0, 1);
    original.set(0, 1, 2);
    original.set(1, 1, 0);

    let img = synth_image(&original, 10);
    let extracted = MockVision::extract_matrix(&img, 10).unwrap();
    assert_eq!(extracted, original);
}

#[test]
fn vision_bad_dimensions() {
    let img = DynamicImage::ImageRgb8(ImageBuffer::new(15, 15));
    assert!(MockVision::extract_matrix(&img, 4).is_err());
}
