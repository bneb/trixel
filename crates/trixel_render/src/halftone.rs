//! # Halftone Constraint Engine
//!
//! Generative optical art constraint builder. Converts an image into a base-3
//! constraint map for the Gaussian parity solver.
//!
//! Uses `imageproc` Sobel edge detection to identify high-frequency features.
//! High-frequency pixels are locked to their closest base-3 physical state
//! (0, 1, or 2). Low-frequency background pixels are set to `None` (free variables)
//! to be sacrificed to the solver for Reed-Solomon parity requirements.

use image::{DynamicImage, GenericImageView, Rgba, Luma};
use imageproc::contrast::equalize_histogram;
use imageproc::gradients::sobel_gradients;
use palette::{Srgb, Hsl, IntoColor};

/// Resolves an image into constrained and free pixels based on structural importance.
pub struct HalftoneEngine {
    /// RGB value used to represent State 0 (Dark)
    pub state_0_rgb: [u8; 3],
    /// RGB value used to represent State 1 (Accent)
    pub state_1_rgb: [u8; 3],
    /// RGB value used to represent State 2 (Light)
    pub state_2_rgb: [u8; 3],
}

impl HalftoneEngine {
    /// Ingests an image and returns an `NxN` grid of `Option<u8>` constraints.
    ///
    /// It ensures that exactly `required_free_trits` are set to `None` by
    /// sacrificing pixels with the lowest Sobel gradient magnitudes.
    pub fn image_to_constraints(
        &self,
        img: &DynamicImage,
        matrix_size: usize,
        required_free_trits: usize,
    ) -> Vec<Vec<Option<u8>>> {
        let total_pixels = matrix_size * matrix_size;
        
        // Safety: don't ask for more free trits than exist in the matrix
        let free_count = required_free_trits.min(total_pixels);

        // Step A: Resize the image
        let scaled = img.resize_exact(
            matrix_size as u32,
            matrix_size as u32,
            image::imageops::FilterType::Lanczos3,
        );

        // Step C: Convert to Luma8, equalize histogram, and run Sobel operator
        // Histogram equalization stretches the dynamic range of the grayscale
        // image, forcing soft gradients (faces, skin tones) to snap into hard,
        // detectable edges. Without this, the Sobel operator treats faces as
        // flat background noise and sacrifices them to the solver.
        let luma = equalize_histogram(&scaled.to_luma8());
        let gradients = sobel_gradients(&luma);

        // Step B & D: Quantize and rank pixel importance
        // Tuple: (x, y, state, magnitude)
        let mut pixels = Vec::with_capacity(total_pixels);

        for y in 0..matrix_size {
            for x in 0..matrix_size {
                let px = x as u32;
                let py = y as u32;
                
                let color = scaled.get_pixel(px, py);
                let state = self.quantize_pixel(color);
                
                // sobel_gradients outputs Luma<u16>
                let grad: Luma<u16> = *gradients.get_pixel(px, py);
                let magnitude = grad[0];
                
                pixels.push((x, y, state, magnitude));
            }
        }

        // Sort by gradient ascending (flattest/most boring regions first)
        // For stable deterministic behavior in tests, tie-break by y, then x
        pixels.sort_by_key(|p| (p.3, p.1, p.0));

        // Output matrix initialized to None
        let mut constraints = vec![vec![None; matrix_size]; matrix_size];

        // Step E & F: Assign constraints based on required sacrifice quota
        for (i, p) in pixels.into_iter().enumerate() {
            let (x, y, state, _magnitude) = p;
            
            // The first `free_count` pixels are the lowest magnitude.
            // They remain `None`. The rest get locked to their quantized state.
            if i >= free_count {
                constraints[y][x] = Some(state);
            }
        }

        constraints
    }

    /// Determines the closest State (0, 1, or 2) for an RGBA pixel using
    /// HSL Lightness bands:
    ///   State 0 (Dark):   L <= 0.35
    ///   State 1 (Middle): 0.35 < L <= 0.65
    ///   State 2 (Light):  L > 0.65
    pub fn quantize_pixel(&self, pixel: Rgba<u8>) -> u8 {
        let srgb = Srgb::new(
            pixel[0] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[2] as f32 / 255.0,
        );
        let hsl: Hsl = srgb.into_color();
        let l = hsl.lightness;

        if l <= 0.35 {
            0 // Dark
        } else if l <= 0.65 {
            1 // Middle
        } else {
            2 // Light
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{RgbImage, Rgb};

    // -----------------------------------------------------------------------
    // HSL Lightness-Band Quantization Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_quantize_pure_black_is_state_0() {
        let engine = HalftoneEngine {
            state_0_rgb: [0, 0, 0],
            state_1_rgb: [128, 128, 128],
            state_2_rgb: [255, 255, 255],
        };
        // Pure black → L=0.0 → State 0
        assert_eq!(engine.quantize_pixel(Rgba([0, 0, 0, 255])), 0);
    }

    #[test]
    fn test_quantize_pure_white_is_state_2() {
        let engine = HalftoneEngine {
            state_0_rgb: [0, 0, 0],
            state_1_rgb: [128, 128, 128],
            state_2_rgb: [255, 255, 255],
        };
        // Pure white → L=1.0 → State 2
        assert_eq!(engine.quantize_pixel(Rgba([255, 255, 255, 255])), 2);
    }

    #[test]
    fn test_quantize_mid_gray_is_state_1() {
        let engine = HalftoneEngine {
            state_0_rgb: [0, 0, 0],
            state_1_rgb: [128, 128, 128],
            state_2_rgb: [255, 255, 255],
        };
        // Medium gray → L≈0.50 → State 1
        assert_eq!(engine.quantize_pixel(Rgba([128, 128, 128, 255])), 1);
    }

    #[test]
    fn test_quantize_dark_navy_is_state_0() {
        let engine = HalftoneEngine {
            state_0_rgb: [0, 0, 0],
            state_1_rgb: [128, 128, 128],
            state_2_rgb: [255, 255, 255],
        };
        // Dark navy → L≈0.15 → State 0
        assert_eq!(engine.quantize_pixel(Rgba([20, 20, 60, 255])), 0);
    }

    #[test]
    fn test_quantize_bright_teal_is_state_1() {
        let engine = HalftoneEngine {
            state_0_rgb: [0, 0, 0],
            state_1_rgb: [128, 128, 128],
            state_2_rgb: [255, 255, 255],
        };
        // Teal #75B3B8 → L≈0.59 → State 1 (middle band)
        assert_eq!(engine.quantize_pixel(Rgba([0x75, 0xB3, 0xB8, 255])), 1);
    }

    #[test]
    fn test_quantize_light_pink_is_state_2() {
        let engine = HalftoneEngine {
            state_0_rgb: [0, 0, 0],
            state_1_rgb: [128, 128, 128],
            state_2_rgb: [255, 255, 255],
        };
        // Light pink → L≈0.88 → State 2
        assert_eq!(engine.quantize_pixel(Rgba([255, 200, 220, 255])), 2);
    }

    #[test]
    fn test_quantize_saturated_red_is_state_1() {
        let engine = HalftoneEngine {
            state_0_rgb: [0, 0, 0],
            state_1_rgb: [128, 128, 128],
            state_2_rgb: [255, 255, 255],
        };
        // Pure red #FF0000 → HSL(0°, 1.0, 0.5) → State 1
        assert_eq!(engine.quantize_pixel(Rgba([255, 0, 0, 255])), 1);
    }

    #[test]
    fn test_quantize_dark_brown_is_state_0() {
        let engine = HalftoneEngine {
            state_0_rgb: [0, 0, 0],
            state_1_rgb: [128, 128, 128],
            state_2_rgb: [255, 255, 255],
        };
        // Dark brown #3A1F00 → L≈0.11 → State 0
        assert_eq!(engine.quantize_pixel(Rgba([0x3A, 0x1F, 0x00, 255])), 0);
    }

    #[test]
    fn test_quantize_boundary_at_035() {
        let engine = HalftoneEngine {
            state_0_rgb: [0, 0, 0],
            state_1_rgb: [128, 128, 128],
            state_2_rgb: [255, 255, 255],
        };
        // Gray at exact boundary: L=0.35 → 0.35*255≈89.25 → Rgb(89,89,89)
        // L ≈ 0.349 → State 0 (just under boundary)
        assert_eq!(engine.quantize_pixel(Rgba([89, 89, 89, 255])), 0);
    }

    #[test]
    fn test_quantize_boundary_at_065() {
        let engine = HalftoneEngine {
            state_0_rgb: [0, 0, 0],
            state_1_rgb: [128, 128, 128],
            state_2_rgb: [255, 255, 255],
        };
        // Gray at the upper boundary: Rgb(166,166,166) → L≈0.651 → State 2
        assert_eq!(engine.quantize_pixel(Rgba([166, 166, 166, 255])), 2);
        // Gray just inside middle band: Rgb(165,165,165) → L≈0.647 → State 1
        assert_eq!(engine.quantize_pixel(Rgba([165, 165, 165, 255])), 1);
    }

    // -----------------------------------------------------------------------
    // Halftone Free Trit Count Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_halftone_free_trit_count() {
        let engine = HalftoneEngine {
            state_0_rgb: [0, 0, 0],
            state_1_rgb: [128, 128, 128],
            state_2_rgb: [255, 255, 255],
        };
        
        // Make a 10x10 image with a sharp edge down the middle
        let mut img = RgbImage::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                let color = if x < 5 { [0, 0, 0] } else { [255, 255, 255] };
                img.put_pixel(x, y, Rgb(color));
            }
        }
        let dyn_img = DynamicImage::ImageRgb8(img);

        let matrix_size = 10;
        let required_free = 15;
        
        let constraints = engine.image_to_constraints(&dyn_img, matrix_size, required_free);
        
        assert_eq!(constraints.len(), matrix_size);
        assert_eq!(constraints[0].len(), matrix_size);

        let mut free_count = 0;
        let mut locked_count = 0;
        
        for row in constraints {
            for cell in row {
                if cell.is_none() {
                    free_count += 1;
                } else {
                    locked_count += 1;
                }
            }
        }

        assert_eq!(free_count, required_free, "Must output exactly the requested number of free trits");
        assert_eq!(locked_count, matrix_size * matrix_size - required_free);
    }
}
