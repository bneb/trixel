//! Perceptual Image Quality Metrics: PSNR and SSIM.

use image::RgbImage;

/// Compute Peak Signal-to-Noise Ratio (PSNR) in dB between two RGB images.
pub fn compute_psnr(img1: &RgbImage, img2: &RgbImage) -> f32 {
    assert_eq!(img1.dimensions(), img2.dimensions());
    let (w, h) = img1.dimensions();
    let total_samples = (w * h * 3) as f64;

    let mut sum_sq_diff = 0.0f64;
    for (p1, p2) in img1.pixels().zip(img2.pixels()) {
        for c in 0..3 {
            let diff = p1[c] as f64 - p2[c] as f64;
            sum_sq_diff += diff * diff;
        }
    }

    let mse = sum_sq_diff / total_samples;
    if mse <= 1e-10 {
        return 100.0; // Identical images
    }

    (10.0 * (255.0 * 255.0 / mse).log10()) as f32
}

/// Compute Mean Structural Similarity Index Measure (SSIM) between two RGB images.
pub fn compute_ssim(img1: &RgbImage, img2: &RgbImage) -> f32 {
    assert_eq!(img1.dimensions(), img2.dimensions());
    let (w, h) = img1.dimensions();
    if w < 8 || h < 8 {
        return 1.0;
    }

    const C1: f64 = (0.01 * 255.0) * (0.01 * 255.0);
    const C2: f64 = (0.03 * 255.0) * (0.03 * 255.0);

    let mut ssim_sum = 0.0f64;
    let mut num_blocks = 0;

    let w_blocks = (w / 8) as usize;
    let h_blocks = (h / 8) as usize;

    for by in 0..h_blocks {
        for bx in 0..w_blocks {
            let mut mean1 = 0.0f64;
            let mut mean2 = 0.0f64;

            // 1. Compute local means
            for dy in 0..8 {
                for dx in 0..8 {
                    let p1 = img1.get_pixel(bx as u32 * 8 + dx, by as u32 * 8 + dy);
                    let p2 = img2.get_pixel(bx as u32 * 8 + dx, by as u32 * 8 + dy);

                    // Rec.601 luminance
                    let y1 = 0.299 * p1[0] as f64 + 0.587 * p1[1] as f64 + 0.114 * p1[2] as f64;
                    let y2 = 0.299 * p2[0] as f64 + 0.587 * p2[1] as f64 + 0.114 * p2[2] as f64;

                    mean1 += y1;
                    mean2 += y2;
                }
            }
            mean1 /= 64.0;
            mean2 /= 64.0;

            // 2. Compute local variances and covariance
            let mut var1 = 0.0f64;
            let mut var2 = 0.0f64;
            let mut covar = 0.0f64;

            for dy in 0..8 {
                for dx in 0..8 {
                    let p1 = img1.get_pixel(bx as u32 * 8 + dx, by as u32 * 8 + dy);
                    let p2 = img2.get_pixel(bx as u32 * 8 + dx, by as u32 * 8 + dy);

                    let y1 = 0.299 * p1[0] as f64 + 0.587 * p1[1] as f64 + 0.114 * p1[2] as f64;
                    let y2 = 0.299 * p2[0] as f64 + 0.587 * p2[1] as f64 + 0.114 * p2[2] as f64;

                    let d1 = y1 - mean1;
                    let d2 = y2 - mean2;

                    var1 += d1 * d1;
                    var2 += d2 * d2;
                    covar += d1 * d2;
                }
            }
            var1 /= 63.0;
            var2 /= 63.0;
            covar /= 63.0;

            let ssim = ((2.0 * mean1 * mean2 + C1) * (2.0 * covar + C2))
                / ((mean1 * mean1 + mean2 * mean2 + C1) * (var1 + var2 + C2));

            ssim_sum += ssim;
            num_blocks += 1;
        }
    }

    (ssim_sum / num_blocks as f64) as f32
}
