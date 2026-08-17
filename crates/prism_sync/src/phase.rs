//! Phase Correlation and Translation Offset Estimation.
//!
//! Uses the Fourier Shift Theorem and Normalized Cross-Power Spectrum to recover
//! sub-pixel spatial translation offsets (\Delta x, \Delta y) between a received
//! image and the canonical reference pilot grid.

use crate::fft::{fft2d, ifft2d};
use crate::pilot::{generate_pilot_grid, PilotConfig};
use crate::detector::SyncError;

/// Estimates the spatial translation offset (dx, dy) in pixels of the embedded pilot
/// relative to the canonical grid using 2D phase correlation.
///
/// Returns `(dx, dy, peak_confidence)` where `dx` and `dy` are signed offsets in pixels.
pub fn estimate_pilot_translation(
    image: &[f32],
    width: usize,
    height: usize,
    config: &PilotConfig,
) -> Result<(f32, f32, f32), SyncError> {
    if !width.is_power_of_two() || !height.is_power_of_two() {
        return Err(SyncError::NonPowerOfTwo { width, height });
    }

    let n = width * height;
    if image.len() != n {
        return Err(SyncError::PeaksNotFound);
    }

    // 1. Generate reference pilot grid template
    let ref_grid = generate_pilot_grid(width, height, config);

    // 2. Compute 2D FFT of input image
    let mut real_f = image.to_vec();
    let mut imag_f = vec![0.0f32; n];
    fft2d(&mut real_f, &mut imag_f, width, height);

    // 3. Compute 2D FFT of reference grid template
    let mut real_g = ref_grid;
    let mut imag_g = vec![0.0f32; n];
    fft2d(&mut real_g, &mut imag_g, width, height);

    // 4. Compute normalized cross-power spectrum: (F * G*) / |F * G*|
    let mut cross_r = vec![0.0f32; n];
    let mut cross_i = vec![0.0f32; n];

    for i in 0..n {
        // (a + bi) * (c - di) = (ac + bd) + i(bc - ad)
        let re = real_f[i] * real_g[i] + imag_f[i] * imag_g[i];
        let im = imag_f[i] * real_g[i] - real_f[i] * imag_g[i];
        let mag = (re * re + im * im).sqrt();
        if mag > 1e-9 {
            cross_r[i] = re / mag;
            cross_i[i] = im / mag;
        } else {
            cross_r[i] = 0.0;
            cross_i[i] = 0.0;
        }
    }

    // 5. Compute Inverse 2D FFT to obtain spatial phase correlation surface
    ifft2d(&mut cross_r, &mut cross_i, width, height);

    // 6. Find correlation peak on cross_r
    let mut best_val = -1.0f32;
    let mut best_x = 0usize;
    let mut best_y = 0usize;

    for y in 0..height {
        for x in 0..width {
            let val = cross_r[y * width + x];
            if val > best_val {
                best_val = val;
                best_x = x;
                best_y = y;
            }
        }
    }

    if best_val <= 0.0 {
        return Err(SyncError::PeaksNotFound);
    }

    // 7. Parabolic sub-pixel peak refinement
    let x_lo = if best_x > 0 { best_x - 1 } else { width - 1 };
    let x_hi = if best_x + 1 < width { best_x + 1 } else { 0 };
    let y_lo = if best_y > 0 { best_y - 1 } else { height - 1 };
    let y_hi = if best_y + 1 < height { best_y + 1 } else { 0 };

    let val_c = cross_r[best_y * width + best_x];
    let val_xl = cross_r[best_y * width + x_lo];
    let val_xr = cross_r[best_y * width + x_hi];
    let val_yl = cross_r[y_lo * width + best_x];
    let val_yr = cross_r[y_hi * width + best_x];

    let sub_dx = parabolic_offset(val_xl, val_c, val_xr);
    let sub_dy = parabolic_offset(val_yl, val_c, val_yr);

    let raw_x = best_x as f32 + sub_dx;
    let raw_y = best_y as f32 + sub_dy;

    let lambda_x = width as f32 / config.ku;
    let lambda_y = height as f32 / config.kv;

    // Wrap to fundamental pilot period [-lambda/2, lambda/2)
    let wrap_period = |val: f32, period: f32| -> f32 {
        let mut r = val.rem_euclid(period);
        if r > period * 0.5 {
            r -= period;
        }
        r
    };

    let dx = wrap_period(raw_x, lambda_x);
    let dy = wrap_period(raw_y, lambda_y);

    Ok((dx, dy, best_val))
}

fn parabolic_offset(y0: f32, y1: f32, y2: f32) -> f32 {
    let denom = y0 - 2.0 * y1 + y2;
    if denom.abs() < 1e-6 {
        0.0
    } else {
        (0.5 * (y0 - y2) / denom).clamp(-0.5, 0.5)
    }
}
