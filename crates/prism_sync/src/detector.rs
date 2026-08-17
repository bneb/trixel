//! 2D Spectral Peak Detector and Transformation Parameter Estimator.

use thiserror::Error;
use crate::pilot::PilotConfig;
use crate::fft::fft2d;

#[derive(Error, Debug)]
pub enum SyncError {
    #[error("Could not find required 4 spectral pilot peaks in FFT spectrum")]
    PeaksNotFound,
    #[error("Dimensions must be power of 2: got {width}x{height}")]
    NonPowerOfTwo { width: usize, height: usize },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeakCoord {
    /// Signed horizontal frequency bin (cycles per image width), sub-bin precision.
    pub u: f32,
    /// Signed vertical frequency bin (cycles per image height), sub-bin precision.
    pub v: f32,
    /// FFT magnitude at the coarse peak bin.
    pub magnitude: f32,
}

/// Detects the 4 conjugate pilot peaks in the 2D FFT magnitude spectrum.
///
/// Returns peaks with *signed* full-spectrum coordinates: quadrant 1 -> (+u, +v),
/// quadrant 2 -> (-u, +v), quadrant 3 -> (-u, -v), quadrant 4 -> (+u, -v).
/// Each peak is refined to sub-bin precision by 3-point parabolic interpolation
/// on the log-magnitude spectrum along each axis, beating the 0.5-bin
/// quantization limit of raw bin detection.
pub fn detect_pilot_peaks(
    image: &[f32],
    width: usize,
    height: usize,
    config: &PilotConfig,
) -> Result<Vec<PeakCoord>, SyncError> {
    if !width.is_power_of_two() || !height.is_power_of_two() {
        return Err(SyncError::NonPowerOfTwo { width, height });
    }

    let mut real = image.to_vec();
    let mut imag = vec![0.0f32; width * height];

    // Apply 2D Hann window to suppress edge spectral leakage
    for y in 0..height {
        let wy = (std::f32::consts::PI * y as f32 / height as f32).sin();
        for x in 0..width {
            let wx = (std::f32::consts::PI * x as f32 / width as f32).sin();
            real[y * width + x] *= wx * wy;
        }
    }

    fft2d(&mut real, &mut imag, width, height);

    // Compute magnitude spectrum
    let mut mag = vec![0.0f32; width * height];
    for i in 0..width * height {
        mag[i] = (real[i] * real[i] + imag[i] * imag[i]).sqrt();
    }

    let target_r = (config.ku * config.ku + config.kv * config.kv).sqrt();
    let r_min = (target_r * 0.80) as usize;
    let r_max = (target_r * 1.20).min((width.min(height) / 2 - 2) as f32) as usize;

    let w = width as f32;
    let h = height as f32;

    let mut peaks = Vec::new();

    // Quadrant 1: x in [1, W/2), y in [1, H/2)  -> (+u, +v)
    if let Some((bx, by, m)) =
        find_quadrant_peak(&mag, width, height, 1, width / 2, 1, height / 2, r_min, r_max)
    {
        let (ux, uy) = refine_peak(&mag, width, height, bx, by);
        peaks.push(PeakCoord { u: ux, v: uy, magnitude: m });
    }
    // Quadrant 2: x in [W/2, W), y in [1, H/2)  -> (-u, +v)
    if let Some((bx, by, m)) =
        find_quadrant_peak(&mag, width, height, width / 2, width, 1, height / 2, r_min, r_max)
    {
        let (ux, uy) = refine_peak(&mag, width, height, bx, by);
        peaks.push(PeakCoord { u: ux - w, v: uy, magnitude: m });
    }
    // Quadrant 3: x in [W/2, W), y in [H/2, H)  -> (-u, -v)
    if let Some((bx, by, m)) =
        find_quadrant_peak(&mag, width, height, width / 2, width, height / 2, height, r_min, r_max)
    {
        let (ux, uy) = refine_peak(&mag, width, height, bx, by);
        peaks.push(PeakCoord { u: ux - w, v: uy - h, magnitude: m });
    }
    // Quadrant 4: x in [1, W/2), y in [H/2, H)  -> (+u, -v)
    if let Some((bx, by, m)) =
        find_quadrant_peak(&mag, width, height, 1, width / 2, height / 2, height, r_min, r_max)
    {
        let (ux, uy) = refine_peak(&mag, width, height, bx, by);
        peaks.push(PeakCoord { u: ux, v: uy - h, magnitude: m });
    }

    if peaks.len() == 4 {
        // Validate conjugate radial symmetry across the 4 quadrants
        let r0 = (peaks[0].u * peaks[0].u + peaks[0].v * peaks[0].v).sqrt();
        let r1 = (peaks[1].u * peaks[1].u + peaks[1].v * peaks[1].v).sqrt();
        let r2 = (peaks[2].u * peaks[2].u + peaks[2].v * peaks[2].v).sqrt();
        let r3 = (peaks[3].u * peaks[3].u + peaks[3].v * peaks[3].v).sqrt();

        let r_mean = 0.25 * (r0 + r1 + r2 + r3);
        let max_r_diff = (r0 - r_mean).abs().max((r1 - r_mean).abs()).max((r2 - r_mean).abs()).max((r3 - r_mean).abs());

        // Max allowable radial deviation across conjugate spikes
        if max_r_diff > 2.5 {
            return Err(SyncError::PeaksNotFound);
        }

        // Validate that estimate_affine yields a physical aspect ratio and scale
        if let Ok((_theta, sx, sy)) = estimate_affine(&peaks, config) {
            let aspect = sx / sy;
            if aspect < 0.70 || aspect > 1.40 || sx < 0.65 || sx > 1.45 || sy < 0.65 || sy > 1.45 {
                return Err(SyncError::PeaksNotFound);
            }
        } else {
            return Err(SyncError::PeaksNotFound);
        }

        Ok(peaks)
    } else {
        Err(SyncError::PeaksNotFound)
    }
}

/// Finds the strongest magnitude bin in a quadrant restricted to the pilot
/// radius band, using local spectral contrast (peak vs local annular background).
fn find_quadrant_peak(
    mag: &[f32],
    width: usize,
    height: usize,
    x_start: usize,
    x_end: usize,
    y_start: usize,
    y_end: usize,
    r_min: usize,
    r_max: usize,
) -> Option<(usize, usize, f32)> {
    let mut best_contrast = 0.0f32;
    let mut best_coord = None;

    for y in y_start..y_end {
        let dy = if y <= height / 2 { y as f32 } else { (height - y) as f32 };
        for x in x_start..x_end {
            let dx = if x <= width / 2 { x as f32 } else { (width - x) as f32 };
            let r = (dx * dx + dy * dy).sqrt();

            if r >= r_min as f32 && r <= r_max as f32 {
                let m = mag[y * width + x];
                if m > 1e-6 {
                    // Measure local background mean around this bin (radius 2 neighborhood excluding self)
                    let mut bg_sum = 0.0f32;
                    let mut bg_count = 0.0f32;
                    for ny in (y.saturating_sub(2))..=(y + 2).min(height - 1) {
                        for nx in (x.saturating_sub(2))..=(x + 2).min(width - 1) {
                            if (nx != x || ny != y) && (nx >= x_start && nx < x_end && ny >= y_start && ny < y_end) {
                                bg_sum += mag[ny * width + nx];
                                bg_count += 1.0;
                            }
                        }
                    }
                    let bg_mean = if bg_count > 0.0 { bg_sum / bg_count } else { 1e-6 };
                    let contrast = m / bg_mean.max(1e-6);

                    if contrast > best_contrast && contrast > 1.25 {
                        best_contrast = contrast;
                        best_coord = Some((x, y, m));
                    }
                }
            }
        }
    }

    best_coord
}

/// Refines an integer-bin peak to sub-bin precision via 3-point parabolic
/// interpolation on the log-magnitude spectrum along each axis.
fn refine_peak(mag: &[f32], width: usize, height: usize, bx: usize, by: usize) -> (f32, f32) {
    let log_mag = |x: usize, y: usize| {
        let xc = x.min(width - 1);
        let yc = y.min(height - 1);
        (mag[yc * width + xc] + 1e-12).ln()
    };

    let x_lo = if bx >= 1 { bx - 1 } else { bx };
    let x_hi = if bx + 1 < width { bx + 1 } else { bx };
    let y_lo = if by >= 1 { by - 1 } else { by };
    let y_hi = if by + 1 < height { by + 1 } else { by };

    let du = parabolic_offset(log_mag(x_lo, by), log_mag(bx, by), log_mag(x_hi, by));
    let dv = parabolic_offset(log_mag(bx, y_lo), log_mag(bx, by), log_mag(bx, y_hi));

    (bx as f32 + du, by as f32 + dv)
}

/// Vertex offset of the parabola through three equally spaced samples.
fn parabolic_offset(y0: f32, y1: f32, y2: f32) -> f32 {
    let denom = y0 - 2.0 * y1 + y2;
    if denom.abs() < 1e-6 {
        0.0
    } else {
        0.5 * (y0 - y2) / denom
    }
}

/// Estimates the signed rotation angle (radians) and per-axis scale factors
/// from the 4 detected conjugate peaks.
///
/// The undistorted pilot spikes sit at (+ku,+kv), (+ku,-kv), (-ku,+kv), (-ku,-kv).
/// For |theta| < 45 degrees the quadrant labeling survives rotation, so the
/// spikes can be identified by sign. The transformed u-axis basis is
/// (B1' + B2')/2 and the v-axis basis is (B1' - B2')/2 for the adjacent spikes
/// B1 = (+ku,+kv) and B2 = (+ku,-kv); averaging all four conjugate spikes gives
/// the same bases with doubled sample count. Then
/// theta = atan2(u_axis.v, u_axis.u), scale_x = |u_axis| / ku,
/// scale_y = |v_axis| / kv.
pub fn estimate_affine(peaks: &[PeakCoord], config: &PilotConfig) -> Result<(f32, f32, f32), SyncError> {
    if peaks.len() != 4 {
        return Err(SyncError::PeaksNotFound);
    }

    let pp = peaks.iter().find(|p| p.u > 0.0 && p.v > 0.0).ok_or(SyncError::PeaksNotFound)?;
    let pn = peaks.iter().find(|p| p.u > 0.0 && p.v < 0.0).ok_or(SyncError::PeaksNotFound)?;
    let np = peaks.iter().find(|p| p.u < 0.0 && p.v > 0.0).ok_or(SyncError::PeaksNotFound)?;
    let nn = peaks.iter().find(|p| p.u < 0.0 && p.v < 0.0).ok_or(SyncError::PeaksNotFound)?;

    // u_axis = (pp + pn - nn - np) / 4, v_axis = (pp - pn - nn + np) / 4
    let ux = 0.25 * (pp.u + pn.u - nn.u - np.u);
    let uy = 0.25 * (pp.v + pn.v - nn.v - np.v);
    let vx = 0.25 * (pp.u - pn.u - nn.u + np.u);
    let vy = 0.25 * (pp.v - pn.v - nn.v + np.v);

    let theta = uy.atan2(ux);
    let scale_x = (ux * ux + uy * uy).sqrt() / config.ku;
    let scale_y = (vx * vx + vy * vy).sqrt() / config.kv;

    Ok((theta, scale_x, scale_y))
}

/// Estimate signed rotation angle (radians) and mean scale factor from the
/// detected conjugate peaks.
pub fn estimate_rotation_and_scale(
    peaks: &[PeakCoord],
    _width: usize,
    _height: usize,
    config: &PilotConfig,
) -> Result<(f32, f32), SyncError> {
    let (theta, scale_x, scale_y) = estimate_affine(peaks, config)?;
    Ok((theta, 0.5 * (scale_x + scale_y)))
}
