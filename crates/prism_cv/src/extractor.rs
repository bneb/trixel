//! PrismCode Camera & Digital Frame Extractor.
//!
//! Extracts QC-LDPC coded Log-Likelihood Ratios (LLRs) from the CIELAB b* residual channel,
//! deinterleaves spatial coordinates, and runs Min-Sum Belief Propagation.

use image::RgbImage;
use prism_core::ldpc::{LdpcConfig, QcLdpcCodec, LdpcError};
use prism_core::interleaver::SpatialInterleaver;
use prism_hvs::color::srgb_to_lab;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ExtractError {
    #[error("Image dimensions ({width}x{height}) too small (minimum 192x128 required)")]
    ImageTooSmall { width: u32, height: u32 },
    #[error("LDPC Decoding error: {0}")]
    Ldpc(#[from] LdpcError),
}

pub const GRID_BLOCKS_X: usize = 32;
pub const GRID_BLOCKS_Y: usize = 24;

pub struct PrismExtractor {
    codec: QcLdpcCodec,
    interleaver: SpatialInterleaver,
}

impl Default for PrismExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl PrismExtractor {
    pub fn new() -> Self {
        let codec = QcLdpcCodec::new(LdpcConfig::rate_one_third_z64());
        let interleaver = SpatialInterleaver::new(GRID_BLOCKS_X, GRID_BLOCKS_Y);

        Self { codec, interleaver }
    }

    /// Extract the full codeword payload from an embedded RGB image.
    ///
    /// Returns the complete decoded payload (32 bytes for the z64 codec),
    /// zero-padding included. Trailing-zero trimming is a display-layer
    /// concern (CLI/WASM) and does not happen here.
    pub fn extract(&self, img: &RgbImage) -> Result<Vec<u8>, ExtractError> {
        let (w, h) = img.dimensions();
        if w < 192 || h < 128 {
            return Err(ExtractError::ImageTooSmall { width: w, height: h });
        }

        let width = w as usize;
        let height = h as usize;

        // 1. Extract CIELAB L* and b* channels
        let mut lab_l = vec![0.0f32; width * height];
        let mut lab_b = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let px = img.get_pixel(x as u32, y as u32);
                let lab = srgb_to_lab(px[0], px[1], px[2]);
                let idx = y * width + x;
                lab_l[idx] = lab.l;
                lab_b[idx] = lab.b;
            }
        }

        // 1.5. Fourier Pilot Detection & Affine Geometric Rectification
        let pilot_config = prism_sync::PilotConfig::default();
        if width.is_power_of_two() && height.is_power_of_two() {
            let peaks_opt = prism_sync::detect_pilot_peaks(&lab_l, width, height, &pilot_config)
                .or_else(|_| prism_sync::detect_pilot_peaks(&lab_b, width, height, &pilot_config));

            match peaks_opt {
                Ok(peaks) => {
                    if let Ok((theta, sx, sy)) = prism_sync::estimate_affine(&peaks, &pilot_config) {
                        eprintln!("PILOT DETECTED -> theta: {:+.2} deg, sx: {:.4}, sy: {:.4}", theta.to_degrees(), sx, sy);
                        if theta.abs() > 0.003 || (sx - 1.0).abs() > 0.015 || (sy - 1.0).abs() > 0.015 {
                            lab_l = rectify_affine(&lab_l, width, height, theta, sx, sy);
                            lab_b = rectify_affine(&lab_b, width, height, theta, sx, sy);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("PILOT PEAKS NOT FOUND: {:?}", e);
                }
            }
        }

        // 2. Extract Residual & Integrate Correlation Energy per Block
        let block_w = width as f32 / GRID_BLOCKS_X as f32;
        let block_h = height as f32 / GRID_BLOCKS_Y as f32;
        let mut interleaved_llrs = vec![0.0f32; GRID_BLOCKS_X * GRID_BLOCKS_Y];

        for by in 0..GRID_BLOCKS_Y {
            for bx in 0..GRID_BLOCKS_X {
                let x_start = (bx as f32 * block_w) as usize;
                let x_end = (((bx + 1) as f32 * block_w) as usize).min(width);
                let y_start = (by as f32 * block_h) as usize;
                let y_end = (((by + 1) as f32 * block_h) as usize).min(height);

                let bw = (x_end - x_start) as f32;
                let bh = (y_end - y_start) as f32;
                let block_pixels = ((x_end - x_start) * (y_end - y_start)) as f32;

                // Evaluate both b* and L* channels with local plane subtraction
                let extract_channel_llr = |channel: &[f32]| -> (f32, f32) {
                    let (cx, cy, a, p, q) = match fit_plane(channel, width, x_start, x_end, y_start, y_end) {
                        Some(plane) => plane,
                        None => {
                            let mut sum = 0.0f32;
                            let mut n = 0.0f32;
                            for y in y_start..y_end {
                                for x in x_start..x_end {
                                    sum += channel[y * width + x];
                                    n += 1.0;
                                }
                            }
                            (0.0, 0.0, sum / n.max(1.0), 0.0, 0.0)
                        }
                    };

                    let mut corr_sum = 0.0f32;
                    let mut energy_sum = 0.0f32;

                    for y in y_start..y_end {
                        let vy = (y - y_start) as f32 / bh;
                        let wy = (4.0 * std::f32::consts::PI * vy).sin();

                        for x in x_start..x_end {
                            let vx = (x - x_start) as f32 / bw;
                            let wx = (4.0 * std::f32::consts::PI * vx).sin();
                            let chip = wx * wy;

                            let idx = y * width + x;
                            let dx = x as f32 - cx;
                            let dy = y as f32 - cy;
                            let residual = channel[idx] - (a + p * dx + q * dy);

                            corr_sum += residual * chip;
                            energy_sum += chip * chip;
                        }
                    }

                    if energy_sum > 1e-4 {
                        let signal = corr_sum / energy_sum;
                        let mut noise_sq_sum = 0.0f32;
                        for y in y_start..y_end {
                            let vy = (y - y_start) as f32 / bh;
                            let wy = (4.0 * std::f32::consts::PI * vy).sin();

                            for x in x_start..x_end {
                                let vx = (x - x_start) as f32 / bw;
                                let wx = (4.0 * std::f32::consts::PI * vx).sin();
                                let chip = wx * wy;

                                let idx = y * width + x;
                                let dx = x as f32 - cx;
                                let dy = y as f32 - cy;
                                let residual = channel[idx] - (a + p * dx + q * dy);
                                let noise = residual - signal * chip;
                                noise_sq_sum += noise * noise;
                            }
                        }
                        let noise_var = (noise_sq_sum / block_pixels).max(0.05);
                        let llr = (signal / noise_var) * 2.5;
                        let snr = signal.abs() / noise_var;
                        (llr.clamp(-15.0, 15.0), snr)
                    } else {
                        (0.0, 0.0)
                    }
                };

                let (llr_b, snr_b) = extract_channel_llr(&lab_b);
                let (llr_l, snr_l) = extract_channel_llr(&lab_l);

                let bit_idx = by * GRID_BLOCKS_X + bx;
                // Maximum Ratio Combining: pick the channel with dominant SNR
                interleaved_llrs[bit_idx] = if snr_l > snr_b * 1.5 {
                    llr_l
                } else {
                    llr_b
                };
            }
        }

        // 4. Deinterleave LLRs
        let codeword_llrs = self.interleaver.deinterleave(&interleaved_llrs);

        let mut pos_count = 0;
        let mut neg_count = 0;
        let mut zero_count = 0;
        for &l in &codeword_llrs {
            if l > 0.05 { pos_count += 1; }
            else if l < -0.05 { neg_count += 1; }
            else { zero_count += 1; }
        }
        eprintln!("Codeword LLR distribution: pos={}, neg={}, near_zero={}", pos_count, neg_count, zero_count);

        // 5. Min-Sum Belief Propagation Decoding
        let decoded_bits = self.codec.decode_min_sum(&codeword_llrs, 60, 0.75)?;

        // 6. Convert decoded bits back to bytes (256 bits -> 32 bytes for z64)
        let num_bytes = decoded_bits.len() / 8;
        let mut payload = vec![0u8; num_bytes];
        for (i, byte) in payload.iter_mut().enumerate() {
            let mut val = 0u8;
            for b in 0..8 {
                val = (val << 1) | (decoded_bits[i * 8 + b] & 1);
            }
            *byte = val;
        }

        Ok(payload)
    }
}

/// Least-squares fit of a local reference plane `a + p*dx + q*dy` over a block,
/// with `dx = x - cx`, `dy = y - cy` centered on the block for conditioning.
///
/// Returns `(cx, cy, a, p, q)`, or `None` for a degenerate system (fewer than
/// 3 samples, or all samples collinear in (x, y)).
fn fit_plane(
    channel: &[f32],
    width: usize,
    x_start: usize,
    x_end: usize,
    y_start: usize,
    y_end: usize,
) -> Option<(f32, f32, f32, f32, f32)> {
    let cx = (x_start as f32 + x_end as f32) * 0.5 - 0.5;
    let cy = (y_start as f32 + y_end as f32) * 0.5 - 0.5;

    // Normal equations for [a, p, q]:
    //   [ n    sx   sy  ] [a]   [sv ]
    //   [ sx  sxx  sxy  ] [p] = [svx]
    //   [ sy  sxy  syy  ] [q]   [svy]
    let mut n = 0.0f32;
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    let mut sxx = 0.0f32;
    let mut syy = 0.0f32;
    let mut sxy = 0.0f32;
    let mut sv = 0.0f32;
    let mut svx = 0.0f32;
    let mut svy = 0.0f32;

    for y in y_start..y_end {
        let dy = y as f32 - cy;
        for x in x_start..x_end {
            let dx = x as f32 - cx;
            let v = channel[y * width + x];
            n += 1.0;
            sx += dx;
            sy += dy;
            sxx += dx * dx;
            syy += dy * dy;
            sxy += dx * dy;
            sv += v;
            svx += v * dx;
            svy += v * dy;
        }
    }

    if n < 3.0 {
        return None;
    }

    // Cramer's rule on the symmetric 3x3 system.
    let det = n * (sxx * syy - sxy * sxy)
        - sx * (sx * syy - sxy * sy)
        + sy * (sx * sxy - sxx * sy);
    if det.abs() < 1e-9 {
        return None;
    }

    let det_a = sv * (sxx * syy - sxy * sxy)
        - sx * (svx * syy - sxy * svy)
        + sy * (svx * sxy - sxx * svy);
    let det_p = n * (svx * syy - sxy * svy)
        - sv * (sx * syy - sxy * sy)
        + sy * (sx * svy - svx * sy);
    let det_q = n * (sxx * svy - svx * sxy)
        - sx * (sx * svy - svx * sy)
        + sv * (sx * sxy - sxx * sy);

    Some((cx, cy, det_a / det, det_p / det, det_q / det))
}

/// Rectifies a 2D scalar channel given forward transformation parameters (theta, scale_x, scale_y).
fn rectify_affine(channel: &[f32], width: usize, height: usize, theta: f32, sx: f32, sy: f32) -> Vec<f32> {
    let cx = (width as f32 - 1.0) * 0.5;
    let cy = (height as f32 - 1.0) * 0.5;
    let (s, c) = theta.sin_cos();
    let mut out = vec![0.0f32; width * height];

    for y in 0..height {
        let dy = y as f32 - cy;
        for x in 0..width {
            let dx = x as f32 - cx;
            let src_x = (c * dx - s * dy) * sx + cx;
            let src_y = (s * dx + c * dy) * sy + cy;
            out[y * width + x] = bilinear_sample_float(channel, width, height, src_x, src_y);
        }
    }

    out
}

/// Bilinear sampling of a 1D-flattened float image grid at fractional coordinates.
fn bilinear_sample_float(img: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x_clamped = x.clamp(0.0, (w - 1) as f32);
    let y_clamped = y.clamp(0.0, (h - 1) as f32);
    let x0 = x_clamped.floor() as usize;
    let y0 = y_clamped.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x_clamped - x0 as f32;
    let fy = y_clamped - y0 as f32;

    let p00 = img[y0 * w + x0];
    let p10 = img[y0 * w + x1];
    let p01 = img[y1 * w + x0];
    let p11 = img[y1 * w + x1];

    p00 * (1.0 - fx) * (1.0 - fy)
        + p10 * fx * (1.0 - fy)
        + p01 * (1.0 - fx) * fy
        + p11 * fx * fy
}

