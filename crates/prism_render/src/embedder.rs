//! PrismCode Forward Perceptual Embedder.
//!
//! Embeds QC-LDPC coded data into CIELAB b* (blue-yellow) channel weighted by Yang-Bovik JND,
//! and inserts a sub-perceptual Fourier pilot grid into CIELAB L*.

use image::RgbImage;
use prism_core::interleaver::SpatialInterleaver;
use prism_core::ldpc::{LdpcConfig, QcLdpcCodec};
use prism_hvs::color::{lab_to_srgb, srgb_to_lab};
use prism_hvs::jnd::compute_spatial_jnd;
use prism_sync::pilot::{generate_pilot_grid, PilotConfig};
use thiserror::Error;

use crate::diffusion::ErrorDiffuser;

#[derive(Error, Debug)]
pub enum EmbedError {
    #[error("Image dimensions ({width}x{height}) too small (minimum 192x128 required)")]
    ImageTooSmall { width: u32, height: u32 },
    #[error("Payload length ({0} bytes) exceeds maximum capacity ({1} bytes)")]
    PayloadTooLarge(usize, usize),
}

pub const GRID_BLOCKS_X: usize = 32;
pub const GRID_BLOCKS_Y: usize = 24;
pub const TOTAL_CODEWORD_BITS: usize = GRID_BLOCKS_X * GRID_BLOCKS_Y; // 768 bits

/// Tuning parameters for the forward embedder.
///
/// These control embedding strength (fidelity vs. decode margin) and the
/// error-diffusion dithering used to suppress banding in low-texture regions.
/// The extractor measures the delivered chip amplitude by correlation, so scaling
/// the carrier is transparent to it as long as the residual stays decodable.
#[derive(Clone, Debug)]
pub struct EmbedConfig {
    /// Fourier pilot amplitude in L* units (spread-spectrum sync signal).
    /// PilotConfig::default() uses 0.5; we default lower to preserve PSNR.
    pub pilot_amplitude: f32,
    /// Multiplier on `jnd * chip` for the b* chroma carrier.
    pub carrier_scale_b: f32,
    /// Multiplier on `jnd * chip` for the L* luma carrier (very dark/bright blocks).
    pub carrier_scale_l: f32,
    /// Upper clamp on the JND value used for carrier amplitude.
    pub jnd_clamp_max: f32,
    /// Quantization grid step (in L*/b* units) for Floyd-Steinberg dithering of the
    /// carrier injection. `<= 0.0` disables diffusion.
    pub diffusion_step: f32,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            pilot_amplitude: 0.25,
            carrier_scale_b: 0.65,
            carrier_scale_l: 0.35,
            jnd_clamp_max: 4.5,
            diffusion_step: 0.5,
        }
    }
}

pub struct PrismEmbedder {
    codec: QcLdpcCodec,
    interleaver: SpatialInterleaver,
    pilot_config: PilotConfig,
    config: EmbedConfig,
}

impl Default for PrismEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl PrismEmbedder {
    pub fn new() -> Self {
        Self::with_config(EmbedConfig::default())
    }

    pub fn with_config(config: EmbedConfig) -> Self {
        let codec = QcLdpcCodec::new(LdpcConfig::rate_one_third_z64());
        let interleaver = SpatialInterleaver::new(GRID_BLOCKS_X, GRID_BLOCKS_Y);
        let pilot_config = PilotConfig {
            amplitude: config.pilot_amplitude,
            ..PilotConfig::default()
        };

        Self {
            codec,
            interleaver,
            pilot_config,
            config,
        }
    }

    /// Embed byte payload into an RGB image.
    ///
    /// Maximum payload: 32 bytes (256 bits).
    /// Returns modified RgbImage.
    pub fn embed(&self, source_img: &RgbImage, payload: &[u8]) -> Result<RgbImage, EmbedError> {
        let (w, h) = source_img.dimensions();
        if w < 192 || h < 128 {
            return Err(EmbedError::ImageTooSmall { width: w, height: h });
        }

        let max_bytes = self.codec.k_info_bits() / 8;
        if payload.len() > max_bytes {
            return Err(EmbedError::PayloadTooLarge(payload.len(), max_bytes));
        }

        // 1. Convert payload to 256 bits with zero-padding
        let mut info_bits = vec![0u8; self.codec.k_info_bits()];
        for (i, &byte) in payload.iter().enumerate() {
            for b in 0..8 {
                info_bits[i * 8 + b] = (byte >> (7 - b)) & 1;
            }
        }

        // 2. LDPC Encode -> 768 codeword bits
        let codeword = self.codec.encode(&info_bits);
        assert_eq!(codeword.len(), TOTAL_CODEWORD_BITS);

        // 3. 2D Spatial Interleaving
        let interleaved_bits = self.interleaver.interleave(&codeword);

        // 4. Color Decomposition to CIELAB & Grayscale Luma
        let width = w as usize;
        let height = h as usize;
        let mut lab_l = vec![0.0f32; width * height];
        let mut lab_a = vec![0.0f32; width * height];
        let mut lab_b = vec![0.0f32; width * height];
        let mut gray_luma = vec![0.0f32; width * height];

        for y in 0..height {
            for x in 0..width {
                let px = source_img.get_pixel(x as u32, y as u32);
                let lab = srgb_to_lab(px[0], px[1], px[2]);
                let idx = y * width + x;
                lab_l[idx] = lab.l;
                lab_a[idx] = lab.a;
                lab_b[idx] = lab.b;
                gray_luma[idx] = 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32;
            }
        }

        // 5. Compute Yang-Bovik JND Map
        let jnd_map = compute_spatial_jnd(&gray_luma, width, height);

        // 6. Inject 2D Fourier Pilot Grid into L* and b* (avoiding one-sided clipping on white)
        let pilot = generate_pilot_grid(width, height, &self.pilot_config);
        for i in 0..width * height {
            if lab_l[i] > 85.0 || lab_l[i] < 15.0 {
                lab_b[i] = (lab_b[i] + pilot[i] * 1.5).clamp(-128.0, 127.0);
            } else {
                lab_l[i] = (lab_l[i] + pilot[i]).clamp(0.0, 100.0);
            }
        }

        // 7. Inject JND-weighted LDPC carrier into b* channel (or L* in extreme shadows/highlights)
        let block_w = width as f32 / GRID_BLOCKS_X as f32;
        let block_h = height as f32 / GRID_BLOCKS_Y as f32;

        // Precompute per-block geometry and the dominant-lightness channel choice.
        // The channel choice must mirror the extractor's decision on the embedded
        // image, so avg_l is measured after pilot injection, exactly as before.
        let mut block_use_luma = vec![false; GRID_BLOCKS_X * GRID_BLOCKS_Y];
        let mut block_x_start = vec![0usize; GRID_BLOCKS_X * GRID_BLOCKS_Y];
        let mut block_y_start = vec![0usize; GRID_BLOCKS_X * GRID_BLOCKS_Y];
        let mut block_w_px = vec![0usize; GRID_BLOCKS_X * GRID_BLOCKS_Y];
        let mut block_h_px = vec![0usize; GRID_BLOCKS_X * GRID_BLOCKS_Y];

        for by in 0..GRID_BLOCKS_Y {
            for bx in 0..GRID_BLOCKS_X {
                let bidx = by * GRID_BLOCKS_X + bx;
                let x_start = (bx as f32 * block_w) as usize;
                let x_end = (((bx + 1) as f32 * block_w) as usize).min(width);
                let y_start = (by as f32 * block_h) as usize;
                let y_end = (((by + 1) as f32 * block_h) as usize).min(height);
                block_x_start[bidx] = x_start;
                block_y_start[bidx] = y_start;
                block_w_px[bidx] = x_end - x_start;
                block_h_px[bidx] = y_end - y_start;

                // Determine dominant block lightness
                let mut block_l_sum = 0.0f32;
                let mut block_count = 0.0f32;
                for y in y_start..y_end {
                    for x in x_start..x_end {
                        block_l_sum += lab_l[y * width + x];
                        block_count += 1.0;
                    }
                }
                let avg_l = if block_count > 0.0 { block_l_sum / block_count } else { 50.0 };
                block_use_luma[bidx] = avg_l < 10.0;
            }
        }

        // Single row-major scan (needed for the Floyd-Steinberg diffuser), one
        // diffuser per channel so untouched pixels are never dithered.
        let mut diffuser_b = ErrorDiffuser::new(width, height);
        let mut diffuser_l = ErrorDiffuser::new(width, height);
        let jnd_max = self.config.jnd_clamp_max;
        let step = self.config.diffusion_step;

        for y in 0..height {
            // Block row containing this pixel row.
            let by = ((y as f32) / block_h) as usize;
            for x in 0..width {
                let bx = ((x as f32) / block_w) as usize;
                let bidx = by.min(GRID_BLOCKS_Y - 1) * GRID_BLOCKS_X + bx.min(GRID_BLOCKS_X - 1);
                let use_luma = block_use_luma[bidx];

                let x_start = block_x_start[bidx];
                let y_start = block_y_start[bidx];
                let bw = block_w_px[bidx] as f32;
                let bh = block_h_px[bidx] as f32;

                let vy = (y - y_start) as f32 / bh;
                let wy = (4.0 * std::f32::consts::PI * vy).sin();
                let vx = (x - x_start) as f32 / bw;
                let wx = (4.0 * std::f32::consts::PI * vx).sin();
                // Zero-mean 2D sinusoidal carrier (2 cycles per block)
                let chip = wx * wy;

                let idx = y * width + x;
                let jnd = jnd_map[idx].clamp(1.5, jnd_max);

                if use_luma {
                    let delta_l = -(interleaved_bits[bidx] as f32 * 2.0 - 1.0)
                        * jnd * chip * self.config.carrier_scale_l;
                    let delivered = diffuser_l.quantize_diffuse(x, y, delta_l, step);
                    lab_l[idx] = (lab_l[idx] + delivered).clamp(0.0, 100.0);
                } else {
                    let delta_b = -(interleaved_bits[bidx] as f32 * 2.0 - 1.0)
                        * jnd * chip * self.config.carrier_scale_b;
                    let delivered = diffuser_b.quantize_diffuse(x, y, delta_b, step);
                    lab_b[idx] = (lab_b[idx] + delivered).clamp(-128.0, 127.0);
                }
            }
        }

        // 8. Reconstruct sRGB Image
        let mut output = RgbImage::new(w, h);
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let rgb = lab_to_srgb(lab_l[idx], lab_a[idx], lab_b[idx]);
                output.put_pixel(x as u32, y as u32, image::Rgb(rgb));
            }
        }

        Ok(output)
    }
}
