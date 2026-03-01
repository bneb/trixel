//! # trixel_cv
//!
//! Computer vision pipeline for the Trixel system.
//! Extracts a `TritMatrix` from a photograph of a printed trixel code.

use image::{DynamicImage, GrayImage};
use trixel_core::TritMatrix;
use trixel_solver::anchor::{ANCHOR_PATTERNS, ANCHOR_SIZE, corner_positions};
use thiserror::Error;

pub mod geometry;

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum VisionError {
    #[error("image dimensions {width}x{height} not divisible by module_size {module_size}")]
    BadDimensions {
        width: u32,
        height: u32,
        module_size: u32,
    },
    #[error("no anchors found in image")]
    NoAnchors,
    #[error("image is empty")]
    EmptyImage,
    #[error("anchor calibration failed: could not determine luminance bands")]
    CalibrationFailed,
}

// ---------------------------------------------------------------------------
// VisionPipeline Trait
// ---------------------------------------------------------------------------

/// Configurable luminance band thresholds for quantizing post-normalized
/// grayscale pixels into trit states. The gaps between bands are guard bands
/// that produce erasures (value 3).
///
/// Default bands (percentage of 0–255 range):
/// - State 0: 0–20%  (0–51)
/// - Guard:   20–40% (52–101)
/// - State 1: 40–60% (102–152)
/// - Guard:   60–80% (153–203)
/// - State 2: 80–100% (204–255)
#[derive(Debug, Clone)]
pub struct LuminanceBands {
    /// Upper bound (inclusive) of the State 0 (black) band.
    pub state_0_upper: u8,
    /// Lower bound (inclusive) of the State 1 (mid) band.
    pub state_1_lower: u8,
    /// Upper bound (inclusive) of the State 1 (mid) band.
    pub state_1_upper: u8,
    /// Lower bound (inclusive) of the State 2 (white) band.
    pub state_2_lower: u8,
}

impl Default for LuminanceBands {
    fn default() -> Self {
        Self {
            state_0_upper: 51,   // 0–20%
            state_1_lower: 102,  // 40%
            state_1_upper: 152,  // 60%
            state_2_lower: 204,  // 80%
        }
    }
}

impl LuminanceBands {
    /// Quantize a grayscale value into a trit (0, 1, 2) or erasure (3).
    pub fn quantize(&self, lum: u8) -> u8 {
        if lum <= self.state_0_upper {
            0 // black band
        } else if lum >= self.state_1_lower && lum <= self.state_1_upper {
            1 // mid band
        } else if lum >= self.state_2_lower {
            2 // white band
        } else {
            3 // guard band → erasure
        }
    }

    /// Create calibrated bands from known anchor luminance samples.
    ///
    /// Given the measured luminance of State 0, State 1, and State 2 from
    /// the anchor crooks, set thresholds at the midpoints between them.
    pub fn calibrate(state_0_lum: u8, state_1_lum: u8, state_2_lum: u8) -> Self {
        let mid_01 = ((state_0_lum as u16 + state_1_lum as u16) / 2) as u8;
        let mid_12 = ((state_1_lum as u16 + state_2_lum as u16) / 2) as u8;

        // State 0: 0 to mid_01
        // State 1: mid_01+1 to mid_12
        // State 2: mid_12+1 to 255
        // Guard bands are eliminated when calibrated (tight thresholds)
        Self {
            state_0_upper: mid_01,
            state_1_lower: mid_01.saturating_add(1),
            state_1_upper: mid_12,
            state_2_lower: mid_12.saturating_add(1),
        }
    }
}

/// Ingests a raw image, locates anchors, un-warps perspective,
/// normalizes luminance, and returns an N×N trit grid.
pub trait VisionPipeline {
    /// Extract the trit matrix from a trixel image.
    /// Returns values 0, 1, 2, or 3 (erasure).
    fn extract_matrix(
        image: &DynamicImage,
        module_pixel_size: u32,
    ) -> Result<TritMatrix, VisionError>;

    /// Returns a debug grayscale view after luminance normalization.
    fn get_normalized_debug_view(image: &DynamicImage) -> Result<GrayImage, VisionError>;
}

// ---------------------------------------------------------------------------
// Mock Implementation
// ---------------------------------------------------------------------------

/// Mock vision pipeline that assumes a digitally-perfect image.
/// Each module is `module_pixel_size × module_pixel_size` pixels.
/// Uses `LuminanceBands::default()` for quantization.
pub struct MockVision;

impl VisionPipeline for MockVision {
    fn extract_matrix(
        image: &DynamicImage,
        module_pixel_size: u32,
    ) -> Result<TritMatrix, VisionError> {
        let gray = image.to_luma8();
        let (img_w, img_h) = gray.dimensions();

        if img_w == 0 || img_h == 0 {
            return Err(VisionError::EmptyImage);
        }
        if img_w % module_pixel_size != 0 || img_h % module_pixel_size != 0 {
            return Err(VisionError::BadDimensions {
                width: img_w,
                height: img_h,
                module_size: module_pixel_size,
            });
        }

        let grid_w = (img_w / module_pixel_size) as usize;
        let grid_h = (img_h / module_pixel_size) as usize;
        let mut matrix = TritMatrix::zeros(grid_w, grid_h);
        let bands = LuminanceBands::default();

        for gy in 0..grid_h {
            for gx in 0..grid_w {
                let avg = sample_module_luminance(&gray, gx, gy, module_pixel_size);
                matrix.set(gx, gy, bands.quantize(avg));
            }
        }

        Ok(matrix)
    }

    fn get_normalized_debug_view(image: &DynamicImage) -> Result<GrayImage, VisionError> {
        let gray = image.to_luma8();
        if gray.dimensions() == (0, 0) {
            return Err(VisionError::EmptyImage);
        }
        Ok(gray)
    }
}

// ---------------------------------------------------------------------------
// Anchor-Aware Vision Pipeline
// ---------------------------------------------------------------------------

/// Production vision pipeline that uses L-bracket anchors for luminance
/// calibration.
///
/// Pipeline:
/// 1. Sample the known anchor positions to measure actual State 0, 1, 2 luminance
/// 2. Build calibrated `LuminanceBands` from anchor measurements
/// 3. Extract all trit values using calibrated bands
pub struct AnchorVision;

impl VisionPipeline for AnchorVision {
    fn extract_matrix(
        image: &DynamicImage,
        module_pixel_size: u32,
    ) -> Result<TritMatrix, VisionError> {
        let gray = image.to_luma8();
        let (img_w, img_h) = gray.dimensions();

        if img_w == 0 || img_h == 0 {
            return Err(VisionError::EmptyImage);
        }
        if img_w % module_pixel_size != 0 || img_h % module_pixel_size != 0 {
            return Err(VisionError::BadDimensions {
                width: img_w,
                height: img_h,
                module_size: module_pixel_size,
            });
        }

        let grid_w = (img_w / module_pixel_size) as usize;
        let grid_h = (img_h / module_pixel_size) as usize;

        if grid_w < ANCHOR_SIZE * 2 || grid_h < ANCHOR_SIZE * 2 {
            return Err(VisionError::NoAnchors);
        }

        // 1. Calibrate luminance from anchor crooks
        let bands = calibrate_from_anchors(&gray, grid_w, module_pixel_size)?;

        // 2. Extract all trits using calibrated bands
        let mut matrix = TritMatrix::zeros(grid_w, grid_h);
        for gy in 0..grid_h {
            for gx in 0..grid_w {
                let avg = sample_module_luminance(&gray, gx, gy, module_pixel_size);
                matrix.set(gx, gy, bands.quantize(avg));
            }
        }

        Ok(matrix)
    }

    fn get_normalized_debug_view(image: &DynamicImage) -> Result<GrayImage, VisionError> {
        let gray = image.to_luma8();
        if gray.dimensions() == (0, 0) {
            return Err(VisionError::EmptyImage);
        }
        Ok(gray)
    }
}

/// Calibrate luminance bands by sampling the known anchor positions.
///
/// The L-bracket anchors contain all 3 states:
/// - State 0 (black): the L-shape itself (e.g., TL pattern[0][0])
/// - State 1 (accent): the color-sync dot (e.g., TL pattern[2][2])
/// - State 2 (white): the quiet zone (e.g., TL pattern[1][1])
///
/// We sample from all 4 corners and take the median for robustness.
fn calibrate_from_anchors(
    gray: &GrayImage,
    grid_size: usize,
    module_pixel_size: u32,
) -> Result<LuminanceBands, VisionError> {
    let mut state_0_samples = Vec::new();
    let mut state_1_samples = Vec::new();
    let mut state_2_samples = Vec::new();

    for &(cx, cy, pi) in &corner_positions(grid_size) {
        let pattern = &ANCHOR_PATTERNS[pi];
        for dy in 0..ANCHOR_SIZE {
            for dx in 0..ANCHOR_SIZE {
                let lum = sample_module_luminance(gray, cx + dx, cy + dy, module_pixel_size);
                match pattern[dy][dx] {
                    0 => state_0_samples.push(lum),
                    1 => state_1_samples.push(lum),
                    2 => state_2_samples.push(lum),
                    _ => {}
                }
            }
        }
    }

    if state_0_samples.is_empty() || state_1_samples.is_empty() || state_2_samples.is_empty() {
        return Err(VisionError::CalibrationFailed);
    }

    // Use median for robustness against noise
    let s0 = median(&mut state_0_samples);
    let s1 = median(&mut state_1_samples);
    let s2 = median(&mut state_2_samples);

    Ok(LuminanceBands::calibrate(s0, s1, s2))
}

/// Sample the average luminance of a module at grid position (gx, gy).
fn sample_module_luminance(
    gray: &GrayImage,
    gx: usize,
    gy: usize,
    module_pixel_size: u32,
) -> u8 {
    let px_x = gx as u32 * module_pixel_size;
    let px_y = gy as u32 * module_pixel_size;
    let mut sum: u64 = 0;
    let count = module_pixel_size as u64 * module_pixel_size as u64;

    for dy in 0..module_pixel_size {
        for dx in 0..module_pixel_size {
            sum += gray.get_pixel(px_x + dx, px_y + dy).0[0] as u64;
        }
    }

    (sum / count) as u8
}

/// Compute median of a mutable slice.
fn median(values: &mut [u8]) -> u8 {
    values.sort_unstable();
    values[values.len() / 2]
}
