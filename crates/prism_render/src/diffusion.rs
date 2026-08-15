//! Dynamic Error Diffusion for the PrismCode embedder.
//!
//! The carrier injection is a smooth 2D sinusoidal chip, quantized to 8-bit RGB at
//! `lab_to_srgb` time. In low-texture regions a smooth sub-LSB gradient can cross
//! quantization boundaries along the chip's contours, producing visible banding.
//!
//! This module pre-quantizes the signed injection onto a finite grid and spreads the
//! signed quantization error to the four unvisited neighbors with the classic
//! Floyd-Steinberg kernel (7/16, 3/16, 5/16, 1/16). The error is zero-mean and
//! decorrelated from the chip, so per-block correlation (and therefore the
//! extractor's LLR) is preserved while smooth contours dissolve into blue noise.

/// Floyd-Steinberg error diffusion state for one image channel.
///
/// Values are processed in row-major order; each call quantizes `desired` onto a
/// grid of `quant_step` and pushes the signed quantization error onto the right and
/// below neighbors, which are guaranteed to be visited later in the scan.
pub struct ErrorDiffuser {
    width: usize,
    height: usize,
    /// Accumulated diffusion error for unvisited pixels.
    buffer: Vec<f32>,
}

impl ErrorDiffuser {
    /// Create a diffuser for a `width x height` image.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            buffer: vec![0.0f32; width * height],
        }
    }

    /// Quantize `desired` onto a grid of `quant_step` and diffuse the error.
    ///
    /// Returns the quantized value to add to the channel. With `quant_step <= 0.0`
    /// the value passes through unquantized and no error is diffused.
    #[inline]
    pub fn quantize_diffuse(
        &mut self,
        x: usize,
        y: usize,
        desired: f32,
        quant_step: f32,
    ) -> f32 {
        if quant_step <= 0.0 {
            return desired;
        }
        let idx = y * self.width + x;
        let v = desired + self.buffer[idx];
        let q = (v / quant_step).round() * quant_step;
        let error = v - q;
        self.buffer[idx] = 0.0;

        // Floyd-Steinberg kernel over the four unvisited neighbors.
        if x + 1 < self.width {
            self.buffer[idx + 1] += error * (7.0 / 16.0);
        }
        if y + 1 < self.height {
            if x > 0 {
                self.buffer[idx + self.width - 1] += error * (3.0 / 16.0);
            }
            self.buffer[idx + self.width] += error * (5.0 / 16.0);
            if x + 1 < self.width {
                self.buffer[idx + self.width + 1] += error * (1.0 / 16.0);
            }
        }
        q
    }
}

/// Standalone convenience: quantize an entire channel in place onto a grid of
/// `quant_step`, diffusing the signed quantization error with the Floyd-Steinberg
/// kernel. `values` must have exactly `width * height` elements.
///
/// With `quant_step <= 0.0` the channel is left unchanged.
pub fn diffuse_error(values: &mut [f32], width: usize, height: usize, quant_step: f32) {
    assert_eq!(values.len(), width * height);
    if quant_step <= 0.0 || width == 0 || height == 0 {
        return;
    }
    let mut diffuser = ErrorDiffuser::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            values[idx] = diffuser.quantize_diffuse(x, y, values[idx], quant_step);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diffuser_preserves_block_dc() {
        // A constant gradient must keep its running mean: with FS diffusion the
        // average of quantized + buffered error over a full image stays near 0.
        let (w, h) = (64, 48);
        let mut values = vec![1.234f32; w * h];
        diffuse_error(&mut values, w, h, 1.0);
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        assert!((mean - 1.234).abs() < 0.05, "mean drifted: {mean}");
        // All values sit on the quant grid (within fp tolerance).
        for &v in &values {
            let rem = (v / 1.0).fract().abs();
            assert!(rem < 1e-3 || rem > 1.0 - 1e-3, "off-grid value {v}");
        }
    }

    #[test]
    fn diffuse_error_pass_through_when_disabled() {
        let mut values = vec![0.5f32; 100];
        diffuse_error(&mut values, 10, 10, 0.0);
        assert!(values.iter().all(|&v| v == 0.5));
    }

    #[test]
    fn diffuser_smooth_ramp_stays_smooth() {
        // A smooth chip ramp quantized to a coarse grid must not accumulate
        // runaway error and must track the local mean.
        let (w, h) = (32, 24);
        let mut diffuser = ErrorDiffuser::new(w, h);
        let mut out = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let desired = (x as f32 / w as f32) * 2.0 - 1.0; // -1..1 ramp
                out[y * w + x] = diffuser.quantize_diffuse(x, y, desired, 0.5);
            }
        }
        // No runaway: every delivered value within one step of the ramp.
        for y in 0..h {
            for x in 0..w {
                let desired = (x as f32 / w as f32) * 2.0 - 1.0;
                let dev = (out[y * w + x] - desired).abs();
                assert!(dev < 1.0, "diffusion runaway at ({x},{y}): {dev}");
            }
        }
        // Mean error over the image is small (zero-mean dither).
        let mut err_sum = 0.0f32;
        for y in 0..h {
            for x in 0..w {
                let desired = (x as f32 / w as f32) * 2.0 - 1.0;
                err_sum += out[y * w + x] - desired;
            }
        }
        assert!(err_sum.abs() < 1.0, "net error too large: {err_sum}");
    }
}
