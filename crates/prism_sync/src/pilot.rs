//! 2D Fourier Spread-Spectrum Pilot Signal Generator.

use std::f32::consts::PI;

#[derive(Clone, Debug)]
pub struct PilotConfig {
    pub ku: f32,       // Horizontal harmonic frequency index
    pub kv: f32,       // Vertical harmonic frequency index
    pub amplitude: f32,// Sub-perceptual Luma amplitude (e.g. 1.5 - 2.0 L*)
}

impl Default for PilotConfig {
    fn default() -> Self {
        Self {
            ku: 16.0,
            kv: 16.0,
            amplitude: 0.5,
        }
    }
}

/// Generates a 2D 4-point conjugate sinusoidal pilot pattern.
///
/// In Fourier domain: 4 discrete delta spikes at (±ku/W, ±kv/H).
/// In Spatial domain: S(x, y) = 2*A * cos(2*pi*ku*x/W) * cos(2*pi*kv*y/H).
pub fn generate_pilot_grid(width: usize, height: usize, config: &PilotConfig) -> Vec<f32> {
    let mut grid = vec![0.0f32; width * height];
    let w_f = width as f32;
    let h_f = height as f32;

    for y in 0..height {
        let cos_y = (2.0 * PI * config.kv * y as f32 / h_f).cos();
        for x in 0..width {
            let cos_x = (2.0 * PI * config.ku * x as f32 / w_f).cos();
            grid[y * width + x] = 2.0 * config.amplitude * cos_x * cos_y;
        }
    }

    grid
}
