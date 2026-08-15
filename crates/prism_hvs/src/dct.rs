//! 2D 8x8 Discrete Cosine Transform (DCT-II) and Inverse Discrete Cosine Transform (IDCT-II).
//!
//! Uses orthonormal basis scaling so that forward and inverse operations are exactly unitary:
//! ||DCT(x)||^2 = ||x||^2.

use std::f32::consts::PI;

pub const BLOCK_SIZE: usize = 8;

/// Precomputed 8x8 DCT-II orthonormal transformation matrix
#[derive(Clone, Debug)]
pub struct Dct8x8Table {
    matrix: [[f32; 8]; 8],
    matrix_t: [[f32; 8]; 8],
}

impl Default for Dct8x8Table {
    fn default() -> Self {
        Self::new()
    }
}

impl Dct8x8Table {
    pub fn new() -> Self {
        let mut matrix = [[0.0f32; 8]; 8];
        let mut matrix_t = [[0.0f32; 8]; 8];

        for u in 0..8 {
            let alpha = if u == 0 {
                (1.0 / 8.0f32).sqrt()
            } else {
                (2.0 / 8.0f32).sqrt()
            };
            for x in 0..8 {
                let coeff = alpha * ((2.0 * x as f32 + 1.0) * u as f32 * PI / 16.0).cos();
                matrix[u][x] = coeff;
                matrix_t[x][u] = coeff;
            }
        }

        Self { matrix, matrix_t }
    }

    /// Forward 2D DCT-II: Y = C * X * C^T
    pub fn forward(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        let mut temp = [0.0f32; 64];

        // 1. Row transforms: temp = X * C^T
        for r in 0..8 {
            for c in 0..8 {
                let mut sum = 0.0f32;
                for k in 0..8 {
                    sum += input[r * 8 + k] * self.matrix_t[k][c];
                }
                temp[r * 8 + c] = sum;
            }
        }

        // 2. Column transforms: Y = C * temp
        for r in 0..8 {
            for c in 0..8 {
                let mut sum = 0.0f32;
                for k in 0..8 {
                    sum += self.matrix[r][k] * temp[k * 8 + c];
                }
                output[r * 8 + c] = sum;
            }
        }
    }

    /// Inverse 2D IDCT-II: X = C^T * Y * C
    pub fn inverse(&self, input: &[f32; 64], output: &mut [f32; 64]) {
        let mut temp = [0.0f32; 64];

        // 1. Row transforms: temp = Y * C
        for r in 0..8 {
            for c in 0..8 {
                let mut sum = 0.0f32;
                for k in 0..8 {
                    sum += input[r * 8 + k] * self.matrix[k][c];
                }
                temp[r * 8 + c] = sum;
            }
        }

        // 2. Column transforms: X = C^T * temp
        for r in 0..8 {
            for c in 0..8 {
                let mut sum = 0.0f32;
                for k in 0..8 {
                    sum += self.matrix_t[r][k] * temp[k * 8 + c];
                }
                output[r * 8 + c] = sum;
            }
        }
    }
}

thread_local! {
    pub static DCT_TABLE: Dct8x8Table = Dct8x8Table::new();
}

/// Convenience forward 2D DCT-II on an 8x8 block.
pub fn dct_8x8(input: &[f32; 64], output: &mut [f32; 64]) {
    DCT_TABLE.with(|table| table.forward(input, output));
}

/// Convenience inverse 2D IDCT-II on an 8x8 block.
pub fn idct_8x8(input: &[f32; 64], output: &mut [f32; 64]) {
    DCT_TABLE.with(|table| table.inverse(input, output));
}
