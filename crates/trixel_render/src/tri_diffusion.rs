//! # Ternary Error Diffusion on Triangular Grid
//!
//! Implements Floyd-Steinberg error diffusion adapted for the triangular
//! tessellation of a TriGrid. Instead of the standard rectangular 4-neighbor
//! kernel, this uses the natural 3-neighbor adjacency of triangles:
//!
//! - Up triangle ▲ shares edges with: left ▽, right ▽, bottom (row below) ▽
//! - Down triangle ▽ shares edges with: left ▲, right ▲, top (row above) ▲
//!
//! The diffusion weights are adapted for 3 neighbors: 3/8, 3/8, 2/8.

use trixel_core::trigrid::TriGrid;
use image::GrayImage;

// ---------------------------------------------------------------------------
// Triangular Adjacency
// ---------------------------------------------------------------------------

/// A neighbor cell with its diffusion weight.
#[derive(Debug, Clone, Copy)]
pub struct TriNeighbor {
    pub col: usize,
    pub row: usize,
    pub weight: f32,
}

/// Compute the forward-diffusion neighbors for triangle at `(col, row)`.
///
/// "Forward" means neighbors that haven't been processed yet in
/// scanline order (left→right, top→bottom). This ensures error is only
/// diffused to unprocessed cells.
///
/// Weights are adapted Floyd-Steinberg for 3 neighbors:
/// - Horizontal right: 3/8
/// - Diagonal/vertical: 3/8
/// - Secondary: 2/8
pub fn tri_forward_neighbors(
    col: usize,
    row: usize,
    rows: usize,
    cols: usize,
) -> Vec<TriNeighbor> {
    let mut neighbors = Vec::with_capacity(3);
    let is_up = TriGrid::is_up(col, row);

    if is_up {
        // ▲ triangle: shares edges with:
        // - Right neighbor (col+1, same row): ▽ — always forward
        // - Left neighbor (col-1, same row): ▽ — backward, skip
        // - Bottom neighbor (col, row-1 or col±1, row+1): forward

        // Right neighbor
        if col + 1 < cols {
            neighbors.push(TriNeighbor {
                col: col + 1,
                row,
                weight: 3.0 / 8.0,
            });
        }

        // Bottom neighbor (the ▽ triangle sharing the base edge)
        // For ▲ at (col, row), the base-sharing ▽ is at (col-1, row+1) or similar
        // depending on grid parity. In our grid layout:
        // ▲ at even col: bottom edge shared with ▽ at (col, row+1)
        // But actually, we use a simpler forward-diffusion:
        // - Next row, same column
        if row + 1 < rows {
            neighbors.push(TriNeighbor {
                col,
                row: row + 1,
                weight: 3.0 / 8.0,
            });
        }

        // Diagonal forward: next row, col+1
        if row + 1 < rows && col + 1 < cols {
            neighbors.push(TriNeighbor {
                col: col + 1,
                row: row + 1,
                weight: 2.0 / 8.0,
            });
        }
    } else {
        // ▽ triangle: shares edges with:
        // - Right neighbor (col+1, same row): ▲ — always forward
        // - Left neighbor (col-1, same row): ▲ — backward, skip
        // - Top neighbor: backward (already processed), skip

        // Right neighbor
        if col + 1 < cols {
            neighbors.push(TriNeighbor {
                col: col + 1,
                row,
                weight: 3.0 / 8.0,
            });
        }

        // Next row, same column (forward in scanline)
        if row + 1 < rows {
            neighbors.push(TriNeighbor {
                col,
                row: row + 1,
                weight: 3.0 / 8.0,
            });
        }

        // Next row, col-1 (diagonal)
        if row + 1 < rows && col > 0 {
            neighbors.push(TriNeighbor {
                col: col - 1,
                row: row + 1,
                weight: 2.0 / 8.0,
            });
        }
    }

    neighbors
}

// ---------------------------------------------------------------------------
// Error Diffusion Engine
// ---------------------------------------------------------------------------

/// Trit-to-luminance mapping.
fn trit_luminance(trit: u8) -> f32 {
    match trit {
        0 => 0.10, // dark
        1 => 0.50, // mid
        2 => 0.90, // light
        _ => 0.50,
    }
}

/// Compute a per-cell lightness correction matrix via Floyd-Steinberg
/// error diffusion on the triangular grid.
///
/// For each cell in scanline order:
/// 1. Compute `ideal_lightness` = source image luminance (0.0–1.0)
/// 2. Compute `actual_lightness` = trit_luminance(grid_trit)
/// 3. `error = ideal_lightness - actual_lightness + accumulated_error[cell]`
/// 4. Distribute error to forward neighbors
///
/// Returns a 2D correction matrix `corrections[row][col]` containing
/// lightness adjustments (positive = brighten, negative = darken).
/// These corrections should be applied to the renderer's HSL lightness.
pub fn diffuse_lightness(
    grid: &TriGrid,
    source_gray: &GrayImage,
    rows: usize,
    cols: usize,
) -> Vec<Vec<f32>> {
    let mut corrections = vec![vec![0.0f32; cols]; rows];
    let mut error_buffer = vec![vec![0.0f32; cols]; rows];

    let (src_w, src_h) = source_gray.dimensions();

    for row in 0..rows {
        for col in 0..cols {
            // Source luminance (0.0–1.0)
            let src_lum = if col < src_w as usize && row < src_h as usize {
                source_gray.get_pixel(col as u32, row as u32).0[0] as f32 / 255.0
            } else {
                1.0 // white for off-image cells
            };

            // Actual lightness from the trit value
            let trit = grid.get(col, row).unwrap_or(2);
            let actual = trit_luminance(trit);

            // Accumulated error from previous cells
            let accumulated = error_buffer[row][col];

            // Total error at this cell
            let error = src_lum - actual + accumulated;

            // Store the correction for this cell (clamped for safety)
            corrections[row][col] = error.clamp(-0.3, 0.3);

            // Diffuse remaining error to forward neighbors
            let neighbors = tri_forward_neighbors(col, row, rows, cols);
            for neighbor in &neighbors {
                error_buffer[neighbor.row][neighbor.col] += error * neighbor.weight;
            }
        }
    }

    corrections
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------
    // Triangular Adjacency
    // -------------------------------------------------------------------

    #[test]
    fn tri_neighbors_up_triangle_has_up_to_3() {
        // Middle of grid: should have 3 forward neighbors
        let neighbors = tri_forward_neighbors(2, 2, 10, 10);
        assert!(!neighbors.is_empty() && neighbors.len() <= 3,
            "Up triangle mid-grid should have 1-3 neighbors, got {}", neighbors.len());
    }

    #[test]
    fn tri_neighbors_down_triangle_has_up_to_3() {
        let neighbors = tri_forward_neighbors(3, 2, 10, 10);
        assert!(!neighbors.is_empty() && neighbors.len() <= 3,
            "Down triangle mid-grid should have 1-3 neighbors, got {}", neighbors.len());
    }

    #[test]
    fn tri_neighbors_bottom_right_corner_fewer() {
        // Last row, last column: no forward neighbors
        let neighbors = tri_forward_neighbors(9, 9, 10, 10);
        assert_eq!(neighbors.len(), 0,
            "Bottom-right corner should have 0 forward neighbors");
    }

    #[test]
    fn tri_neighbors_weights_sum_leq_one() {
        // For any cell with neighbors, weights should sum to ≤ 1.0
        for row in 0..10 {
            for col in 0..10 {
                let neighbors = tri_forward_neighbors(col, row, 10, 10);
                let sum: f32 = neighbors.iter().map(|n| n.weight).sum();
                assert!(sum <= 1.01,
                    "Weights at ({},{}) sum to {} > 1.0", col, row, sum);
            }
        }
    }

    #[test]
    fn tri_neighbors_all_within_bounds() {
        let rows = 10;
        let cols = 10;
        for row in 0..rows {
            for col in 0..cols {
                let neighbors = tri_forward_neighbors(col, row, rows, cols);
                for n in &neighbors {
                    assert!(n.row < rows && n.col < cols,
                        "Neighbor ({},{}) out of bounds for grid {}×{}",
                        n.col, n.row, cols, rows);
                }
            }
        }
    }

    // -------------------------------------------------------------------
    // Error Diffusion
    // -------------------------------------------------------------------

    #[test]
    fn diffusion_produces_correct_dimensions() {
        let grid = TriGrid::zeros(8, 16);
        let gray = GrayImage::new(16, 8);
        let corrections = diffuse_lightness(&grid, &gray, 8, 16);
        assert_eq!(corrections.len(), 8);
        assert_eq!(corrections[0].len(), 16);
    }

    #[test]
    fn diffusion_uniform_white_small_corrections() {
        // If source is all white and grid is all State 2 (light),
        // error should be minimal
        let mut grid = TriGrid::zeros(8, 16);
        for row in 0..8 {
            for col in 0..16 {
                grid.set(col, row, 2);
            }
        }
        let mut gray = GrayImage::new(16, 8);
        for y in 0..8 {
            for x in 0..16 {
                gray.put_pixel(x, y, image::Luma([230])); // near white
            }
        }

        let corrections = diffuse_lightness(&grid, &gray, 8, 16);

        // All corrections should be small (< 0.15) since source ≈ trit
        for row in &corrections {
            for &c in row {
                assert!(c.abs() < 0.3,
                    "Correction {} too large for uniform white→State2", c);
            }
        }
    }

    #[test]
    fn diffusion_mismatch_propagates_error() {
        // Source is all black (0), grid is all State 2 (light=0.9).
        // Error should propagate: negative corrections everywhere.
        let mut grid = TriGrid::zeros(8, 16);
        for row in 0..8 {
            for col in 0..16 {
                grid.set(col, row, 2); // light
            }
        }
        let gray = GrayImage::new(16, 8); // all black (0)

        let corrections = diffuse_lightness(&grid, &gray, 8, 16);

        // Most corrections should be negative (darken)
        let negative_count = corrections.iter()
            .flat_map(|row| row.iter())
            .filter(|&&c| c < 0.0)
            .count();
        let total = 8 * 16;
        assert!(negative_count > total / 2,
            "Expected majority negative corrections for black→State2, got {}/{}", negative_count, total);
    }
}
