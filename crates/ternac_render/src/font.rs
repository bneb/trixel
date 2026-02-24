//! # FontEngine Implementation
//!
//! Rasterizes text strings into `ConstraintMask` arrays using the
//! Constrained Halo glyph design. Only `Some(0)` and `Some(2)` cells
//! produce constraints; `None` cells are left free for Z3.

use crate::glyphs::{self, GLYPH_HEIGHT, GLYPH_WIDTH};
use ternac_solver::ConstraintMask;

/// Production font engine using 5×7 Constrained Halo glyphs.
///
/// - Stroke pixels (`Some(0)`) → `ConstraintMask { required_state: 0 }`
/// - Halo pixels (`Some(2)`) → `ConstraintMask { required_state: 2 }`
/// - Free pixels (`None`) → **no constraint emitted** (Z3 chooses)
/// - 1-column gap between characters: all `Some(2)` (light separator)
pub struct TernacFont;

impl super::FontEngine for TernacFont {
    fn string_to_constraints(
        text: &str,
        start_x: usize,
        start_y: usize,
    ) -> Vec<ConstraintMask> {
        let mut constraints = Vec::new();
        let mut cursor_x = start_x;

        for (i, ch) in text.chars().enumerate() {
            // Add 1-column gap between characters (not before first)
            if i > 0 {
                for row in 0..GLYPH_HEIGHT {
                    constraints.push(ConstraintMask {
                        x: cursor_x,
                        y: start_y + row,
                        required_state: 2, // gap = light
                    });
                }
                cursor_x += 1;
            }

            // Rasterize the glyph — only constrained cells
            let glyph = glyphs::get_glyph(ch);
            for row in 0..GLYPH_HEIGHT {
                for col in 0..GLYPH_WIDTH {
                    if let Some(state) = glyph[row][col] {
                        constraints.push(ConstraintMask {
                            x: cursor_x + col,
                            y: start_y + row,
                            required_state: state,
                        });
                    }
                    // None → no constraint, Z3 is free to choose
                }
            }
            cursor_x += GLYPH_WIDTH;
        }

        constraints
    }
}
