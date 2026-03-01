//! # FontEngine Implementation
//!
//! Rasterizes text strings into `ConstraintMask` arrays using the
//! Constrained Halo glyph design. Only `Some(0)` and `Some(2)` cells
//! produce constraints; `None` cells are left free for the solver.
//!
//! After rasterizing all glyphs, a 1-cell light backing plate border
//! is emitted around the text bounding box for readability.

use crate::glyphs::{self, GLYPH_HEIGHT, GLYPH_WIDTH};
use trixel_solver::ConstraintMask;

/// Production font engine using 5×7 Constrained Halo glyphs.
///
/// - Stroke pixels (`Some(0)`) → `ConstraintMask { required_state: 0 }`
/// - Halo pixels (`Some(2)`) → `ConstraintMask { required_state: 2 }`
/// - Free pixels (`None`) → **no constraint emitted** (solver chooses)
/// - 1-column transparent gap between characters
/// - 1-cell light backing plate border around the entire text block
pub struct TrixelFont;

impl super::FontEngine for TrixelFont {
    fn string_to_constraints(
        text: &str,
        start_x: usize,
        start_y: usize,
    ) -> Vec<ConstraintMask> {
        let mut constraints = Vec::new();
        let mut cursor_x = start_x;
        let char_count = text.chars().count();

        for (i, ch) in text.chars().enumerate() {
            // 1-column transparent gap between characters (no constraints emitted)
            if i > 0 {
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
                    // None → no constraint, solver is free to choose
                }
            }
            cursor_x += GLYPH_WIDTH;
        }

        // ---------------------------------------------------------------
        // Backing Plate: 1-cell light border around the text bounding box
        // ---------------------------------------------------------------
        if char_count > 0 && start_x > 0 && start_y > 0 {
            let text_width = char_count * GLYPH_WIDTH + (char_count - 1); // glyphs + gaps
            let text_height = GLYPH_HEIGHT;

            // Border coordinates (1 cell outside each edge)
            let bx0 = start_x - 1;             // left border
            let bx1 = start_x + text_width;    // right border
            let by0 = start_y - 1;             // top border
            let by1 = start_y + text_height;   // bottom border

            // Collect border positions, avoiding duplicates with glyph cells
            let glyph_positions: std::collections::HashSet<(usize, usize)> =
                constraints.iter().map(|c| (c.x, c.y)).collect();

            // Top row: y=by0, x from bx0 to bx1
            for x in bx0..=bx1 {
                if !glyph_positions.contains(&(x, by0)) {
                    constraints.push(ConstraintMask { x, y: by0, required_state: 2 });
                }
            }
            // Bottom row: y=by1, x from bx0 to bx1
            for x in bx0..=bx1 {
                if !glyph_positions.contains(&(x, by1)) {
                    constraints.push(ConstraintMask { x, y: by1, required_state: 2 });
                }
            }
            // Left column: x=bx0, y from start_y to start_y+text_height-1
            for y in start_y..start_y + text_height {
                if !glyph_positions.contains(&(bx0, y)) {
                    constraints.push(ConstraintMask { x: bx0, y, required_state: 2 });
                }
            }
            // Right column: x=bx1, y from start_y to start_y+text_height-1
            for y in start_y..start_y + text_height {
                if !glyph_positions.contains(&(bx1, y)) {
                    constraints.push(ConstraintMask { x: bx1, y, required_state: 2 });
                }
            }
        }

        constraints
    }
}
