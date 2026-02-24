use ternac_render::{FontEngine, TernacFont};
use ternac_render::glyphs::{self, GLYPH_WIDTH, GLYPH_HEIGHT, free_cell_count};
use ternac_solver::ConstraintMask;

// ---------------------------------------------------------------------------
// Glyph Bitmap Tests
// ---------------------------------------------------------------------------

#[test]
fn glyph_a_has_correct_dimensions() {
    let glyph = glyphs::get_glyph('A');
    assert_eq!(glyph.len(), GLYPH_HEIGHT);
    for row in &glyph {
        assert_eq!(row.len(), GLYPH_WIDTH);
    }
}

#[test]
fn glyph_a_top_row_matches_halo_pattern() {
    let glyph = glyphs::get_glyph('A');
    // Top of 'A': F L D L F
    assert_eq!(glyph[0][0], None,    "A[0][0] should be free");
    assert_eq!(glyph[0][1], Some(2), "A[0][1] should be light halo");
    assert_eq!(glyph[0][2], Some(0), "A[0][2] should be dark stroke");
    assert_eq!(glyph[0][3], Some(2), "A[0][3] should be light halo");
    assert_eq!(glyph[0][4], None,    "A[0][4] should be free");
}

#[test]
fn glyph_a_crossbar_is_all_dark() {
    let glyph = glyphs::get_glyph('A');
    // Row 3 of 'A' should be all dark: D D D D D
    for col in 0..GLYPH_WIDTH {
        assert_eq!(glyph[3][col], Some(0), "A crossbar at row 3, col {col}");
    }
}

#[test]
fn glyph_a_has_free_cells() {
    let glyph = glyphs::get_glyph('A');
    let free = free_cell_count(&glyph);
    assert!(free > 0, "glyph 'A' should have free cells for Z3");
    // Bottom row is all free × 5, plus corners = at least 9
    assert!(free >= 9, "glyph 'A' should have at least 9 free cells, got {free}");
}

#[test]
fn all_defined_glyphs_are_5x7() {
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .!?-:#".chars() {
        let glyph = glyphs::get_glyph(ch);
        assert_eq!(glyph.len(), GLYPH_HEIGHT, "glyph '{ch}' height");
        for (r, row) in glyph.iter().enumerate() {
            assert_eq!(row.len(), GLYPH_WIDTH, "glyph '{ch}' row {r} width");
        }
    }
}

#[test]
fn all_glyphs_use_only_valid_states() {
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .!?-:#".chars() {
        let glyph = glyphs::get_glyph(ch);
        for row in &glyph {
            for &cell in row {
                match cell {
                    None | Some(0) | Some(2) => {} // valid
                    Some(s) => panic!("glyph '{ch}' has invalid state {s} (only 0, 2, None allowed)"),
                }
            }
        }
    }
}

#[test]
fn all_letter_glyphs_have_free_cells() {
    // Every letter glyph should have at least some free cells for Z3
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ".chars() {
        let glyph = glyphs::get_glyph(ch);
        let free = free_cell_count(&glyph);
        assert!(free > 0, "glyph '{ch}' should have free cells, got 0");
    }
}

#[test]
fn unknown_char_returns_blank_glyph() {
    let glyph = glyphs::get_glyph('€');
    // Blank glyph: all None (free)
    for row in &glyph {
        for &cell in row {
            assert_eq!(cell, None, "unknown char glyph should be all free");
        }
    }
}

// ---------------------------------------------------------------------------
// FontEngine Tests
// ---------------------------------------------------------------------------

#[test]
fn single_char_constraint_count_less_than_total() {
    // Constrained Halo: not all cells produce constraints (free cells don't)
    let constraints = TernacFont::string_to_constraints("A", 0, 0);
    let total_cells = GLYPH_WIDTH * GLYPH_HEIGHT;
    assert!(
        constraints.len() < total_cells,
        "halo font should produce fewer constraints ({}) than total cells ({total_cells})",
        constraints.len()
    );
    assert!(!constraints.is_empty(), "should have some constraints");
}

#[test]
fn constraints_only_contain_state_0_and_2() {
    let constraints = TernacFont::string_to_constraints("HELLO", 0, 0);
    for c in &constraints {
        assert!(c.required_state == 0 || c.required_state == 2,
            "constraint at ({},{}) has state {} (only 0 and 2 allowed)",
            c.x, c.y, c.required_state
        );
    }
}

#[test]
fn constraints_have_both_dark_and_light() {
    let constraints = TernacFont::string_to_constraints("A", 0, 0);
    let dark = constraints.iter().filter(|c| c.required_state == 0).count();
    let light = constraints.iter().filter(|c| c.required_state == 2).count();
    assert!(dark > 0, "should have dark constraints (strokes)");
    assert!(light > 0, "should have light constraints (halos)");
}

#[test]
fn constraints_respect_start_offset() {
    let constraints = TernacFont::string_to_constraints("A", 5, 10);
    let min_x = constraints.iter().map(|c| c.x).min().unwrap();
    let min_y = constraints.iter().map(|c| c.y).min().unwrap();
    assert_eq!(min_x, 5, "constraints should start at x=5");
    assert_eq!(min_y, 10, "constraints should start at y=10");
}

#[test]
fn multi_char_gap_is_constrained_light() {
    // "AB" has a 1-column gap between A and B at x=5
    let constraints = TernacFont::string_to_constraints("AB", 0, 0);
    let gap_x = GLYPH_WIDTH; // column 5 = gap
    let gap_constraints: Vec<&ConstraintMask> = constraints.iter()
        .filter(|c| c.x == gap_x)
        .collect();
    assert_eq!(gap_constraints.len(), GLYPH_HEIGHT, "gap should have {GLYPH_HEIGHT} constraints");
    for c in &gap_constraints {
        assert_eq!(c.required_state, 2, "gap cell at ({},{}) should be light", c.x, c.y);
    }
}

#[test]
fn lowercase_treated_as_uppercase() {
    let upper = TernacFont::string_to_constraints("A", 0, 0);
    let lower = TernacFont::string_to_constraints("a", 0, 0);
    assert_eq!(upper, lower);
}
