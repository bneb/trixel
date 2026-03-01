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
    // Top of 'A' with true transparency: L L D L L
    // (corners are halo because they're 8-adjacent to the apex stroke)
    assert_eq!(glyph[0][0], Some(2), "A[0][0] should be halo (adjacent to stroke)");
    assert_eq!(glyph[0][1], Some(2), "A[0][1] should be halo");
    assert_eq!(glyph[0][2], Some(0), "A[0][2] should be dark stroke");
    assert_eq!(glyph[0][3], Some(2), "A[0][3] should be halo");
    assert_eq!(glyph[0][4], Some(2), "A[0][4] should be halo (adjacent to stroke)");
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
    // With true transparency, 'A' has 2 free cells (bottom corners)
    assert!(free > 0, "glyph 'A' should have free cells, got {free}");
}

#[test]
fn all_defined_glyphs_are_5x7() {
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .!?-:#/".chars() {
        let glyph = glyphs::get_glyph(ch);
        assert_eq!(glyph.len(), GLYPH_HEIGHT, "glyph '{ch}' height");
        for (r, row) in glyph.iter().enumerate() {
            assert_eq!(row.len(), GLYPH_WIDTH, "glyph '{ch}' row {r} width");
        }
    }
}

#[test]
fn all_glyphs_use_only_valid_states() {
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .!?-:#/".chars() {
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
fn most_letter_glyphs_have_free_cells() {
    // Most letters should have at least some free/transparent cells.
    // Very dense letters (G, Q, R) may have 0 — that's correct with true
    // transparency, since every non-stroke pixel is still halo-adjacent.
    let mut total_free = 0;
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ".chars() {
        let glyph = glyphs::get_glyph(ch);
        total_free += free_cell_count(&glyph);
    }
    // Collectively, glyphs should have meaningful transparency
    assert!(total_free > 30,
        "letters should have significant transparency, got total free = {total_free}");
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
    // The backing plate border extends 1 cell outside the glyph origin
    assert_eq!(min_x, 4, "backing plate border should start at x=4 (glyph x=5 minus 1)");
    assert_eq!(min_y, 9, "backing plate border should start at y=9 (glyph y=10 minus 1)");
}

#[test]
fn multi_char_gap_is_transparent() {
    // "AB" has a 1-column transparent gap between A and B at x=5.
    // With true transparency, the gap emits NO constraints.
    let constraints = TernacFont::string_to_constraints("AB", 0, 0);
    let gap_x = GLYPH_WIDTH; // column 5 = gap
    let gap_constraints: Vec<&ConstraintMask> = constraints.iter()
        .filter(|c| c.x == gap_x)
        .collect();
    assert_eq!(gap_constraints.len(), 0,
        "transparent gap should have 0 constraints, got {}", gap_constraints.len());
}

#[test]
fn lowercase_treated_as_uppercase() {
    let upper = TernacFont::string_to_constraints("A", 0, 0);
    let lower = TernacFont::string_to_constraints("a", 0, 0);
    assert_eq!(upper, lower);
}

// ---------------------------------------------------------------------------
// Backing Plate Tests
// ---------------------------------------------------------------------------

#[test]
fn text_has_light_backing_plate_border() {
    // Place "A" at (5, 5). The glyph is 5×7.
    // The backing plate border should add 1-cell padding on all sides:
    //   x range: 4 to 10  (5-1 to 5+5)
    //   y range: 4 to 12  (5-1 to 5+7)
    let constraints = TernacFont::string_to_constraints("A", 5, 5);

    // Build a lookup set of all constrained positions
    let constrained: std::collections::HashMap<(usize, usize), u8> = constraints.iter()
        .map(|c| ((c.x, c.y), c.required_state))
        .collect();

    // Top border row (y=4): all cells from x=4..=10 should be State 2
    for x in 4..=10 {
        assert_eq!(constrained.get(&(x, 4)), Some(&2),
            "top border at ({}, 4) should be State 2", x);
    }

    // Bottom border row (y=12): all cells from x=4..=10 should be State 2
    for x in 4..=10 {
        assert_eq!(constrained.get(&(x, 12)), Some(&2),
            "bottom border at ({}, 12) should be State 2", x);
    }

    // Left border column (x=4): all cells from y=5..=11 should be State 2
    for y in 5..=11 {
        assert_eq!(constrained.get(&(4, y)), Some(&2),
            "left border at (4, {}) should be State 2", y);
    }

    // Right border column (x=10): all cells from y=5..=11 should be State 2
    for y in 5..=11 {
        assert_eq!(constrained.get(&(10, y)), Some(&2),
            "right border at (10, {}) should be State 2", y);
    }
}

#[test]
fn multi_char_backing_plate_spans_full_width() {
    // "AB" at (2, 3): width = 5+1+5 = 11, height = 7  
    // Backing plate: x from 1 to 13, y from 2 to 10
    let constraints = TernacFont::string_to_constraints("AB", 2, 3);

    let constrained: std::collections::HashMap<(usize, usize), u8> = constraints.iter()
        .map(|c| ((c.x, c.y), c.required_state))
        .collect();

    // Top border row (y=2)
    for x in 1..=13 {
        assert_eq!(constrained.get(&(x, 2)), Some(&2),
            "top border at ({}, 2) should be State 2", x);
    }

    // Bottom border row (y=10)
    for x in 1..=13 {
        assert_eq!(constrained.get(&(x, 10)), Some(&2),
            "bottom border at ({}, 10) should be State 2", x);
    }
}
