//! Tests for GF(3^6) Galois Field arithmetic.
//!
//! These tests are written BEFORE the implementation (TDD red phase).
//! They define the exact mathematical contract the field engine must satisfy.

use ternac_core::gf3::{self, GF3};

// ---------------------------------------------------------------------------
// Table Integrity
// ---------------------------------------------------------------------------

#[test]
fn exp_table_generates_all_728_nonzero_elements() {
    let gf = GF3::new();
    let mut seen = [false; 729];
    for i in 0..728 {
        let val = gf.exp(i);
        assert!(
            val > 0 && val < 729,
            "exp[{i}] = {val}, expected 1..728"
        );
        assert!(
            !seen[val as usize],
            "exp[{i}] = {val} is a duplicate (primitive polynomial is not primitive!)"
        );
        seen[val as usize] = true;
    }
    // 0 must never appear in the exp table
    assert!(!seen[0], "0 appeared in exp table");
    // Every non-zero element must appear
    for v in 1u16..729 {
        assert!(seen[v as usize], "element {v} missing from exp table");
    }
}

#[test]
fn exp_0_is_one() {
    let gf = GF3::new();
    assert_eq!(gf.exp(0), 1, "α^0 must be 1 (multiplicative identity)");
}

#[test]
fn log_exp_are_inverses() {
    let gf = GF3::new();
    for a in 1u16..729 {
        let l = gf.log(a);
        assert_eq!(gf.exp(l), a, "exp(log({a})) = {} ≠ {a}", gf.exp(l));
    }
}

#[test]
fn exp_wraps_at_728() {
    let gf = GF3::new();
    // α^728 = α^0 = 1 (Fermat's little theorem in finite fields)
    assert_eq!(gf.exp(0), gf.exp(728), "exp(728) must equal exp(0) = 1");
    // α^729 = α^1
    assert_eq!(gf.exp(1), gf.exp(729), "exp(729) must equal exp(1)");
}

// ---------------------------------------------------------------------------
// Addition / Subtraction (trit-wise mod 3)
// ---------------------------------------------------------------------------

#[test]
fn add_identity() {
    let gf = GF3::new();
    for a in 0u16..729 {
        assert_eq!(gf.add(a, 0), a, "add({a}, 0) must be {a}");
        assert_eq!(gf.add(0, a), a, "add(0, {a}) must be {a}");
    }
}

#[test]
fn add_commutative() {
    let gf = GF3::new();
    for a in 0u16..729 {
        for b in 0u16..729 {
            assert_eq!(
                gf.add(a, b),
                gf.add(b, a),
                "add({a},{b}) ≠ add({b},{a})"
            );
        }
    }
}

#[test]
fn add_sub_inverse_exhaustive() {
    let gf = GF3::new();
    // For all a, b: sub(add(a, b), b) == a
    for a in 0u16..729 {
        for b in 0u16..729 {
            let sum = gf.add(a, b);
            let recovered = gf.sub(sum, b);
            assert_eq!(
                recovered, a,
                "sub(add({a},{b}), {b}) = sub({sum},{b}) = {recovered} ≠ {a}"
            );
        }
    }
}

#[test]
fn additive_inverse_exists() {
    let gf = GF3::new();
    // For every element a, there exists -a such that a + (-a) = 0
    for a in 0u16..729 {
        let neg_a = gf.sub(0, a);
        assert_eq!(
            gf.add(a, neg_a),
            0,
            "add({a}, neg({a})={neg_a}) must be 0"
        );
    }
}

// ---------------------------------------------------------------------------
// Multiplication / Division (via log/exp tables)
// ---------------------------------------------------------------------------

#[test]
fn mul_identity() {
    let gf = GF3::new();
    for a in 0u16..729 {
        assert_eq!(gf.mul(a, 1), a, "mul({a}, 1) must be {a}");
        assert_eq!(gf.mul(1, a), a, "mul(1, {a}) must be {a}");
    }
}

#[test]
fn mul_zero() {
    let gf = GF3::new();
    for a in 0u16..729 {
        assert_eq!(gf.mul(a, 0), 0, "mul({a}, 0) must be 0");
        assert_eq!(gf.mul(0, a), 0, "mul(0, {a}) must be 0");
    }
}

#[test]
fn mul_commutative() {
    let gf = GF3::new();
    for a in 1u16..729 {
        for b in 1u16..729 {
            assert_eq!(
                gf.mul(a, b),
                gf.mul(b, a),
                "mul({a},{b}) ≠ mul({b},{a})"
            );
        }
    }
}

#[test]
fn mul_div_inverse_exhaustive() {
    let gf = GF3::new();
    // For all non-zero a: mul(a, div(1, a)) == 1
    for a in 1u16..729 {
        let inv_a = gf.div(1, a);
        let product = gf.mul(a, inv_a);
        assert_eq!(
            product, 1,
            "mul({a}, inv({a})={inv_a}) = {product} ≠ 1"
        );
    }
}

#[test]
fn div_self_is_one() {
    let gf = GF3::new();
    for a in 1u16..729 {
        assert_eq!(gf.div(a, a), 1, "div({a}, {a}) must be 1");
    }
}

#[test]
fn mul_associative_spot_check() {
    let gf = GF3::new();
    // Check associativity: (a*b)*c == a*(b*c) for a sample of values
    let values = [1u16, 2, 3, 42, 100, 500, 727, 728];
    for &a in &values {
        for &b in &values {
            for &c in &values {
                let ab_c = gf.mul(gf.mul(a, b), c);
                let a_bc = gf.mul(a, gf.mul(b, c));
                assert_eq!(
                    ab_c, a_bc,
                    "associativity failed: ({a}*{b})*{c} = {ab_c} ≠ {a}*({b}*{c}) = {a_bc}"
                );
            }
        }
    }
}

#[test]
fn distributive_spot_check() {
    let gf = GF3::new();
    // a * (b + c) == a*b + a*c
    let values = [0u16, 1, 2, 3, 42, 100, 500, 727, 728];
    for &a in &values {
        for &b in &values {
            for &c in &values {
                let lhs = gf.mul(a, gf.add(b, c));
                let rhs = gf.add(gf.mul(a, b), gf.mul(a, c));
                assert_eq!(
                    lhs, rhs,
                    "distributivity failed: {a}*({b}+{c}) = {lhs} ≠ {a}*{b}+{a}*{c} = {rhs}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Power / Inverse
// ---------------------------------------------------------------------------

#[test]
fn pow_basic() {
    let gf = GF3::new();
    for a in 1u16..729 {
        assert_eq!(gf.pow(a, 0), 1, "pow({a}, 0) must be 1");
        assert_eq!(gf.pow(a, 1), a, "pow({a}, 1) must be {a}");
    }
    assert_eq!(gf.pow(0, 0), 1, "pow(0, 0) = 1 by convention");
    assert_eq!(gf.pow(0, 5), 0, "pow(0, n>0) must be 0");
}

#[test]
fn pow_matches_repeated_mul() {
    let gf = GF3::new();
    let values = [2u16, 3, 5, 100, 500, 728];
    for &a in &values {
        let mut expected = 1u16;
        for n in 0u32..10 {
            assert_eq!(
                gf.pow(a, n),
                expected,
                "pow({a}, {n}) = {} ≠ expected {expected}",
                gf.pow(a, n)
            );
            expected = gf.mul(expected, a);
        }
    }
}

#[test]
fn inv_is_div_one() {
    let gf = GF3::new();
    for a in 1u16..729 {
        assert_eq!(gf.inv(a), gf.div(1, a), "inv({a}) must equal div(1, {a})");
    }
}

// ---------------------------------------------------------------------------
// Symbol ↔ Trit Conversion
// ---------------------------------------------------------------------------

#[test]
fn symbol_trit_roundtrip_exhaustive() {
    for s in 0u16..729 {
        let trits = gf3::symbol_to_trits(s);
        // All trits must be 0, 1, or 2
        for (i, &t) in trits.iter().enumerate() {
            assert!(t <= 2, "symbol_to_trits({s})[{i}] = {t}, expected 0-2");
        }
        let back = gf3::trits_to_symbol(&trits);
        assert_eq!(back, s, "trits_to_symbol(symbol_to_trits({s})) = {back} ≠ {s}");
    }
}

#[test]
fn symbol_0_is_all_zero_trits() {
    let trits = gf3::symbol_to_trits(0);
    assert_eq!(trits, [0, 0, 0, 0, 0, 0]);
}

#[test]
fn symbol_728_is_max() {
    let trits = gf3::symbol_to_trits(728);
    // 728 in base 3: 728 = 2+2*3+2*9+2*27+2*81+2*243 = 2*(1+3+9+27+81+243) = 2*364 = 728
    assert_eq!(trits, [2, 2, 2, 2, 2, 2]);
}

#[test]
fn known_trit_values() {
    // 1 = 1*3^0
    assert_eq!(gf3::symbol_to_trits(1), [1, 0, 0, 0, 0, 0]);
    // 3 = 1*3^1
    assert_eq!(gf3::symbol_to_trits(3), [0, 1, 0, 0, 0, 0]);
    // 5 = 2*3^0 + 1*3^1
    assert_eq!(gf3::symbol_to_trits(5), [2, 1, 0, 0, 0, 0]);
    // 243 = 1*3^5
    assert_eq!(gf3::symbol_to_trits(243), [0, 0, 0, 0, 0, 1]);
}
