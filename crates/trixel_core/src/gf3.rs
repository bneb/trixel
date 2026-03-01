//! # GF(3⁶) Galois Field Arithmetic
//!
//! Finite field with 729 elements (0–728). Each element is a degree-≤5
//! polynomial over GF(3), stored as a `u16` in base-3:
//!
//!   value = c₀ + c₁·3 + c₂·9 + c₃·27 + c₄·81 + c₅·243
//!
//! Addition/subtraction are trit-wise mod 3.
//! Multiplication/division are O(1) via pre-computed exp/log tables.

/// Order of the multiplicative group: 3⁶ - 1 = 728.
pub const FIELD_ORDER: usize = 728;

/// Number of elements in the field: 3⁶ = 729.
pub const FIELD_SIZE: usize = 729;

/// Number of trits per symbol.
pub const TRITS_PER_SYMBOL: usize = 6;

/// Primitive polynomial over GF(3): p(x) = x⁶ + x⁵ + 2.
///
/// Verified primitive by exhaustive search: α has multiplicative order exactly 728.
///
/// Reduction rule: x⁶ ≡ −(x⁵ + 2) ≡ 2x⁵ + 1  (mod 3).
///
/// Stored as the coefficients of x⁶ mod p(x), for degrees 0..5.
const REDUCE: [u8; 6] = [1, 0, 0, 0, 0, 2];

// ---------------------------------------------------------------------------
// GF3 Engine
// ---------------------------------------------------------------------------

/// Pre-computed GF(3⁶) lookup tables for O(1) multiplication and division.
pub struct GF3 {
    /// exp_table[i] = α^i for i in 0..728. Indices ≥ 728 wrap via mod.
    exp_table: [u16; FIELD_ORDER],
    /// log_table[v] = i where α^i = v, for v in 1..728. log_table[0] is unused.
    log_table: [u16; FIELD_SIZE],
}

impl GF3 {
    /// Initialize the field by generating exp/log tables from the primitive polynomial.
    pub fn new() -> Self {
        let mut exp_table = [0u16; FIELD_ORDER];
        let mut log_table = [0u16; FIELD_SIZE];

        // α^0 = 1
        let mut current = 1u16;
        for i in 0..FIELD_ORDER {
            exp_table[i] = current;
            log_table[current as usize] = i as u16;
            current = mul_by_alpha(current);
        }

        GF3 {
            exp_table,
            log_table,
        }
    }

    /// Look up α^i. Wraps at 728 (since α^728 = 1).
    #[inline]
    pub fn exp(&self, i: usize) -> u16 {
        self.exp_table[i % FIELD_ORDER]
    }

    /// Look up log_α(a). Panics if a == 0 (log(0) is undefined).
    #[inline]
    pub fn log(&self, a: u16) -> usize {
        debug_assert!(a > 0, "log(0) is undefined in GF(3^6)");
        self.log_table[a as usize] as usize
    }

    /// Trit-wise addition mod 3.
    #[inline]
    pub fn add(&self, a: u16, b: u16) -> u16 {
        trit_add(a, b)
    }

    /// Trit-wise subtraction mod 3: sub(a, b) = a + (-b).
    #[inline]
    pub fn sub(&self, a: u16, b: u16) -> u16 {
        trit_sub(a, b)
    }

    /// Multiplication via log/exp tables. O(1).
    #[inline]
    pub fn mul(&self, a: u16, b: u16) -> u16 {
        if a == 0 || b == 0 {
            return 0;
        }
        let log_sum = self.log(a) + self.log(b);
        self.exp(log_sum)
    }

    /// Division via log/exp tables. O(1). Panics if b == 0.
    #[inline]
    pub fn div(&self, a: u16, b: u16) -> u16 {
        debug_assert!(b != 0, "division by zero in GF(3^6)");
        if a == 0 {
            return 0;
        }
        let log_diff = self.log(a) + FIELD_ORDER - self.log(b);
        self.exp(log_diff)
    }

    /// Multiplicative inverse: inv(a) = 1/a. Panics if a == 0.
    #[inline]
    pub fn inv(&self, a: u16) -> u16 {
        self.div(1, a)
    }

    /// Exponentiation: a^n in GF(3⁶).
    #[inline]
    pub fn pow(&self, a: u16, n: u32) -> u16 {
        if n == 0 {
            return 1; // a^0 = 1, even for a=0 by convention
        }
        if a == 0 {
            return 0;
        }
        let log_a = self.log(a) as u64;
        let exp = (log_a * n as u64) % FIELD_ORDER as u64;
        self.exp(exp as usize)
    }
}

// ---------------------------------------------------------------------------
// Trit-level Operations (standalone, no tables needed)
// ---------------------------------------------------------------------------

/// Trit-wise addition of two GF(3⁶) elements mod 3.
#[inline]
fn trit_add(a: u16, b: u16) -> u16 {
    let mut result = 0u16;
    let mut a = a;
    let mut b = b;
    let mut power = 1u16;
    for _ in 0..TRITS_PER_SYMBOL {
        let ta = a % 3;
        let tb = b % 3;
        result += ((ta + tb) % 3) * power;
        a /= 3;
        b /= 3;
        power *= 3;
    }
    result
}

/// Trit-wise subtraction of two GF(3⁶) elements mod 3.
#[inline]
fn trit_sub(a: u16, b: u16) -> u16 {
    let mut result = 0u16;
    let mut a = a;
    let mut b = b;
    let mut power = 1u16;
    for _ in 0..TRITS_PER_SYMBOL {
        let ta = a % 3;
        let tb = b % 3;
        result += ((ta + 3 - tb) % 3) * power;
        a /= 3;
        b /= 3;
        power *= 3;
    }
    result
}

/// Multiply a field element by the primitive element α (= x).
///
/// Given element = c₀ + c₁x + c₂x² + c₃x³ + c₄x⁴ + c₅x⁵,
/// compute element × x, reducing x⁶ via the primitive polynomial.
///
/// For p(x) = x⁶ + 2x + 1:  x⁶ ≡ x + 2  (mod 3).
/// So element × x = c₅·(REDUCE) + shift-up of c₀..c₄.
fn mul_by_alpha(val: u16) -> u16 {
    let mut c = [0u8; TRITS_PER_SYMBOL];
    let mut v = val;
    for trit in c.iter_mut() {
        *trit = (v % 3) as u8;
        v /= 3;
    }

    let high = c[5]; // coefficient being shifted out (multiplied by x → x⁶)

    // Shift coefficients up: c[i] = c[i-1] for i = 5..1, c[0] = 0
    let mut new_c = [0u8; TRITS_PER_SYMBOL];
    for i in 1..TRITS_PER_SYMBOL {
        new_c[i] = c[i - 1];
    }

    // Add the reduction: high * REDUCE[i] for each position
    for i in 0..TRITS_PER_SYMBOL {
        new_c[i] = (new_c[i] + high * REDUCE[i]) % 3;
    }

    // Pack back to u16
    let mut result = 0u16;
    let mut power = 1u16;
    for &t in &new_c {
        result += t as u16 * power;
        power *= 3;
    }
    result
}

// ---------------------------------------------------------------------------
// Symbol ↔ Trit Conversion (public API)
// ---------------------------------------------------------------------------

/// Convert a GF(3⁶) symbol (0–728) to its 6-trit representation (LSB first).
pub fn symbol_to_trits(s: u16) -> [u8; TRITS_PER_SYMBOL] {
    debug_assert!(s < FIELD_SIZE as u16, "symbol {s} out of range");
    let mut trits = [0u8; TRITS_PER_SYMBOL];
    let mut v = s;
    for t in trits.iter_mut() {
        *t = (v % 3) as u8;
        v /= 3;
    }
    trits
}

/// Convert a 6-trit array (LSB first) to a GF(3⁶) symbol.
/// Each trit must be 0, 1, or 2.
pub fn trits_to_symbol(trits: &[u8]) -> u16 {
    debug_assert!(trits.len() >= TRITS_PER_SYMBOL);
    let mut val = 0u16;
    let mut power = 1u16;
    for &t in trits.iter().take(TRITS_PER_SYMBOL) {
        debug_assert!(t <= 2, "invalid trit value {t}");
        val += t as u16 * power;
        power *= 3;
    }
    val
}
