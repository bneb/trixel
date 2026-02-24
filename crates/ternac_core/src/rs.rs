//! # Reed-Solomon Encoder/Decoder over GF(3⁶)
//!
//! **Codeword format**: `c[0..parity]` = parity, `c[parity..n]` = data.
//! The polynomial c(x) is divisible by the generator g(x).
//!
//! - **Encoding**: Polynomial long division.
//! - **Decoding**: Sugiyama's Extended Euclidean Algorithm (handles errors + erasures).

use crate::gf3::{GF3, FIELD_ORDER};

/// Reed-Solomon codec parameterized by the number of parity symbols.
pub struct ReedSolomon {
    pub parity_count: usize,
    generator: Vec<u16>,
}

impl ReedSolomon {
    /// Construct an RS codec. Generator: g(x) = ∏_{i=1}^{parity_count} (x − α^i).
    pub fn new(gf: &GF3, parity_count: usize) -> Self {
        ReedSolomon {
            parity_count,
            generator: build_generator(gf, parity_count),
        }
    }

    /// Systematic RS encoding: returns `[parity || data]`.
    pub fn encode(&self, gf: &GF3, data: &[u16]) -> Vec<u16> {
        let n = data.len() + self.parity_count;
        let mut codeword = vec![0u16; n];

        for (i, &d) in data.iter().enumerate() {
            codeword[self.parity_count + i] = d;
        }

        if data.is_empty() {
            return codeword;
        }

        // Polynomial long division: divide data*x^{2t} by g(x)
        let mut remainder = codeword.clone();
        let gen_lead_inv = gf.inv(self.generator[self.parity_count]);

        for i in (self.parity_count..n).rev() {
            let coeff = remainder[i];
            if coeff == 0 {
                continue;
            }
            let factor = gf.mul(coeff, gen_lead_inv);
            for j in 0..=self.parity_count {
                let term = gf.mul(factor, self.generator[j]);
                remainder[i - self.parity_count + j] = gf.sub(
                    remainder[i - self.parity_count + j],
                    term,
                );
            }
        }

        for i in 0..self.parity_count {
            codeword[i] = gf.sub(0, remainder[i]);
        }

        codeword
    }

    /// Decode a codeword, correcting errors and erasures.
    ///
    /// Uses Sugiyama's Extended Euclidean Algorithm to solve the key equation,
    /// then Chien search + Forney to locate and correct.
    ///
    /// Capacity: 2e + E ≤ 2t (e=errors, E=erasures, 2t=parity_count).
    pub fn decode(
        &self,
        gf: &GF3,
        codeword: &[u16],
        erasure_positions: &[usize],
    ) -> Result<Vec<u16>, RsError> {
        let n = codeword.len();
        let two_t = self.parity_count;

        // 1. Syndromes: S[i] = c(α^{i+1})
        let syndromes = compute_syndromes(gf, codeword, two_t);

        if syndromes.iter().all(|&s| s == 0) {
            return Ok(codeword[self.parity_count..].to_vec());
        }

        let erasure_count = erasure_positions.len();
        if erasure_count > two_t {
            return Err(RsError::TooManyErasures {
                erasures: erasure_count,
                capacity: two_t,
            });
        }

        // 2. Build erasure locator
        let erasure_locator = build_erasure_locator(gf, erasure_positions);

        // 3. Build syndrome polynomial: S(x) = S[0] + S[1]*x + ... + S[2t-1]*x^{2t-1}
        //    Modified: T(x) = S(x) * Λ_e(x) mod x^{2t}
        let syndrome_poly = syndromes.clone();
        let modified = poly_mod_xn(gf, &poly_mul(gf, &syndrome_poly, &erasure_locator), two_t);

        // 4. Solve key equation via Extended Euclidean Algorithm
        //    Find σ(x), ω(x) such that S(x)*σ(x) ≡ ω(x) mod x^{2t}
        //    Stop when deg(ω) < (2t + erasure_count) / 2
        let stop_degree = (two_t + erasure_count) / 2;
        let (sigma, _omega) = extended_euclidean(gf, two_t, &modified, stop_degree);

        // Normalize so σ(0) = 1
        let sigma0 = sigma[0];
        if sigma0 == 0 {
            return Err(RsError::ForneyDegeneracy);
        }
        let sigma0_inv = gf.inv(sigma0);
        let sigma_norm: Vec<u16> = sigma.iter().map(|&c| gf.mul(c, sigma0_inv)).collect();

        // 5. Full locator = σ(x) * Λ_e(x)
        let full_locator = poly_mul(gf, &sigma_norm, &erasure_locator);

        // 6. Full error evaluator: Ω(x) = S(x) * Λ_full(x) mod x^{2t}
        let full_omega = poly_mod_xn(
            gf,
            &poly_mul(gf, &syndrome_poly, &full_locator),
            two_t,
        );

        let total_degree = poly_degree(&full_locator);
        let error_count = total_degree.saturating_sub(erasure_count);

        if 2 * error_count + erasure_count > two_t {
            return Err(RsError::TooManyErrors {
                errors: error_count,
                erasures: erasure_count,
                capacity: two_t,
            });
        }

        // 7. Chien search
        let found = chien_search(gf, &full_locator, n);
        if found.len() != total_degree {
            return Err(RsError::ChienSearchFailed {
                expected: total_degree,
                found: found.len(),
            });
        }

        // 8. Forney correction
        let mut corrected = codeword.to_vec();
        let locator_deriv = poly_formal_derivative(gf, &full_locator);

        for &pos in &found {
            let xi_inv = gf.exp(FIELD_ORDER - (pos % FIELD_ORDER));
            let omega_val = poly_eval(gf, &full_omega, xi_inv);
            let deriv_val = poly_eval(gf, &locator_deriv, xi_inv);

            if deriv_val == 0 {
                return Err(RsError::ForneyDegeneracy);
            }

            let magnitude = gf.div(omega_val, deriv_val);
            corrected[pos] = gf.add(corrected[pos], magnitude);
        }

        Ok(corrected[self.parity_count..].to_vec())
    }
}

// ---------------------------------------------------------------------------
// Error Type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum RsError {
    TooManyErrors {
        errors: usize,
        erasures: usize,
        capacity: usize,
    },
    TooManyErasures {
        erasures: usize,
        capacity: usize,
    },
    ChienSearchFailed {
        expected: usize,
        found: usize,
    },
    ForneyDegeneracy,
}

impl std::fmt::Display for RsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RsError::TooManyErrors { errors, erasures, capacity } =>
                write!(f, "too many errors: 2*{errors} + {erasures} > {capacity}"),
            RsError::TooManyErasures { erasures, capacity } =>
                write!(f, "too many erasures: {erasures} > {capacity}"),
            RsError::ChienSearchFailed { expected, found } =>
                write!(f, "Chien search found {found} roots, expected {expected}"),
            RsError::ForneyDegeneracy =>
                write!(f, "Forney algorithm encountered zero derivative"),
        }
    }
}

impl std::error::Error for RsError {}

// ---------------------------------------------------------------------------
// Generator Polynomial
// ---------------------------------------------------------------------------

fn build_generator(gf: &GF3, parity_count: usize) -> Vec<u16> {
    let mut g = vec![1u16];
    for i in 1..=parity_count {
        let alpha_i = gf.exp(i);
        let neg_alpha_i = gf.sub(0, alpha_i);
        let mut new_g = vec![0u16; g.len() + 1];
        for (j, &coeff) in g.iter().enumerate() {
            new_g[j + 1] = gf.add(new_g[j + 1], coeff);
            new_g[j] = gf.add(new_g[j], gf.mul(neg_alpha_i, coeff));
        }
        g = new_g;
    }
    g
}

// ---------------------------------------------------------------------------
// Syndrome Calculation
// ---------------------------------------------------------------------------

fn compute_syndromes(gf: &GF3, codeword: &[u16], two_t: usize) -> Vec<u16> {
    (0..two_t).map(|i| poly_eval(gf, codeword, gf.exp(i + 1))).collect()
}

// ---------------------------------------------------------------------------
// Erasure Locator
// ---------------------------------------------------------------------------

fn build_erasure_locator(gf: &GF3, erasure_positions: &[usize]) -> Vec<u16> {
    let mut locator = vec![1u16];
    for &pos in erasure_positions {
        let factor = vec![1u16, gf.sub(0, gf.exp(pos))];
        locator = poly_mul(gf, &locator, &factor);
    }
    locator
}

// ---------------------------------------------------------------------------
// Extended Euclidean Algorithm (Sugiyama)
// ---------------------------------------------------------------------------

/// Solve the key equation via the Extended Euclidean Algorithm.
///
/// Input: syndrome polynomial `s(x)` (length ≤ 2t).
/// Computes `gcd(x^{2t}, s(x))` until `deg(remainder) < stop_degree`.
/// Returns `(locator σ(x), evaluator ω(x))`.
fn extended_euclidean(
    gf: &GF3,
    two_t: usize,
    syndrome_poly: &[u16],
    stop_degree: usize,
) -> (Vec<u16>, Vec<u16>) {
    // r_prev = x^{2t}
    let mut r_prev = vec![0u16; two_t + 1];
    r_prev[two_t] = 1;

    // r_curr = syndrome polynomial
    let mut r_curr = syndrome_poly.to_vec();

    // t_prev = 0, t_curr = 1  (tracks the locator)
    let mut t_prev: Vec<u16> = vec![0u16];
    let mut t_curr: Vec<u16> = vec![1u16];

    while poly_degree(&r_curr) >= stop_degree {
        let (quotient, remainder) = poly_div(gf, &r_prev, &r_curr);

        // r_prev, r_curr = r_curr, remainder
        r_prev = r_curr;
        r_curr = remainder;

        // t_prev, t_curr = t_curr, t_prev - quotient * t_curr
        let qt = poly_mul(gf, &quotient, &t_curr);
        let new_t = poly_sub(gf, &t_prev, &qt);
        t_prev = t_curr;
        t_curr = new_t;
    }

    // σ(x) = t_curr,  ω(x) = r_curr
    (poly_trim(&t_curr), poly_trim(&r_curr))
}

// ---------------------------------------------------------------------------
// Polynomial Arithmetic
// ---------------------------------------------------------------------------

fn poly_eval(gf: &GF3, poly: &[u16], x: u16) -> u16 {
    let mut result = 0u16;
    for &coeff in poly.iter().rev() {
        result = gf.add(gf.mul(result, x), coeff);
    }
    result
}

fn poly_mul(gf: &GF3, a: &[u16], b: &[u16]) -> Vec<u16> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut result = vec![0u16; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] = gf.add(result[i + j], gf.mul(ai, bj));
        }
    }
    result
}

/// Polynomial subtraction: a(x) - b(x).
fn poly_sub(gf: &GF3, a: &[u16], b: &[u16]) -> Vec<u16> {
    let len = a.len().max(b.len());
    let mut result = vec![0u16; len];
    for i in 0..a.len() {
        result[i] = a[i];
    }
    for i in 0..b.len() {
        result[i] = gf.sub(result[i], b[i]);
    }
    result
}

/// Polynomial long division: a / b → (quotient, remainder).
fn poly_div(gf: &GF3, a: &[u16], b: &[u16]) -> (Vec<u16>, Vec<u16>) {
    let a_deg = poly_degree(a);
    let b_deg = poly_degree(b);

    if a_deg < b_deg || b[b_deg] == 0 {
        return (vec![0u16], a.to_vec());
    }

    let mut remainder = a.to_vec();
    let b_lead_inv = gf.inv(b[b_deg]);
    let q_len = a_deg - b_deg + 1;
    let mut quotient = vec![0u16; q_len];

    for i in (0..q_len).rev() {
        let idx = i + b_deg;
        if idx >= remainder.len() {
            continue;
        }
        let coeff = gf.mul(remainder[idx], b_lead_inv);
        quotient[i] = coeff;
        if coeff != 0 {
            for j in 0..=b_deg {
                let term = gf.mul(coeff, b[j]);
                remainder[i + j] = gf.sub(remainder[i + j], term);
            }
        }
    }

    (quotient, remainder)
}

/// Truncate polynomial mod x^n (keep only terms of degree < n).
fn poly_mod_xn(_gf: &GF3, poly: &[u16], n: usize) -> Vec<u16> {
    let len = poly.len().min(n);
    poly[..len].to_vec()
}

/// Remove trailing zero coefficients.
fn poly_trim(poly: &[u16]) -> Vec<u16> {
    let mut end = poly.len();
    while end > 1 && poly[end - 1] == 0 {
        end -= 1;
    }
    poly[..end].to_vec()
}

/// Formal derivative over GF(3). Coefficient k multiplied by (k mod 3).
fn poly_formal_derivative(gf: &GF3, poly: &[u16]) -> Vec<u16> {
    if poly.len() <= 1 {
        return vec![0];
    }
    let mut deriv = vec![0u16; poly.len() - 1];
    for i in 1..poly.len() {
        let k_mod3 = (i % 3) as u16;
        if k_mod3 != 0 {
            deriv[i - 1] = gf.mul(poly[i], k_mod3);
        }
    }
    deriv
}

fn poly_degree(poly: &[u16]) -> usize {
    for i in (0..poly.len()).rev() {
        if poly[i] != 0 {
            return i;
        }
    }
    0
}

fn chien_search(gf: &GF3, locator: &[u16], n: usize) -> Vec<usize> {
    let mut positions = Vec::new();
    for i in 0..n {
        let alpha_neg_i = if i == 0 { 1 } else { gf.exp(FIELD_ORDER - (i % FIELD_ORDER)) };
        if poly_eval(gf, locator, alpha_neg_i) == 0 {
            positions.push(i);
        }
    }
    positions
}
