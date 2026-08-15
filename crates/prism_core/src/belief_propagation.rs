//! Belief Propagation decoders over Log-Likelihood Ratios (LLRs).
//!
//! A CSR (compressed sparse row) view of the QC-LDPC parity-check matrix is
//! built once in [`BeliefPropagation::new`], then reused across decode runs.
//! Two message-passing decoders are provided:
//!
//! - [`BeliefPropagation::decode_min_sum`] — Normalized Min-Sum with a tunable
//!   normalization factor `alpha`. This is the exact algorithm previously
//!   inlined in `QcLdpcCodec::decode_min_sum`, which now delegates here.
//! - [`BeliefPropagation::decode_sum_product`] — Sum-Product in the tanh /
//!   phi domain.
//!
//! Both run a hard-decision syndrome check every iteration and stop early on
//! convergence, bounded by the configured `max_iterations`.

use crate::ldpc::{LdpcError, QcLdpcCodec};

/// Messages with |LLR| below this carry no information (LLR 0.0 is an erased
/// bit); the Sum-Product check update treats them as infinite-uncertainty edges.
const ERASURE_EPS: f32 = 1e-6;

/// Hard cap on outgoing check-to-variable message magnitudes. A check whose
/// incoming messages are all saturated would otherwise emit ±inf (phi-sum of
/// zero), which can propagate NaN through variable-node sums.
const MSG_CLAMP: f32 = 30.0;

/// Belief Propagation decoder over a CSR representation of the parity-check
/// matrix.
pub struct BeliefPropagation {
    n_vars: usize,
    n_checks: usize,
    k_info_bits: usize,
    max_iterations: usize,
    /// CSR row offsets over H, in check-major order (len `n_checks + 1`).
    row_ptr: Vec<usize>,
    /// Variable index of each edge, in CSR order (len `total_edges`).
    col_idx: Vec<usize>,
    /// Global edge id of each CSR position (len `total_edges`).
    edge_id: Vec<usize>,
    /// Per-variable edge-list offsets (len `n_vars + 1`).
    var_ptr: Vec<usize>,
    /// Global edge id of each variable-side edge (len `total_edges`).
    var_edge: Vec<usize>,
}

impl BeliefPropagation {
    /// Builds a decoder for `codec`, copying only the parity-check topology
    /// into CSR form. `max_iterations` caps each decode run.
    pub fn new(codec: QcLdpcCodec, max_iterations: usize) -> Self {
        let n_vars = codec.n_vars;
        let n_checks = codec.n_checks;
        let total_edges = codec.total_edges;
        let k_info_bits = codec.k_info_bits();

        // Check-major (CSR) view: edges ordered by check row, then by the
        // codec's own edge-id assignment (row-major over the base matrix).
        let mut row_ptr = Vec::with_capacity(n_checks + 1);
        let mut col_idx = Vec::with_capacity(total_edges);
        let mut edge_id = Vec::with_capacity(total_edges);
        row_ptr.push(0);
        for check in &codec.check_nodes {
            for (i, &v) in check.var_indices.iter().enumerate() {
                col_idx.push(v);
                edge_id.push(check.edge_indices[i]);
            }
            row_ptr.push(col_idx.len());
        }

        // Variable-major view: edge ids grouped per variable, in ascending
        // edge-id order (identical to the codec's variable node lists, so the
        // Min-Sum message accumulation order is preserved exactly).
        let mut var_ptr = Vec::with_capacity(n_vars + 1);
        let mut var_edge = Vec::with_capacity(total_edges);
        var_ptr.push(0);
        for var in &codec.variable_nodes {
            for &e in &var.edge_indices {
                var_edge.push(e);
            }
            var_ptr.push(var_edge.len());
        }

        Self {
            n_vars,
            n_checks,
            k_info_bits,
            max_iterations,
            row_ptr,
            col_idx,
            edge_id,
            var_ptr,
            var_edge,
        }
    }

    #[inline]
    pub fn k_info_bits(&self) -> usize {
        self.k_info_bits
    }

    #[inline]
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    #[inline]
    pub fn n_checks(&self) -> usize {
        self.n_checks
    }

    #[inline]
    pub fn total_edges(&self) -> usize {
        self.edge_id.len()
    }

    /// Degree (number of incident edges) of every check node, in check order.
    pub fn check_degrees(&self) -> Vec<usize> {
        self.row_ptr.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// Degree (number of incident edges) of every variable node, in variable
    /// order.
    pub fn variable_degrees(&self) -> Vec<usize> {
        self.var_ptr.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// Check if hard decision bits satisfy H * c^T = 0 mod 2.
    pub fn verify_syndrome(&self, bits: &[u8]) -> bool {
        if bits.len() != self.n_vars {
            return false;
        }
        for c in 0..self.n_checks {
            let mut sum = 0u8;
            for j in self.row_ptr[c]..self.row_ptr[c + 1] {
                sum ^= bits[self.col_idx[j]];
            }
            if sum != 0 {
                return false;
            }
        }
        true
    }

    /// Normalized Min-Sum decoding over channel LLRs. Returns the extracted
    /// k info bits on success (syndrome-verified hard decision).
    pub fn decode_min_sum(
        &self,
        channel_llrs: &[f32],
        alpha: f32,
    ) -> Result<Vec<u8>, LdpcError> {
        if channel_llrs.len() != self.n_vars {
            return Err(LdpcError::InvalidSize {
                expected: self.n_vars,
                actual: channel_llrs.len(),
            });
        }

        let mut v2c = vec![0.0f32; self.total_edges()];
        let mut c2v = vec![0.0f32; self.total_edges()];

        // Initialize V2C messages with channel LLRs.
        for c in 0..self.n_checks {
            for j in self.row_ptr[c]..self.row_ptr[c + 1] {
                v2c[self.edge_id[j]] = channel_llrs[self.col_idx[j]];
            }
        }

        let mut hard_bits = vec![0u8; self.n_vars];
        let mut total_llrs = vec![0.0f32; self.n_vars];

        for _ in 0..self.max_iterations {
            // --- 1. Check node update (normalized min-sum) ---
            for c in 0..self.n_checks {
                let lo = self.row_ptr[c];
                let hi = self.row_ptr[c + 1];
                if lo == hi {
                    continue;
                }

                let mut min1 = f32::MAX;
                let mut min2 = f32::MAX;
                let mut min1_pos = lo;
                let mut sign_prod = 1.0f32;

                for j in lo..hi {
                    let val = v2c[self.edge_id[j]];
                    if val < 0.0 {
                        sign_prod = -sign_prod;
                    }
                    let abs_val = val.abs();
                    if abs_val < min1 {
                        min2 = min1;
                        min1 = abs_val;
                        min1_pos = j;
                    } else if abs_val < min2 {
                        min2 = abs_val;
                    }
                }

                for j in lo..hi {
                    let e = self.edge_id[j];
                    let val = v2c[e];
                    let s = if val < 0.0 { -sign_prod } else { sign_prod };
                    let m = if j == min1_pos { min2 } else { min1 };
                    c2v[e] = alpha * s * m;
                }
            }

            // --- 2. Variable node update & marginal LLR ---
            total_llrs.copy_from_slice(channel_llrs);
            for (v, span) in self.var_ptr.windows(2).enumerate() {
                for j in span[0]..span[1] {
                    total_llrs[v] += c2v[self.var_edge[j]];
                }
                hard_bits[v] = if total_llrs[v] < 0.0 { 1 } else { 0 };
            }

            // --- 3. Syndrome verification ---
            if self.verify_syndrome(&hard_bits) {
                return Ok(hard_bits[..self.k_info_bits].to_vec());
            }

            // --- 4. Update V2C for the next iteration ---
            for (v, span) in self.var_ptr.windows(2).enumerate() {
                for j in span[0]..span[1] {
                    let e = self.var_edge[j];
                    v2c[e] = total_llrs[v] - c2v[e];
                }
            }
        }

        Err(LdpcError::MaxIterationsExceeded(self.max_iterations))
    }

    /// Sum-Product decoding over channel LLRs (tanh domain via the phi
    /// transform). Returns the extracted k info bits on success.
    pub fn decode_sum_product(&self, channel_llrs: &[f32]) -> Result<Vec<u8>, LdpcError> {
        if channel_llrs.len() != self.n_vars {
            return Err(LdpcError::InvalidSize {
                expected: self.n_vars,
                actual: channel_llrs.len(),
            });
        }

        let mut v2c = vec![0.0f32; self.total_edges()];
        let mut c2v = vec![0.0f32; self.total_edges()];

        for c in 0..self.n_checks {
            for j in self.row_ptr[c]..self.row_ptr[c + 1] {
                v2c[self.edge_id[j]] = channel_llrs[self.col_idx[j]];
            }
        }

        let mut hard_bits = vec![0u8; self.n_vars];
        let mut total_llrs = vec![0.0f32; self.n_vars];

        for _ in 0..self.max_iterations {
            // --- 1. Check node update (Sum-Product, phi domain) ---
            for c in 0..self.n_checks {
                let lo = self.row_ptr[c];
                let hi = self.row_ptr[c + 1];
                if lo == hi {
                    continue;
                }

                let mut sum_phi = 0.0f32;
                let mut sign_prod = 1.0f32;
                let mut erased = 0usize;
                let mut erased_pos = lo;

                for j in lo..hi {
                    let val = v2c[self.edge_id[j]];
                    if val < 0.0 {
                        sign_prod = -sign_prod;
                    }
                    if val.abs() < ERASURE_EPS {
                        erased += 1;
                        erased_pos = j;
                    } else {
                        sum_phi += phi(val);
                    }
                }

                match erased {
                    // No erasures: exact tanh-domain combination.
                    0 => {
                        for j in lo..hi {
                            let e = self.edge_id[j];
                            let val = v2c[e];
                            let s = if val < 0.0 { -sign_prod } else { sign_prod };
                            c2v[e] = s * phi_inv(sum_phi - phi(val));
                        }
                    }
                    // One erasure: it receives info from the other edges
                    // (their tanh product); everyone else gets a zero message
                    // (their product includes tanh(0) = 0).
                    1 => {
                        for j in lo..hi {
                            if j != erased_pos {
                                c2v[self.edge_id[j]] = 0.0;
                            }
                        }
                        let e = self.edge_id[erased_pos];
                        // Exclude the erased edge's own (noise-level) sign.
                        let own_sign = if v2c[e] < 0.0 { -1.0f32 } else { 1.0f32 };
                        c2v[e] = (sign_prod / own_sign) * phi_inv(sum_phi);
                    }
                    // Two or more erasures: every product contains a zero.
                    _ => {
                        for j in lo..hi {
                            c2v[self.edge_id[j]] = 0.0;
                        }
                    }
                }
            }

            // --- 2. Variable node update & marginal LLR ---
            total_llrs.copy_from_slice(channel_llrs);
            for (v, span) in self.var_ptr.windows(2).enumerate() {
                for j in span[0]..span[1] {
                    total_llrs[v] += c2v[self.var_edge[j]];
                }
                hard_bits[v] = if total_llrs[v] < 0.0 { 1 } else { 0 };
            }

            // --- 3. Syndrome verification ---
            if self.verify_syndrome(&hard_bits) {
                return Ok(hard_bits[..self.k_info_bits].to_vec());
            }

            // --- 4. Update V2C for the next iteration ---
            for (v, span) in self.var_ptr.windows(2).enumerate() {
                for j in span[0]..span[1] {
                    let e = self.var_edge[j];
                    v2c[e] = total_llrs[v] - c2v[e];
                }
            }
        }

        Err(LdpcError::MaxIterationsExceeded(self.max_iterations))
    }
}

/// phi(x) = -ln(tanh(|x| / 2)); maps a certainty (large |x|) to 0 and an
/// uncertainty to a large positive value. Only called with |x| >= ERASURE_EPS,
/// so no ln(0).
fn phi(x: f32) -> f32 {
    let ax = x.abs();
    -((ax * 0.5).tanh()).ln()
}

/// Inverse of `phi`: phi_inv(0) is a saturated certainty, capped at
/// [`MSG_CLAMP`] to keep the message state finite.
fn phi_inv(y: f32) -> f32 {
    if y >= 1e-6 {
        let e = (-y).exp();
        ((1.0 + e) / (1.0 - e)).ln()
    } else {
        MSG_CLAMP
    }
}
