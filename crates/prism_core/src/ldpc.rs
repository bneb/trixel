//! Quasi-Cyclic Low-Density Parity-Check (QC-LDPC) Codec.
//!
//! Features:
//! - Lower-triangular base matrix for O(N) linear-time systematic encoding.
//! - Belief Propagation decoding over Log-Likelihood Ratios (LLRs), implemented
//!   in [`crate::belief_propagation`] (Normalized Min-Sum and Sum-Product).
//! - Syndrome-checked early stopping.

use thiserror::Error;

use crate::belief_propagation::BeliefPropagation;

#[derive(Error, Debug)]
pub enum LdpcError {
    #[error("Maximum decoding iterations ({0}) exceeded without convergence")]
    MaxIterationsExceeded(usize),
    #[error("Invalid input size: expected {expected}, got {actual}")]
    InvalidSize { expected: usize, actual: usize },
}

#[derive(Clone, Debug)]
pub struct LdpcConfig {
    pub z: usize,                // Circulant block size (e.g. 32)
    pub num_info_blocks: usize,  // K_b (e.g. 4)
    pub num_parity_blocks: usize,// M_b (e.g. 8)
    pub base_matrix: Vec<Vec<i32>>, // M_b x N_b matrix of cyclic shifts (-1 = zero block)
}

impl LdpcConfig {
    /// Rate-1/3 QC-LDPC code with Z=32 (K=128 info bits / 16 bytes, N=384 codeword bits).
    /// Optimized for high coding gain under heavy AWGN and camera optical noise.
    pub fn rate_one_third_z32() -> Self {
        let z = 32;
        let num_info_blocks = 4;
        let num_parity_blocks = 8;

        // 8x12 Base Proto-Matrix: Information columns 0..4, Parity columns 4..12 (Lower-Triangular)
        let base_matrix: Vec<Vec<i32>> = vec![
            // Row 0
            vec![16,  7, 25, 12,   0, -1, -1, -1, -1, -1, -1, -1],
            // Row 1
            vec![23, 19,  4, 28,   3,  0, -1, -1, -1, -1, -1, -1],
            // Row 2
            vec![ 5, 14, 21,  9,  -1,  7,  0, -1, -1, -1, -1, -1],
            // Row 3
            vec![18, 30,  2, 17,  -1, -1, 11,  0, -1, -1, -1, -1],
            // Row 4
            vec![11,  8, 29, 22,  -1, -1, -1, 15,  0, -1, -1, -1],
            // Row 5
            vec![27, 13, 10,  6,  -1, -1, -1, -1, 19,  0, -1, -1],
            // Row 6
            vec![ 3, 26, 15, 31,  -1, -1, -1, -1, -1, 23,  0, -1],
            // Row 7
            vec![14,  1, 18, 24,  -1, -1, -1, -1, -1, -1, 27,  0],
        ];

        Self {
            z,
            num_info_blocks,
            num_parity_blocks,
            base_matrix,
        }
    }

    /// Rate-1/3 QC-LDPC code with Z=64 (K=256 info bits / 32 bytes, N=768 codeword bits).
    /// Accommodates standard web URLs up to 32 bytes with high coding gain.
    pub fn rate_one_third_z64() -> Self {
        let z = 64;
        let num_info_blocks = 4;
        let num_parity_blocks = 8;

        let base_matrix: Vec<Vec<i32>> = vec![
            vec![32, 15, 51, 24,   0, -1, -1, -1, -1, -1, -1, -1],
            vec![47, 39,  8, 57,   7,  0, -1, -1, -1, -1, -1, -1],
            vec![11, 29, 43, 19,  -1, 15,  0, -1, -1, -1, -1, -1],
            vec![37, 61,  5, 35,  -1, -1, 23,  0, -1, -1, -1, -1],
            vec![23, 17, 59, 45,  -1, -1, -1, 31,  0, -1, -1, -1],
            vec![55, 27, 21, 13,  -1, -1, -1, -1, 39,  0, -1, -1],
            vec![ 7, 53, 31, 63,  -1, -1, -1, -1, -1, 47,  0, -1],
            vec![29,  3, 37, 49,  -1, -1, -1, -1, -1, -1, 55,  0],
        ];

        Self {
            z,
            num_info_blocks,
            num_parity_blocks,
            base_matrix,
        }
    }
}

/// Structure representing edges connected to a check node.
#[derive(Clone, Debug)]
pub(crate) struct CheckNodeEdges {
    pub(crate) var_indices: Vec<usize>,
    pub(crate) edge_indices: Vec<usize>,
}

/// Structure representing edges connected to a variable node.
#[derive(Clone, Debug)]
pub(crate) struct VariableNodeEdges {
    pub(crate) edge_indices: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct QcLdpcCodec {
    config: LdpcConfig,
    pub(crate) n_vars: usize,
    pub(crate) n_checks: usize,
    pub(crate) check_nodes: Vec<CheckNodeEdges>,
    pub(crate) variable_nodes: Vec<VariableNodeEdges>,
    pub(crate) total_edges: usize,
}

impl QcLdpcCodec {
    pub fn new(config: LdpcConfig) -> Self {
        let z = config.z;
        let m_b = config.num_parity_blocks;
        let n_b = config.num_info_blocks + config.num_parity_blocks;
        let n_vars = n_b * z;
        let n_checks = m_b * z;

        let mut check_nodes = vec![CheckNodeEdges { var_indices: Vec::new(), edge_indices: Vec::new() }; n_checks];
        let mut variable_nodes = vec![VariableNodeEdges { edge_indices: Vec::new() }; n_vars];
        let mut edge_count = 0;

        for r_b in 0..m_b {
            for c_b in 0..n_b {
                let shift = config.base_matrix[r_b][c_b];
                if shift >= 0 {
                    let s = shift as usize % z;
                    for k in 0..z {
                        let check_idx = r_b * z + k;
                        let var_idx = c_b * z + (k + s) % z;
                        let e = edge_count;

                        check_nodes[check_idx].var_indices.push(var_idx);
                        check_nodes[check_idx].edge_indices.push(e);
                        variable_nodes[var_idx].edge_indices.push(e);

                        edge_count += 1;
                    }
                }
            }
        }

        Self {
            config,
            n_vars,
            n_checks,
            check_nodes,
            variable_nodes,
            total_edges: edge_count,
        }
    }

    #[inline]
    pub fn k_info_bits(&self) -> usize {
        self.config.num_info_blocks * self.config.z
    }

    #[inline]
    pub fn n_code_bits(&self) -> usize {
        self.n_vars
    }

    /// Systematic encode: takes k_info_bits and returns n_code_bits.
    /// Fast O(N) linear time via lower-triangular forward-substitution.
    pub fn encode(&self, info_bits: &[u8]) -> Vec<u8> {
        let z = self.config.z;
        let k_b = self.config.num_info_blocks;
        let m_b = self.config.num_parity_blocks;
        assert_eq!(info_bits.len(), k_b * z, "Info bits length mismatch");

        let mut codeword = vec![0u8; (k_b + m_b) * z];
        codeword[..k_b * z].copy_from_slice(info_bits);

        // Solve for each parity block p_0, p_1, ..., p_{m_b-1}
        for r_b in 0..m_b {
            let mut acc = vec![0u8; z];

            // 1. Contribution from info blocks
            for c_b in 0..k_b {
                let shift = self.config.base_matrix[r_b][c_b];
                if shift >= 0 {
                    let s = shift as usize % z;
                    let block = &codeword[c_b * z..(c_b + 1) * z];
                    for k in 0..z {
                        acc[k] ^= block[(k + s) % z];
                    }
                }
            }

            // 2. Contribution from previously solved parity blocks
            for c_b in 0..r_b {
                let shift = self.config.base_matrix[r_b][k_b + c_b];
                if shift >= 0 {
                    let s = shift as usize % z;
                    let block = &codeword[(k_b + c_b) * z..(k_b + c_b + 1) * z];
                    for k in 0..z {
                        acc[k] ^= block[(k + s) % z];
                    }
                }
            }

            // Since base_matrix[r_b][k_b + r_b] == 0 (Identity), p_{r_b} is directly acc!
            let p_start = (k_b + r_b) * z;
            codeword[p_start..p_start + z].copy_from_slice(&acc);
        }

        codeword
    }

    /// Check if hard decision bits satisfy H * c^T = 0 mod 2.
    pub fn verify_syndrome(&self, bits: &[u8]) -> bool {
        if bits.len() != self.n_vars {
            return false;
        }
        for check in &self.check_nodes {
            let mut sum = 0u8;
            for &v in &check.var_indices {
                sum ^= bits[v];
            }
            if sum != 0 {
                return false;
            }
        }
        true
    }

    /// Normalized Min-Sum Belief Propagation Decoder over Channel LLRs.
    /// Returns extracted k info bits on success.
    ///
    /// Delegates to [`BeliefPropagation::decode_min_sum`] (identical
    /// algorithm: same message schedule and normalization).
    pub fn decode_min_sum(
        &self,
        channel_llrs: &[f32],
        max_iterations: usize,
        alpha: f32,
    ) -> Result<Vec<u8>, LdpcError> {
        BeliefPropagation::new(self.clone(), max_iterations).decode_min_sum(channel_llrs, alpha)
    }
}
