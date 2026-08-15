//! # prism_render
//!
//! Forward perceptual embedding engine for PrismCode.
//! Combines LDPC coding, 2D spatial interleaving, Yang-Bovik JND masking, and Fourier pilot generation.

pub mod diffusion;
pub mod embedder;
pub mod metrics;

pub use diffusion::{diffuse_error, ErrorDiffuser};
pub use embedder::{
    EmbedConfig, EmbedError, PrismEmbedder, GRID_BLOCKS_X, GRID_BLOCKS_Y, TOTAL_CODEWORD_BITS,
};
pub use metrics::{compute_psnr, compute_ssim};
