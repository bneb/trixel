//! # prism_hvs
//!
//! Human Visual System (HVS) perceptual modeling and frequency-domain transforms.
//! Provides sRGB <-> CIELAB color conversions, 8x8 2D-DCT-II transforms, and Yang-Bovik JND masking.

pub mod color;
pub mod dct;
pub mod jnd;

pub use color::{srgb_to_lab, lab_to_srgb, Lab};
pub use dct::{dct_8x8, idct_8x8, Dct8x8Table, BLOCK_SIZE};
pub use jnd::compute_spatial_jnd;
