//! # prism_sync
//!
//! Fourier spread-spectrum spatial synchronization and homography estimation.
//! Provides 2D pilot pattern generation, fast in-place 2D FFT, spectral peak
//! detection with signed sub-bin affine recovery, and DLT homography estimation.

pub mod pilot;
pub mod fft;
pub mod detector;
pub mod phase;
pub mod homography;

pub use pilot::{generate_pilot_grid, PilotConfig};
pub use fft::{fft2d, ifft2d};
pub use detector::{detect_pilot_peaks, estimate_affine, estimate_rotation_and_scale, SyncError, PeakCoord};
pub use phase::estimate_pilot_translation;
pub use homography::{apply_homography, invert_homography, solve_dlt};
