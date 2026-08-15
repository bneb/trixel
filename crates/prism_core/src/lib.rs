//! # prism_core
//!
//! Information theory and soft-decision channel coding engine for PrismCode.
//! Provides Quasi-Cyclic LDPC (QC-LDPC) codecs, Belief Propagation, and 2D spatial interleaving.

pub mod belief_propagation;
pub mod interleaver;
pub mod ldpc;

pub use belief_propagation::BeliefPropagation;
pub use interleaver::SpatialInterleaver;
pub use ldpc::{LdpcConfig, QcLdpcCodec, LdpcError};
