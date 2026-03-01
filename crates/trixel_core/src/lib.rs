//! # trixel_core
//!
//! Core types and math for the Trixel ternary matrix encoding system.
//! Provides the byte↔trit codec, error-correction traits, and shared data types.

pub mod gf3;
pub mod rs;

use thiserror::Error;

// ---------------------------------------------------------------------------
// Shared Types
// ---------------------------------------------------------------------------

/// A 2D grid of trit values (0, 1, 2). Value 3 represents an erasure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TritMatrix {
    pub width: usize,
    pub height: usize,
    /// Row-major data. Valid values: 0, 1, 2, or 3 (erasure).
    pub data: Vec<u8>,
}

impl TritMatrix {
    /// Create a new matrix filled with zeros.
    pub fn zeros(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0; width * height],
        }
    }

    /// Get the trit at `(x, y)`.
    pub fn get(&self, x: usize, y: usize) -> Option<u8> {
        if x < self.width && y < self.height {
            Some(self.data[y * self.width + x])
        } else {
            None
        }
    }

    /// Set the trit at `(x, y)`.
    pub fn set(&mut self, x: usize, y: usize, val: u8) {
        if x < self.width && y < self.height {
            self.data[y * self.width + x] = val;
        }
    }
}

// ---------------------------------------------------------------------------
// Error Types
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("invalid trit value {0} at index {1}")]
    InvalidTrit(u8, usize),
    #[error("trit sequence length {0} is not a multiple of 6")]
    BadLength(usize),
    #[error("trit value {0} overflows a byte")]
    Overflow(u32),
}

#[derive(Debug, Error)]
pub enum EccError {
    #[error("ecc_capacity must be in [0.0, 1.0], got {0}")]
    InvalidCapacity(f32),
    #[error("too many errors to recover ({errors} errors, {erasures} erasures, capacity {capacity})")]
    Unrecoverable {
        errors: usize,
        erasures: usize,
        capacity: usize,
    },
    #[error("payload is empty")]
    EmptyPayload,
}

// ---------------------------------------------------------------------------
// TernaryCodec Trait
// ---------------------------------------------------------------------------

/// Converts between raw bytes and trit arrays.
pub trait TernaryCodec {
    /// Packs standard UTF-8/binary data into an optimized trit array.
    /// Each byte is expanded into 6 trits (⌈log₃ 256⌉ = 6).
    fn encode_bytes(data: &[u8]) -> Result<Vec<u8>, CodecError>;

    /// Reconstructs bytes from a trit array.
    fn decode_trits(trits: &[u8]) -> Result<Vec<u8>, CodecError>;
}

// ---------------------------------------------------------------------------
// ErrorCorrection Trait
// ---------------------------------------------------------------------------

/// Reed-Solomon-style error correction over GF(3ⁿ).
pub trait ErrorCorrection {
    /// Appends parity trits to the payload.
    /// `ecc_capacity` is the fraction of the total output dedicated to parity (0.0–1.0).
    fn apply_parity(payload: &[u8], ecc_capacity: f32) -> Result<Vec<u8>, EccError>;

    /// Attempts to recover the original payload.
    /// Input may contain erasures (value 3).
    fn correct_errors(raw_read: &[u8]) -> Result<Vec<u8>, EccError>;
}

// ---------------------------------------------------------------------------
// Mock Implementations
// ---------------------------------------------------------------------------

/// Mock codec: simple base-3 expansion (6 trits per byte).
pub struct MockCodec;

impl TernaryCodec for MockCodec {
    fn encode_bytes(data: &[u8]) -> Result<Vec<u8>, CodecError> {
        let mut trits = Vec::with_capacity(data.len() * 6);
        for &byte in data {
            let mut val = byte as u32;
            for _ in 0..6 {
                trits.push((val % 3) as u8);
                val /= 3;
            }
        }
        Ok(trits)
    }

    fn decode_trits(trits: &[u8]) -> Result<Vec<u8>, CodecError> {
        if trits.len() % 6 != 0 {
            return Err(CodecError::BadLength(trits.len()));
        }
        let mut bytes = Vec::with_capacity(trits.len() / 6);
        for chunk in trits.chunks(6) {
            let mut val: u32 = 0;
            let mut base: u32 = 1;
            for (i, &t) in chunk.iter().enumerate() {
                if t > 2 {
                    let abs_idx = bytes.len() * 6 + i;
                    return Err(CodecError::InvalidTrit(t, abs_idx));
                }
                val += t as u32 * base;
                base *= 3;
            }
            if val > 255 {
                return Err(CodecError::Overflow(val));
            }
            bytes.push(val as u8);
        }
        Ok(bytes)
    }
}

/// Mock ECC: pads with zeros for parity, strips them on decode.
/// Stores the original payload length as a 12-trit (base-3) prefix.
/// 12 trits can represent 0–531,440 which is more than sufficient.
pub struct MockEcc;

pub const LENGTH_PREFIX_TRITS: usize = 12;

/// Encode a `usize` into a fixed-width base-3 trit sequence (LSB first).
pub fn encode_length(mut len: usize) -> [u8; LENGTH_PREFIX_TRITS] {
    let mut trits = [0u8; LENGTH_PREFIX_TRITS];
    for t in trits.iter_mut() {
        *t = (len % 3) as u8;
        len /= 3;
    }
    trits
}

/// Decode a fixed-width base-3 trit sequence (LSB first) into a `usize`.
pub fn decode_length(trits: &[u8]) -> usize {
    let mut val: usize = 0;
    let mut base: usize = 1;
    for &t in trits.iter().take(LENGTH_PREFIX_TRITS) {
        val += (t as usize) * base;
        base *= 3;
    }
    val
}

impl ErrorCorrection for MockEcc {
    fn apply_parity(payload: &[u8], ecc_capacity: f32) -> Result<Vec<u8>, EccError> {
        if !(0.0..=1.0).contains(&ecc_capacity) {
            return Err(EccError::InvalidCapacity(ecc_capacity));
        }
        if payload.is_empty() {
            return Err(EccError::EmptyPayload);
        }
        let prefixed_len = LENGTH_PREFIX_TRITS + payload.len();
        // Total = prefixed / (1 - ecc_capacity). Parity fills the rest with zeros.
        let total = if ecc_capacity < 1.0 {
            (prefixed_len as f32 / (1.0 - ecc_capacity)).ceil() as usize
        } else {
            prefixed_len * 2
        };
        let parity_len = total.saturating_sub(prefixed_len);

        let len_trits = encode_length(payload.len());
        let mut out = Vec::with_capacity(prefixed_len + parity_len);
        out.extend_from_slice(&len_trits);
        out.extend_from_slice(payload);
        out.resize(prefixed_len + parity_len, 0);
        Ok(out)
    }

    fn correct_errors(raw_read: &[u8]) -> Result<Vec<u8>, EccError> {
        if raw_read.len() < LENGTH_PREFIX_TRITS {
            return Err(EccError::EmptyPayload);
        }
        let len = decode_length(&raw_read[..LENGTH_PREFIX_TRITS]);
        let start = LENGTH_PREFIX_TRITS;
        if start + len > raw_read.len() {
            return Err(EccError::Unrecoverable {
                errors: 0,
                erasures: 0,
                capacity: raw_read.len().saturating_sub(start),
            });
        }
        Ok(raw_read[start..start + len].to_vec())
    }
}

// ---------------------------------------------------------------------------
// Production ECC: Reed-Solomon over GF(3^6)
// ---------------------------------------------------------------------------

use gf3::{GF3, TRITS_PER_SYMBOL};
use rs::ReedSolomon;

/// Production Reed-Solomon error correction over GF(3⁶).
///
/// Converts flat trit arrays ↔ GF(3⁶) symbols, applies RS encoding/decoding.
/// Uses a 3-symbol (18-trit) header:
///   [0] = original_trit_count % 729
///   [1] = original_trit_count / 729
///   [2] = parity_count
pub struct RsEcc;

pub const RS_HEADER_SYMBOLS: usize = 3;

impl ErrorCorrection for RsEcc {
    fn apply_parity(payload: &[u8], ecc_capacity: f32) -> Result<Vec<u8>, EccError> {
        if !(0.0..=1.0).contains(&ecc_capacity) {
            return Err(EccError::InvalidCapacity(ecc_capacity));
        }
        if payload.is_empty() {
            return Err(EccError::EmptyPayload);
        }

        // Convert trits → GF(3^6) symbols (pad to multiple of 6)
        let original_len = payload.len();
        let padded_len =
            ((original_len + TRITS_PER_SYMBOL - 1) / TRITS_PER_SYMBOL) * TRITS_PER_SYMBOL;
        let mut padded = payload.to_vec();
        padded.resize(padded_len, 0);

        let data_symbols: Vec<u16> = padded
            .chunks(TRITS_PER_SYMBOL)
            .map(|chunk| gf3::trits_to_symbol(chunk))
            .collect();

        // Calculate parity count from ecc_capacity
        let msg_len = RS_HEADER_SYMBOLS + data_symbols.len();
        let total_symbols = if ecc_capacity < 1.0 {
            (msg_len as f32 / (1.0 - ecc_capacity)).ceil() as usize
        } else {
            msg_len * 2
        };
        let mut parity_count = total_symbols.saturating_sub(msg_len).max(2);
        if parity_count % 2 != 0 {
            parity_count += 1;
        }
        // Parity count must fit in one GF(3^6) symbol (0–728)
        let parity_count = parity_count.min(728);

        // Build message: [len_low, len_high, parity_count, data...]
        let mut message = Vec::with_capacity(msg_len);
        message.push((original_len % 729) as u16);
        message.push((original_len / 729) as u16);
        message.push(parity_count as u16);
        message.extend_from_slice(&data_symbols);

        // RS encode
        let gf = GF3::new();
        let rs = ReedSolomon::new(&gf, parity_count);
        let codeword = rs.encode(&gf, &message);

        // Convert symbols → trits
        let codeword_trits_len = codeword.len() * TRITS_PER_SYMBOL;
        let len_trits = encode_length(codeword_trits_len);

        let mut out = Vec::with_capacity(LENGTH_PREFIX_TRITS + codeword_trits_len);
        out.extend_from_slice(&len_trits);
        for &sym in &codeword {
            out.extend_from_slice(&gf3::symbol_to_trits(sym));
        }

        Ok(out)
    }

    fn correct_errors(raw_read: &[u8]) -> Result<Vec<u8>, EccError> {
        if raw_read.len() < LENGTH_PREFIX_TRITS {
            return Err(EccError::EmptyPayload);
        }

        // Read outer length prefix to separate codeword from matrix padding
        let codeword_trits_len = decode_length(&raw_read[..LENGTH_PREFIX_TRITS]);
        let start = LENGTH_PREFIX_TRITS;
        let end = start + codeword_trits_len;

        if end > raw_read.len() || codeword_trits_len == 0 {
            return Err(EccError::EmptyPayload);
        }

        let raw_codeword = &raw_read[start..end];

        // Convert trits → symbols, tracking erasure positions
        let mut symbols = Vec::with_capacity(raw_codeword.len() / TRITS_PER_SYMBOL);
        let mut erasure_positions = Vec::new();

        for (idx, chunk) in raw_codeword.chunks(TRITS_PER_SYMBOL).enumerate() {
            if chunk.iter().any(|&t| t == 3) {
                erasure_positions.push(idx);
                symbols.push(0);
            } else {
                symbols.push(gf3::trits_to_symbol(chunk));
            }
        }

        let n = symbols.len();

        // The parity count is stored as the 3rd data symbol.
        // In the codeword layout [parity || data], data starts at index parity_count.
        // So data[2] = symbols[parity_count + 2].
        // We don't know parity_count yet, so we read from the end:
        // message = symbols[parity_count..n], message[2] = parity_count
        // → symbols[parity_count + 2] = parity_count
        // → n - msg_len + 2 has the parity count, where msg_len = n - parity_count
        // This is circular. Instead, scan for a valid parity count.
        //
        // Since parity_count < n/2, try each even value and check if
        // symbols[parity_count + 2] == parity_count (self-consistent).
        let gf = GF3::new();

        for pc in (2..=n / 2).step_by(2) {
            let rs = ReedSolomon::new(&gf, pc);
            if let Ok(data_with_header) = rs.decode(&gf, &symbols, &erasure_positions) {
                // The data might be padded at the front. Scan for a valid header sequence.
                // We know parity_count must equal `pc`.
                for offset in 0..data_with_header.len().saturating_sub(RS_HEADER_SYMBOLS - 1) {
                    if data_with_header[offset + 2] as usize == pc {
                        // Found a potential header. Validate original_len.
                        let original_len = data_with_header[offset] as usize + data_with_header[offset + 1] as usize * 729;
                        let data_symbols = &data_with_header[offset + RS_HEADER_SYMBOLS..];
                        
                        let max_capacity = data_symbols.len() * TRITS_PER_SYMBOL;
                        // Reject original_len == 0 (black padding false-positive) and
                        // non-multiple-of-6 (halftone noise false-positive).
                        // The encoder always writes whole symbols, so original_len % 6 == 0.
                        if original_len > 0 && original_len % TRITS_PER_SYMBOL == 0 && original_len <= max_capacity {
                            let mut trits = Vec::with_capacity(max_capacity);
                            for &sym in data_symbols {
                                trits.extend_from_slice(&gf3::symbol_to_trits(sym));
                            }
                            trits.truncate(original_len);
                            
                            // Byte-range validation: every 6-trit chunk must decode to ≤ 255.
                            // This filters out false-positive headers from high-entropy halftone
                            // noise where symbols=[len_lo, len_hi, pc] appears by coincidence.
                            let all_valid_bytes = trits.chunks(TRITS_PER_SYMBOL).all(|chunk| {
                                let mut val: u32 = 0;
                                let mut base: u32 = 1;
                                for &t in chunk {
                                    val += t as u32 * base;
                                    base *= 3;
                                }
                                val <= 255
                            });
                            
                            if all_valid_bytes {
                                return Ok(trits);
                            }
                        }
                    }
                }
            }
        }

        Err(EccError::Unrecoverable {
            errors: 0,
            erasures: erasure_positions.len(),
            capacity: 0,
        })
    }
}
