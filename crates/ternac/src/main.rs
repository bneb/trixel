//! # ternac CLI
//!
//! Command-line interface for the Ternac ternary matrix encoding system.
//! Supports `encode` (string → PNG) and `decode` (PNG → string) subcommands.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use ternac_core::{ErrorCorrection, MockCodec, RsEcc, TernaryCodec};
use ternac_cv::{AnchorVision, VisionPipeline};
use ternac_render::{AnchorRenderer, Renderer};
use ternac_solver::{AnchorSolver, MatrixSolver};
use ternac_solver::anchor::ANCHOR_SIZE;

// ---------------------------------------------------------------------------
// CLI Structure
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "ternac", version, about = "Ternary matrix encoder/decoder")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Encode a string into a Ternac PNG image.
    Encode {
        /// The data string to encode.
        #[arg(long)]
        data: String,

        /// Output PNG file path.
        #[arg(long)]
        output: PathBuf,

        /// Hex color for State 1 modules (e.g. "#FF00FF").
        #[arg(long, default_value = "#000000")]
        color: String,

        /// Pixel size of each trit module.
        #[arg(long, default_value_t = 10)]
        module_size: u32,

        /// ECC capacity as a fraction (0.0 to 1.0).
        #[arg(long, default_value_t = 0.3)]
        ecc: f32,
    },
    /// Decode a Ternac PNG image back to a string.
    Decode {
        /// Input PNG file path.
        #[arg(long)]
        input: PathBuf,

        /// Pixel size of each trit module (must match the encode setting).
        #[arg(long, default_value_t = 10)]
        module_size: u32,
    },
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a hex color string like "#FF00FF" or "FF00FF" into [R, G, B].
fn parse_hex_color(s: &str) -> Result<[u8; 3], String> {
    let s = s.trim_start_matches('#');
    if s.len() != 6 {
        return Err(format!("expected 6 hex digits, got '{s}'"));
    }
    let r = u8::from_str_radix(&s[0..2], 16).map_err(|e| e.to_string())?;
    let g = u8::from_str_radix(&s[2..4], 16).map_err(|e| e.to_string())?;
    let b = u8::from_str_radix(&s[4..6], 16).map_err(|e| e.to_string())?;
    Ok([r, g, b])
}

/// Compute the smallest square matrix side length that fits `n` payload trits
/// plus 4 anchor blocks (each ANCHOR_SIZE × ANCHOR_SIZE = 9 cells).
fn min_square_side(n: usize) -> usize {
    let anchor_cells = 4 * ANCHOR_SIZE * ANCHOR_SIZE;
    let total = n + anchor_cells;
    let s = (total as f64).sqrt().ceil() as usize;
    // Must be at least 2 * ANCHOR_SIZE to fit non-overlapping corners
    let min = ANCHOR_SIZE * 2;
    let side = if s * s >= total { s } else { s + 1 };
    side.max(min)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Encode {
            data,
            output,
            color,
            module_size,
            ecc,
        } => {
            let color_rgb = parse_hex_color(&color).unwrap_or_else(|e| {
                eprintln!("error: bad color '{}': {}", color, e);
                std::process::exit(1);
            });

            // 1. Bytes → trits
            let trits = MockCodec::encode_bytes(data.as_bytes()).unwrap_or_else(|e| {
                eprintln!("error: codec encode failed: {e}");
                std::process::exit(1);
            });

            // 2. Apply ECC parity
            let with_parity = RsEcc::apply_parity(&trits, ecc).unwrap_or_else(|e| {
                eprintln!("error: ECC apply failed: {e}");
                std::process::exit(1);
            });

            // 3. Pack into anchor-aware square matrix
            let side = min_square_side(with_parity.len());
            let matrix =
                AnchorSolver::resolve_matrix(&with_parity, side, &[]).unwrap_or_else(|e| {
                    eprintln!("error: solver failed: {e}");
                    std::process::exit(1);
                });

            // 4. Render to image (anchors are already in the matrix)
            let img =
                AnchorRenderer::render_png(&matrix, module_size, color_rgb).unwrap_or_else(|e| {
                    eprintln!("error: render failed: {e}");
                    std::process::exit(1);
                });

            img.save(&output).unwrap_or_else(|e| {
                eprintln!("error: failed to save '{}': {e}", output.display());
                std::process::exit(1);
            });

            println!(
                "Encoded {} bytes → {}×{} matrix → {}",
                data.len(),
                matrix.width,
                matrix.height,
                output.display()
            );
        }

        Commands::Decode {
            input,
            module_size,
        } => {
            // 1. Load image
            let img = image::open(&input).unwrap_or_else(|e| {
                eprintln!("error: failed to open '{}': {e}", input.display());
                std::process::exit(1);
            });

            // 2. Extract trit matrix (with anchor-calibrated luminance)
            let matrix =
                AnchorVision::extract_matrix(&img, module_size).unwrap_or_else(|e| {
                    eprintln!("error: vision extract failed: {e}");
                    std::process::exit(1);
                });

            // 3. Extract payload from non-anchor cells
            let payload_trits = extract_payload_from_matrix(&matrix);

            // 4. Error correction
            let clean = RsEcc::correct_errors(&payload_trits).unwrap_or_else(|e| {
                eprintln!("error: ECC correction failed: {e}");
                std::process::exit(1);
            });

            // 5. Trits → bytes
            let bytes = MockCodec::decode_trits(&clean).unwrap_or_else(|e| {
                eprintln!("error: codec decode failed: {e}");
                std::process::exit(1);
            });

            let text = String::from_utf8_lossy(&bytes);
            println!("{text}");
        }
    }
}

/// Extract payload trits from an anchor-bearing matrix by skipping anchor regions.
fn extract_payload_from_matrix(matrix: &ternac_core::TritMatrix) -> Vec<u8> {
    use ternac_solver::anchor::is_in_anchor_region;
    let n = matrix.width; // assumes square
    let mut payload = Vec::new();
    for y in 0..matrix.height {
        for x in 0..matrix.width {
            if !is_in_anchor_region(x, y, n) {
                payload.push(matrix.get(x, y).unwrap_or(0));
            }
        }
    }
    payload
}
