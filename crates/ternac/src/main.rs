//! # ternac CLI
//!
//! Command-line interface for the Ternac ternary matrix encoding system.
//! Supports `encode` (string → PNG) and `decode` (PNG → string) subcommands.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use ternac_core::{ErrorCorrection, MockCodec, RsEcc, TernaryCodec};
use ternac_cv::{AnchorVision, VisionPipeline};
use ternac_render::{AnchorRenderer, FontEngine, Renderer, TernacFont};
use ternac_solver::{GaussSolver, MatrixSolver};
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
        #[arg(long, default_value = "#808080")]
        color: String,

        /// Pixel size of each trit module.
        #[arg(long, default_value_t = 10)]
        module_size: u32,

        /// Optional visible text to embed in the matrix.
        /// Font constraints are woven into the RS codeword.
        #[arg(long)]
        text: Option<String>,

        /// X position for embedded text (in grid cells).
        #[arg(long, default_value_t = 4)]
        text_x: usize,

        /// Y position for embedded text (in grid cells).
        #[arg(long, default_value_t = 4)]
        text_y: usize,

        /// Optional image path to use as a Base-3 halftone art background.
        #[arg(long)]
        image: Option<PathBuf>,

        /// Minimum matrix side length. Larger = higher illustration fidelity.
        /// Defaults to 60 when --image is used, otherwise auto-calculated.
        #[arg(long)]
        min_side: Option<usize>,
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
/// plus 4 anchor blocks (each ANCHOR_SIZE × ANCHOR_SIZE = 9 cells) and RS parity.
fn min_square_side(n: usize) -> usize {
    let anchor_cells = 4 * ANCHOR_SIZE * ANCHOR_SIZE;
    // Reserve ~3× payload for RS parity + header overhead
    let total = n * 3 + anchor_cells;
    let s = (total as f64).sqrt().ceil() as usize;
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
            text,
            text_x,
            text_y,
            image,
            min_side,
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

            // 2. Build font constraints (if --text is provided)
            let text_constraints = if let Some(ref txt) = text {
                TernacFont::string_to_constraints(txt, text_x, text_y)
            } else {
                vec![]
            };

            // 3. Determine matrix size (grow until text constraints clear anchors)
            let constraint_overhead = text_constraints.len();
            let mut side = min_square_side(trits.len() + constraint_overhead);
            
            // Apply user-specified or image-default minimum side
            let effective_min = min_side.unwrap_or(if image.is_some() { 60 } else { 0 });
            side = side.max(effective_min);
            
            if !text_constraints.is_empty() {
                use ternac_solver::anchor::is_in_anchor_region;
                loop {
                    let any_conflict = text_constraints.iter().any(|c| is_in_anchor_region(c.x, c.y, side));
                    if !any_conflict { break; }
                    side += 1;
                }
                // Ensure minimum side of 20 when text is present
                side = side.max(20);
            }

            // 4. Halftone & Priority Compositing
            let mut final_constraints_map = std::collections::HashMap::new();

            if let Some(ref img_path) = image {
                let dyn_img = image::open(img_path).unwrap_or_else(|e| {
                    eprintln!("error: failed to open image '{}': {e}", img_path.display());
                    std::process::exit(1);
                });

                // Calculate required free trits based on GaussSolver's internal formula
                let cell_coords = ternac_solver::gauss_solver::grid_to_flat_coords(side);
                use ternac_core::LENGTH_PREFIX_TRITS;
                let trits_for_codeword = cell_coords.len().saturating_sub(LENGTH_PREFIX_TRITS);
                let num_symbols = trits_for_codeword / 6; // TRITS_PER_SYMBOL
                let msg_symbols = ternac_core::RS_HEADER_SYMBOLS + ((trits.len() + 5) / 6);
                
                let mut parity_count = (num_symbols as f32 * 0.3).ceil() as usize;
                if parity_count % 2 != 0 { parity_count += 1; }
                let parity_count = parity_count.min(728).max(2);
                let parity_trits = parity_count * 6;
                let max_offset = num_symbols.saturating_sub(parity_count + msg_symbols);

                // Predict the EXACT offset GaussSolver will use
                let mut best_offset = None;
                for offset in 0..=max_offset {
                    let locked_msg_start = parity_trits + offset * 6;
                    let locked_msg_end = locked_msg_start + msg_symbols * 6;

                    let mut conflict = false;
                    for (flat_idx, &(x, y)) in cell_coords.iter().enumerate().take(LENGTH_PREFIX_TRITS + num_symbols * 6) {
                        if text_constraints.iter().any(|c| c.x == x && c.y == y) {
                            if flat_idx < LENGTH_PREFIX_TRITS { conflict = true; break; }
                            let cw_idx = flat_idx - LENGTH_PREFIX_TRITS;
                            if cw_idx >= locked_msg_start && cw_idx < locked_msg_end { conflict = true; break; }
                        }
                    }
                    if !conflict {
                        best_offset = Some(offset);
                        break;
                    }
                }
                
                let message_offset = best_offset.expect("Failed to find valid offset for text overlay");
                let locked_msg_start = parity_trits + message_offset * 6;
                let locked_msg_end = locked_msg_start + msg_symbols * 6;

                // Build a set of exactly where the payload goes
                let mut fixed_payload_cells = std::collections::HashSet::new();
                for (flat_idx, &(x, y)) in cell_coords.iter().enumerate() {
                    if flat_idx < LENGTH_PREFIX_TRITS {
                        fixed_payload_cells.insert((x, y));
                    } else {
                        let cw_idx = flat_idx - LENGTH_PREFIX_TRITS;
                        if cw_idx >= locked_msg_start && cw_idx < locked_msg_end {
                            fixed_payload_cells.insert((x, y));
                        }
                    }
                }
                
                // Inflate required free trits: parity + payload + text + 20% buffer
                let base_eqs = parity_count * 6;
                let mut required_free_trits = base_eqs 
                    + fixed_payload_cells.len() 
                    + text_constraints.len() 
                    + (base_eqs / 5);

                let engine = ternac_render::HalftoneEngine {
                    state_0_rgb: [0, 0, 0],
                    state_1_rgb: color_rgb,
                    state_2_rgb: [255, 255, 255],
                };
                
                // --- The Compensation Loop ---
                // We must guarantee that AFTER compositing the Font constraints and emptying the
                // payload footprint, we STILL have enough `None` variables (free trits) in the 
                // Halftone map to solve `base_eqs` (the Reed-Solomon parity check matrix).
                loop {
                    final_constraints_map.clear();
                    let ht_matrix = engine.image_to_constraints(&dyn_img, side, required_free_trits);
                    
                    // 1. Load halftone constraints, skipping anchor and payload regions.
                    for y in 0..side {
                        for x in 0..side {
                            if !ternac_solver::anchor::is_in_anchor_region(x, y, side) {
                                // Leave the payload footprint completely empty in the constraint map
                                if fixed_payload_cells.contains(&(x, y)) {
                                    continue;
                                }
                                if let Some(state) = ht_matrix[y][x] {
                                    final_constraints_map.insert(
                                        (x, y), 
                                        ternac_solver::ConstraintMask { x, y, required_state: state }
                                    );
                                }
                            }
                        }
                    }

                    // 2. Layer text constraints on top (Priority 1 & 2)
                    for c in &text_constraints {
                        final_constraints_map.insert((c.x, c.y), c.clone());
                    }

                    // 3. Count how many free variables actually remain for the solver
                    let total_cells = side * side;
                    let anchor_cells = 4 * ternac_solver::anchor::ANCHOR_SIZE * ternac_solver::anchor::ANCHOR_SIZE;
                    let locked_cells = final_constraints_map.len();
                    // Payload is handled internally by GaussSolver, but they are NOT free for parity matching
                    let free_cells = total_cells.saturating_sub(anchor_cells + locked_cells + fixed_payload_cells.len());

                    if free_cells >= base_eqs {
                        break; // We have enough free trits for the parity equations
                    }
                    
                    // Starvation detected (Font overlapped too many `None` regions).
                    // Sacrifice another batch of pixels.
                    required_free_trits += 50; 
                }
            }

            let constraints: Vec<_> = final_constraints_map.into_values().collect();

            // 4. Solve: GaussSolver produces a valid RS codeword via
            //    Gaussian elimination over GF(3). No Z3, no black boxes.
            let matrix =
                GaussSolver::resolve_matrix(&trits, side, &constraints).unwrap_or_else(|e| {
                    eprintln!("error: solver failed: {e}");
                    std::process::exit(1);
                });

            // 5. Render to image
            let img = if let Some(ref img_path) = image {
                // Build the font mask grid for Typography Immunity
                let mut font_mask: Vec<Vec<Option<u8>>> = vec![vec![None; side]; side];
                for c in &text_constraints {
                    if c.x < side && c.y < side {
                        font_mask[c.y][c.x] = Some(c.required_state);
                    }
                }

                // Context-aware halftone renderer with font immunity
                let dyn_img = image::open(img_path).unwrap_or_else(|e| {
                    eprintln!("error: failed to open image '{}': {e}", img_path.display());
                    std::process::exit(1);
                });
                AnchorRenderer::render_halftone_png(&matrix, module_size, &dyn_img, &font_mask)
                    .unwrap_or_else(|e| {
                        eprintln!("error: halftone render failed: {e}");
                        std::process::exit(1);
                    })
            } else {
                // Standard flat-color renderer
                AnchorRenderer::render_png(&matrix, module_size, color_rgb).unwrap_or_else(|e| {
                    eprintln!("error: render failed: {e}");
                    std::process::exit(1);
                })
            };

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
            if let Some(ref txt) = text {
                println!("Embedded text: \"{}\" ({} constraints)", txt, constraints.len());
            }
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

            // 4. Error correction (RS decode)
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
