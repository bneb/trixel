use trixel_core::{ErrorCorrection, MockCodec, RsEcc, TernaryCodec, LENGTH_PREFIX_TRITS, gf3};
use trixel_cv::{AnchorVision, VisionPipeline};
use trixel_solver::anchor::is_in_anchor_region;

#[test]
fn debug_color_decode_detailed() {
    let input = "/tmp/halftone_color.png";
    if !std::path::Path::new(input).exists() { return; }
    let img = image::open(input).unwrap();
    
    let matrix = AnchorVision::extract_matrix(&img, 10).unwrap();
    let n = matrix.width;
    
    let mut flat = Vec::new();
    for y in 0..matrix.height {
        for x in 0..n {
            if !is_in_anchor_region(x, y, n) {
                flat.push(matrix.get(x, y).unwrap_or(0));
            }
        }
    }
    
    let cw_len = trixel_core::decode_length(&flat[..LENGTH_PREFIX_TRITS]);
    eprintln!("Flat: {}, Length prefix: {}", flat.len(), cw_len);
    
    match RsEcc::correct_errors(&flat) {
        Ok(clean) => {
            eprintln!("RS OK: {} trits", clean.len());
            eprintln!("First 30 trits: {:?}", &clean[..clean.len().min(30)]);
            
            // Check if all trits are valid (0, 1, or 2)
            let invalid = clean.iter().filter(|&&t| t > 2).count();
            eprintln!("Invalid trits (>2): {}", invalid);
            
            match MockCodec::decode_trits(&clean) {
                Ok(b) => eprintln!("Decoded: '{}'", String::from_utf8_lossy(&b)),
                Err(e) => {
                    eprintln!("Codec error: {:?}", e);
                    // Show which chunk failed
                    for (i, chunk) in clean.chunks(6).enumerate() {
                        let mut val: u32 = 0;
                        let mut base: u32 = 1;
                        for &t in chunk {
                            val += t as u32 * base;
                            base *= 3;
                        }
                        if val > 255 {
                            eprintln!("  Overflow at chunk {}: {:?} = {}", i, chunk, val);
                        }
                    }
                }
            }
        }
        Err(e) => eprintln!("RS error: {:?}", e),
    }
}
