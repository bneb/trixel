use image;
use ternac_core::{ErrorCorrection, RsEcc, TritMatrix, MockCodec, TernaryCodec, gf3, TRITS_PER_SYMBOL, LENGTH_PREFIX_TRITS};
use ternac_cv::{AnchorVision, VisionPipeline};
use ternac_solver::anchor::is_in_anchor_region;

fn main() {
    let input = "/tmp/halftone_test.png";
    let img = image::open(input).unwrap();
    let matrix = AnchorVision::extract_matrix(&img, 10).unwrap();
    
    let w = matrix.width;
    let mut flat = Vec::new();
    for y in 0..matrix.height {
        for x in 0..matrix.width {
            if !is_in_anchor_region(x, y, w) {
                flat.push(matrix.get(x, y).unwrap_or(0));
            }
        }
    }

    println!("Extracted {} non-anchor trits", flat.len());

    let cw_len = ternac_core::decode_length(&flat[..LENGTH_PREFIX_TRITS]);
    println!("Length prefix says codeword is {} trits", cw_len);
    
    if cw_len == 0 || cw_len > flat.len() - LENGTH_PREFIX_TRITS {
        println!("Invalid length prefix!");
        return;
    }

    let cw_trits = &flat[LENGTH_PREFIX_TRITS..LENGTH_PREFIX_TRITS + cw_len];
    let symbols: Vec<u16> = cw_trits.chunks(TRITS_PER_SYMBOL)
        .map(|c| gf3::trits_to_symbol(c))
        .collect();

    println!("Codeword symbols: {}", symbols.len());

    let clean = RsEcc::correct_errors(&flat).unwrap();
    println!("RsEcc correction successful, clean payload length: {}", clean.len());
    
    let bytes = MockCodec::decode_trits(&clean).unwrap();
    println!("Decoded bytes length: {}", bytes.len());
    println!("Text: '{}'", String::from_utf8_lossy(&bytes));
}
