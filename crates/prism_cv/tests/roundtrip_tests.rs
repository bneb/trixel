use prism_render::embedder::PrismEmbedder;
use prism_render::metrics::{compute_psnr, compute_ssim};
use prism_cv::extractor::PrismExtractor;
use image::{RgbImage, Rgb};

/// Full codeword payload size for the z64 codec (256 info bits).
const FULL_PAYLOAD_BYTES: usize = 32;

fn synthetic_source(width: u32, height: u32) -> RgbImage {
    let mut source_img = RgbImage::new(width, height);

    // Create realistic photographic gradient & texture pattern
    for y in 0..height {
        for x in 0..width {
            let texture = ((x * 17 + y * 29) % 23) as f32;
            let r = (((x as f32 / width as f32) * 180.0 + 30.0) + texture).clamp(0.0, 255.0) as u8;
            let g = (((y as f32 / height as f32) * 160.0 + 40.0) + texture).clamp(0.0, 255.0) as u8;
            let b = ((((x + y) as f32 / (width + height) as f32) * 140.0 + 50.0) + texture).clamp(0.0, 255.0) as u8;
            source_img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    source_img
}

/// The 32-byte codeword-payload encoding of `payload` (zero-padded to 256 bits).
fn padded_payload(payload: &[u8]) -> Vec<u8> {
    let mut expected = vec![0u8; FULL_PAYLOAD_BYTES];
    expected[..payload.len()].copy_from_slice(payload);
    expected
}

#[test]
fn test_prism_end_to_end_clean_roundtrip() {
    let width = 384u32;
    let height = 256u32;
    let source_img = synthetic_source(width, height);

    let payload = b"https://trixel.to";
    let embedder = PrismEmbedder::new();
    let extractor = PrismExtractor::new();

    // 1. Embed payload
    let embedded_img = embedder.embed(&source_img, payload).expect("Embedding must succeed");

    // 2. Measure Perceptual Quality Metrics
    let psnr = compute_psnr(&source_img, &embedded_img);
    let ssim = compute_ssim(&source_img, &embedded_img);

    eprintln!("PrismCode Visual Metrics -> PSNR: {:.2} dB, SSIM: {:.4}", psnr, ssim);

    assert!(psnr >= 40.0, "PSNR must be >= 40.0 dB (imperceptible), got {:.2} dB", psnr);
    assert!(ssim >= 0.985, "SSIM must be >= 0.985 (high structural fidelity), got {:.4}", ssim);

    // 3. Extract & Decode: extract() returns the FULL codeword payload
    let extracted = extractor.extract(&embedded_img).expect("Extraction must succeed");
    eprintln!("Decoded payload: '{}'", String::from_utf8_lossy(&extracted));

    assert_eq!(
        extracted,
        padded_payload(payload),
        "Extracted payload must equal the full 32-byte codeword payload"
    );
}

#[test]
fn test_prism_roundtrip_with_hero_image() {
    let hero_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .join("web/hero-matrix.png");

    let dyn_img = image::open(&hero_path).expect("web/hero-matrix.png must exist");
    let rgb_img = dyn_img.to_rgb8();
    let payload = b"https://trixel.to";

    let embedder = PrismEmbedder::new();
    let extractor = PrismExtractor::new();

    let embedded = embedder.embed(&rgb_img, payload).expect("Embedding on hero-matrix must succeed");

    let psnr = compute_psnr(&rgb_img, &embedded);
    let ssim = compute_ssim(&rgb_img, &embedded);
    eprintln!("Hero-matrix Metrics -> PSNR: {:.2} dB, SSIM: {:.4}", psnr, ssim);

    assert!(psnr >= 40.0);
    assert!(ssim >= 0.985);

    let decoded = extractor.extract(&embedded).expect("Extraction on hero-matrix must succeed");
    assert_eq!(
        decoded,
        padded_payload(payload),
        "Extracted payload must equal the full 32-byte codeword payload"
    );
}

#[test]
fn test_payload_with_trailing_zero_bytes_survives_roundtrip() {
    // Ends in 0x00: the extractor must not trim it, and the LDPC zero-padding
    // tail must not be mistaken for a terminator.
    let payload = b"ends-in-nul\x00";
    let embedded = PrismEmbedder::new()
        .embed(&synthetic_source(384, 256), payload)
        .expect("Embedding must succeed");

    let extracted = PrismExtractor::new()
        .extract(&embedded)
        .expect("Extraction must succeed");

    assert_eq!(
        extracted.len(),
        FULL_PAYLOAD_BYTES,
        "extract() must return the full 32-byte codeword payload, got {} bytes",
        extracted.len()
    );
    assert_eq!(
        extracted,
        padded_payload(payload),
        "Trailing 0x00 bytes must survive the roundtrip verbatim"
    );
}
