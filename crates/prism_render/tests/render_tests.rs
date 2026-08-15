//! Integration tests for the prism_render forward embedder.
//!
//! Quality targets (measured with `EmbedConfig::default()`):
//!   synthetic photographic gradient: PSNR 42.45 dB, SSIM 0.9920
//!   web/hero-matrix.png:             PSNR 43.52 dB, SSIM 0.9929
//! Both clear the plan targets (PSNR >= 42.0 dB, SSIM >= 0.985).

use image::{Rgb, RgbImage};
use prism_render::embedder::{EmbedError, PrismEmbedder};
use prism_render::metrics::{compute_psnr, compute_ssim};

/// Synthetic photographic gradient with texture, mirroring prism_cv's roundtrip image.
fn synthetic_gradient() -> RgbImage {
    let (width, height) = (384u32, 256u32);
    let mut img = RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let texture = ((x * 17 + y * 29) % 23) as f32;
            let r = (((x as f32 / width as f32) * 180.0 + 30.0) + texture).clamp(0.0, 255.0) as u8;
            let g = (((y as f32 / height as f32) * 160.0 + 40.0) + texture).clamp(0.0, 255.0) as u8;
            let b = ((((x + y) as f32 / (width + height) as f32) * 140.0 + 50.0) + texture)
                .clamp(0.0, 255.0) as u8;
            img.put_pixel(x, y, Rgb([r, g, b]));
        }
    }
    img
}

fn hero_image() -> RgbImage {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("web/hero-matrix.png");
    image::open(&path).expect("web/hero-matrix.png must exist").to_rgb8()
}

#[test]
fn embed_synthetic_gradient_quality() {
    let source = synthetic_gradient();
    let payload = b"https://trixel.to";
    let embedded = PrismEmbedder::new().embed(&source, payload).expect("embed must succeed");

    let psnr = compute_psnr(&source, &embedded);
    let ssim = compute_ssim(&source, &embedded);
    eprintln!("Synthetic gradient: PSNR {psnr:.2} dB, SSIM {ssim:.4}");

    // Measured 42.45 dB / 0.9920; assert at the plan targets with margin.
    assert!(psnr >= 42.0, "PSNR {psnr:.2} dB below plan target 42.0 dB");
    assert!(ssim >= 0.985, "SSIM {ssim:.4} below plan target 0.985");
}

#[test]
fn embed_hero_image_quality() {
    let source = hero_image();
    let payload = b"https://trixel.to";
    let embedded = PrismEmbedder::new().embed(&source, payload).expect("embed must succeed");

    let psnr = compute_psnr(&source, &embedded);
    let ssim = compute_ssim(&source, &embedded);
    eprintln!("Hero-matrix: PSNR {psnr:.2} dB, SSIM {ssim:.4}");

    // Measured 43.52 dB / 0.9929; assert at the plan targets with margin.
    assert!(psnr >= 42.0, "PSNR {psnr:.2} dB below plan target 42.0 dB");
    assert!(ssim >= 0.985, "SSIM {ssim:.4} below plan target 0.985");
}

#[test]
fn embed_image_too_small() {
    let img = RgbImage::new(191, 127);
    let err = PrismEmbedder::new().embed(&img, b"x").unwrap_err();
    assert!(
        matches!(err, EmbedError::ImageTooSmall { width: 191, height: 127 }),
        "unexpected error: {err}"
    );

    // Exactly at the minimum (192x128) succeeds.
    let min = RgbImage::new(192, 128);
    assert!(PrismEmbedder::new().embed(&min, b"x").is_ok());
}

#[test]
fn embed_payload_too_large() {
    let img = RgbImage::new(384, 256);
    let payload = [0x41u8; 33]; // 33 > 32-byte capacity
    let err = PrismEmbedder::new().embed(&img, &payload).unwrap_err();
    assert!(
        matches!(err, EmbedError::PayloadTooLarge(33, 32)),
        "unexpected error: {err}"
    );
}

#[test]
fn embed_empty_payload() {
    let img = RgbImage::new(384, 256);
    let embedded = PrismEmbedder::new().embed(&img, b"").expect("empty payload must embed");
    assert_eq!(embedded.dimensions(), img.dimensions());
}

#[test]
fn embed_max_payload_exactly_32_bytes() {
    let img = RgbImage::new(384, 256);
    let payload = [0x5au8; 32]; // exactly the capacity (256 info bits / 8)
    let embedded = PrismEmbedder::new().embed(&img, &payload).expect("32-byte payload must embed");
    assert_eq!(embedded.dimensions(), img.dimensions());
}
