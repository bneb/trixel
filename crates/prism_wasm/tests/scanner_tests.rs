//! Native scanner pipeline tests (no wasm runtime needed): roundtrip via
//! embed + decode on a synthetic frame, trailing-NUL trimming, invalid-size
//! rejection, and scratch-buffer reuse across payloads.

use image::{Rgb, RgbImage};
use prism_render::embedder::PrismEmbedder;
use prism_wasm::scanner::FrameScanner;

/// Maximum payload capacity of the z64 codec (32 bytes).
const PAYLOAD_CAPACITY: usize = 32;

fn synthetic_source(width: u32, height: u32) -> RgbImage {
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

fn embed_payload(img: &RgbImage, payload: &[u8]) -> RgbImage {
    PrismEmbedder::new()
        .embed(img, payload)
        .expect("embedding must succeed")
}

fn to_rgba(img: &RgbImage) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((img.width() * img.height() * 4) as usize);
    for px in img.pixels() {
        rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
    }
    rgba
}

/// The 32-byte zero-padded codeword payload `extract()` returns for `payload`.
fn padded(payload: &[u8]) -> Vec<u8> {
    assert!(payload.len() <= PAYLOAD_CAPACITY);
    let mut v = vec![0u8; PAYLOAD_CAPACITY];
    v[..payload.len()].copy_from_slice(payload);
    v
}

#[test]
fn roundtrip_via_embed_and_decode_on_synthetic_frame() {
    let frame = to_rgba(&embed_payload(&synthetic_source(384, 256), b"https://trixel.to"));
    let mut scanner = FrameScanner::new();
    assert_eq!(
        scanner.decode_camera_frame(&frame, 384, 256),
        Some("https://trixel.to".to_string()),
        "embedded payload must survive a synthetic roundtrip"
    );
}

#[test]
fn decode_trims_trailing_nuls_from_display_string() {
    // Short payload: the codeword zero-pads to 32 bytes, which must not leak
    // into the display string.
    let frame = to_rgba(&embed_payload(&synthetic_source(384, 256), b"hi"));
    let mut scanner = FrameScanner::new();
    assert_eq!(scanner.decode_camera_frame(&frame, 384, 256), Some("hi".to_string()));

    // Payload ending in a real 0x00 byte: the NUL-trim contract strips it for
    // display (the raw payload is preserved verbatim by scan_into).
    let frame = to_rgba(&embed_payload(&synthetic_source(384, 256), b"nul-end\x00"));
    assert_eq!(
        scanner.decode_camera_frame(&frame, 384, 256),
        Some("nul-end".to_string())
    );
}

#[test]
fn scan_into_returns_full_padded_payload() {
    let frame = to_rgba(&embed_payload(&synthetic_source(384, 256), b"hi"));
    let mut scanner = FrameScanner::new();
    let mut out = Vec::new();
    assert!(scanner.scan_into(&frame, 384, 256, &mut out));
    assert_eq!(out, padded(b"hi"), "scan_into must return the full 32-byte payload");
}

#[test]
fn invalid_frames_are_rejected() {
    let mut scanner = FrameScanner::new();
    let mut out = Vec::new();

    // Below the 192x128 minimum.
    assert!(!scanner.scan_into(&vec![0u8; 100 * 100 * 4], 100, 100, &mut out));
    assert!(scanner.decode_camera_frame(&vec![0u8; 100 * 100 * 4], 100, 100).is_none());

    // Buffer smaller than width * height * 4.
    let small = vec![0u8; 192 * 128 * 4 - 1];
    assert!(!scanner.scan_into(&small, 192, 128, &mut out));
    assert!(scanner.decode_camera_frame(&small, 192, 128).is_none());

    // Buffer exactly the required size but garbage: decode failure (LDPC
    // non-convergence) must be a clean `false`/`None`, never a panic.
    let noise = vec![0u8; 192 * 128 * 4];
    assert!(!scanner.scan_into(&noise, 192, 128, &mut out));
    assert!(scanner.decode_camera_frame(&noise, 192, 128).is_none());
}

#[test]
fn scan_into_reuses_scratch_buffers_across_payloads() {
    let mut scanner = FrameScanner::new();
    let mut out = Vec::new();

    // First payload primes the scratch buffers.
    let frame_a = to_rgba(&embed_payload(&synthetic_source(384, 256), b"first-payload"));
    assert!(scanner.scan_into(&frame_a, 384, 256, &mut out));
    assert_eq!(out, padded(b"first-payload"));

    // Second, different payload on the same scanner: must decode correctly
    // (no stale state from the previous frame) without panics.
    let frame_b = to_rgba(&embed_payload(&synthetic_source(384, 256), b"second-payload!!"));
    assert!(scanner.scan_into(&frame_b, 384, 256, &mut out));
    assert_eq!(out, padded(b"second-payload!!"));

    // Same-size frames keep the same staging image; different sizes trigger a
    // resize but must still decode.
    let frame_c = to_rgba(&embed_payload(&synthetic_source(768, 512), b"third"));
    assert!(scanner.scan_into(&frame_c, 768, 512, &mut out));
    assert_eq!(out, padded(b"third"));
}
