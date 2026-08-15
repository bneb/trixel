use clap::{Parser, Subcommand};
use prism_cv::extractor::PrismExtractor;
use prism_render::embedder::PrismEmbedder;
use prism_render::metrics::{compute_psnr, compute_ssim};
use prism_sync::homography::{apply_homography, invert_homography, solve_dlt};
use std::path::PathBuf;
use std::time::Instant;

use image::{Rgb, RgbImage};

#[derive(Parser, Debug)]
#[command(name = "prism", about = "PrismCode — Next-Generation Perceptual Optical Codec")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Embed data payload into an image
    Encode {
        #[arg(short, long)]
        image: PathBuf,
        #[arg(short, long)]
        data: String,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Extract data payload from an image
    Decode {
        #[arg(short, long)]
        input: PathBuf,
    },
    /// Benchmark visual quality, decode timing, and optical distortion robustness
    Benchmark {
        #[arg(short, long)]
        image: PathBuf,
        #[arg(short, long)]
        data: Option<String>,
        /// Apply Gaussian blur with this sigma before decoding
        #[arg(long)]
        blur: Option<f32>,
        /// Re-encode through JPEG at this quality (1-100) before decoding
        #[arg(long, value_parser = clap::value_parser!(u8).range(1..=100))]
        jpeg_quality: Option<u8>,
        /// Apply a perspective yaw tilt of this many degrees before decoding
        #[arg(long)]
        warp: Option<f32>,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Encode { image, data, output } => {
            let img = image::open(&image)?.to_rgb8();
            let embedder = PrismEmbedder::new();
            let start = Instant::now();
            let embedded = embedder.embed(&img, data.as_bytes())?;
            let elapsed = start.elapsed();

            embedded.save(&output)?;

            let psnr = compute_psnr(&img, &embedded);
            let ssim = compute_ssim(&img, &embedded);

            println!("✨ PrismCode Embedded Successfully!");
            println!("   Payload:  '{}' ({} bytes)", data, data.len());
            println!("   Output:   {}", output.display());
            println!("   Time:     {:.2} ms", elapsed.as_secs_f64() * 1000.0);
            println!("   PSNR:     {:.2} dB", psnr);
            println!("   SSIM:     {:.4}", ssim);
        }
        Commands::Decode { input } => {
            let img = image::open(&input)?.to_rgb8();
            let extractor = PrismExtractor::new();
            let start = Instant::now();
            let payload = extractor.extract(&img)?;
            let elapsed = start.elapsed();

            // The extractor returns the full 32-byte codeword payload with
            // zero padding; trim trailing NULs for display.
            let text = String::from_utf8_lossy(trim_trailing_nuls(&payload));
            println!("{}", text);
            eprintln!("(Decoded in {:.2} ms)", elapsed.as_secs_f64() * 1000.0);
        }
        Commands::Benchmark {
            image,
            data,
            blur,
            jpeg_quality,
            warp,
        } => {
            let img = image::open(&image)?.to_rgb8();
            let text = data.unwrap_or_else(|| "https://trixel.to".to_string());
            let embedder = PrismEmbedder::new();
            let extractor = PrismExtractor::new();

            let enc_start = Instant::now();
            let embedded = embedder.embed(&img, text.as_bytes())?;
            let enc_time = enc_start.elapsed();

            let dec_start = Instant::now();
            let decoded = extractor.extract(&embedded)?;
            let dec_time = dec_start.elapsed();

            let psnr = compute_psnr(&img, &embedded);
            let ssim = compute_ssim(&img, &embedded);

            println!("=== PrismCode Benchmark Results ===");
            println!("Resolution:   {}x{}", img.width(), img.height());
            println!("Payload:      '{}'", text);
            println!("Encode Time:  {:.2} ms", enc_time.as_secs_f64() * 1000.0);
            println!("Decode Time:  {:.2} ms", dec_time.as_secs_f64() * 1000.0);
            println!("PSNR:         {:.2} dB", psnr);
            println!("SSIM:         {:.4}", ssim);
            println!(
                "Match:        {}",
                if trim_trailing_nuls(&decoded) == text.as_bytes() {
                    "100% PERFECT"
                } else {
                    "MISMATCH"
                }
            );

            run_distortions(&extractor, &embedded, &text, blur, jpeg_quality, warp);
        }
    }

    Ok(())
}

/// Applies each requested distortion to the embedded image and attempts a
/// decode, reporting PASS/FAIL and decode timing per distortion. Decode
/// timing is the native binary's; the WASM 10 ms/frame target needs a browser
/// runner (measured natively as the honest proxy).
fn run_distortions(
    extractor: &PrismExtractor,
    embedded: &RgbImage,
    text: &str,
    blur: Option<f32>,
    jpeg_quality: Option<u8>,
    warp: Option<f32>,
) {
    let mut ran_any = false;
    if let Some(sigma) = blur {
        ran_any = true;
        let distorted = gaussian_blur(embedded, sigma);
        report_distortion(&format!("blur sigma={sigma}"), extractor, &distorted, text);
    }
    if let Some(quality) = jpeg_quality {
        ran_any = true;
        match jpeg_reencode(embedded, quality) {
            Ok(distorted) => {
                report_distortion(&format!("jpeg q={quality}"), extractor, &distorted, text)
            }
            Err(e) => println!("[jpeg q={quality}]      ERROR  ({e})"),
        }
    }
    if let Some(degrees) = warp {
        ran_any = true;
        match warp_yaw(embedded, degrees) {
            Ok(distorted) => {
                report_distortion(&format!("warp {degrees} deg"), extractor, &distorted, text)
            }
            Err(e) => println!("[warp {degrees} deg]    ERROR  ({e})"),
        }
    }
    if ran_any {
        println!(
            "(Decode timing is native; the WASM 10 ms/frame target requires a browser runner)"
        );
    }
}

fn report_distortion(label: &str, extractor: &PrismExtractor, distorted: &RgbImage, text: &str) {
    let start = Instant::now();
    let result = extractor.extract(distorted);
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    match result {
        Ok(payload) => {
            let ok = trim_trailing_nuls(&payload) == text.as_bytes();
            let preview = String::from_utf8_lossy(trim_trailing_nuls(&payload));
            println!(
                "[{label}] decode {} ({ms:.2} ms) -> '{}'",
                if ok { "PASS" } else { "FAIL" },
                preview
            );
        }
        Err(e) => println!("[{label}] decode FAIL ({ms:.2} ms) -> {e}"),
    }
}

/// Returns the payload slice without trailing 0x00 bytes (codeword zero padding).
fn trim_trailing_nuls(payload: &[u8]) -> &[u8] {
    let end = payload
        .iter()
        .rposition(|&b| b != 0)
        .map(|i| i + 1)
        .unwrap_or(0);
    &payload[..end]
}

/// Separable Gaussian blur with a hand-rolled 1D kernel (clamped edges).
fn gaussian_blur(img: &RgbImage, sigma: f32) -> RgbImage {
    let (w, h) = img.dimensions();
    let r = (sigma * 3.0).ceil() as i32;

    let mut kernel = Vec::new();
    let mut sum = 0.0f32;
    for i in -r..=r {
        let v = (-(i * i) as f32 / (2.0 * sigma * sigma)).exp();
        sum += v;
        kernel.push(v);
    }
    for k in kernel.iter_mut() {
        *k /= sum;
    }

    // Horizontal pass
    let mut horiz = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for (i, &k) in kernel.iter().enumerate() {
                let sx = (x as i32 + i as i32 - r).clamp(0, w as i32 - 1) as u32;
                let px = img.get_pixel(sx, y);
                acc[0] += k * px[0] as f32;
                acc[1] += k * px[1] as f32;
                acc[2] += k * px[2] as f32;
            }
            horiz.put_pixel(
                x,
                y,
                Rgb([
                    acc[0].clamp(0.0, 255.0) as u8,
                    acc[1].clamp(0.0, 255.0) as u8,
                    acc[2].clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }

    // Vertical pass
    let mut out = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for (i, &k) in kernel.iter().enumerate() {
                let sy = (y as i32 + i as i32 - r).clamp(0, h as i32 - 1) as u32;
                let px = horiz.get_pixel(x, sy);
                acc[0] += k * px[0] as f32;
                acc[1] += k * px[1] as f32;
                acc[2] += k * px[2] as f32;
            }
            out.put_pixel(
                x,
                y,
                Rgb([
                    acc[0].clamp(0.0, 255.0) as u8,
                    acc[1].clamp(0.0, 255.0) as u8,
                    acc[2].clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }
    out
}

/// Re-encodes through JPEG at the given quality via a temp file, returning
/// the reloaded image. The temp file is always removed before returning.
fn jpeg_reencode(img: &RgbImage, quality: u8) -> Result<RgbImage, Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join(format!("prism_cli_jpeg_{}_{}.jpg", std::process::id(), quality));
    {
        let file = std::fs::File::create(&path)?;
        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(file, quality);
        encoder.encode_image(img)?;
    }
    let result = image::open(&path).map(|d| d.to_rgb8());
    let _ = std::fs::remove_file(&path);
    Ok(result?)
}

/// Yaw-tilts the image by `degrees` about the vertical axis using a DLT
/// homography (focal length = image width), rendering into the destination
/// bounding box with inverse-mapped bilinear sampling.
fn warp_yaw(img: &RgbImage, degrees: f32) -> Result<RgbImage, &'static str> {
    let (w, h) = img.dimensions();
    let wf = w as f32;
    let hf = h as f32;

    let theta = degrees.to_radians();
    let (cos_t, sin_t) = (theta.cos(), theta.sin());
    let f = wf;
    let project = |x: f32, y: f32| {
        let denom = f + x * sin_t;
        (f * x * cos_t / denom, f * y / denom)
    };

    let src_corners = [(0.0f32, 0.0f32), (wf, 0.0), (wf, hf), (0.0, hf)];
    let dst_corners: Vec<(f32, f32)> = src_corners.iter().map(|&(x, y)| project(x, y)).collect();
    let correspondences: Vec<(f32, f32, f32, f32)> = src_corners
        .iter()
        .zip(dst_corners.iter())
        .map(|(&(sx, sy), &(dx, dy))| (sx, sy, dx, dy))
        .collect();
    let h = solve_dlt(&correspondences).ok_or("DLT could not solve the yaw homography")?;
    let h_inv = invert_homography(&h).ok_or("yaw homography is singular")?;

    let max_x = dst_corners.iter().map(|c| c.0).fold(0.0f32, f32::max);
    let max_y = dst_corners.iter().map(|c| c.1).fold(0.0f32, f32::max);
    let cw = max_x.ceil().max(1.0) as u32;
    let ch = max_y.ceil().max(1.0) as u32;
    eprintln!("Warp canvas: {cw}x{ch} (source {wf}x{hf}, {degrees} deg yaw)");

    let mut warped = RgbImage::new(cw, ch);
    for oy in 0..ch {
        for ox in 0..cw {
            let (sx, sy) = apply_homography(&h_inv, ox as f32, oy as f32);
            warped.put_pixel(ox, oy, sample_bilinear(img, sx, sy));
        }
    }
    Ok(warped)
}

/// Bilinear sample of `img` at possibly-fractional coordinates, clamped edges.
fn sample_bilinear(img: &RgbImage, x: f32, y: f32) -> Rgb<u8> {
    let (w, h) = img.dimensions();
    let x = x.clamp(0.0, w as f32 - 1.0);
    let y = y.clamp(0.0, h as f32 - 1.0);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let p00 = img.get_pixel(x0, y0);
    let p10 = img.get_pixel(x1, y0);
    let p01 = img.get_pixel(x0, y1);
    let p11 = img.get_pixel(x1, y1);

    let mut out = [0u8; 3];
    for c in 0..3 {
        let v = p00[c] as f32 * (1.0 - fx) * (1.0 - fy)
            + p10[c] as f32 * fx * (1.0 - fy)
            + p01[c] as f32 * (1.0 - fx) * fy
            + p11[c] as f32 * fx * fy;
        out[c] = v.clamp(0.0, 255.0) as u8;
    }
    Rgb(out)
}
