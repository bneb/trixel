//! Color space transformation between standard sRGB (gamma-encoded) and CIELAB (D65).

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lab {
    pub l: f32, // 0.0 to 100.0 (Lightness)
    pub a: f32, // -128.0 to 127.0 (Green to Magenta)
    pub b: f32, // -128.0 to 127.0 (Blue to Yellow)
}

// D65 Standard Illuminant reference white point
const XN: f32 = 0.95047;
const YN: f32 = 1.00000;
const ZN: f32 = 1.08883;

#[inline]
fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

#[inline]
fn f_lab(t: f32) -> f32 {
    const DELTA: f32 = 6.0 / 29.0;
    if t > DELTA * DELTA * DELTA {
        t.cbrt()
    } else {
        t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}

#[inline]
fn f_inv_lab(t: f32) -> f32 {
    const DELTA: f32 = 6.0 / 29.0;
    if t > DELTA {
        t * t * t
    } else {
        3.0 * DELTA * DELTA * (t - 4.0 / 29.0)
    }
}

/// Convert 8-bit sRGB to CIELAB (D65).
pub fn srgb_to_lab(r: u8, g: u8, b: u8) -> Lab {
    let r_lin = srgb_to_linear(r as f32 / 255.0);
    let g_lin = srgb_to_linear(g as f32 / 255.0);
    let b_lin = srgb_to_linear(b as f32 / 255.0);

    // sRGB D65 to XYZ matrix
    let x = 0.4124564 * r_lin + 0.3575761 * g_lin + 0.1804375 * b_lin;
    let y = 0.2126729 * r_lin + 0.7151522 * g_lin + 0.0721750 * b_lin;
    let z = 0.0193339 * r_lin + 0.1191920 * g_lin + 0.9503041 * b_lin;

    let fx = f_lab(x / XN);
    let fy = f_lab(y / YN);
    let fz = f_lab(z / ZN);

    let l = (116.0 * fy - 16.0).clamp(0.0, 100.0);
    let a = (500.0 * (fx - fy)).clamp(-128.0, 127.0);
    let b = (200.0 * (fy - fz)).clamp(-128.0, 127.0);

    Lab { l, a, b }
}

/// Convert CIELAB (D65) to 8-bit sRGB with soft gamut clipping.
pub fn lab_to_srgb(l: f32, a: f32, b: f32) -> [u8; 3] {
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;

    let x = XN * f_inv_lab(fx);
    let y = YN * f_inv_lab(fy);
    let z = ZN * f_inv_lab(fz);

    // XYZ to sRGB D65 inverse matrix
    let r_lin =  3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let g_lin = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    let b_lin =  0.0556434 * x - 0.2040259 * y + 1.0572252 * z;

    let r = (linear_to_srgb(r_lin.clamp(0.0, 1.0)) * 255.0).round().clamp(0.0, 255.0) as u8;
    let g = (linear_to_srgb(g_lin.clamp(0.0, 1.0)) * 255.0).round().clamp(0.0, 255.0) as u8;
    let b = (linear_to_srgb(b_lin.clamp(0.0, 1.0)) * 255.0).round().clamp(0.0, 255.0) as u8;

    [r, g, b]
}
