//! Yang-Bovik Spatial Just-Noticeable Difference (JND) Model.
//!
//! Models the visibility threshold of the Human Visual System (HVS) considering:
//! 1. Background Luminance Adaptation (single-hump curve; ~zero at flat white/black,
//!    per implementation plan Phase 2 — see `compute_spatial_jnd`)
//! 2. Texture & Spatial Edge Masking (Contrast Sensitivity Function)

/// Compute the spatial-domain Just-Noticeable Difference (JND) threshold map for an image.
///
/// Returns a map of same width and height where each value represents the maximum
/// permissible pixel perturbation amplitude (in 0-255 scale) that remains imperceptible.
pub fn compute_spatial_jnd(gray_image: &[f32], width: usize, height: usize) -> Vec<f32> {
    assert_eq!(gray_image.len(), width * height);
    let mut jnd_map = vec![3.0f32; width * height];

    if width < 5 || height < 5 {
        return jnd_map;
    }

    for y in 2..height - 2 {
        for x in 2..width - 2 {
            // 1. Local average background luminance and variance (5x5 neighborhood)
            let mut luma_sum = 0.0f32;
            let mut luma_sq_sum = 0.0f32;
            for dy in -2..=2 {
                for dx in -2..=2 {
                    let px = (y as isize + dy) as usize * width + (x as isize + dx) as usize;
                    let val = gray_image[px];
                    luma_sum += val;
                    luma_sq_sum += val * val;
                }
            }
            let bg_luma = luma_sum / 25.0f32;
            let variance = (luma_sq_sum / 25.0f32 - bg_luma * bg_luma).max(0.0);
            let local_std_dev = variance.sqrt();

            // Luminance adaptation masking T_L.
            //
            // Plan (implementation_plan.md, Phase 2, prism_hvs): the JND map must
            // allocate ~zero energy to flat white/black areas while keeping a
            // reasonable threshold at mid-gray. The classic Yang-Bovik T_L grows at
            // low luminance (T_L(0) = 20, T_L(255) = 6), the opposite of that intent,
            // so the curve is inverted: a single hump peaking at ~6.5 near mid-gray
            // and falling to the clamp floor (1.5) at both extremes.
            let norm = bg_luma / 255.0;
            let t_l = 1.5 + 5.0 * (4.0 * norm * (1.0 - norm));

            // 2. Texture & edge gradient masking T_T
            let p_c = y * width + x;
            let center_val = gray_image[p_c];
            let mut grad_max = 0.0f32;
            for &(dx, dy) in &[(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)] {
                let n_idx = (y as isize + dy) as usize * width + (x as isize + dx) as usize;
                let diff = (gray_image[n_idx] - center_val).abs();
                if diff > grad_max {
                    grad_max = diff;
                }
            }

            let texture_energy = local_std_dev.max(grad_max);
            let t_t = texture_energy * 0.28;

            // 3. Combined non-linear masking threshold
            let jnd = t_l + t_t - 0.3 * t_l.min(t_t);
            jnd_map[p_c] = jnd.clamp(1.5, 32.0);
        }
    }

    // Edge padding
    for y in 0..height {
        jnd_map[y * width] = jnd_map[y * width + 2];
        jnd_map[y * width + 1] = jnd_map[y * width + 2];
        jnd_map[y * width + width - 1] = jnd_map[y * width + width - 3];
        jnd_map[y * width + width - 2] = jnd_map[y * width + width - 3];
    }
    for x in 0..width {
        jnd_map[x] = jnd_map[2 * width + x];
        jnd_map[width + x] = jnd_map[2 * width + x];
        jnd_map[(height - 1) * width + x] = jnd_map[(height - 3) * width + x];
        jnd_map[(height - 2) * width + x] = jnd_map[(height - 3) * width + x];
    }

    jnd_map
}
