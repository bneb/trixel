//! In-place 1D and 2D Radix-2 Fast Fourier Transform (FFT) and Inverse FFT (IFFT).

use std::f32::consts::PI;

/// In-place 1D Radix-2 Decimation-in-Time FFT.
pub fn fft1d(real: &mut [f32], imag: &mut [f32], inverse: bool) {
    let n = real.len();
    assert_eq!(imag.len(), n);
    assert!(n.is_power_of_two(), "FFT length must be a power of 2");

    // Bit-reversal permutation
    let mut j = 0;
    for i in 0..n {
        if i < j {
            real.swap(i, j);
            imag.swap(i, j);
        }
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
    }

    // Cooley-Tukey butterflies
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle_sign = if inverse { 1.0 } else { -1.0 };
        let angle = angle_sign * 2.0 * PI / len as f32;
        let w_step_re = angle.cos();
        let w_step_im = angle.sin();

        for i in (0..n).step_by(len) {
            let mut w_re = 1.0f32;
            let mut w_im = 0.0f32;

            for k in 0..half {
                let u_idx = i + k;
                let v_idx = i + k + half;

                let u_re = real[u_idx];
                let u_im = imag[u_idx];

                let v_re = real[v_idx] * w_re - imag[v_idx] * w_im;
                let v_im = real[v_idx] * w_im + imag[v_idx] * w_re;

                real[u_idx] = u_re + v_re;
                imag[u_idx] = u_im + v_im;
                real[v_idx] = u_re - v_re;
                imag[v_idx] = u_im - v_im;

                let next_w_re = w_re * w_step_re - w_im * w_step_im;
                let next_w_im = w_re * w_step_im + w_im * w_step_re;
                w_re = next_w_re;
                w_im = next_w_im;
            }
        }
        len <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / n as f32;
        for i in 0..n {
            real[i] *= inv_n;
            imag[i] *= inv_n;
        }
    }
}

/// In-place 2D FFT.
pub fn fft2d(real: &mut [f32], imag: &mut [f32], width: usize, height: usize) {
    assert_eq!(real.len(), width * height);
    assert_eq!(imag.len(), width * height);
    assert!(width.is_power_of_two() && height.is_power_of_two());

    // Row transforms
    let mut row_r = vec![0.0f32; width];
    let mut row_i = vec![0.0f32; width];
    for y in 0..height {
        let start = y * width;
        row_r.copy_from_slice(&real[start..start + width]);
        row_i.copy_from_slice(&imag[start..start + width]);

        fft1d(&mut row_r, &mut row_i, false);

        real[start..start + width].copy_from_slice(&row_r);
        imag[start..start + width].copy_from_slice(&row_i);
    }

    // Column transforms
    let mut col_r = vec![0.0f32; height];
    let mut col_i = vec![0.0f32; height];
    for x in 0..width {
        for y in 0..height {
            col_r[y] = real[y * width + x];
            col_i[y] = imag[y * width + x];
        }

        fft1d(&mut col_r, &mut col_i, false);

        for y in 0..height {
            real[y * width + x] = col_r[y];
            imag[y * width + x] = col_i[y];
        }
    }
}

/// In-place 2D Inverse FFT.
pub fn ifft2d(real: &mut [f32], imag: &mut [f32], width: usize, height: usize) {
    assert_eq!(real.len(), width * height);
    assert_eq!(imag.len(), width * height);
    assert!(width.is_power_of_two() && height.is_power_of_two());

    // Row inverse transforms
    let mut row_r = vec![0.0f32; width];
    let mut row_i = vec![0.0f32; width];
    for y in 0..height {
        let start = y * width;
        row_r.copy_from_slice(&real[start..start + width]);
        row_i.copy_from_slice(&imag[start..start + width]);

        fft1d(&mut row_r, &mut row_i, true);

        real[start..start + width].copy_from_slice(&row_r);
        imag[start..start + width].copy_from_slice(&row_i);
    }

    // Column inverse transforms
    let mut col_r = vec![0.0f32; height];
    let mut col_i = vec![0.0f32; height];
    for x in 0..width {
        for y in 0..height {
            col_r[y] = real[y * width + x];
            col_i[y] = imag[y * width + x];
        }

        fft1d(&mut col_r, &mut col_i, true);

        for y in 0..height {
            real[y * width + x] = col_r[y];
            imag[y * width + x] = col_i[y];
        }
    }
}
