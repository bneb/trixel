use prism_core::ldpc::{LdpcConfig, QcLdpcCodec};
use prism_core::interleaver::SpatialInterleaver;

#[test]
fn test_ldpc_encode_satisfies_parity() {
    let config = LdpcConfig::rate_one_third_z32();
    let codec = QcLdpcCodec::new(config);

    let k_bits = codec.k_info_bits();
    let message: Vec<u8> = (0..k_bits).map(|i| (i % 2) as u8).collect();

    let codeword = codec.encode(&message);
    assert_eq!(codeword.len(), codec.n_code_bits());

    // Codeword must satisfy H * c^T = 0 mod 2
    assert!(codec.verify_syndrome(&codeword), "Encoded codeword must have zero syndrome");
}

#[test]
fn test_ldpc_min_sum_decodes_clean_signal() {
    let config = LdpcConfig::rate_one_third_z32();
    let codec = QcLdpcCodec::new(config);

    let k_bits = codec.k_info_bits();
    let message: Vec<u8> = (0..k_bits).map(|i| ((i * 7 + 3) % 2) as u8).collect();
    let codeword = codec.encode(&message);

    // Convert bits to clean LLRs: bit 0 -> +10.0, bit 1 -> -10.0
    let llrs: Vec<f32> = codeword.iter().map(|&b| if b == 0 { 10.0 } else { -10.0 }).collect();

    let decoded = codec.decode_min_sum(&llrs, 30, 0.8125).expect("Clean LLRs must decode");
    assert_eq!(decoded, message, "Decoded message must match original");
}

#[test]
fn test_ldpc_min_sum_corrects_channel_noise() {
    let config = LdpcConfig::rate_one_third_z32();
    let codec = QcLdpcCodec::new(config);

    let _k_bits = codec.k_info_bits();
    let message: Vec<u8> = vec![
        1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1,
        0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0,
        1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0,
        0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1,
        1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1,
        0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0,
        1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0,
        0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1,
    ];
    let codeword = codec.encode(&message);

    let sigma: f32 = 0.65; // AWGN standard deviation (~3.7 dB SNR)
    let mut rng_state: u64 = 123456789;
    let mut rand_uniform = || -> f32 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng_state >> 32) as u32 as f32 + 1.0) / (u32::MAX as f32 + 2.0)
    };

    let mut llrs = Vec::with_capacity(codeword.len());
    let mut errors_injected = 0;
    for &b in &codeword {
        let u1 = rand_uniform();
        let u2 = rand_uniform();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        let s = if b == 0 { 1.0f32 } else { -1.0f32 };
        let y = s + sigma * z0;
        if (y < 0.0 && b == 0) || (y > 0.0 && b == 1) {
            errors_injected += 1;
        }
        // Channel LLR for AWGN: L = 2*y / sigma^2
        llrs.push(2.0 * y / (sigma * sigma));
    }
    eprintln!("AWGN Channel: {} raw bit errors injected / {}", errors_injected, codeword.len());

    let decoded = codec.decode_min_sum(&llrs, 50, 0.8125).expect("Belief propagation must correct errors");
    assert_eq!(decoded, message, "Decoded message must match original despite channel noise");
}

#[test]
fn test_ldpc_min_sum_recovers_burst_erasures() {
    let config = LdpcConfig::rate_one_third_z32();
    let codec = QcLdpcCodec::new(config);

    let k_bits = codec.k_info_bits();
    let message: Vec<u8> = (0..k_bits).map(|i| (i % 2) as u8).collect();
    let codeword = codec.encode(&message);

    let mut llrs: Vec<f32> = codeword.iter().map(|&b| if b == 0 { 4.0 } else { -4.0 }).collect();

    // Erase a burst of 15% of the codeword (LLR = 0.0 means complete uncertainty)
    for i in 50..100 {
        llrs[i] = 0.0;
    }

    let decoded = codec.decode_min_sum(&llrs, 30, 0.8125).expect("LDPC must recover burst erasures");
    assert_eq!(decoded, message);
}

#[test]
fn test_spatial_interleaver_is_bijective() {
    let num_bits = 384;
    let width = 24;
    let height = 16;
    assert_eq!(width * height, num_bits);

    let interleaver = SpatialInterleaver::new(width, height);
    let coords: Vec<(usize, usize)> = (0..num_bits).map(|i| interleaver.index_to_coord(i)).collect();

    // Check all coordinates are within bounds
    for &(x, y) in &coords {
        assert!(x < width);
        assert!(y < height);
    }

    // Check uniqueness (bijection)
    let mut seen = std::collections::HashSet::new();
    for &c in &coords {
        assert!(seen.insert(c), "Coordinate {:?} was visited more than once", c);
    }
    assert_eq!(seen.len(), num_bits);
}
