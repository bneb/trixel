//! Belief Propagation module tests: Tanner regularity, circulant encode
//! structure, AWGN threshold sweep, and burst/random erasure recovery.

use prism_core::belief_propagation::BeliefPropagation;
use prism_core::ldpc::{LdpcConfig, QcLdpcCodec};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

/// Normalized Min-Sum normalization factor from the plan.
const ALPHA: f32 = 0.8125;

/// Standard normal sample via Box-Muller (deterministic per seed).
fn gaussian(rng: &mut StdRng) -> f32 {
    let u1: f64 = rng.gen::<f64>();
    let u2: f64 = rng.gen::<f64>();
    let u1 = u1.clamp(1e-12, 1.0 - 1e-12);
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
}

/// Channel LLRs for BPSK (+/-1) over AWGN: L = 2*y/sigma^2.
fn awgn_llrs(codeword: &[u8], sigma: f32, rng: &mut StdRng) -> Vec<f32> {
    codeword
        .iter()
        .map(|&b| {
            let s = if b == 0 { 1.0f32 } else { -1.0f32 };
            let y = s + sigma * gaussian(rng);
            2.0 * y / (sigma * sigma)
        })
        .collect()
}

fn random_message(rng: &mut StdRng, k: usize) -> Vec<u8> {
    (0..k).map(|_| rng.gen::<u8>() & 1).collect()
}

fn snr_db(sigma: f32) -> f32 {
    10.0 * (1.0 / (sigma * sigma)).log10()
}

#[test]
fn tanner_graph_regularity() {
    for config in [LdpcConfig::rate_one_third_z32(), LdpcConfig::rate_one_third_z64()] {
        let z = config.z;
        let m_b = config.num_parity_blocks;
        let n_b = config.num_info_blocks + config.num_parity_blocks;

        let codec = QcLdpcCodec::new(config.clone());
        let bp = BeliefPropagation::new(codec, 30);
        let check_degrees = bp.check_degrees();
        let var_degrees = bp.variable_degrees();

        // Every check node's degree equals its base-matrix row weight.
        for r_b in 0..m_b {
            let expected = config.base_matrix[r_b].iter().filter(|&&s| s >= 0).count();
            for k in 0..z {
                assert_eq!(
                    check_degrees[r_b * z + k],
                    expected,
                    "check ({r_b},{k}) degree for Z={z}"
                );
            }
        }

        // Every variable node's degree equals its base-matrix column weight,
        // and no variable node is isolated (degree 0).
        for c_b in 0..n_b {
            let expected = (0..m_b).filter(|&r_b| config.base_matrix[r_b][c_b] >= 0).count();
            assert!(expected > 0, "base column {c_b} must have at least one 1");
            for k in 0..z {
                assert_eq!(
                    var_degrees[c_b * z + k],
                    expected,
                    "variable ({c_b},{k}) degree for Z={z}"
                );
            }
        }

        // Total edge count consistent across both views and the base matrix.
        let base_edges = config
            .base_matrix
            .iter()
            .map(|row| row.iter().filter(|&&s| s >= 0).count())
            .sum::<usize>()
            * z;
        assert_eq!(check_degrees.iter().sum::<usize>(), bp.total_edges());
        assert_eq!(var_degrees.iter().sum::<usize>(), bp.total_edges());
        assert_eq!(base_edges, bp.total_edges(), "total edges vs base matrix for Z={z}");
    }
}

#[test]
fn circulant_lower_triangular_encode_structure() {
    for config in [LdpcConfig::rate_one_third_z32(), LdpcConfig::rate_one_third_z64()] {
        let z = config.z;
        let k_b = config.num_info_blocks;
        let m_b = config.num_parity_blocks;

        let codec = QcLdpcCodec::new(config.clone());
        let k = codec.k_info_bits();

        for seed in 1..=5u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let message = random_message(&mut rng, k);
            let codeword = codec.encode(&message);
            assert_eq!(codeword.len(), codec.n_code_bits());
            assert!(codec.verify_syndrome(&codeword), "Z={z} seed={seed}: encoded codeword must satisfy H*c^T=0");

            // Lower-triangular identity structure: the diagonal parity block
            // of each row has shift 0, so parity block p_r must equal the
            // accumulated row sum over info blocks and earlier parity blocks.
            for r_b in 0..m_b {
                let mut acc = vec![0u8; z];
                for c_b in 0..k_b {
                    let shift = config.base_matrix[r_b][c_b];
                    if shift >= 0 {
                        let s = shift as usize % z;
                        let block = &codeword[c_b * z..(c_b + 1) * z];
                        for kk in 0..z {
                            acc[kk] ^= block[(kk + s) % z];
                        }
                    }
                }
                for c_b in 0..r_b {
                    let shift = config.base_matrix[r_b][k_b + c_b];
                    if shift >= 0 {
                        let s = shift as usize % z;
                        let block = &codeword[(k_b + c_b) * z..(k_b + c_b + 1) * z];
                        for kk in 0..z {
                            acc[kk] ^= block[(kk + s) % z];
                        }
                    }
                }
                let parity = &codeword[(k_b + r_b) * z..(k_b + r_b + 1) * z];
                assert_eq!(
                    parity,
                    acc.as_slice(),
                    "Z={z} seed={seed}: parity block p_{r_b} must equal the accumulated row sum"
                );
            }
        }
    }
}

#[test]
fn awgn_sweep_min_sum_100_percent_threshold() {
    let config = LdpcConfig::rate_one_third_z32();
    let codec = QcLdpcCodec::new(config);
    let k = codec.k_info_bits();
    let bp = BeliefPropagation::new(codec.clone(), 100);

    // sigma ~= 0.841 is 1.5 dB SNR; entries are ordered strongest noise first.
    // Points below 0.9 dB probe beyond the plan target to find the true threshold.
    let sigmas = [1.2, 1.15, 1.1, 1.05, 1.0, 0.95, 0.9, 0.85, 0.841, 0.8, 0.75, 0.7];
    const N_SEEDS: u64 = 5;

    let mut results: Vec<(f32, usize, usize)> = Vec::new();
    for &sigma in &sigmas {
        let mut failures = 0usize;
        let mut wrong = 0usize;
        for seed in 1..=N_SEEDS {
            let mut rng = StdRng::seed_from_u64(seed);
            let message = random_message(&mut rng, k);
            let codeword = codec.encode(&message);
            let llrs = awgn_llrs(&codeword, sigma, &mut rng);
            match bp.decode_min_sum(&llrs, ALPHA) {
                Ok(dec) if dec == message => {}
                Ok(_) => wrong += 1,
                Err(_) => failures += 1,
            }
        }
        results.push((sigma, failures, wrong));
    }

    println!("AWGN sweep (min-sum, Z=32, K={k}, alpha={ALPHA}, {} seeds per sigma):", N_SEEDS);
    println!("sigma    SNR(dB)   failures   wrong-codeword");
    for (sigma, failures, wrong) in &results {
        println!("{sigma:.3}   {snr:6.2}   {failures:>8}   {wrong:>12}", snr = snr_db(*sigma));
    }

    // Strongest noise level (largest sigma) at which every seed decodes to
    // the exact message.
    let threshold = results
        .iter()
        .find(|(_, failures, wrong)| *failures == 0 && *wrong == 0)
        .unwrap_or_else(|| panic!("min-sum failed to reach 100% even at the weakest noise level"));
    println!(
        "threshold: 100% min-sum recovery at sigma = {:.3} ({:.2} dB)",
        threshold.0,
        snr_db(threshold.0)
    );

    // Assert the measured threshold (strongest noise level actually achieved).
    // Measured: 100% at sigma = 1.000 (0.00 dB), 5/5 seeds, which exceeds the
    // 1.5 dB plan target by 1.5 dB. Failures begin at sigma 1.05 (-0.42 dB).
    let measured_sigma = threshold.0;
    assert!(
        measured_sigma >= 1.0,
        "min-sum must reach 100% at sigma 1.0 (0.00 dB); measured threshold {measured_sigma}"
    );
}

#[test]
fn awgn_sweep_sum_product_100_percent_threshold() {
    let config = LdpcConfig::rate_one_third_z32();
    let codec = QcLdpcCodec::new(config);
    let k = codec.k_info_bits();
    let bp = BeliefPropagation::new(codec.clone(), 100);

    let sigmas = [1.2, 1.15, 1.1, 1.05, 1.0, 0.95, 0.9, 0.85, 0.841, 0.8, 0.75, 0.7];
    const N_SEEDS: u64 = 5;

    let mut results: Vec<(f32, usize, usize)> = Vec::new();
    for &sigma in &sigmas {
        let mut failures = 0usize;
        let mut wrong = 0usize;
        for seed in 1..=N_SEEDS {
            let mut rng = StdRng::seed_from_u64(seed);
            let message = random_message(&mut rng, k);
            let codeword = codec.encode(&message);
            let llrs = awgn_llrs(&codeword, sigma, &mut rng);
            match bp.decode_sum_product(&llrs) {
                Ok(dec) if dec == message => {}
                Ok(_) => wrong += 1,
                Err(_) => failures += 1,
            }
        }
        results.push((sigma, failures, wrong));
    }

    println!("AWGN sweep (sum-product, Z=32, K={k}, {} seeds per sigma):", N_SEEDS);
    println!("sigma    SNR(dB)   failures   wrong-codeword");
    for (sigma, failures, wrong) in &results {
        println!("{sigma:.3}   {snr:6.2}   {failures:>8}   {wrong:>12}", snr = snr_db(*sigma));
    }

    let threshold = results
        .iter()
        .find(|(_, failures, wrong)| *failures == 0 && *wrong == 0)
        .unwrap_or_else(|| panic!("sum-product failed to reach 100% even at the weakest noise level"));
    println!(
        "threshold: 100% sum-product recovery at sigma = {:.3} ({:.2} dB)",
        threshold.0,
        snr_db(threshold.0)
    );

    let measured_sigma = threshold.0;
    assert!(
        measured_sigma >= 1.0,
        "sum-product must reach 100% at sigma 1.0 (0.00 dB); measured threshold {measured_sigma}"
    );
}

#[test]
fn min_sum_recovers_15_percent_burst_erasures() {
    let config = LdpcConfig::rate_one_third_z32();
    let codec = QcLdpcCodec::new(config);
    let k = codec.k_info_bits();
    let bp = BeliefPropagation::new(codec.clone(), 60);

    let mut rng = StdRng::seed_from_u64(7);
    let message = random_message(&mut rng, k);
    let codeword = codec.encode(&message);

    let burst_len = (codeword.len() as f32 * 0.15) as usize;
    assert_eq!(burst_len, 57, "15% of N=384 is 57.6 -> 57");

    let mut llrs: Vec<f32> = codeword.iter().map(|&b| if b == 0 { 4.0 } else { -4.0 }).collect();
    for llr in llrs[50..50 + burst_len].iter_mut() {
        *llr = 0.0;
    }

    let decoded = bp
        .decode_min_sum(&llrs, ALPHA)
        .expect("min-sum must recover a 15% burst erasure");
    assert_eq!(decoded, message, "min-sum burst erasure recovery");
}

#[test]
fn sum_product_recovers_15_percent_burst_erasures() {
    let config = LdpcConfig::rate_one_third_z32();
    let codec = QcLdpcCodec::new(config);
    let k = codec.k_info_bits();
    let bp = BeliefPropagation::new(codec.clone(), 60);

    let mut rng = StdRng::seed_from_u64(11);
    let message = random_message(&mut rng, k);
    let codeword = codec.encode(&message);

    let burst_len = (codeword.len() as f32 * 0.15) as usize;
    let mut llrs: Vec<f32> = codeword.iter().map(|&b| if b == 0 { 4.0 } else { -4.0 }).collect();
    for llr in llrs[50..50 + burst_len].iter_mut() {
        *llr = 0.0;
    }

    let decoded = bp
        .decode_sum_product(&llrs)
        .expect("sum-product must recover a 15% burst erasure");
    assert_eq!(decoded, message, "sum-product burst erasure recovery");
}

#[test]
fn both_decoders_recover_20_percent_random_erasures() {
    let config = LdpcConfig::rate_one_third_z32();
    let codec = QcLdpcCodec::new(config);
    let k = codec.k_info_bits();
    let bp = BeliefPropagation::new(codec.clone(), 60);

    let mut rng = StdRng::seed_from_u64(13);
    let message = random_message(&mut rng, k);
    let codeword = codec.encode(&message);

    let n_erase = (codeword.len() as f32 * 0.20) as usize;
    assert_eq!(n_erase, 76, "20% of N=384 is 76.8 -> 76");

    // Deterministic random subset of positions to erase.
    let mut positions: Vec<usize> = (0..codeword.len()).collect();
    positions.shuffle(&mut rng);

    let mut llrs: Vec<f32> = codeword.iter().map(|&b| if b == 0 { 4.0 } else { -4.0 }).collect();
    for &i in positions.iter().take(n_erase) {
        llrs[i] = 0.0;
    }

    let decoded = bp
        .decode_min_sum(&llrs, ALPHA)
        .expect("min-sum must recover 20% random erasures");
    assert_eq!(decoded, message, "min-sum random erasure recovery");

    let decoded = bp
        .decode_sum_product(&llrs)
        .expect("sum-product must recover 20% random erasures");
    assert_eq!(decoded, message, "sum-product random erasure recovery");
}

#[test]
fn belief_propagation_rejects_wrong_llr_length() {
    let config = LdpcConfig::rate_one_third_z32();
    let codec = QcLdpcCodec::new(config);
    let bp = BeliefPropagation::new(codec, 10);

    let err = bp.decode_min_sum(&[0.0f32; 10], ALPHA).unwrap_err();
    assert!(err.to_string().contains("Invalid input size"));
    let err = bp.decode_sum_product(&[0.0f32; 10]).unwrap_err();
    assert!(err.to_string().contains("Invalid input size"));
}
