use ternac_cv::LuminanceBands;

#[test]
fn default_bands_quantize_state_0() {
    let bands = LuminanceBands::default();
    assert_eq!(bands.quantize(0), 0);
    assert_eq!(bands.quantize(25), 0);
    assert_eq!(bands.quantize(51), 0);
}

#[test]
fn default_bands_quantize_state_1() {
    let bands = LuminanceBands::default();
    assert_eq!(bands.quantize(102), 1);
    assert_eq!(bands.quantize(128), 1);
    assert_eq!(bands.quantize(152), 1);
}

#[test]
fn default_bands_quantize_state_2() {
    let bands = LuminanceBands::default();
    assert_eq!(bands.quantize(204), 2);
    assert_eq!(bands.quantize(230), 2);
    assert_eq!(bands.quantize(255), 2);
}

#[test]
fn default_bands_guard_bands_produce_erasure() {
    let bands = LuminanceBands::default();
    // Gap between State 0 upper (51) and State 1 lower (102)
    assert_eq!(bands.quantize(52), 3);
    assert_eq!(bands.quantize(80), 3);
    assert_eq!(bands.quantize(101), 3);
    // Gap between State 1 upper (152) and State 2 lower (204)
    assert_eq!(bands.quantize(153), 3);
    assert_eq!(bands.quantize(180), 3);
    assert_eq!(bands.quantize(203), 3);
}

#[test]
fn custom_bands() {
    let bands = LuminanceBands {
        state_0_upper: 30,
        state_1_lower: 80,
        state_1_upper: 170,
        state_2_lower: 220,
    };
    assert_eq!(bands.quantize(15), 0);
    assert_eq!(bands.quantize(50), 3);  // guard
    assert_eq!(bands.quantize(125), 1);
    assert_eq!(bands.quantize(200), 3); // guard
    assert_eq!(bands.quantize(240), 2);
}
