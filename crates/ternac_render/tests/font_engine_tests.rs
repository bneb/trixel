use ternac_render::{MockFontEngine, FontEngine};

#[test]
fn font_engine_mock_returns_empty() {
    let constraints = MockFontEngine::string_to_constraints("HELLO", 0, 0);
    assert!(constraints.is_empty());
}
