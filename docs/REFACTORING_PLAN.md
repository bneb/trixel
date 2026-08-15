# Trixel Refactoring Plan — Square-Grid Legacy Removal, Renames, Feature Backfill

Goal: make the triangular-grid pipeline the only pipeline. Delete every square-grid
legacy path, drop the `Tri*` prefixes, and backfill the three features that only the
square path currently provides (CLI decode, `--text` typography, dimension handling).

Status: steps 1–4 of §5 are complete (RS-math extraction, square-legacy cut,
Tri-prefix rename, CLI triangular decode). §4.2 (typography), §4.3 (scanner
dimension detection), and step 7's web cleanup (`trixel-web/`) are pending;
`README.md` and the `SPEC.md` source-of-truth file list are updated. Note:
§1.11/§1.13/§1.14 were superseded — the legacy 5×8 Tri-Anchors were kept and
hardened (Phase 3, commit `5c2bdbc`) and remain the calibration source in `cv.rs`.

Context facts the plan is built on (verified against source):

- `docs/SPEC.md` and `CLAUDE.md` both declare the radial triangular architecture
  (3 Macro-Anchors, masked radial spiral) the standard; the legacy 5×8 Tri-Anchors
  and the square-grid code are deprecated dead ends.
- Dependency chain: `trixel_core` ← `trixel_solver` ← `trixel_render` / `trixel_cv`
  ← `trixel_scanner` (wasm) and `trixel` (CLI). The CLI does **not** depend on
  `trixel_scanner` — its decode backfill must use `trixel_cv` directly.
- The square API is imported by **all** downstream crates (`trixel_solver::anchor`
  is used by main.rs, scanner, cv, render), so the square cut is one atomic
  workspace-wide change; it cannot be done crate-by-crate with green builds in
  between (detailed in §5).

---

## 1. Files to DELETE (square-grid legacy)

### 1.1 `crates/trixel_solver/src/anchor.rs` — entire file
3×3 L-bracket square anchors (`ANCHOR_SIZE`, `ANCHOR_PATTERNS`, `corner_positions`,
`is_in_anchor_region`, `scan_for_false_anchors`). Every consumer is square-legacy:
`GaussSolver`, `MockSolver`/`AnchorSolver`, `AnchorVision`, `AnchorRenderer`,
scanner square decode, main.rs square path, and 4 test files. Nothing in the tri
pipeline references it.

### 1.2 `crates/trixel_solver/src/gauss_solver.rs` — entire file
Square `GaussSolver::resolve_matrix` + `grid_to_flat_coords` (row-major square
mapping). **Exception:** `compute_transform_matrix` and `build_parity_check_matrix`
are shared — the tri solver (`gauss_solver.rs`) imports both. Move them out first (to
`gauss.rs` or a new `rs_math.rs`) before deleting this file (see §5 step 1).

### 1.3 `crates/trixel_solver/src/lib.rs` — delete contents, not file
Remove `MockSolver`, `AnchorSolver`, the `MatrixSolver` trait, the `TritMatrix`
import, and the `pub mod anchor;` line. Keep `ConstraintMask`, `SolverError`,
`gauss`, `anchor`, `gauss_solver`, `header`, `layout`
(`ConstraintMask` is used by every tri solver method; `gauss` supplies
`Gf3Matrix`/`solve_gf3_with_default`/`solve_gf3_with_targets` to the tri solver).

### 1.4 `crates/trixel_render/src/halftone.rs` — entire file
`HalftoneEngine` (square `image_to_constraints`, edge detection via
`imageproc::sobel_gradients`/`equalize_histogram`). Only consumer: main.rs square
path (the "compensation loop"). The tri path does its own image-guided target
map inline in main.rs.

### 1.5 `crates/trixel_render/src/font.rs` — entire file
`TrixelFont` (5×7 glyphs → square `ConstraintMask`s + backing plate). Square-only;
typography for the tri path is re-implemented in §4.2 (reusing the glyph bitmap
data, which is geometry-neutral).

### 1.6 `crates/trixel_render/src/lib.rs` — delete contents, not file
Remove `Renderer` trait, `MockRenderer`, `AnchorRenderer` (both `render_png` and
`render_halftone_png`), `FontEngine` trait, `MockFontEngine`, and the
`palette::{Srgb, Hsl, IntoColor}` imports. Keep `RenderError`, module decls, and
`pub use font::TrixelFont;` only if the tri font keeps that name (§4.2).

### 1.7 `crates/trixel_cv/src/lib.rs` — delete contents, not file
Remove `VisionPipeline` trait, `MockVision`, `AnchorVision`,
`calibrate_from_anchors`, `sample_module_luminance`, `median`, and the
`trixel_solver::anchor` import. Keep `LuminanceBands` (+ its unit API — cv.rs
uses it), `VisionError`, `geometry`, `cv` modules.

### 1.8 `crates/trixel_core/src/lib.rs` — delete contents, not file
- `TritMatrix` struct (only square consumers: MockSolver/AnchorSolver,
  GaussSolver, AnchorVision, AnchorRenderer, scanner square path, main square
  path, tests). Dead after the cut.
- `MockEcc` (no production consumers at all; only `ecc_tests.rs`).
- `EccError::InvalidCapacity` variant (used only by `MockEcc`).
Keep: `MockCodec` + `TernaryCodec`, `RsEcc` + `ErrorCorrection`,
`encode_length`/`decode_length`, `LENGTH_PREFIX_TRITS`, `RS_HEADER_SYMBOLS` —
all used by the tri pipeline (scanner, CLI, tests).

### 1.9 `crates/trixel_scanner/src/lib.rs` — delete contents, not file
Remove `decode_image`, `decode_png_bytes`, `decode_png`, `decode_png_auto`,
`extract_payload`, and the square imports (`TritMatrix`,
`trixel_solver::anchor::is_in_anchor_region`, `AnchorVision`). Keep the tri
pipeline: `decode_tri_image`, `try_decode_tri`, `decode_tri_png_bytes`,
`try_decode_tri_camera`, `decode_rgba_frame`, `decode_camera_frame`,
`decode_tri_png`, `PipelineError`/`PipelineStage`.

### 1.10 `crates/trixel/src/main.rs` — delete contents, not file
Remove the whole square Encode path (lines ~322–535), the square Decode path
(lines ~537–571), `min_square_side`, `extract_payload_from_matrix`, and the
square imports (`GaussSolver`, `MatrixSolver`, `anchor::ANCHOR_SIZE`,
`is_in_anchor_region`, `AnchorVision`, `VisionPipeline`, `AnchorRenderer`,
`Renderer`, `FontEngine`, `TrixelFont`, `HalftoneEngine`). The `--triangular`
flag becomes the only mode (flag can stay as a no-op or be removed — see §4.1).

### 1.11 `crates/trixel_solver/src/anchor.rs` — delete section, not file
Delete the legacy 5×8 Tri-Anchor block: `TRI_ANCHOR_ROWS`, `TRI_ANCHOR_COLS`,
`CORNER_IDS`, `TRI_ANCHOR_PATTERNS`, `compute_anchor_crc`, `verify_anchor_crc`,
`detect_corner_id`, `tri_corner_positions`, `is_in_tri_anchor_region`. Keep the
Macro-Anchor and Mini-Anchor sections (used by radial/clockwise paths,
`layout`, `header`, `cv`, `render`).

### 1.12 `crates/trixel_cv/src/geometry.rs` — delete functions, not file
Delete `is_l_shape` and `classify_corners` (square 4-corner CV; only used by
`vision_anchor_tests.rs`). Verify and delete `triangle_area_signed` and
`distance_sq` if they have no remaining callers (neither is imported by
`cv.rs`). Keep everything `cv.rs` imports (listed at its line 19):
`Point, douglas_peucker, triangle_area, is_valid_triangle, sort_macro_anchors,
affine_from_triangles, affine_transform, centroid, perspective_from_4points,
perspective_transform, warp_perspective, extract_four_corners,
order_quad_corners`.

### 1.13 `crates/trixel_solver/src/gauss_solver.rs` — delete methods, not file
Delete the legacy 5×8-Tri-Anchor methods `resolve` (~lines 45–291) and
`resolve_image_guided` (~lines 292–551) and their unit tests (~lines
1880–2060). These are the only non-test users of the 5×8 anchor code. Keep the
radial / clockwise / interleaved methods and `grid_to_flat_coords`.

### 1.14 `crates/trixel_cv/src/cv.rs` — replace function
Replace `calibrate_from_anchors` (line ~709, samples the 4 legacy 5×8
anchors) with calibration from the 3 Macro-Anchors (`macro_corner_positions` +
`MACRO_ANCHOR_PATTERNS`-equivalent cells: shell = state 0, fixed interior = 2,
sync dot at relative `(3,5)` = 1 — the same all-3-states coverage the 5×8 block
gave). `extract_digital` is its only caller.

### 1.15 Test files to delete (they test deleted code)

| File | Tests |
|---|---|
| `crates/trixel_core/tests/trit_matrix_tests.rs` | `TritMatrix` |
| `crates/trixel_core/tests/ecc_tests.rs` | `MockEcc` |
| `crates/trixel_solver/tests/anchor_tests.rs` | square anchors |
| `crates/trixel_solver/tests/solver_tests.rs` | `MockSolver` |
| `crates/trixel_solver/tests/constraint_tests.rs` | `AnchorSolver` + `TrixelFont` — rewrite as tri-constraint tests instead (see §4.2) |
| `crates/trixel_solver/tests/gauss_tests.rs` | square `GaussSolver` + shared math — keep the `compute_transform_matrix`/`build_parity_check_matrix` cases, drop the rest |
| `crates/trixel_render/tests/renderer_tests.rs`, `render_anchor_tests.rs`, `halftone_renderer_tests.rs`, `font_tests.rs`, `font_engine_tests.rs` | square renderer / halftone / font |
| `crates/trixel_cv/tests/vision_anchor_tests.rs`, `vision_tests.rs` | square `AnchorVision` (geometry square fns live in `vision_anchor_tests.rs`) |
| `crates/trixel/tests/anchor_pipeline_tests.rs`, `pipeline_tests.rs`, `halftone_decode_test.rs`, `gauss_pipeline_tests.rs` | square end-to-end pipeline |
| `crates/trixel/tests/miserable_work_test.rs` | mixed — strip square parts, keep tri parts |
| `crates/trixel_scanner/tests/scanner_tests.rs` | mostly tri already; strip any square references |

### 1.16 Web artifacts
- `web/scanner/scanner.js` — remove the `decode_png_auto` import and its upload
  fallback branch (line ~191). Keep `decode_tri_png` / `decode_camera_frame`.
- `web/scanner/pkg/` — rebuild via `wasm-pack build crates/trixel_scanner --target
  web --out-dir ../../web/scanner/pkg` and commit (Cloudflare Pages GitOps).
- `trixel-web/` (untracked) — stale duplicate deploy copy with its own wasm
  bundle (dated Mar 4 vs. current Aug). Sync or delete; do not let it diverge.

---

## 2. Files to MODIFY (remove dead code, update imports)

Per-crate list of import/use changes beyond the deletions in §1:

### `crates/trixel/src/main.rs`
- Encode: drop the square branch; the tri branch becomes the body. Pass tri
  constraints from the typography backfill (§4.2) into
  `GaussSolver::resolve_radial(..., &constraints)` and build the
  `font_mask` for `render_halftone` (it already accepts one — currently
  always `None`).
- Decode: replace `AnchorVision::extract_matrix` + `extract_payload_from_matrix`
  + `RsEcc` with the tri pipeline (§4.1). Imports to update:
  `use trixel_cv::cv::CvPipeline;`, `use trixel_solver::{header, layout};`.

### `crates/trixel_scanner/src/lib.rs`
- Drop `use trixel_solver::anchor::is_in_anchor_region;` and `TritMatrix`/
  `AnchorVision` from the imports (line 15–17).
- `extract_payload` (square row-major skip-anchors) is deleted; the tri paths
  already use `layout::masked_radial_sequence`.

### `crates/trixel_cv/src/lib.rs`
- Drop `use trixel_solver::anchor::{ANCHOR_PATTERNS, ANCHOR_SIZE,
  corner_positions};` (line 8).
- `VisionError` stays; `LuminanceBands` stays.

### `crates/trixel_cv/src/cv.rs`
- Update `calibrate_from_anchors` → macro-anchor calibration (§1.14).
- `use crate::geometry::{...}` import stays (all ten symbols are tri-used).

### `crates/trixel_render/src/lib.rs`
- Drop `use trixel_solver::anchor;` (line 10) and `use trixel_core::TritMatrix;`
  (line 8). Keep `ConstraintMask` (tri font will emit these).

### `crates/trixel_render/src/render.rs`
- No square imports. `anchor` usage stays. After §1.13 the solver no longer
  emits 5×8 anchors; rendering already uses macro anchors.

### `crates/trixel_solver/src/lib.rs`, `gauss_solver.rs`, `layout.rs`,
`header.rs`
- `gauss_solver.rs`: the shared math moved to `gauss.rs` in §5 step 1; the
  import is now `use crate::gauss::{compute_transform_matrix,
  build_parity_check_matrix};`.
- Remove references to `anchor` (square) — `gauss_solver.rs` has none
  (it uses the renamed `anchor` module), and the square module declaration
  in `lib.rs` was removed.

### Tests
- `crates/trixel_solver/tests/gauss_tests.rs`: retain the
  `compute_transform_matrix` / `build_parity_check_matrix` tests, delete the
  `GaussSolver`/`grid_to_flat_coords`/`TrixelFont` tests.
- `crates/trixel/tests/miserable_work_test.rs`: keep tri portions only.

---

## 3. Rename map (drop the `Tri` prefix)

Do this **after** the square cut (§5 step 3) so freed-up names don't collide.

### 3.1 File renames

| Old | New | Notes |
|---|---|---|
| `crates/trixel_core/src/trigrid.rs` | `crates/trixel_core/src/grid.rs` | module `trigrid` → `grid` |
| `crates/trixel_solver/src/tri_anchor.rs` | `crates/trixel_solver/src/anchor.rs` | square `anchor.rs` deleted in step 2, name is free |
| `crates/trixel_solver/src/tri_gauss_solver.rs` | `crates/trixel_solver/src/gauss_solver.rs` | square `gauss_solver.rs` deleted in step 2 |
| `crates/trixel_solver/src/tri_header.rs` | `crates/trixel_solver/src/header.rs` | |
| `crates/trixel_solver/src/tri_layout.rs` | `crates/trixel_solver/src/layout.rs` | |
| `crates/trixel_render/src/tri_render.rs` | `crates/trixel_render/src/render.rs` | |
| `crates/trixel_render/src/tri_diffusion.rs` | `crates/trixel_render/src/diffusion.rs` | |
| `crates/trixel_cv/src/tri_cv.rs` | `crates/trixel_cv/src/cv.rs` | |

### 3.2 Symbol renames

| Old | New |
|---|---|
| `TriGrid` | `Grid` |
| `TriGrid::is_up` | `Grid::is_up` |
| `TriGaussSolver` | `GaussSolver` (square `GaussSolver` is gone) |
| `TriGaussSolver::resolve_trigrid_radial` | `GaussSolver::resolve_radial` |
| `TriGaussSolver::resolve_trigrid_radial_image_guided` | `GaussSolver::resolve_radial_image_guided` |
| `TriGaussSolver::resolve_trigrid_clockwise` | `GaussSolver::resolve_clockwise` |
| `TriGaussSolver::resolve_trigrid_clockwise_image_guided` | `GaussSolver::resolve_clockwise_image_guided` |
| `TriGaussSolver::resolve_trigrid_interleaved_image_guided` | `GaussSolver::resolve_interleaved_image_guided` |
| `tri_grid_to_flat_coords` | `grid_to_flat_coords` |
| `TriAnchorRenderer` | `AnchorRenderer` (square `AnchorRenderer` is gone) |
| `TriAnchorRenderer::render_trigrid` | `AnchorRenderer::render` |
| `TriAnchorRenderer::render_halftone_trigrid` | `AnchorRenderer::render_halftone` |
| `TriCvPipeline` | `CvPipeline` |
| `TriCvPipeline::extract_trigrid_digital` | `CvPipeline::extract_grid_digital` |
| `TriCvPipeline::extract_trigrid_camera` | `CvPipeline::extract_grid_camera` |
| `TriNeighbor` | `Neighbor` (in `diffusion.rs`) |
| `tri_forward_neighbors` | `forward_neighbors` |
| `tri_anchor::tri_corner_positions` (5×8) | deleted (§1.11) |
| `tri_anchor::is_in_tri_anchor_region` (5×8) | deleted (§1.11) |
| `tri_anchor::write_macro_anchor` | `anchor::write_macro_anchor` (module path only) |
| `tri_anchor::is_in_macro_anchor_region` | `anchor::is_in_macro_anchor_region` (module path only) |
| `trixel_core::trigrid::TriGrid` in imports | `trixel_core::grid::Grid` |

Keep `MaskProfile`, `encode_header`/`decode_header`,
`masked_radial_sequence`, `clockwise_perimeter_sequence`,
`interleaved_sequence`, `macro_anchor_*`, `MINI_ANCHOR_*`, `TELOMERE_SYMBOL`
names as-is — they have no `Tri` prefix. Keep the `MINI_ANCHOR_*` prefix (it
denotes the Mini-Anchor architecture, not a legacy prefix).

Doc mentions: `CLAUDE.md` and `README.md` already describe the target state
(trixel_* crate list); the `SPEC.md` source-of-truth file list now points at
the renamed modules.

---

## 4. Feature backfill

### 4.1 CLI triangular decode (`trixel` `decode` subcommand)

Current state: `decode` is square-only (`AnchorVision::extract_matrix` +
`extract_payload_from_matrix` + `RsEcc::correct_errors` + `MockCodec`). The
`--triangular` flag exists only on `encode`.

Plan (mirrors `crates/trixel_scanner/src/lib.rs::try_decode_tri`, which is the
proven path, but self-contained — the CLI cannot depend on the wasm crate):

1. In main.rs, add `--triangular` (or `--auto`; keep `--module-size` accepted).
2. Load image, then derive candidate `(rows, cols)` from geometry:
   `cell_h` from `module_size` if given, else try `cell_h` in 3..=24 with
   `rows = img_h / cell_h`, `cols = rows * 2` (2:1 ratio, §1.3 of SPEC), keeping
   candidates with `rows ≥ 16` and `cols ≥ 26` (radial minimums) and whose
   predicted width `cols·cell_h/2 + cell_h/2` is close to `img_w`.
3. Per candidate: `CvPipeline::extract_digital(image, rows, cols,
   cell_h)` → `header::extract_header_from_grid` +
   `header::decode_header` (validate MaskProfile) →
   `layout::masked_radial_sequence` unroll →
   `RsEcc::correct_errors` (self-describing, §6.5 of SPEC) →
   `MockCodec::decode_trits` → print UTF-8 to stdout.
4. Track the deepest failure stage (dimension → extraction → header → ECC →
   codec → UTF-8) and report it, same as the scanner's `PipelineError`.
5. Factor: if the decode body becomes shared with the scanner later, extract it
   into `trixel_cv` (cv already depends on core + solver, so `RsEcc`/`MockCodec`
   are reachable) — optional, not required.

### 4.2 Typography (`--text`) for the triangular path

Current state: `--text`/`--text_x`/`--text_y` are ignored in the tri encode
path — `font_mask` is all-`None` and no constraints are passed to the solver.
All tri solver methods already accept `constraints: &[ConstraintMask]`, validate
them against anchor regions, and weave them into the codeword as fixed trits
with message-offset sliding (`gauss_solver.rs` lines 563–589, 1096–1106,
1354–1364, 1633–1643). So this is a rasterizer + wiring task:

1. Keep `crates/trixel_render/src/glyphs.rs` (the 5×7 bitmap data is
   geometry-neutral) but rename to `glyphs.rs` data-only, or move into the new
   tri font module. Delete only the square constraint emitter (`font.rs`).
2. New `TriFont` (in a new `trixel_render/src/font.rs` or `tri_font.rs`):
   rasterize glyphs onto the triangle grid at 2:1 scale —
   a glyph pixel at virtual `(gx, gy)` covers triangle columns
   `[start_col + 2·gx, start_col + 2·gx + 1]` on row `start_row + gy`
   (emit a constraint for the triangle whose centroid is inside the stroke
   pixel). Emit `ConstraintMask { x: col, y: row, required_state }` for stroke
   (0) and halo (2) cells, leave `None` free — same contract as the square
   `TrixelFont`.
3. Return both the constraint list and the `font_mask: Vec<Vec<Option<u8>>>`
   (row-major over the grid) that `render_halftone` already consumes
   (font immunity, `render.rs` lines 298–299).
4. Wire into main.rs tri path: constraints → solver, font_mask → renderer.
5. Tests: rewrite `constraint_tests.rs` as tri-constraint tests (constraints
   never overlap macro/mini anchor regions; payload round-trips with text
   present; off-grid + backing-plate behavior).

### 4.3 Scanner header-based dimension detection

Current state: `TRI_GRID_SIZES` is a hardcoded 6-entry list
(`scanner/lib.rs` lines 93–100); every decode brute-forces all six.

Plan:

1. Replace `TRI_GRID_SIZES` with `derive_candidate_dims(img_w, img_h)`:
   for `cell_h` in 3..=24, `rows = img_h / cell_h`, `cols = 2·rows`; filter
   radial minimums (`rows ≥ 16`, `cols ≥ 26`); sort by `|img_w − (cols·cell_h/2
   + cell_h/2)|`. Keeps arbitrary `--min-side` encodes decodable without an
   exhaustive list.
2. Clockwise architecture is already self-describing: the 36-trit inline header
   (`encode_clockwise_header`, `header.rs` §5.4) carries rows and cols with
   TMR. After a candidate extraction, decode the inline header; if its
   rows/cols disagree with the candidate, re-run extraction with the header's
   dimensions (true header-based dimension detection). Add
   `extract_dims_from_clockwise_header(gray, rows, cols)` to `cv.rs`.
3. Radial architecture: cross-check the decoded `MaskProfile` and the 12-trit
   length prefix (codeword length in trits) against candidate capacity —
   `usable = rows·cols − GQZ − 3 macro anchors`; a candidate whose `prefix +
   codeword` exceeds usable is rejected before RS decode. This prunes wrong
   candidates at the cheap length-prefix stage instead of at RS decode.
4. Keep the `PipelineStage` error taxonomy; add
   `decode_tri_png_auto(png_bytes)` as a wasm export and switch
   `web/scanner/scanner.js` uploads to it (replacing the `decode_png_auto`
   square fallback, which is being deleted).

---

## 5. Execution order (each step leaves `cargo test --workspace` green)

The square API spans all six crates, so the cut itself is one atomic step; the
steps around it are staged to keep the tree compilable and the rename trivial.

1. **Extract shared RS math.** Move `compute_transform_matrix` and
   `build_parity_check_matrix` from `gauss_solver.rs` into `gauss.rs` (rename to
   `gauss.rs` shared helpers or add `rs_math.rs`); update
   `gauss_solver.rs` + `gauss_tests.rs` imports. No behavior change.
   *(Do first so §2 can delete `gauss_solver.rs` wholesale.)*
2. **Atomic square-legacy cut (one commit).** Delete per §1: solver
   (`anchor.rs`, `gauss_solver.rs`, Mock/AnchorSolver, `MatrixSolver`, legacy
   5×8 methods in `gauss_solver.rs`, legacy 5×8 section in `anchor.rs`),
   core (`TritMatrix`, `MockEcc`, `InvalidCapacity`), render (`halftone.rs`,
   `font.rs`, square lib.rs code), cv (square lib.rs code, `is_l_shape`,
   `classify_corners`), scanner (square decode + `decode_png_auto`), main.rs
   (square encode + decode paths); replace `calibrate_from_anchors` with
   macro-anchor calibration; delete/replace the square test files (§1.15).
   Result: workspace = triangular only, `--triangular` encode path intact.
3. **Rename pass (mechanical).** File renames + symbol renames per §3 (sed
   across crates, then fix module paths). Pure rename — verify with
   `cargo test --workspace` and a smoke `encode`/`decode` round-trip via the
   new CLI decode from step 4.
4. **Backfill: CLI triangular decode** (§4.1). Add `--triangular`/auto decode;
   replace the old square decode tests with a tri round-trip test
   (encode → PNG → decode → bytes equal).
5. **Backfill: scanner dimension detection** (§4.3). Replace `TRI_GRID_SIZES`,
   add header-based dims, add `decode_tri_png_auto`.
6. **Backfill: tri typography** (§4.2). `TriFont` rasterizer, solver wiring,
   `font_mask` wiring, rewrite `constraint_tests.rs` as tri tests.
7. **Web + docs.** Update `web/scanner/scanner.js` (drop `decode_png_auto`),
   rebuild and commit `web/scanner/pkg/`, sync or remove the stale
   `trixel-web/` copy. `README.md` and the SPEC.md source-of-truth file list
   are updated (trixel_* crates, renamed modules). Delete
   `debug_halftone.rs` / `debug_test.py` / `scripts/composite.py` only if they
   still reference the square API after step 2 (check; the root `bneb_trixel*.png`
   and `trixel` binary are artifacts, not code).

Optional follow-up (explicitly out of scope per CLAUDE.md, which says "don't
build on them"): removing the deprecated clockwise/interleaved layouts entirely
would let `anchor.rs` mini-anchor code, `layout.rs` non-radial
sequences, and `header.rs` clockwise header code also go. The plan keeps
them because they are triangular, not square — but nothing new should be built
on them.
