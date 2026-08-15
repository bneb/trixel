# Trixel

Base-3 optical data encoding standard: triangular-grid matrix codes where each
cell ("trixel") is a dark/mid/light triangle carrying one trit, protected by
Reed-Solomon over GF(3⁶). Formal spec: `docs/SPEC.md`.

## Build / Test

```bash
cargo test --workspace        # all tests (unit tests in each crate's src/, integration in tests/)
cargo build --release         # release build
cargo run --release -- encode --data "Hello" --output hello.png
cargo run --release -- decode --input hello.png
```

Scanner WASM (Cloudflare Pages deploys the committed bundle):

```bash
wasm-pack build crates/trixel_scanner --target web --out-dir ../../web/scanner/pkg
```

## Architecture

6 crates, dependency chain: core ← solver ← render/cv ← {scanner, CLI}

| Crate | Modules | Role |
|---|---|---|
| `trixel_core` | `gf3`, `rs`, `grid` | GF(3⁶) arithmetic, Reed-Solomon codec, `Grid` data structure |
| `trixel_solver` | `gauss`, `gauss_solver`, `anchor`, `header`, `layout` | Constraint-based packing: Gaussian elimination over GF(3), Macro-Anchor patterns, MaskProfile header, radial-spiral layout |
| `trixel_render` | `render`, `diffusion` | Grid → PNG with CIELAB halftoning + Floyd-Steinberg error diffusion |
| `trixel_cv` | `cv`, `geometry` | PNG → Grid: anchor detection, perspective warp, luminance calibration |
| `trixel_scanner` | `lib` (wasm) | WASM decoder: `decode_camera_frame`, `decode_tri_png` |
| `trixel` | `main` (bin) | CLI: `encode`, `decode` subcommands |

## Conventions

- Trits ∈ {0, 1, 2}; 3 is erasure, never encoded.
- Luma thresholds: 0–89 dark, 89–165 mid, >165 light.
- Parity/RS math is Gaussian elimination over GF(3) (not GF(2)).
- RS codec works over GF(3⁶) with a 3-symbol header (len_lo, len_hi, parity_count).
- Grid is `rows × cols`, 2:1 column:row ratio (typical 60×30, 80×40).
- Cell ▲ iff `(col + row) mod 2 == 0`, ▽ otherwise.

## Before Changing Code

- **The pipeline is triangular-grid only.** The legacy square-grid code was removed.
- **Encode**: bytes → trits → RS encode → solver (parity via GF(3) elimination) → render → PNG.
- **Decode**: PNG → CV anchor detection → radial spiral unroll → RS decode → bytes.
- **Only radial layout is standard** (3 Macro-Anchors, masked radial spiral). Clockwise and interleaved are deprecated dead ends — don't build on them.
- **Scanner WASM is committed** to `web/scanner/pkg/` for Cloudflare Pages GitOps. Source changes need a rebuild + commit of the bundle.
- Tests live in each crate's `tests/` directory.

## Current State

Square-grid legacy removed. Phase 4 (layout protocol) committed. Working tree:
post-refactoring cleanup — renamed modules (dropped `Tri` prefix), unified
triangular-only pipeline, CLI decode added.
