# Trixel

A base-3 optical data encoding standard using triangular grids, Reed-Solomon
error correction over GF(3⁶), Macro-Anchor detection, and CIELAB perceptual
halftone rendering.

Reads like a QR code, looks like a picture.

## Architecture

```
trixel/
├── crates/
│   ├── trixel_core/    # GF(3⁶) arithmetic, RS codec, ECC traits, Grid data structure
│   ├── trixel_solver/  # Gaussian elimination over GF(3) with image-guided free variables
│   ├── trixel_render/  # Grid → PNG with CIELAB halftoning + Floyd-Steinberg diffusion
│   ├── trixel_cv/      # PNG → Grid with anchor-calibrated luminance + perspective warp
│   ├── trixel_scanner/ # WASM camera scanner (browser + mobile)
│   └── trixel/         # CLI: encode & decode subcommands
├── web/                # Cloudflare Pages site (landing + scanner PWA)
└── docs/               # Formal specification + refactoring plan
```

## Quick Start

```bash
# Encode text to a triangular PNG
cargo run --release -- encode --data "Hello" --output hello.png

# Encode with a guide image (looks like the image, carries the data)
cargo run --release -- encode --data "https://example.com" --output qr.png --image photo.jpg

# Decode
cargo run --release -- decode --input hello.png
```

## Tests

```bash
cargo test --workspace
```

## Docs

- Formal spec: [docs/SPEC.md](docs/SPEC.md)
- Architecture decisions: [CLAUDE.md](CLAUDE.md)

## License

MIT
