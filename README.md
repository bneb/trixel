# Ternac

A base-3 optical data encoding standard with Reed-Solomon error correction, L-bracket anchor detection, and dynamic luminance calibration.

## Architecture

```
ternac/
├── crates/
│   ├── ternac_core/    # GF(3⁶) arithmetic, RS codec, ECC traits
│   ├── ternac_solver/  # Constraint-based matrix packing with anchor anti-pattern enforcement
│   ├── ternac_render/  # TritMatrix → PNG with L-bracket anchors
│   ├── ternac_cv/      # PNG → TritMatrix with anchor-calibrated luminance
│   └── ternac/         # CLI: encode & decode subcommands
```

## Quick Start

```bash
# Encode
cargo run --release -- encode --data "Hello" --output hello.png --color 808080 --ecc 0.3

# Decode
cargo run --release -- decode --input hello.png
```

## Tests

```bash
cargo test --workspace
```

## License

MIT
