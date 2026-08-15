# Trixel Triangular Grid Format — Formal Specification

Version 1.0 · Status: In-formalization (matches implementation as of commit `5c2bdbc`)

Trixel is a 2D matrix-code format in which each module ("trixel") is a triangle
of one of three states — dark, mid, or light — and in which payload bytes are
carried as **trits** (base-3 digits), protected by a Reed–Solomon code over the
field GF(3⁶).

Source of truth files:

- `crates/trixel_core/src/grid.rs` — grid data structure
- `crates/trixel_solver/src/anchor.rs` — anchor patterns
- `crates/trixel_solver/src/header.rs` — allocation header encoding
- `crates/trixel_solver/src/layout.rs` — read-order iterators
- `crates/trixel_solver/src/gauss_solver.rs` — parity solver
- `crates/trixel_core/src/gf3.rs`, `crates/trixel_core/src/rs.rs` — RS math
- `crates/trixel_render/src/render.rs`, `diffusion.rs` — rendering
- `crates/trixel_cv/src/cv.rs`, `geometry.rs` — scanning

---

## 1. Grid Geometry

### 1.1 Triangle Tessellation

A Trixel grid is a `rows × cols` array of triangles. Adjacent rows are
staggered so that each triangle shares full edges with its neighbors:

```
Row 0:  ▲₀ ▽₁ ▲₂ ▽₃ ▲₄ ▽₅
Row 1:  ▽₀ ▲₁ ▽₂ ▲₃ ▽₄ ▲₅
Row 2:  ▲₀ ▽₁ ▲₂ ▽₃ ▲₄ ▽₅
```

A triangle at `(col, row)` is:

- **Up-pointing (▲)** iff `(col + row) mod 2 == 0`
- **Down-pointing (▽)** iff `(col + row) mod 2 == 1`

`cols` should be even for clean tessellation. Cells are stored row-major:
`data[row * cols + col]`.

### 1.2 Coordinate System

- Origin `(0, 0)` is the top-left triangle.
- `col` increases rightward, `row` increases downward.
- Each cell holds exactly one **trit** ∈ {0, 1, 2}. Value 3 is used only by the
  reader to mark an erasure; it is never encoded.

### 1.3 Pixel Geometry and Column:Row Ratio

Rendering (see §7) uses a per-triangle bounding box `cell_w × cell_h` with
`cell_w = cell_h` (each triangle is as wide as it is tall — a flat-bottomed
isoceles shape, not fully equilateral):

- Column pitch: `cell_w / 2` px (two triangle columns per cell width)
- Row pitch: `cell_h` px
- Image size: `width  = (cols · cell_w) / 2 + cell_w / 2`
  `height = rows · cell_h`
- A ▲/▽ pair occupies a square `cell_w × cell_h` region → the grid's
  **column:row ratio is 2:1** at the triangle level (2 columns per unit width,
  1 row per unit height). Typical grids are 2:1 or wider, e.g. 60×30, 80×40,
  120×60.

## 2. Trit States and Color Semantics

| State | Name   | Scanner Rec.601 luma (Y) | Render (anchor cells) |
|-------|--------|--------------------------|------------------------|
| 0     | dark   | Y ≤ 89                   | `(0, 0, 0)` black      |
| 1     | mid    | 89 < Y ≤ 165             | `(128,128,128)` gray   |
| 2     | light  | Y > 165                  | `(255,255,255)` white  |

Thresholds: `LUMA_THRESHOLD_01 = 89.0`, `LUMA_THRESHOLD_12 = 165.0`.

## 3. Anchor System

Three anchor families exist. The **radial** architecture (primary) uses three
Macro-Anchors; the **clockwise** architecture uses four Mini-Anchors; the
legacy architecture uses four 5×8 Tri-Anchors.

### 3.1 Tri-Anchor Patterns (5×8, corner-ID + CRC)

Legacy anchors occupy `5 × 8 = 40` triangles at the four corners
(rows ≥ 10, cols ≥ 16):

```
TL (0,0)              TR (cols−8, 0)
BL (0, rows−5)        BR (cols−8, rows−5)
```

Pattern structure (T = 0 black border, W = 2 white, C = corner-ID trit,
K = CRC trit, S = sync dot):

```
Row 0: 0 0 0 0 0 0 0 0    (full black border)
Row 1: 0 W W C C K K 0    (quiet zone + corner ID + CRC)
Row 2: 0 W W W S W W 0    (quiet zone + sync dot)
Row 3: 0 W W K C C W 0    (CRC + corner ID, mirrored)
Row 4: 0 0 0 0 0 0 0 0    (full black border)
```

**Corner ID**: 2 trits, base-3 encoding of the corner index:

| Corner | ID trits |
|--------|----------|
| TL     | (0, 0)   |
| TR     | (0, 1)   |
| BL     | (0, 2)   |
| BR     | (1, 0)   |

**CRC-3**: `CRC = (id[0] + id[1]) mod 3`, embedded (replicated) in the
pattern so a detected 5×8 block can be self-verified against each corner ID
before being trusted as that corner.

Exact patterns (`TRI_ANCHOR_PATTERNS`):

```
TL:                      TR:
0 0 0 0 0 0 0 0          0 0 0 0 0 0 0 0
0 2 2 0 0 0 0 0          0 0 0 0 1 1 2 0
0 2 2 2 1 2 2 0          0 2 2 1 2 2 2 0
0 2 2 0 0 0 2 0          0 2 1 1 0 2 2 0
0 0 0 0 0 0 0 0          0 0 0 0 0 0 0 0

BL:                      BR:
0 0 0 0 0 0 0 0          0 0 0 0 0 0 0 0
0 2 2 0 2 2 2 0          0 2 1 1 1 0 2 0
0 2 2 2 1 2 2 0          0 2 2 1 2 2 2 0
0 2 2 2 0 2 2 0          0 2 2 1 0 2 2 0
0 0 0 0 0 0 0 0          0 0 0 0 0 0 0 0
```

Each anchor uses all three states (0 border for contour detection, 2 for
quiet-zone contrast, 1 sync dot as a unique orientation marker per corner).

### 3.2 Macro-Anchors (7×12, TL/TR/BL)

The radial architecture uses **three** Macro-Anchors — TL, TR, BL — with **no
BR anchor** (the missing anchor disambiguates orientation). Each is a
`7 × 12 = 84`-trixel block, inset 1 trixel from the grid edge:

```
TL at (1, 1)
TR at (cols − 13, 1)
BL at (1, rows − 8)
```

Internal layout (S = shell, W = quiet zone, H = header core, D = sync dot):

```
Row 0: S S S S S S S S S S S S   (shell ring: all State 0)
Row 1: S W W H H W W W W W W S
Row 2: S W W W W W W W W W W S
Row 3: S W W W W D W W W W W S   (sync dot at relative (3,5), State 1)
Row 4: S W W W W W W W W W W S
Row 5: S W W H H H H W W W W S
Row 6: S S S S S S S S S S S S   (shell ring: all State 0)
```

- **Shell**: outermost ring (all 34 perimeter cells) fixed to State 0 — the
  solid contour used for detection.
- **Core**: exactly 6 dynamic trits per anchor, at relative offsets
  `(1,3), (1,4), (5,3), (5,4), (5,5), (5,6)` (offset `(row, col)`), carrying
  the allocation header (3 anchors × 6 = 18 header trits, §5).
- **Fixed interior**: all remaining non-shell cells State 2, except the single
  sync dot at relative `(3,5)` which is State 1.

### 3.3 Mini-Anchors (3×5, clockwise architecture)

The clockwise architecture uses four minimalist `3 × 5` anchors at
`(1,1)`, `(cols−6,1)`, `(1,rows−4)`, `(cols−6,rows−4)`:

```
TL (solid black = orientation key):   TR (notch col 3):   BL (notch col 1):   BR (notch col 2):
0 0 0 0 0                             0 0 0 0 0           0 0 0 0 0           0 0 0 0 0
0 0 0 0 0                             0 2 2 1 0           0 1 2 2 0           0 2 1 2 0
0 0 0 0 0                             0 0 0 0 0           0 0 0 0 0           0 0 0 0 0
```

TL is a solid State-0 block (the unique orientation key); TR/BL/BR have a
State-0 border, State-2 interior, and a single State-1 notch in a unique
column. Matching the TL pattern against all four corners yields the absolute
rotation (0, 90°, 180°, 270°).

## 4. Read Order (Layout)

### 4.1 Masked Radial Spiral (primary)

`masked_radial_sequence(rows, cols)` enumerates every `(col, row)` that is
**not** in the Global Quiet Zone (§9) and **not** inside a Macro-Anchor region
(§3.2), sorted by:

1. `dist² = (col + 0.5 − cols/2)² + (row + 0.5 − rows/2)²` ascending
   (cell centers measured from grid center), then
2. `angle = atan2(row + 0.5 − rows/2, col + 0.5 − cols/2)` ascending as the
   tie-breaker, producing a smooth center-outward spiral.

The sequence is deterministic and contains every eligible cell exactly once.

**Flat unroll format** (positions along the spiral):

```
offset 0 .. 11                : length prefix, 12 trits (§6.2)
offset 12 .. 12+P·6 − 1       : RS parity block, P symbols (§6.4)
offset 12+P·6 .. M−1          : message block, M symbols
      [0]  = len_lo  = payload_trit_count mod 729
      [1]  = len_hi  = payload_trit_count / 729
      [2]  = parity_count P
      [3+] = payload data symbols (zero-padded to whole symbols)
      (the message may be preceded by an offset of free padding symbols,
       chosen to avoid fixed-image constraints)
remaining                      : free padding symbols (solved, any value)
cells beyond the codeword      : hardcoded State 2 (light)
```

The parity block is the *first* P symbols of the RS codeword; the message
block is placed at `12 + P·6 + offset·6` trits into the spiral.

### 4.2 Clockwise Perimeter (alternate)

`clockwise_perimeter_sequence(rows, cols)` walks concentric rectangular rings
from the outer perimeter inward (top edge left→right, then right edge
top→bottom, then bottom edge right→left, then left edge bottom→top), skipping
the quiet zone and all Mini-Anchor regions. The first 36 cells of this
sequence carry the inline header (§5.4); payload follows.

### 4.3 Interleaved (PRNG shuffle)

`interleaved_sequence(rows, cols)` collects all eligible cells (outside quiet
zone and Mini-Anchors) and Fisher–Yates shuffles them with a
`ChaCha8Rng` seeded by `seed = rows · 65537 + cols`. Same dimensions always
produce the identical permutation.

### 4.4 Parity Zone Placement

The active `MaskProfile` (§5.2) defines which cells are **parity-zone** cells.
In image-guided encoding, parity-zone cells are left as free variables (they
are solved to satisfy the RS parity equations) while non-parity-zone cells are
fixed to source-image-derived trit targets. Cells inside Macro-Anchor regions
are never parity-zone cells.

| Profile     | Membership test                                             |
|-------------|-------------------------------------------------------------|
| `Border2px` | `row < 2` or `row ≥ rows−2` or `col < 2` or `col ≥ cols−2`  |
| `Border3px` | `row < 3` or `row ≥ rows−3` or `col < 3` or `col ≥ cols−3`  |
| `Radial(r)` | `sqrt((col+0.5−cols/2)² + (row+0.5−rows/2)²) > r`           |

## 5. Allocation Header (MaskProfile)

### 5.1 MaskProfile Encoding

The header encodes one `MaskProfile` as a u16:

| Profile     | Value        |
|-------------|--------------|
| `Border2px` | 0            |
| `Border3px` | 1            |
| `Radial(r)` | `2 + r`, r ∈ 0..726 |

Maximum serialized value: `MAX_MASK_PROFILE = 728` (= 3⁶ − 1).

### 5.2 TMR Encoding (18 trits in Macro-Anchor cores)

The u16 value is expanded to 6 base-3 trits, least-significant trit first:
`data[i] = (value / 3^i) mod 3`. The full 18-trit header is built as
**Triple-Modular Redundancy**:

```
header[0..6]   = data    (6 trits)
header[6..12]  = replica (same 6 trits)
header[12..18] = parity  (parity[i] = (2 · data[i]) mod 3)
```

Because `data[i] + parity[i] ≡ 0 (mod 3)`, the decoder corrects any single
error per trit position by majority logic:

- `a == b` → take `a`
- else if `(a + p) mod 3 == 0` → take `a`
- else if `(b + p) mod 3 == 0` → take `b`
- else → header is corrupted

The 18 trits are routed to the Macro-Anchor cores in order: anchor 0 (TL)
trits 0–5, anchor 1 (TR) trits 6–11, anchor 2 (BL) trits 12–17, each in core
coordinate order `(1,3),(1,4),(5,3),(5,4),(5,5),(5,6)` relative to the anchor
origin. A reader reconstructs the header by reading the same 18 cells,
decoding with the TMR vote, and mapping the u16 back to a `MaskProfile`.

### 5.3 Parity Zone Determination

Given a decoded profile, `is_in_parity_zone(row, col, profile, rows, cols)`
returns true per §4.4 (cells inside Macro-Anchor regions return false first).

### 5.4 Clockwise Inline Header (36 trits)

The clockwise architecture replaces the Macro-Anchor header with an inline
header occupying the first 36 trits of the perimeter sequence — six GF(3⁶)
symbols:

```
symbols 0–1: MaskProfile   (6 data trits + 6 replica trits; TMR vote)
symbols 2–3: grid rows     (6 data trits + 6 replica trits; TMR vote)
symbols 4–5: grid cols     (6 data trits + 6 replica trits; TMR vote)
```

Each value is 6 base-3 trits LSB-first (range 0–728). On replica
disagreement, replica 0 wins (documented trade-off: no correction for replica-0
corruption).

**Telomere stop-codon**: the 6-trit symbol `TELOMERE_SYMBOL = [2,2,2,0,0,0]`
marks end-of-payload. A decoder reading the clockwise track halts at this
symbol and ignores all subsequent parity data.

## 6. Data Encoding and Error Correction

### 6.1 Byte → Trit Conversion

Each byte is expanded into exactly 6 trits, base-3, least-significant trit
first:

```
trits[i] = (byte / 3^i) mod 3,  i = 0..5
```

6 trits hold 729 values ≥ 256, so every byte has a unique 6-trit image and
unused codes are detectable. Decoding a 6-trit chunk yields a value that must
be ≤ 255, else the chunk is invalid (decoders reject it; 6-trit chunks with
any trit > 2 are also invalid).

### 6.2 Length Prefix

A fixed 12-trit base-3 prefix (LSB first) precedes the RS codeword in the
unroll. It encodes the **codeword length in trits** (not payload length).
12 trits cover 0..531,440. `encode_length` / `decode_length` are the
canonical codec.

### 6.3 GF(3⁶) Field

- Field size: 3⁶ = **729** elements (0–728); multiplicative group order
  **728**.
- An element is a polynomial over GF(3) of degree ≤ 5, stored as a `u16` in
  base-3: `value = c₀ + c₁·3 + c₂·9 + c₃·27 + c₄·81 + c₅·243`.
- Primitive polynomial: **p(x) = x⁶ + x⁵ + 2** (verified primitive by
  exhaustive search; α has multiplicative order exactly 728).
- Addition/subtraction are trit-wise mod 3; multiplication/division are O(1)
  via precomputed exp/log tables.
- **Symbol ↔ trits**: `symbol_to_trits` / `trits_to_symbol` use the base-3
  LSB-first encoding of §6.1; 6 trits = 1 GF(3⁶) symbol.

### 6.4 Reed–Solomon Codeword

Systematic RS code over GF(3⁶), parameterized by `parity_count`:

- Generator: `g(x) = ∏_{i=1}^{parity_count} (x − α^i)`
- Codeword layout: `c[0..parity_count]` = parity symbols, `c[parity_count..n]`
  = message symbols.
- Encoding (library path): polynomial long division.
- Decoding: Sugiyama's Extended Euclidean Algorithm for the key equation,
  then Chien search + Forney correction.
- **Correction capacity**: `2·e + E ≤ parity_count`, where `e` = symbol
  errors, `E` = symbol erasures (erasure positions are passed by the reader).

**Message structure** (3-symbol RS header + data):

```
message = [ len_lo,        = payload_trit_count mod 729
            len_hi,        = payload_trit_count / 729
            parity_count,  = P (even, ≥ 2, ≤ 728)
            data symbols..., zero-padded to whole 6-trit symbols ]
```

`parity_count` is chosen per grid: `num_symbols ≤ 728` (GF(3⁶) codeword cap),
and P is derived from the capacity headroom (`num_symbols − message_symbols −
constraint slack − 12`), rounded down to even, minimum 2. The legacy solver
used a fixed 30% parity fraction instead. Max payload: 728 symbols ≈ 728
bytes.

### 6.5 Self-Describing Decode

The reader does not know P in advance (it is a message symbol), so it scans:

1. Read the 12-trit length prefix → slice the codeword trits.
2. Convert to symbols, recording erasure positions (any 6-trit chunk
   containing trit 3 becomes an erasure).
3. For `pc` in `{2, 4, …, n−3}`: attempt RS decode with `pc` parity symbols.
4. Accept iff `data[offset+2] == pc` (parity-count self-consistency),
   `original_len > 0`, `original_len mod 6 == 0` (whole symbols),
   `original_len ≤ capacity`, and every 6-trit chunk of the payload decodes
   to a byte value ≤ 255 (filters halftone false positives).

### 6.6 Grid-Level Parity Solving

The radial/row-major encoders solve the parity equations **in the grid** via
Gaussian elimination over GF(3) at the trit level:

- Build the parity-check matrix H (P equations over GF(3⁶) expanded to
  `P · 6` trit-level equations per codeword symbol).
- Message trits and constraint-truncated cells are fixed; the remaining
  variables are free.
- Free variables are solved with default 2 (light) to avoid dark voids in
  halftone rendering; the full codeword trits are then placed into the grid
  in layout order (§4.1).
- Cells beyond the codeword are filled with State 2.

## 7. Rendering

### 7.1 Canvas

Triangles are rasterized per §1.3. `cell_w = cell_h`; a pixel belongs to the
triangle at normalized cell coordinates `(fx, fy)` when:

```
▲: |fx − 0.5| ≤ 0.5·fy        ▽: |fx − 0.5| ≤ 0.5·(1 − fy)
```

Background defaults to white. Halftone input is alpha-flattened over white
**before** resizing (prevents transparent black pixels) and resized to
`cols × rows` (1 px per cell, Lanczos3).

### 7.2 CIELAB Perceptual Encoding

Data cells preserve source chromaticity while forcing scanner-readable luma:

1. Convert source sRGB → CIELAB; compute the source's natural Rec.601 luma
   `Y = 0.299·R + 0.587·G + 0.114·B`.
2. If Y already lies in the target state's band (§2), return the source
   color **unchanged**.
3. Otherwise target the **nearest band boundary** plus a 3-luma-point safety
   margin: `y_target = y_min + 3` (too dark) or `y_max − 3` (too light).
4. Add the Floyd–Steinberg correction: `y_target += correction · 128`,
   clamped to `[y_min + 3, y_max − 3]`.
5. **Chroma attenuation** for large shifts: soft threshold at CIELAB chroma
   `C = 30`; `excess = clamp((C − 30)/30, 0, 1)` (0 at C ≤ 30, 1.0 at C = 60);
   `chroma_scale = 1 − excess² · (|Y_shift|/255) · 15`, clamped to
   `[0.15, 1.0]`. Moderate-chroma colors (skin tones ~30–35) are not
   attenuated.
6. Bisect `L*` (20 iterations) to hit `y_target` with the attenuated
   `(a*, b*)`; if the final RGB misses the band, fall back to grayscale at
   `y_target`.

Anchor cells are rendered with **anchor immunity**: strict `(0,0,0)`,
`(128,128,128)`, `(255,255,255)` for states 0/1/2, independent of the source
image. Font-mask state-2 cells also get the CIELAB shift.

### 7.3 Floyd–Steinberg Error Diffusion on Triangle Adjacency

Adapted to the natural 3-neighbor adjacency of triangles, weights
`3/8, 3/8, 2/8`:

```
▲ at (col,row) →  (col+1, row)  w=3/8   (right ▽)
                  (col, row+1)  w=3/8   (below)
                  (col+1, row+1) w=2/8  (diagonal)
▽ at (col,row) →  (col+1, row)  w=3/8   (right ▲)
                  (col, row+1)  w=3/8   (below)
                  (col−1, row+1) w=2/8  (diagonal)
```

Trit luminance midpoints (aligned to scanner bands):

| State | Midpoint (0–1) |
|-------|----------------|
| 0     | 0.175          |
| 1     | 0.500          |
| 2     | 0.825          |

Per cell in scanline order: `error = src_lum − actual + accumulated`;
correction stored as `clamp(error, −0.5, +0.5)` and the remainder is
distributed to forward neighbors. Corrections feed the renderer as the
`correction · 128` luma term of §7.2.

## 8. Scanning (Extraction)

### 8.1 Digital Path (rendered images, known cell size)

1. Grayscale the image.
2. **Luminance calibration** from known anchor cells: sample the centroid of
   every cell of all four 5×8 Tri-Anchors (or Mini-Anchors), collect the
   luminance of state-0, state-1, state-2 cells, take the **median** of each
   class, and build calibrated `LuminanceBands` at the midpoints (§8.4).
3. For each cell, sample the triangle centroid and quantize:

```
px_x = col · cell_w/2 + cell_w/2
py   = row · cell_h
cy   = py + 2·cell_h/3   (▲)      cy = py + cell_h/3   (▽)
```

### 8.2 Camera Path (perspective)

1. Grayscale → binary mask at a **high threshold (≥ 200)** — this isolates
   the white quiet-zone border plus other state-2 cells and ignores internal
   dark/mid data, avoiding Otsu holes. The bright region must cover ≥ 2% of
   the frame.
2. Extract the 4 extreme corners of the bright pixel cloud via
   `min/max(x+y)` → TL/BR and `min/max(x−y)` → BL/TR; order as TL, TR, BL, BR
   by centroid quadrant.
3. Compute the perspective homography (DLT, 8×8 system, h₂₂ = 1) from ideal
   corners `(0,0), (ideal_w,0), (0,ideal_h), (ideal_w,ideal_h)` to the
   detected corners; invert it; warp the image by inverse mapping with
   nearest-neighbor sampling at an ideal `cell_h = 10` px.
4. Hand off to the digital path (§8.1).

Legacy alternative: Otsu threshold → contour detection →
Douglas–Peucker simplification (2% of perimeter as epsilon) → filter to
4–30 vertices, solidity ≥ 0.7, bounding aspect ratio 0.4–4.0; dedupe
centroids within 20 px; take the top-3 by solidity as Macro-Anchors.

### 8.3 Anchor Identification (rotation-invariant)

Given 3 candidate centroids, sort to `[TL, TR, BL]` using pure geometry:

1. TL is the vertex opposite the longest pairwise edge (the hypotenuse TR–BL
   — the right-angle corner of the L).
2. Of the two remaining vertices, TR is the one that is clockwise from the
   other when viewed from TL (positive cross product in Y-down image
   coordinates); BL is the remainder.

This works for any camera rotation because it uses only the L-shape
structure. With > 3 candidates, pick the triple with maximum enclosing area.

Header and payload extraction map each layout-order cell centroid (§8.1
formulas, or grid coordinates `(col+0.5, row+0.333/0.667)`) through the
inverse affine/homography into pixel space; out-of-bounds samples become
erasure trit 3.

### 8.4 LuminanceBands

Defaults (percentages of 0–255) with **guard bands** that quantize to
erasure 3:

| Band            | Range    |
|-----------------|----------|
| State 0         | 0–51     |
| Guard           | 52–101   |
| State 1         | 102–152  |
| Guard           | 153–203  |
| State 2         | 204–255  |

`calibrate(s0, s1, s2)` sets thresholds at the midpoints between the measured
anchor medians: `state_0_upper = (s0+s1)/2`, `state_1_upper = (s1+s2)/2`,
eliminating guard bands (tight thresholds).

## 9. Quiet Zone and Borders

- **Global Quiet Zone (GQZ)**: the outermost perimeter — `row 0`,
  `row rows−1`, `col 0`, `col cols−1` — is hardcoded to State 2 (white) on
  encode. It provides the bright boundary that camera detection keys on and
  guarantees contrast around every anchor. It is excluded from all read
  orders (§4).
- **Anchor inset**: Macro-, Mini- and legacy Tri-Anchors are placed at least
  1 trixel inside the grid edge so they never touch the GQZ.
- **Border/parity zones**: `Border2px`/`Border3px` dedicate the outermost
  2 or 3 rows/cols to parity (free) cells; `Radial(r)` dedicates everything
  outside safe radius r (§4.4).
- **Minimum grid sizes**:
  - Radial architecture: `rows ≥ MACRO_ANCHOR_ROWS·2 + 2 = 16`,
    `cols ≥ MACRO_ANCHOR_COLS·2 + 2 = 26`.
  - Legacy: `rows ≥ 10`, `cols ≥ 16` (two 5×8 anchors per dimension).
  - Clockwise: `rows ≥ 10`, `cols ≥ 16` (two 3×5 anchors per dimension).
- **Codeword bounds**: a grid's usable cells minus the 12-trit prefix must
  fit `num_symbols ≤ 728` RS symbols (GF(3⁶) limit) and leave at least
  `3 + 1` message symbols (RS header + one data symbol) for the code to be
  non-degenerate.

## Appendix A: Constant Reference

| Constant               | Value |
|------------------------|-------|
| TRITS_PER_SYMBOL       | 6     |
| LENGTH_PREFIX_TRITS    | 12    |
| RS_HEADER_SYMBOLS      | 3     |
| FIELD_SIZE (3⁶)        | 729   |
| FIELD_ORDER (3⁶ − 1)   | 728   |
| MAX_MASK_PROFILE       | 728   |
| HEADER_TRITS (TMR)     | 18    |
| TRITS_PER_ANCHOR       | 6     |
| TRI_ANCHOR_ROWS/COLS   | 5 / 8 |
| MACRO_ANCHOR_ROWS/COLS | 7 / 12|
| MACRO_CORE_TRITS       | 6     |
| MINI_ANCHOR_ROWS/COLS  | 3 / 5 |
| CLOCKWISE_HEADER_TRITS | 36    |
| LUMA_THRESHOLD_01/12   | 89 / 165 |
| TELOMERE_SYMBOL        | [2,2,2,0,0,0] |
