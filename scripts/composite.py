#!/usr/bin/env python3
"""
Composite a Ternac matrix PNG onto the laptop screen in the "Miserable Work" illustration.

Usage:
    python3 scripts/composite.py \
        --illustration /path/to/illustration.png \
        --matrix /tmp/miserable_matrix.png \
        --output /tmp/miserable_composite.png

The Ternac matrix is scaled and placed onto the laptop lid in the illustration.
State 1 (teal #75B3B8) blends with the laptop screen color, making the barcode
look like a deliberate design element rather than a sticker.
"""

import argparse
from PIL import Image


def composite(illustration_path: str, matrix_path: str, output_path: str):
    """Overlay the Ternac matrix onto the illustration's laptop screen."""
    bg = Image.open(illustration_path).convert("RGBA")
    matrix = Image.open(matrix_path).convert("RGBA")

    bg_w, bg_h = bg.size

    # The laptop screen region in the illustration (approximate bounding box).
    # These coordinates were eyeballed from the ~400x400 illustration:
    #   Top-left of the laptop screen area, and the target width/height.
    screen_x = int(bg_w * 0.46)
    screen_y = int(bg_h * 0.08)
    screen_w = int(bg_w * 0.35)
    screen_h = int(bg_w * 0.30)

    # Scale the matrix to fit the laptop screen
    matrix_resized = matrix.resize((screen_w, screen_h), Image.NEAREST)

    # Create a semi-transparent version so the illustration shows through slightly
    # at the edges where the matrix has white (State 2) modules
    bg.paste(matrix_resized, (screen_x, screen_y), matrix_resized)

    # Save as RGB (no alpha needed for final output)
    bg.convert("RGB").save(output_path)
    print(f"Composite saved to {output_path}")
    print(f"  Illustration: {illustration_path} ({bg_w}x{bg_h})")
    print(f"  Matrix: {matrix_path} ({matrix.size[0]}x{matrix.size[1]})")
    print(f"  Placed at: ({screen_x}, {screen_y}), scaled to {screen_w}x{screen_h}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Composite Ternac matrix onto illustration")
    parser.add_argument("--illustration", required=True, help="Path to the background illustration")
    parser.add_argument("--matrix", required=True, help="Path to the Ternac matrix PNG")
    parser.add_argument("--output", required=True, help="Output composite image path")
    args = parser.parse_args()

    composite(args.illustration, args.matrix, args.output)
