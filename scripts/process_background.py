#!/usr/bin/env python3
"""
Process the airplane image into a light, frosted, blurred background suitable for UI

Input: 1427611-1920x1080-desktop-1080p-malaysia-airlines-background-image.jpg (project root)
Output: prod/assets/processed_bg.jpg

This script uses Pillow to:
- apply a smooth Gaussian blur
- increase brightness
- lower contrast
- apply a semi-transparent white overlay (frosted glass effect)

Run: python scripts/process_background.py
"""

from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "1427611-1920x1080-desktop-1080p-malaysia-airlines-background-image.jpg"
OUT_DIR = ROOT / "prod" / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "processed_bg.jpg"

if not SRC.exists():
    print("Source image not found at", SRC)
    raise SystemExit(1)

print("Processing", SRC)
img = Image.open(SRC).convert("RGB")

# Ensure a reasonable working size (keep original if already large)
max_w = 1920
if img.width > max_w:
    new_h = int(img.height * (max_w / img.width))
    img = img.resize((max_w, new_h), Image.LANCZOS)

# Smooth blur to make image unobtrusive but still recognizable
img = img.filter(ImageFilter.GaussianBlur(radius=14))

# Brighten and lower contrast for airy feel
img = ImageEnhance.Brightness(img).enhance(1.18)
img = ImageEnhance.Contrast(img).enhance(0.72)

# Apply semi-transparent white overlay (frosted glass) to lift the image
overlay_strength = 0.36
overlay = Image.new("RGB", img.size, (255, 255, 255))
img = Image.blend(img, overlay, overlay_strength)

# Gentle additional blur to smooth artifacts
img = img.filter(ImageFilter.GaussianBlur(radius=2))

# Save output
img.save(OUT, quality=92)
print("Saved processed background to", OUT)
