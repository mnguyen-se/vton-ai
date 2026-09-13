"""
compose.py
-----------
Blends warped garment layers (top, bottom) onto the base mannequin photo.

Uses alpha compositing + light Poisson-blend-style edge softening so the
garment edges don't look pasted-on. This is the baseline compositor;
swap this out later for a learned generator (GAN/diffusion refinement
pass) once you have training data, without changing the rest of the
pipeline.
"""

from __future__ import annotations
import numpy as np
import cv2
from PIL import Image, ImageFilter


def _feather_alpha(alpha: np.ndarray, radius: int = 3) -> np.ndarray:
    """Slightly blur the alpha mask edge so the garment blends in smoothly."""
    img = Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(radius))
    return np.array(img)


def alpha_composite_layer(base_rgb: np.ndarray, layer_rgba: np.ndarray) -> np.ndarray:
    """
    Composite a single RGBA garment layer onto the base RGB mannequin image.
    """
    layer_rgb = layer_rgba[:, :, :3].astype(np.float32)
    alpha = layer_rgba[:, :, 3].astype(np.float32)
    alpha = _feather_alpha(alpha.astype(np.uint8)).astype(np.float32) / 255.0
    alpha = alpha[:, :, None]

    base = base_rgb.astype(np.float32)
    out = layer_rgb * alpha + base * (1 - alpha)
    return out.astype(np.uint8)


def compose_outfit(mannequin_rgb: Image.Image, top_layer: Image.Image | None,
                    bottom_layer: Image.Image | None) -> Image.Image:
    """
    mannequin_rgb: base mannequin photo (RGB)
    top_layer / bottom_layer: warped RGBA garment images, same canvas size
                               as mannequin_rgb (output of warp.tps_warp)
    Order matters: bottom placed first, then top (so top overlaps waistband
    naturally like a real outfit).
    """
    base = np.array(mannequin_rgb.convert("RGB"))

    if bottom_layer is not None:
        base = alpha_composite_layer(base, np.array(bottom_layer))
    if top_layer is not None:
        base = alpha_composite_layer(base, np.array(top_layer))

    return Image.fromarray(base, mode="RGB")


def apply_variation(rgba_layer: Image.Image, seed: int) -> Image.Image:
    """
    Small controlled random variation between the 3 generated outfits:
    slight brightness/contrast jitter + tiny horizontal offset so the 3
    outputs aren't pixel-identical. Replace with real sampling diversity
    (e.g. diffusion noise seed) once the generator is learned.
    """
    rng = np.random.default_rng(seed)
    arr = np.array(rgba_layer).astype(np.float32)

    brightness = rng.uniform(0.96, 1.04)
    arr[:, :, :3] = np.clip(arr[:, :, :3] * brightness, 0, 255)

    shift_x = int(rng.integers(-3, 4))
    arr = np.roll(arr, shift_x, axis=1)

    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")
