"""
segmentation.py
----------------
Handles background removal for product images (top/bottom garments) and
gives us a clean RGBA garment cutout with alpha mask.

Uses `rembg` (U2-Net based, pretrained, runs locally, no API calls).
"""

from __future__ import annotations
import numpy as np
from PIL import Image
from rembg import remove, new_session

# u2netp = smaller/faster variant, good enough for product photos on low VRAM.
# Switch to "u2net" for higher quality if you have more VRAM/time.
_SESSION = new_session("u2netp")


def remove_background(image: Image.Image) -> Image.Image:
    """
    Remove background from a garment product photo.
    Returns an RGBA PIL Image with transparent background.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    result = remove(image, session=_SESSION)
    return result


def get_garment_mask(rgba_image: Image.Image) -> np.ndarray:
    """
    Extract a clean binary mask (0/255) from an RGBA garment cutout.
    """
    alpha = np.array(rgba_image)[:, :, 3]
    mask = (alpha > 10).astype(np.uint8) * 255
    return mask


def get_garment_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """
    Returns (x_min, y_min, x_max, y_max) tight bounding box of the garment
    within its own image, based on the alpha mask.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Empty garment mask - background removal likely failed")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def crop_to_content(rgba_image: Image.Image, mask: np.ndarray, padding: int = 5) -> tuple[Image.Image, np.ndarray]:
    """
    Crop the garment image + mask tightly around the non-transparent content.
    """
    x0, y0, x1, y1 = get_garment_bbox(mask)
    w, h = rgba_image.size
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(w, x1 + padding)
    y1 = min(h, y1 + padding)
    cropped_img = rgba_image.crop((x0, y0, x1, y1))
    cropped_mask = mask[y0:y1, x0:x1]
    return cropped_img, cropped_mask
