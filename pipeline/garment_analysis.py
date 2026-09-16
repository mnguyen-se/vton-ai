"""
garment_analysis.py
--------------------
Lightweight heuristic to guess whether a top garment has short or long
sleeves, purely from its product photo. Used by phase2_diffusion/generate.py
to decide how far outside the torso the inpainting mask should extend:
long sleeves add width down the sides of the body all the way toward the
wrist, short sleeves barely extend past the shoulder cap.

CAVEAT - this is a simple width/height heuristic, NOT a trained classifier:
  - Works reasonably for garment photos shot flat/laid-out or on a hanger/
    ghost-mannequin with sleeves roughly extended (the common e-commerce
    product-photo style).
  - Can misclassify: sleeves folded/crossed over the body, extreme crops,
    dresses, tank tops photographed at an angle, etc.
  - Always available as an escape hatch: pass --sleeve_length short|long
    to generate.py to override the auto-guess when you know it's wrong,
    instead of relying on the heuristic.
"""
from __future__ import annotations
from PIL import Image
from . import segmentation


def classify_sleeve_length(garment_image: Image.Image, long_sleeve_aspect_threshold: float = 0.85) -> str:
    """
    Returns "short" or "long".

    Heuristic: remove the background, take the tight bounding box of the
    garment, and compare width to height. A long-sleeve top photographed
    with sleeves extended is proportionally wider (arms add width on both
    sides of the torso); a short-sleeve top is proportionally taller/
    narrower since the sleeves barely extend past the shoulders.
    """
    rgba = segmentation.remove_background(garment_image.convert("RGB"))
    mask = segmentation.get_garment_mask(rgba)
    x0, y0, x1, y1 = segmentation.get_garment_bbox(mask)
    w, h = (x1 - x0), (y1 - y0)
    if h <= 0:
        return "long"  # degenerate fallback - err on the side of more coverage
    aspect = w / h
    return "long" if aspect >= long_sleeve_aspect_threshold else "short"