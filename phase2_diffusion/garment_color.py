"""Foreground-only color hints for diffusion; never recolors output pixels."""
import colorsys
import numpy as np
from PIL import Image


def describe_color(rgba):
    """Estimate the dominant color of an already segmented product image.

    Ignore transparent edges and use a median to resist small logos/highlights.
    This is a heuristic for predominantly single-color product photographs.
    """
    pixels = np.asarray(rgba.convert("RGBA"))
    rgb = pixels[..., :3][pixels[..., 3] >= 230]
    if len(rgb) < 32:
        raise ValueError("Too few foreground pixels to estimate garment color; use --<kind>_color.")
    r, g, b = np.median(rgb, axis=0) / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.27:
        return "black"
    if s < 0.12:
        return "white" if v > 0.85 else "dark gray" if v < 0.48 else "light gray" if v > 0.7 else "gray"
    hue = h * 360
    name = ("red" if hue < 15 or hue >= 345 else "orange" if hue < 45
            else "yellow" if hue < 70 else "green" if hue < 170
            else "cyan" if hue < 195 else "blue" if hue < 260
            else "purple" if hue < 290 else "pink")
    if name == "orange" and v < 0.65:
        return "brown"
    return ("dark " if v < 0.45 else "light " if v > 0.65 and s < 0.4 else "") + name


def color_prompt(kind, color):
    item = {"top": "shirt", "bottom": "pants", "hat": "hat", "shoes": "shoes"}[kind]
    prompt = (f"A mannequin wearing {color} {item}. {color} fabric, "
              "matching the garment reference color and design, realistic fabric folds, product photography.")
    negative = "wrong garment color, discolored fabric, washed out fabric"
    if color.strip().lower() == "black":
        prompt += " Deep black fabric with subtle natural highlights."
        negative += f", gray {item}, light gray {item}, white {item}"
    return prompt, negative
