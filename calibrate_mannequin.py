"""
calibrate_mannequin.py
------------------------
One-time manual keypoint calibration tool for a mannequin base photo.
Run this once per mannequin photo you plan to reuse (e.g. once for your
male mannequin, once for your female mannequin). Click points in order
when the window/plot appears; result is saved to
models/mannequin_keypoints/<mannequin_id>.json and reused by the
pipeline every time after that.

Usage:
    python calibrate_mannequin.py --image sample_data/mannequin_female.png --id female_default

In Colab (no GUI window), use matplotlib's ginput in a notebook cell instead -
see the Colab notebook cell "Calibrate mannequins" in README.md.
"""

import argparse
import matplotlib.pyplot as plt
from PIL import Image
from pipeline.mannequin_pose import BodyKeypoints, save_manual_keypoints

POINT_ORDER = [
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
]


def calibrate(image_path: str, mannequin_id: str):
    img = Image.open(image_path)
    w, h = img.size
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Click in order: " + " -> ".join(POINT_ORDER))
    pts = plt.ginput(n=len(POINT_ORDER), timeout=0)
    plt.close(fig)

    # Save as FRACTIONS of this image's width/height (not raw pixels), so the
    # keypoints work correctly later even if generate.py is given a
    # mannequin_bare photo at a different resolution than this exact file.
    kwargs = {name: (float(x) / w, float(y) / h) for name, (x, y) in zip(POINT_ORDER, pts)}
    keypoints = BodyKeypoints(**kwargs)
    path = save_manual_keypoints(mannequin_id, keypoints)
    print(f"Saved keypoints for '{mannequin_id}' -> {path}")
    return keypoints


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--id", required=True, help="e.g. male_default, female_default, unisex_default")
    args = parser.parse_args()
    calibrate(args.image, args.id)