"""
warp.py
--------
Warps a flat garment image (RGBA) so it matches the target body region on
the mannequin, using Thin Plate Spline (TPS) transform.

This is the classic (pre-deep-learning) approach used as the geometric
matching step in CP-VTON / VITON-style pipelines. It's not as good as a
learned warping network, but it's a solid, fast, trainable-free baseline
to get an end-to-end pipeline working before you invest in a GAN/diffusion
generator.
"""

from __future__ import annotations
import cv2
import numpy as np
from PIL import Image


def _place_on_canvas(rgba_garment: Image.Image, mask: np.ndarray, target_box: tuple[int, int, int, int],
                      canvas_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize the garment to roughly fit the target box, then paste it onto a
    transparent canvas the same size as the mannequin photo, at the target
    box's position. This guarantees the working image is already
    canvas-sized before TPS fine-warps it (avoids relying on cv2's
    warpImage output-size handling, which varies across OpenCV builds).
    """
    canvas_w, canvas_h = canvas_size
    x0, y0, x1, y1 = target_box
    box_w, box_h = max(1, x1 - x0), max(1, y1 - y0)

    garment = rgba_garment.resize((box_w, box_h), Image.LANCZOS)
    garment_mask = cv2.resize(mask, (box_w, box_h), interpolation=cv2.INTER_LINEAR)

    canvas_rgba = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    canvas_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(canvas_w, x1), min(canvas_h, y1)
    gw, gh = x1c - x0c, y1c - y0c
    if gw <= 0 or gh <= 0:
        return canvas_rgba, canvas_mask

    garment_arr = np.array(garment)[: gh, : gw]
    canvas_rgba[y0c:y1c, x0c:x1c] = garment_arr
    canvas_mask[y0c:y1c, x0c:x1c] = garment_mask[: gh, : gw]

    return canvas_rgba, canvas_mask


def _grid_points(mask: np.ndarray, n_cols: int, n_rows: int) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        # empty mask fallback: grid over whole canvas edges (identity-ish, avoids crash)
        h, w = mask.shape
        x0, x1, y0, y1 = 0, w - 1, 0, h - 1
    else:
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    xs_grid = np.linspace(x0, x1, n_cols)
    ys_grid = np.linspace(y0, y1, n_rows)
    return np.array([[x, y] for y in ys_grid for x in xs_grid], dtype=np.float32)


def _shape_target_points(src_pts: np.ndarray, n_cols: int, n_rows: int, garment_type: str) -> np.ndarray:
    """
    Nudge the canvas-space grid to better hug body proportions:
    tops taper slightly narrower at the shoulder row and flare a little
    at the waist row; bottoms taper slightly at the ankle row. This is a
    lightweight heuristic deformation on top of the already-placed grid -
    replace with a learned warping network for real garment-specific fit.
    """
    pts = src_pts.reshape(n_rows, n_cols, 2).copy()
    center_x = pts[:, :, 0].mean()

    if garment_type == "top":
        row_scale = np.linspace(0.94, 1.04, n_rows)  # shoulders narrower -> hem slightly wider
    else:
        row_scale = np.linspace(1.02, 0.96, n_rows)  # waist wider -> ankle narrower

    for r in range(n_rows):
        pts[r, :, 0] = center_x + (pts[r, :, 0] - center_x) * row_scale[r]

    return pts.reshape(-1, 2).astype(np.float32)


def tps_warp(rgba_garment: Image.Image, mask: np.ndarray, target_box: tuple[int, int, int, int],
             canvas_size: tuple[int, int], garment_type: str = "top") -> Image.Image:
    """
    Place the garment onto the mannequin-sized canvas at the target body
    region, then apply a light TPS fit-adjustment so it hugs body
    proportions instead of sitting as a plain rectangle.
    """
    n_cols, n_rows = 3, 4
    canvas_rgba, canvas_mask = _place_on_canvas(rgba_garment, mask, target_box, canvas_size)

    src_pts = _grid_points(canvas_mask, n_cols, n_rows)
    dst_pts = _shape_target_points(src_pts, n_cols, n_rows, garment_type)

    tps = cv2.createThinPlateSplineShapeTransformer()
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
    tps.estimateTransformation(dst_pts.reshape(1, -1, 2), src_pts.reshape(1, -1, 2), matches)

    warped = tps.warpImage(canvas_rgba, flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    return Image.fromarray(warped, mode="RGBA")
