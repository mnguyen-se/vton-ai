"""
orchestrator.py
-----------------
Main entry point tying every module together:

  product (top) + product (bottom)
        -> background removal (segmentation.py)
        -> gender check -> pick mannequin (gender.py)
        -> load mannequin body keypoints (mannequin_pose.py)
        -> TPS warp garments onto body regions (warp.py)
        -> composite layers onto mannequin photo (compose.py)
        -> repeat x3 with variation -> return 3 images

This is the BASELINE (non-learned) version: no training required to run
end-to-end today. Swap `compose_outfit` internals for a GAN/diffusion
generator later without touching the rest of this file.
"""

from __future__ import annotations
from dataclasses import dataclass
from PIL import Image

from . import segmentation, warp, compose, gender, mannequin_pose


@dataclass
class GenerationResult:
    images: list[Image.Image]
    gender_used: str
    mannequin_id: str


def generate_outfits(
    top_image: Image.Image | None,
    bottom_image: Image.Image | None,
    mannequin_photos: dict[str, Image.Image],   # {"male": PIL.Image, "female": PIL.Image, "unisex": PIL.Image}
    product_metadata: dict,
    n_variants: int = 3,
    clip_classifier=None,
) -> GenerationResult:

    # 1. Decide gender / which mannequin to use
    ref_image = top_image if top_image is not None else bottom_image
    selected_gender = gender.classify_gender(product_metadata, ref_image, clip_classifier)
    if selected_gender not in mannequin_photos:
        selected_gender = "unisex"
    mannequin_img = mannequin_photos[selected_gender]
    mannequin_id = f"{selected_gender}_default"

    # 2. Load body keypoints for this mannequin (manual calibration recommended)
    keypoints = mannequin_pose.load_manual_keypoints(mannequin_id)
    canvas_size = mannequin_img.size

    # 3. Segment + prep each garment
    top_layer_base = None
    bottom_layer_base = None

    if top_image is not None:
        top_rgba = segmentation.remove_background(top_image)
        top_mask = segmentation.get_garment_mask(top_rgba)
        top_rgba, top_mask = segmentation.crop_to_content(top_rgba, top_mask)
        top_layer_base = warp.tps_warp(
            top_rgba, top_mask, keypoints.torso_box(), canvas_size, garment_type="top"
        )

    if bottom_image is not None:
        bottom_rgba = segmentation.remove_background(bottom_image)
        bottom_mask = segmentation.get_garment_mask(bottom_rgba)
        bottom_rgba, bottom_mask = segmentation.crop_to_content(bottom_rgba, bottom_mask)
        bottom_layer_base = warp.tps_warp(
            bottom_rgba, bottom_mask, keypoints.lower_body_box(canvas_size[1]), canvas_size, garment_type="bottom"
        )

    # 4. Generate N variants with small controlled variation, composite each
    results = []
    for i in range(n_variants):
        seed = 1000 + i
        top_v = compose.apply_variation(top_layer_base, seed) if top_layer_base is not None else None
        bottom_v = compose.apply_variation(bottom_layer_base, seed + 1) if bottom_layer_base is not None else None
        final = compose.compose_outfit(mannequin_img, top_v, bottom_v)
        results.append(final)

    return GenerationResult(images=results, gender_used=selected_gender, mannequin_id=mannequin_id)
