"""
prepare_dataset.py (Phase 2)
------------------------------
Builds the training set for LoRA fine-tuning of the diffusion VTON model.

Each training sample needs THREE things:
  1. target.jpg     - ground truth: mannequin ALREADY wearing the garment
                       (this is what the model learns to generate)
  2. masked.jpg      - the same mannequin WITHOUT that garment (bare, or
                        wearing something else) - what the model paints INTO
  3. mask.png        - binary mask of the region that gets repainted
                        (torso for tops, legs for bottoms)
  4. garment.jpg     - the flat/product photo of the garment (conditioning
                        signal fed through IP-Adapter)

WHERE DOES THIS DATA COME FROM?
Realistically you have two sources, use both:
  A. Real photos you already have or shoot: mannequin bare + mannequin
     wearing a real garment = perfect (target, masked) pair for that garment.
  B. Bootstrap using Phase 1 (baseline TPS pipeline): run the baseline
     pipeline to paste garments onto mannequins -> these become rough
     "target" images. They won't be photorealistic, but they still teach
     the model garment identity / rough placement, and mixing in even a
     small number of real photos (source A) teaches it realism. This is
     the standard trick when real paired data is scarce.

Folder layout expected as INPUT to this script:
  raw_data/
    sample_0001/
      mannequin_bare.jpg
      mannequin_wearing.jpg
      garment.jpg
      body_region.json     # optional: {"box": [x0,y0,x1,y1]}, else auto from keypoints
    sample_0002/
      ...

Produces OUTPUT in `data/train/` ready for train_lora.py.
"""
import argparse
import json
import os
import shutil
import sys
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pipeline import segmentation  # reuse Phase 1's background removal for the mask fallback


def make_mask_from_box(box, size) -> Image.Image:
    x0, y0, x1, y1 = box
    mask = Image.new("L", size, 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x0, y0, x1, y1], fill=255)
    return mask


def make_mask_from_diff(bare: Image.Image, wearing: Image.Image, threshold: int = 30) -> Image.Image:
    """
    Fallback: if no body_region.json given, estimate the mask as the region
    that visibly changed between bare and wearing mannequin photos.
    Works best when the two photos are pixel-aligned (same camera position).
    """
    a = np.array(bare.convert("RGB")).astype(np.int16)
    b = np.array(wearing.convert("RGB")).astype(np.int16)
    diff = np.abs(a - b).sum(axis=-1)
    mask = (diff > threshold).astype(np.uint8) * 255

    # clean up: dilate a bit so edges are fully covered
    import cv2
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return Image.fromarray(mask, mode="L")


def clean_garment(garment_path: str, image_size: int) -> Image.Image:
    """Run a garment product photo through Phase 1's background remover for a
    consistent, isolated conditioning signal."""
    garment_raw = Image.open(garment_path).convert("RGB")
    garment_rgba = segmentation.remove_background(garment_raw)
    garment_clean = Image.new("RGB", garment_rgba.size, (255, 255, 255))
    garment_clean.paste(garment_rgba, mask=garment_rgba.split()[3])
    return garment_clean.resize((image_size, image_size))


def split_mask_top_bottom(mask: Image.Image, waist_frac: float = 0.55):
    """
    Split a full diff mask into a "top" half (torso/shirt region) and a
    "bottom" half (legs/pants region) using an approximate waist line.
    Used for combo photos where both a top and a bottom changed at once,
    so a single diff mask would otherwise cover both garments.
    """
    w, h = mask.size
    split_y = int(h * waist_frac)

    arr = np.array(mask)
    top_arr = arr.copy()
    top_arr[split_y:, :] = 0
    bottom_arr = arr.copy()
    bottom_arr[:split_y, :] = 0

    return Image.fromarray(top_arr, mode="L"), Image.fromarray(bottom_arr, mode="L")


def _write_example(out_dir: str, example_id: str, wearing: Image.Image, bare: Image.Image,
                    mask: Image.Image, garment_clean: Image.Image):
    out = os.path.join(out_dir, example_id)
    os.makedirs(out, exist_ok=True)
    wearing.save(os.path.join(out, "target.jpg"), quality=95)
    bare.save(os.path.join(out, "masked.jpg"), quality=95)
    mask.save(os.path.join(out, "mask.png"))
    garment_clean.save(os.path.join(out, "garment.jpg"), quality=95)


_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _find_image(sample_dir: str, basename: str) -> str | None:
    """Look for <basename>.<ext> under any of the common image extensions,
    so raw_data samples work whether their photos are .jpg or .png (mixed
    extensions from different phone/export sources used to make this script
    silently skip whole samples)."""
    for ext in _IMG_EXTS:
        p = os.path.join(sample_dir, basename + ext)
        if os.path.exists(p):
            return p
    return None


def process_sample(sample_dir: str, out_dir: str, sample_id: str, image_size: int = 512):
    bare_path = _find_image(sample_dir, "mannequin_bare")
    wearing_path = _find_image(sample_dir, "mannequin_wearing")
    region_path = os.path.join(sample_dir, "body_region.json")

    garment_path = _find_image(sample_dir, "garment")
    garment_top_path = _find_image(sample_dir, "garment_top")
    garment_bottom_path = _find_image(sample_dir, "garment_bottom")

    if not (bare_path and wearing_path):
        print(f"[skip] {sample_dir}: missing mannequin_bare.* / mannequin_wearing.* "
              f"(looked for {_IMG_EXTS})")
        return False

    bare = Image.open(bare_path).convert("RGB").resize((image_size, image_size))
    wearing = Image.open(wearing_path).convert("RGB").resize((image_size, image_size))

    has_combo = bool(garment_top_path or garment_bottom_path)

    if has_combo:
        # Combo sample: photo shows the mannequin wearing TWO garments at
        # once (e.g. a top + a bottom). Produce one training example per
        # garment, each using only its half of the diff mask so the model
        # doesn't learn to paint the other garment too.
        full_mask = make_mask_from_diff(bare, wearing)
        top_mask, bottom_mask = split_mask_top_bottom(full_mask)

        wrote_any = False
        if garment_top_path:
            garment_clean = clean_garment(garment_top_path, image_size)
            _write_example(out_dir, f"{sample_id}_top", wearing, bare, top_mask, garment_clean)
            wrote_any = True
        if garment_bottom_path:
            garment_clean = clean_garment(garment_bottom_path, image_size)
            _write_example(out_dir, f"{sample_id}_bottom", wearing, bare, bottom_mask, garment_clean)
            wrote_any = True
        return wrote_any

    if not garment_path:
        print(f"[skip] {sample_dir}: missing garment.* (or garment_top.*/garment_bottom.*)")
        return False

    garment_clean = clean_garment(garment_path, image_size)

    if os.path.exists(region_path):
        with open(region_path) as f:
            box = json.load(f)["box"]
        mask = make_mask_from_box(box, (image_size, image_size))
    else:
        mask = make_mask_from_diff(bare, wearing)

    _write_example(out_dir, sample_id, wearing, bare, mask, garment_clean)
    return True


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    samples = sorted(os.listdir(args.raw_dir))
    ok, skipped = 0, 0
    for s in samples:
        sample_dir = os.path.join(args.raw_dir, s)
        if not os.path.isdir(sample_dir):
            continue
        success = process_sample(sample_dir, args.out_dir, s, args.image_size)
        ok += int(success)
        skipped += int(not success)

    print(f"Done. {ok} samples prepared, {skipped} skipped -> {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="raw_data")
    parser.add_argument("--out_dir", default="data/train")
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()
    main(args)
