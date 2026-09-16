"""
prepare_dataset.py (Phase 2)
------------------------------
Builds the training set for LoRA fine-tuning of the diffusion VTON model.

--- v2: pose-detected masking, no "bare" reference photo required ---
The original version built the inpaint mask by pixel-diffing
mannequin_bare.jpg against mannequin_wearing.jpg. That ONLY works if both
photos are the exact same mannequin/pose/camera/crop - a real before/after
pair. Real-world data (product photos pulled from different listings/shops)
almost never satisfies that even when you reuse one generic "bare" photo
across samples: different aspect ratios, poses, crops, even different
mannequins/models entirely. Diffing unrelated photos produces garbage masks
and - worse - trains the model on (masked_input, target) pairs that don't
actually correspond to the same scene, which teaches it to hallucinate
mismatched bodies (this is what produced the bloated-mannequin-with-human-feet
and floating-garment-on-blank-background failures).

New approach: detect body keypoints DIRECTLY on each mannequin_wearing photo
(via MediaPipe Pose, see pipeline/pose_detect.py), build the mask from that,
and build the "masked" training input by blanking the mask region OUT OF
THE SAME WEARING PHOTO (not a different bare photo). This is the standard
scheme for fine-tuning inpainting models and requires no matched pair at
all - mannequin_bare.jpg is no longer used at training time (only needed
later, at inference, as the actual photo you want to dress).

Folder layout expected as INPUT to this script:
  raw_data/
    sample_0001/
      mannequin_wearing.jpg        # required - the only photo actually used
      garment.jpg                  # single-garment sample (treated as "top"), OR any of:
      garment_hat.jpg              # sample shows a hat worn
      garment_top.jpg              # sample shows a top worn
      garment_bottom.jpg           # sample shows a bottom worn
      garment_shoes.jpg            # sample shows shoes worn
      mannequin_bare.jpg           # optional, ignored here (kept for inference use)
      body_region.json             # optional manual override, see below
    sample_0002/
      ...

A sample can include ANY COMBINATION of hat/top/bottom/shoes (1 to 4 of
them) as long as mannequin_wearing.jpg actually shows the mannequin wearing
each one included - you do NOT need every sample to have all 4 items.
One training example is written per item present in the sample.

body_region.json (optional, only if auto pose-detection fails/misfires for
a given photo): {"box_<kind>": [x0, y0, x1, y1]} as FRACTIONS (0.0-1.0) of
the ORIGINAL wearing photo's width/height, one key per item kind present
("box_hat", "box_top", "box_bottom", "box_shoes"). For a legacy single
"garment.jpg" sample, use {"box": [...]}.

Produces OUTPUT in `data/train/` ready for train_lora.py.
"""
import argparse
import json
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pipeline import segmentation   # Phase 1's background remover, reused for garment conditioning image
from pipeline import pose_detect    # MediaPipe-based auto keypoint detection (this fix's core piece)

_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
ITEM_KINDS = ("hat", "top", "bottom", "shoes", "dress")
# A dress covers what top+bottom would separately cover - mutually
# exclusive with them (see process_sample below), never combined.
_MUTUALLY_EXCLUSIVE = {"dress": ("top", "bottom"), "top": ("dress",), "bottom": ("dress",)}

# How far to pad the raw keypoint box, as a fraction of box width/height.
# Single fixed value (unlike generate.py's sleeve-aware padding) because the
# dataset should teach the model a range of sleeve/hem lengths across many
# samples - it doesn't need to be exact per-sample. Hats and shoes get their
# own (smaller) padding since head_box/feet_box are already sized to the
# item's natural footprint, not a coarse torso-scale estimate.
_PAD_X_FRAC = {"hat": 0.15, "top": 0.30, "bottom": 0.30, "shoes": 0.15, "dress": 0.25}
_PAD_Y_TOP_FRAC = {"hat": 0.10, "top": 0.08, "bottom": 0.05, "shoes": 0.05, "dress": 0.06}
_PAD_Y_BOTTOM_FRAC = {"hat": 0.05, "top": 0.05, "bottom": 0.05, "shoes": 0.10, "dress": 0.05}


def _find_file(sample_dir: str, base_name: str) -> str | None:
    """Return the first existing path for base_name with any of _EXTS, or None."""
    for ext in _EXTS:
        p = os.path.join(sample_dir, base_name + ext)
        if os.path.exists(p):
            return p
    return None


def letterbox(img: Image.Image, size: int, fill=(255, 255, 255)):
    """
    Resize preserving aspect ratio, padding to a size x size square instead
    of stretching. Stretching a 1792x2400 portrait product photo and a
    987x1024 near-square photo to the same 512x512 square (the old
    behavior) distorts body proportions by a DIFFERENT amount per sample -
    this is a big part of why generated bodies came out bloated/warped.
    Returns (canvas, scale, pad_x, pad_y) - the transform is only needed if
    you must map coordinates from the ORIGINAL image into canvas space;
    here we instead run pose detection straight on the canvas, so callers
    that don't have pre-existing original-space coordinates can ignore them.
    """
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), fill)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def clean_garment(garment_path: str, image_size: int) -> Image.Image:
    """Run a garment product photo through Phase 1's background remover for a
    consistent, isolated conditioning signal."""
    garment_raw = Image.open(garment_path).convert("RGB")
    garment_rgba = segmentation.remove_background(garment_raw)
    garment_clean = Image.new("RGB", garment_rgba.size, (255, 255, 255))
    garment_clean.paste(garment_rgba, mask=garment_rgba.split()[3])
    return garment_clean.resize((image_size, image_size))


def _pad_box(box_frac: tuple[float, float, float, float], size: int, kind: str) -> tuple[int, int, int, int]:
    """Turn a fractional (x0,y0,x1,y1) box into a padded pixel box on a size x size canvas."""
    x0, y0, x1, y1 = [v * size for v in box_frac]
    w, h = x1 - x0, y1 - y0
    x0 -= w * _PAD_X_FRAC[kind]
    x1 += w * _PAD_X_FRAC[kind]
    y0 -= h * _PAD_Y_TOP_FRAC[kind]
    y1 += h * _PAD_Y_BOTTOM_FRAC[kind]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(size, int(x1)), min(size, int(y1))
    if x1 <= x0:
        x0, x1 = max(0, x0 - 1), min(size, x0 + 1)
    if y1 <= y0:
        y0, y1 = max(0, y0 - 1), min(size, y0 + 1)
    return x0, y0, x1, y1


def _box_for_kind(kp, kind: str) -> tuple[float, float, float, float]:
    return {
        "hat": kp.head_box,
        "top": kp.torso_box,
        "bottom": kp.lower_body_box,
        "shoes": kp.feet_box,
        "dress": kp.dress_box,
    }[kind]()


def _mask_and_masked(wearing: Image.Image, box: tuple[int, int, int, int], image_size: int):
    mask = Image.new("L", (image_size, image_size), 0)
    ImageDraw.Draw(mask).rectangle(list(box), fill=255)

    # Standard SD-inpainting training convention: masked_image = image with
    # the masked region zeroed out (this matches what
    # StableDiffusionInpaintPipeline does internally at inference, so
    # training sees the same kind of input it will get at inference time).
    wearing_arr = np.array(wearing)
    mask_arr = np.array(mask)
    masked_arr = wearing_arr.copy()
    masked_arr[mask_arr > 127] = 0
    masked = Image.fromarray(masked_arr, mode="RGB")
    return mask, masked


def _write_example(out_dir: str, example_id: str, target: Image.Image, masked: Image.Image,
                    mask: Image.Image, garment_clean: Image.Image):
    out = os.path.join(out_dir, example_id)
    os.makedirs(out, exist_ok=True)
    target.save(os.path.join(out, "target.jpg"), quality=95)
    masked.save(os.path.join(out, "masked.jpg"), quality=95)
    mask.save(os.path.join(out, "mask.png"))
    garment_clean.save(os.path.join(out, "garment.jpg"), quality=95)


def process_sample(sample_dir: str, out_dir: str, sample_id: str, image_size: int = 512,
                    min_confidence: float = 0.3) -> bool:
    wearing_path = _find_file(sample_dir, "mannequin_wearing")
    if not wearing_path:
        print(f"[skip] {sample_dir}: missing mannequin_wearing (any of {_EXTS})")
        return False

    garment_path = _find_file(sample_dir, "garment")  # legacy single-item naming, always "top"
    item_paths = {kind: _find_file(sample_dir, f"garment_{kind}") for kind in ITEM_KINDS}
    has_named_items = any(item_paths.values())

    if not has_named_items and not garment_path:
        print(f"[skip] {sample_dir}: no garment files found "
              f"(expected garment.jpg or garment_hat/top/bottom/shoes/dress.*)")
        return False

    if item_paths["dress"] and (item_paths["top"] or item_paths["bottom"]):
        print(f"[skip] {sample_dir}: has both garment_dress and garment_top/bottom - a dress "
              f"already covers the torso+leg region, combining it with a separate top/bottom "
              f"would give two conflicting masks for overlapping skin. Split into two samples instead.")
        return False

    wearing_raw = Image.open(wearing_path).convert("RGB")
    wearing, _, _, _ = letterbox(wearing_raw, image_size)

    region_path = os.path.join(sample_dir, "body_region.json")
    manual = None
    if os.path.exists(region_path):
        with open(region_path) as f:
            manual = json.load(f)

    _kp_cache = {}

    def get_box(kind: str):
        """Prefers a manual body_region.json override, else auto pose detection (cached per sample)."""
        if manual is not None:
            key = f"box_{kind}" if has_named_items else "box"
            if key in manual:
                return _pad_box(tuple(manual[key]), image_size, kind)

        if "kp" not in _kp_cache:
            _kp_cache["kp"] = pose_detect.detect_keypoints(wearing)
        kp = _kp_cache["kp"]
        if kp is None:
            print(f"[skip] {sample_dir} ({kind}): no pose detected in mannequin_wearing "
                  f"(headless/ghost-mannequin photo, extreme crop, or occlusion). "
                  f"Add a body_region.json override to use this sample.")
            return None
        if kp.confidence < min_confidence:
            print(f"[skip] {sample_dir} ({kind}): pose detection confidence too low "
                  f"({kp.confidence:.2f} < {min_confidence}) - box would likely be unreliable. "
                  f"Add a body_region.json override to use this sample anyway.")
            return None
        return _pad_box(_box_for_kind(kp, kind), image_size, kind)

    if has_named_items:
        wrote_any = False
        for kind in ITEM_KINDS:
            path = item_paths[kind]
            if not path:
                continue
            box = get_box(kind)
            if not box:
                continue
            mask, masked = _mask_and_masked(wearing, box, image_size)
            garment_clean = clean_garment(path, image_size)
            _write_example(out_dir, f"{sample_id}_{kind}", wearing, masked, mask, garment_clean)
            wrote_any = True
        return wrote_any

    # Legacy path: a single unlabeled "garment.jpg" is always treated as a top.
    box = get_box("top")
    if not box:
        return False
    mask, masked = _mask_and_masked(wearing, box, image_size)
    garment_clean = clean_garment(garment_path, image_size)
    _write_example(out_dir, sample_id, wearing, masked, mask, garment_clean)
    return True


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    samples = sorted(os.listdir(args.raw_dir))
    ok, skipped = 0, 0
    for s in samples:
        sample_dir = os.path.join(args.raw_dir, s)
        if not os.path.isdir(sample_dir):
            continue
        success = process_sample(sample_dir, args.out_dir, s, args.image_size, args.min_confidence)
        ok += int(success)
        skipped += int(not success)

    print(f"Done. {ok} samples prepared, {skipped} skipped -> {args.out_dir}")
    if skipped:
        print(f"({skipped} skipped - see [skip] lines above for why each one was excluded)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="raw_data")
    parser.add_argument("--out_dir", default="data/train")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--min_confidence", type=float, default=0.3,
                         help="Minimum mean MediaPipe landmark visibility to trust the auto-detected "
                              "pose box. Lower this if too many real samples get skipped; raise it if "
                              "you're seeing bad boxes get through.")
    args = parser.parse_args()
    main(args)