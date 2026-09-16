"""
generate.py (Phase 2 - production inference / "API 1")
--------------------------------------------------------
Loads the base inpainting model + your fine-tuned LoRA weights + IP-Adapter,
and dresses a bare mannequin photo with 1-4 items: hat, top, bottom, shoes.

--- v2: auto pose-detection instead of manual calibration ---
Mask placement now defaults to MediaPipe pose detection run directly on the
mannequin_bare photo you pass in (via pipeline/pose_detect.py), the same
approach used in prepare_dataset.py for training. This:
  - Adds hat/shoes support for free (manual calibrate_mannequin.py only ever
    captured shoulders/hips/knees - no head or foot landmarks to build a
    hat or shoe box from).
  - Removes the need to calibrate every mannequin photo by hand before use.
  - Keeps inference geometrically consistent with how the model was trained.
Manual calibration (pipeline/mannequin_pose.py, --mannequin_id) is kept as
an explicit FALLBACK for top/bottom only, for the rare case a specific bare
photo doesn't auto-detect well.

Two ways to run this:
  1. SINGLE ITEM (legacy, still supported): --garment_type + --garment
  2. MULTI ITEM ("dress the mannequin" - the real point of this file now):
     any combination of --hat / --top / --bottom / --shoes. Each requested
     item is inpainted in sequence (hat, then top, then bottom, then shoes),
     each step building on the previous step's output, so by the end the
     mannequin is wearing everything you asked for in one output image.

Usage (single item, unchanged from before):
    python generate.py \
        --lora_path lora_output/final \
        --mannequin_bare mannequin_female_bare.jpg \
        --garment_type top --garment shirt.jpg \
        --n_variants 3 --out_dir results/

Usage (full outfit - hat + top + bottom + shoes in one call):
    python generate.py \
        --lora_path lora_output/final \
        --mannequin_bare mannequin_female_bare.jpg \
        --hat cap.jpg --top shirt.jpg --bottom jeans.jpg --shoes sneakers.jpg \
        --n_variants 3 --out_dir results/
"""
import argparse
import os
import sys
import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionInpaintPipeline

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pipeline import pose_detect               # auto keypoint detection (default path)
from pipeline import mannequin_pose            # manual calibration (fallback, top/bottom only)
from pipeline.garment_analysis import classify_sleeve_length

ITEM_KINDS = ("hat", "top", "bottom", "shoes")
# Dressing order matters: each step's output becomes the next step's input,
# so this is effectively the order you'd physically get dressed in - hat
# last would visually cover hair/neckline oddly, shoes first would get
# painted over by pants. Top-down, "inner layers first" order avoids that.
DRESS_ORDER = ("hat", "top", "bottom", "shoes")

# Same padding constants as prepare_dataset.py, so a box built here at
# inference matches the geometry the model was actually trained on.
_PAD_X_FRAC = {"hat": 0.15, "top": 0.30, "bottom": 0.30, "shoes": 0.15}
_PAD_Y_TOP_FRAC = {"hat": 0.10, "top": 0.08, "bottom": 0.05, "shoes": 0.05}
_PAD_Y_BOTTOM_FRAC = {"hat": 0.05, "top": 0.05, "bottom": 0.05, "shoes": 0.10}


class PoseUnavailable(Exception):
    """Raised when neither auto pose-detection nor a manual fallback could
    produce keypoints for the requested item kind on this photo."""


def _get_fractional_keypoints(mannequin_bare: Image.Image, kind: str,
                               mannequin_id: str | None, min_confidence: float):
    """
    Returns an object with .torso_box() / .lower_body_box() / .head_box() /
    .feet_box() methods returning FRACTIONAL (0-1) boxes, resolved from
    (in priority order): auto pose detection, then manual calibration
    fallback (top/bottom only) if --mannequin_id was given.
    """
    kp = pose_detect.detect_keypoints(mannequin_bare)
    if kp is not None and kp.confidence >= min_confidence:
        return kp

    reason = "no pose detected" if kp is None else f"low confidence ({kp.confidence:.2f})"
    if mannequin_id:
        if kind not in ("top", "bottom"):
            raise PoseUnavailable(
                f"Auto pose detection failed on this mannequin photo ({reason}), and manual "
                f"calibration (--mannequin_id) only covers shoulders/hips/knees - it can't "
                f"provide a '{kind}' box. Try a clearer/more front-facing mannequin_bare photo."
            )
        print(f"[pose] auto-detection failed ({reason}) - falling back to manual "
              f"calibration for mannequin_id='{mannequin_id}'")
        manual_kp = mannequin_pose.load_manual_keypoints(mannequin_id)
        return manual_kp.to_pixels(1, 1)  # already fractional in storage; to_pixels(1,1) is a no-op passthrough
    raise PoseUnavailable(
        f"Auto pose detection failed on this mannequin photo ({reason}), and no --mannequin_id "
        f"fallback was given. Either use a clearer mannequin_bare photo, or calibrate one with "
        f"calibrate_mannequin.py and pass --mannequin_id."
    )


def build_mask_for_kind(kp, kind: str, image_size: tuple[int, int],
                         garment_image: Image.Image | None = None,
                         sleeve_length: str = "auto") -> Image.Image:
    """
    kp: either a pose_detect.DetectedKeypoints (auto, has head_box/feet_box)
        or a mannequin_pose.BodyKeypoints (manual fallback, top/bottom only).
    kind: one of ITEM_KINDS.
    """
    is_manual = isinstance(kp, mannequin_pose.BodyKeypoints)

    if kind == "hat":
        x0, y0, x1, y1 = kp.head_box()
    elif kind == "top":
        x0, y0, x1, y1 = kp.torso_box()
    elif kind == "bottom":
        # mannequin_pose.BodyKeypoints.lower_body_box() predates the
        # fractional pose_detect API and takes an image_height pixel arg;
        # passing 1 keeps its "* image_height" math a no-op so the result
        # stays in the same 0-1 fractional space as everything else here.
        x0, y0, x1, y1 = kp.lower_body_box(1) if is_manual else kp.lower_body_box()
    elif kind == "shoes":
        x0, y0, x1, y1 = kp.feet_box()
    else:
        raise ValueError(f"Unknown item kind: {kind}")

    w, h = x1 - x0, y1 - y0
    resolved_sleeve = None
    if kind == "top":
        resolved_sleeve = sleeve_length
        if resolved_sleeve == "auto":
            resolved_sleeve = classify_sleeve_length(garment_image) if garment_image is not None else "long"
        print(f"[mask:{kind}] sleeve_length={resolved_sleeve} "
              f"({'auto-detected' if sleeve_length == 'auto' else 'forced by --sleeve_length'})")
        pad_x_frac = 0.15 if resolved_sleeve == "short" else 0.45
    else:
        pad_x_frac = _PAD_X_FRAC[kind]

    x0 -= w * pad_x_frac
    x1 += w * pad_x_frac
    y0 -= h * _PAD_Y_TOP_FRAC[kind]
    y1 += h * _PAD_Y_BOTTOM_FRAC[kind]

    x0, y0, x1, y1 = [v for v in (x0, y0, x1, y1)]
    px0 = max(0, min(int(x0 * image_size[0]), image_size[0]))
    px1 = max(0, min(int(x1 * image_size[0]), image_size[0]))
    py0 = max(0, min(int(y0 * image_size[1]), image_size[1]))
    py1 = max(0, min(int(y1 * image_size[1]), image_size[1]))
    if px1 <= px0:
        px0, px1 = max(0, px0 - 1), min(image_size[0], px0 + 1)
    if py1 <= py0:
        py0, py1 = max(0, py0 - 1), min(image_size[1], py0 + 1)

    area_frac = ((px1 - px0) * (py1 - py0)) / (image_size[0] * image_size[1])
    print(f"[mask:{kind}] box=({px0},{py0},{px1},{py1}) on image {image_size} -> covers {area_frac*100:.1f}% of image")
    if area_frac < 0.01:
        print(f"[mask:{kind}] WARNING: mask covers under 1% of the image - the detected/calibrated "
              f"keypoints likely don't match this mannequin photo. Double-check the photo before trusting output.")

    mask = Image.new("L", image_size, 0)
    ImageDraw.Draw(mask).rectangle([px0, py0, px1, py1], fill=255)
    return mask


def load_pipeline(base_model: str, lora_path: str, device: str, ip_adapter_scale: float = 1.4):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    pipe.set_ip_adapter_scale(ip_adapter_scale)

    if lora_path and os.path.exists(lora_path):
        # train_lora.py injects LoRA via unet.add_adapter() (PEFT) and saves
        # with unet.save_lora_adapter() - load with the matching PEFT-based
        # method. unet.load_attn_procs() is a different legacy format
        # (Custom Diffusion attention processors) and will silently produce
        # an unmodified/broken UNet against a PEFT-saved checkpoint.
        if hasattr(pipe.unet, "load_lora_adapter"):
            # prefix=None: save_lora_adapter() was called directly on the
            # unet object (not the full pipeline), so its saved state dict
            # keys have no "unet." prefix. load_lora_adapter()'s default
            # prefix="unet" then can't match any key, silently loading the
            # adapter with ZERO effect (no error, but LoRA is inert).
            pipe.unet.load_lora_adapter(lora_path, weight_name="pytorch_lora_weights.safetensors", prefix=None)
        else:
            pipe.unet.load_attn_procs(lora_path)
        print(f"Loaded LoRA weights from {lora_path}")
    else:
        print("WARNING: no LoRA path given/found - running base model only (garment identity will be weak).")

    pipe = pipe.to(device)
    return pipe


def dress_mannequin(pipe, mannequin_bare: Image.Image, items: dict[str, str | Image.Image],
                     n_variants: int = 3, num_inference_steps: int = 30, guidance_scale: float = 4.5,
                     resolution: int = 512, base_seed: int = 42, sleeve_length: str = "auto",
                     mannequin_id: str | None = None, min_confidence: float = 0.3) -> list[Image.Image]:
    """
    THE core "API 1" function: dress a bare mannequin in any combination of
    hat/top/bottom/shoes in a single call.

    items: e.g. {"top": "shirt.jpg", "bottom": PIL.Image(...), "shoes": "sneakers.jpg"}
           Only include the kinds you actually want dressed (1 to 4 of them).
    Returns n_variants final images, each with ALL requested items applied.
    """
    kinds = [k for k in DRESS_ORDER if items.get(k)]
    if not kinds:
        raise ValueError(f"items must include at least one of {ITEM_KINDS}")

    mannequin_bare = mannequin_bare.convert("RGB").resize((resolution, resolution))
    garment_images = {
        k: (v if isinstance(v, Image.Image) else Image.open(v)).convert("RGB")
        for k, v in items.items() if v
    }

    # Detect pose ONCE from the original bare photo. Re-detecting after each
    # dressing step would be unreliable (e.g. a hat painted in step 1 can
    # cover the nose landmark needed to detect anything in step 2) - the
    # mannequin's body doesn't move between steps, only what's drawn on it.
    kp = _get_fractional_keypoints(mannequin_bare, kinds[0], mannequin_id, min_confidence)
    masks = {
        k: build_mask_for_kind(kp, k, mannequin_bare.size,
                                garment_image=garment_images.get(k), sleeve_length=sleeve_length)
        for k in kinds
    }

    results = []
    for i in range(n_variants):
        current = mannequin_bare
        seed = base_seed + i
        for kind in kinds:
            generator = torch.Generator(device=pipe.device).manual_seed(seed)
            current = pipe(
                prompt="",
                image=current,
                mask_image=masks[kind],
                ip_adapter_image=garment_images[kind],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
        results.append(current)

    return results


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: running on CPU - diffusion inference will be very slow. Use a GPU.")

    pipe = load_pipeline(args.base_model, args.lora_path, device, ip_adapter_scale=args.ip_adapter_scale)
    mannequin = Image.open(args.mannequin_bare)

    # Multi-item mode: any of --hat/--top/--bottom/--shoes given.
    multi_items = {k: getattr(args, k) for k in ITEM_KINDS if getattr(args, k)}

    if multi_items:
        images = dress_mannequin(
            pipe, mannequin, multi_items, n_variants=args.n_variants,
            sleeve_length=args.sleeve_length, mannequin_id=args.mannequin_id,
            min_confidence=args.min_confidence,
        )
    elif args.garment_type and args.garment:
        # Legacy single-item path, kept for backward compatibility / quick testing.
        images = dress_mannequin(
            pipe, mannequin, {args.garment_type: args.garment}, n_variants=args.n_variants,
            sleeve_length=args.sleeve_length, mannequin_id=args.mannequin_id,
            min_confidence=args.min_confidence,
        )
    else:
        raise ValueError("Provide either --hat/--top/--bottom/--shoes (any combination), "
                          "or the legacy --garment_type + --garment for a single item.")

    os.makedirs(args.out_dir, exist_ok=True)
    for i, img in enumerate(images):
        path = os.path.join(args.out_dir, f"outfit_{i+1}.png")
        img.save(path)
        print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--lora_path", default="lora_output/final")
    parser.add_argument("--mannequin_bare", required=True)
    parser.add_argument("--mannequin_id", default=None,
                         help="Optional FALLBACK manual calibration (pipeline/mannequin_pose.py), only "
                              "used if auto pose-detection fails, and only covers top/bottom.")
    parser.add_argument("--min_confidence", type=float, default=0.3,
                         help="Minimum mean pose-landmark visibility to trust auto-detection before "
                              "falling back to --mannequin_id (if given) or raising an error.")

    # Multi-item mode (the main way to use this script now):
    parser.add_argument("--hat", default=None, help="Path to a hat product photo")
    parser.add_argument("--top", default=None, help="Path to a top product photo")
    parser.add_argument("--bottom", default=None, help="Path to a bottom product photo")
    parser.add_argument("--shoes", default=None, help="Path to a shoes product photo")

    # Legacy single-item mode (kept for backward compatibility):
    parser.add_argument("--garment_type", default=None, choices=list(ITEM_KINDS))
    parser.add_argument("--garment", default=None)

    parser.add_argument("--sleeve_length", default="auto", choices=["auto", "short", "long"],
                         help="Only used for the 'top' item. 'auto' guesses from the garment photo "
                              "(see pipeline/garment_analysis.py); override if the guess looks wrong.")
    parser.add_argument("--ip_adapter_scale", type=float, default=1.4)
    parser.add_argument("--n_variants", type=int, default=3)
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()
    main(args)