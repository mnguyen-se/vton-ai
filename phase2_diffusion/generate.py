"""
generate.py (Phase 2 - production inference)
-----------------------------------------------
Loads the base inpainting model + your fine-tuned LoRA weights + IP-Adapter,
and generates outfit images the same way demo.py did for Phase 1 - but with
the learned generator instead of TPS warp + alpha blend.

This is the function to call from your actual product (API endpoint, batch
job, etc.) once you have a LoRA checkpoint you're happy with.

Two ways to supply the inpainting mask:
  1. --mask path/to/mask.png            (a mask file you already have)
  2. --mannequin_id female_default --garment_type top
     (auto-builds the mask from the SAME keypoints you calibrated in
     Phase 1 with calibrate_mannequin.py - recommended, keeps Phase 1
     and Phase 2 consistent with zero extra work)

Usage:
    python generate.py \
        --lora_path lora_output/final \
        --mannequin_bare mannequin_female_bare.jpg \
        --mannequin_id female_default --garment_type top \
        --garment shirt.jpg \
        --n_variants 3 \
        --out_dir results/
"""
import argparse
import os
import sys
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pipeline import mannequin_pose  # reuse Phase 1 keypoints for mask generation


def build_mask_from_keypoints(mannequin_id: str, garment_type: str, image_size: tuple[int, int],
                               sleeve_length: str = "short") -> Image.Image:
    kp = mannequin_pose.load_manual_keypoints(mannequin_id)
    if garment_type == "top":
        # torso_box() alone excludes the arms entirely, so a sleeved garment
        # can never be painted there - use torso_and_arms_box() instead.
        box = kp.torso_and_arms_box(sleeve_length=sleeve_length)
    else:
        box = kp.lower_body_box(image_size[1])

    # pad the box a bit so the model has room to blend garment edges naturally
    x0, y0, x1, y1 = box
    pad_x, pad_y = int((x1 - x0) * 0.15), int((y1 - y0) * 0.08)
    x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    x1, y1 = min(image_size[0], x1 + pad_x), min(image_size[1], y1 + pad_y)

    mask = Image.new("L", image_size, 0)
    from PIL import ImageDraw
    ImageDraw.Draw(mask).rectangle([x0, y0, x1, y1], fill=255)
    return mask


def load_pipeline(base_model: str, lora_path: str, device: str, ip_adapter_scale: float = 1.4,
                   low_vram: bool = False):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    # 1.0 (the diffusers default) left garment identity too weak in testing -
    # 1.4 keeps pattern/color much closer to the source garment photo.
    pipe.set_ip_adapter_scale(ip_adapter_scale)

    if lora_path and os.path.exists(lora_path):
        pipe.unet.load_attn_procs(lora_path)
        print(f"Loaded LoRA weights from {lora_path}")
    else:
        print("WARNING: no LoRA path given/found - running base model only (garment identity will be weak).")

    if low_vram:
        # Trades a bit of speed for a lot less peak VRAM - useful on <=6GB cards
        # (e.g. RTX 3050 Ti) for a quick smoke-test before running for real on
        # a bigger GPU. Not needed on 8GB+ cards.
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        pipe.enable_sequential_cpu_offload()  # keeps this from also calling .to(device) below
        print("low_vram: attention/vae slicing + sequential CPU offload enabled")
        return pipe

    pipe = pipe.to(device)
    return pipe


def generate_variants(pipe, mannequin_bare: Image.Image, mask: Image.Image, garment: Image.Image,
                       n_variants: int = 3, num_inference_steps: int = 30, guidance_scale: float = 4.5,
                       resolution: int = 512, base_seed: int = 42,
                       prompt: str = "photo of a person wearing the exact garment shown, same fabric print, "
                                     "same colors and pattern, same sleeve length, high detail, studio lighting",
                       negative_prompt: str = "different pattern, different color, blurry, distorted, extra limbs, "
                                               "sleeveless, cropped sleeve, low quality") -> list[Image.Image]:
    """
    Mirrors Phase 1's generate_outfits() output contract: returns a list of
    N generated images, one per variant, using different seeds for diversity.

    NOTE: previously this called the pipe with prompt="", so garment fidelity
    relied 100% on the IP-Adapter image embedding, which only captures a
    loose "style" of the garment (this is why colors/pattern could drift
    between variants). Pairing IP-Adapter with a descriptive text prompt +
    negative_prompt keeps the sampler anchored closer to the source photo.
    """
    mannequin_bare = mannequin_bare.convert("RGB").resize((resolution, resolution))
    mask = mask.convert("L").resize((resolution, resolution))
    garment = garment.convert("RGB")

    results = []
    for i in range(n_variants):
        generator = torch.Generator(device=pipe.device).manual_seed(base_seed + i)
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=mannequin_bare,
            mask_image=mask,
            ip_adapter_image=garment,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        results.append(out)

    return results


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: running on CPU - diffusion inference will be very slow. Use a Colab GPU runtime.")

    pipe = load_pipeline(args.base_model, args.lora_path, device, ip_adapter_scale=args.ip_adapter_scale,
                         low_vram=args.low_vram)

    mannequin = Image.open(args.mannequin_bare)
    garment = Image.open(args.garment)

    if args.mask:
        mask = Image.open(args.mask)
    elif args.mannequin_id and args.garment_type:
        mask = build_mask_from_keypoints(args.mannequin_id, args.garment_type, mannequin.size,
                                          sleeve_length=args.sleeve_length)
    else:
        raise ValueError("Provide either --mask, or both --mannequin_id and --garment_type")

    images = generate_variants(pipe, mannequin, mask, garment, n_variants=args.n_variants)
    # Debug aid: save the mask too so you can *see* whether the arm area is
    # actually included before waiting on a full diffusion run again.
    os.makedirs(args.out_dir, exist_ok=True)
    mask.save(os.path.join(args.out_dir, "_debug_mask.png"))

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
    parser.add_argument("--mask", default=None, help="Path to a mask file. Omit to auto-build from keypoints instead.")
    parser.add_argument("--mannequin_id", default=None, help="Used with --garment_type to auto-build mask from Phase 1 keypoints")
    parser.add_argument("--garment_type", default=None, choices=["top", "bottom"])
    parser.add_argument("--sleeve_length", default="short", choices=["none", "short", "long"],
                         help="How far down/out the auto-built mask should extend for a 'top' garment, "
                              "so the diffusion model has room to paint the sleeve. Match this to the "
                              "garment photo (e.g. short-sleeve shirt -> 'short').")
    parser.add_argument("--garment", required=True)
    parser.add_argument("--n_variants", type=int, default=3)
    parser.add_argument("--ip_adapter_scale", type=float, default=1.4,
                         help="How strongly the garment photo drives the output (higher = closer to source "
                              "pattern/color, too high can look pasted-on). 1.4 worked best in testing.")
    parser.add_argument("--low_vram", action="store_true",
                         help="Enable attention/VAE slicing + CPU offload for GPUs with <=6GB VRAM "
                              "(e.g. RTX 3050 Ti). Slower, but avoids OOM for a smoke test.")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()
    main(args)
