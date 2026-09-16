"""
train_lora.py (Phase 2)
--------------------------
Fine-tunes a Stable Diffusion INPAINTING model with LoRA, conditioned on
the garment reference image via IP-Adapter, so it learns to paint YOUR
garments onto YOUR mannequins realistically.

Design choice / why this architecture:
  - Base: runwayml/stable-diffusion-inpainting (pretrained, open-source,
    downloaded once from HuggingFace - no ongoing API calls, everything
    runs locally after download).
  - Garment conditioning: IP-Adapter (pretrained image encoder + adapter
    layers, also open-source/local) turns the garment product photo into
    a conditioning embedding, instead of relying on a text prompt alone.
  - Only LoRA adapters on the UNet's attention layers are trained -
    everything else (VAE, text encoder, IP-Adapter image encoder) stays
    frozen. This is what makes fine-tuning feasible on a single consumer
    GPU (Colab T4/A100) instead of needing to train a model from scratch.

Run (in Colab, with a GPU runtime):
    accelerate launch train_lora.py \
        --data_dir data/train \
        --output_dir lora_output \
        --resolution 512 \
        --train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 30 \
        --learning_rate 1e-4

Notes on VRAM:
  - resolution 512, batch_size 1, fp16, gradient checkpointing ON:
    fits comfortably on a 16GB GPU (Colab T4/L4), tight but workable on
    12GB, NOT recommended below that.
  - If you only have ~8-12GB, lower resolution to 384 and/or increase
    gradient_accumulation_steps instead of batch size.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from accelerate import Accelerator
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
from peft import LoraConfig
from torchvision import transforms


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class VtonDataset(Dataset):
    """
    Expects data_dir/<sample_id>/{target.jpg, masked.jpg, mask.png, garment.jpg}
    as produced by prepare_dataset.py
    """

    def __init__(self, data_dir: str, resolution: int = 512):
        self.samples = sorted(
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        )
        self.data_dir = data_dir
        self.resolution = resolution
        self.img_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # -> [-1, 1], matches SD VAE input range
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),  # -> [0, 1]
        ])
        self.garment_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # CLIP image encoder input size for IP-Adapter
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                  std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d = os.path.join(self.data_dir, self.samples[idx])
        target = Image.open(os.path.join(d, "target.jpg")).convert("RGB")
        masked = Image.open(os.path.join(d, "masked.jpg")).convert("RGB")
        mask = Image.open(os.path.join(d, "mask.png")).convert("L")
        garment = Image.open(os.path.join(d, "garment.jpg")).convert("RGB")

        return {
            "target": self.img_transform(target),
            "masked_image": self.img_transform(masked),
            "mask": self.mask_transform(mask),
            "garment": self.garment_transform(garment),
        }


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def main(args):
    accelerator = Accelerator(
        mixed_precision="fp16" if torch.cuda.is_available() else "no",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    print("Loading base pipeline (downloads pretrained weights on first run, then cached locally)...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
    )

    # Attach pretrained IP-Adapter for garment-image conditioning (frozen weights).
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="models",
        weight_name="ip-adapter_sd15.bin",
    )
    pipe.set_ip_adapter_scale(1.0)

    vae, unet, text_encoder, tokenizer = pipe.vae, pipe.unet, pipe.text_encoder, pipe.tokenizer
    image_encoder = pipe.image_encoder

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # text_encoder is frozen and never passed through accelerator.prepare(),
    # so it must be moved to the training device manually (same reason
    # vae/image_encoder get an explicit .to() below) - otherwise the empty
    # prompt embedding lookup runs with weights on CPU against GPU input ids.
    text_encoder.to(accelerator.device)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # --- Attach LoRA adapters to the UNet's cross/self-attention layers ---
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
    )
    unet.add_adapter(lora_config)

    # AMP's GradScaler (used for mixed_precision="fp16") requires trainable
    # parameters to be fp32 so it can unscale their gradients - it errors
    # with "Attempting to unscale FP16 gradients" otherwise. The base UNet
    # stays frozen in fp16; only cast the newly-added, trainable LoRA
    # adapter weights up to fp32.
    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    print(f"Trainable LoRA params: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")

    dataset = VtonDataset(args.data_dir, resolution=args.resolution)
    print(f"Dataset size: {len(dataset)} samples")
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    # Empty text prompt embedding reused for every step - we condition on
    # the garment image via IP-Adapter, not on text.
    with torch.no_grad():
        empty_input_ids = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length,
                                     truncation=True, return_tensors="pt").input_ids
        empty_text_embeds = text_encoder(empty_input_ids.to(accelerator.device))[0]

    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    vae.to(accelerator.device)
    image_encoder.to(accelerator.device)

    global_step = 0
    for epoch in range(args.num_train_epochs):
        unet.train()
        epoch_loss = 0.0

        for batch in dataloader:
            with accelerator.accumulate(unet):
                target = batch["target"].to(accelerator.device, dtype=vae.dtype)
                masked_image = batch["masked_image"].to(accelerator.device, dtype=vae.dtype)
                mask = batch["mask"].to(accelerator.device, dtype=vae.dtype)
                garment = batch["garment"].to(accelerator.device, dtype=vae.dtype)

                bsz = target.shape[0]

                # Encode target -> latents (this is what we train the UNet to denoise)
                latents = vae.encode(target).latent_dist.sample() * vae.config.scaling_factor
                masked_latents = vae.encode(masked_image).latent_dist.sample() * vae.config.scaling_factor
                mask_latent = F.interpolate(mask, size=latents.shape[-2:])

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                           device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # SD inpainting UNet expects 9 channels: noisy_latents(4) + mask(1) + masked_latents(4)
                unet_input = torch.cat([noisy_latents, mask_latent, masked_latents], dim=1)

                # Garment image -> IP-Adapter conditioning embedding
                garment_embeds = image_encoder(garment).image_embeds
                # IP-Adapter's MultiIPAdapterImageProjection wrapper expects
                # each entry in the `image_embeds` list to be 3D:
                # [batch_size, num_images, embedding_dim]. It only
                # auto-adds that middle dimension when you pass a bare
                # tensor (not wrapped in a list); since we pass a list,
                # we must add it ourselves (num_images=1 here) or it reads
                # the embedding dim itself as num_images and corrupts the
                # later reshape.
                garment_embeds = garment_embeds.view(bsz, -1).unsqueeze(1)
                text_embeds = empty_text_embeds.repeat(bsz, 1, 1)

                added_cond_kwargs = {"image_embeds": [garment_embeds]}
                model_pred = unet(unet_input, timesteps, encoder_hidden_states=text_embeds,
                                   added_cond_kwargs=added_cond_kwargs).sample

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1

        print(f"Epoch {epoch+1}/{args.num_train_epochs} - avg loss: {epoch_loss/len(dataloader):.4f}")

        if (epoch + 1) % args.save_every == 0:
            _save_lora(unet, args.output_dir, f"checkpoint-epoch{epoch+1}")

    _save_lora(unet, args.output_dir, "final")
    print(f"Training complete. LoRA weights saved to {args.output_dir}")


def _save_lora(unet, output_dir, name):
    save_path = os.path.join(output_dir, name)
    os.makedirs(save_path, exist_ok=True)
    # unet.add_adapter() above injects LoRA via PEFT, so it must be saved
    # with the matching PEFT-based save method. save_attn_procs() is a
    # separate legacy path only valid for Custom Diffusion attention
    # processors and will raise ValueError against a PEFT-injected model.
    if hasattr(unet, "save_lora_adapter"):
        unet.save_lora_adapter(save_path)
    elif hasattr(unet, "save_attn_procs"):
        unet.save_attn_procs(save_path)
    else:
        unet.save_pretrained(save_path)
    print(f"  saved checkpoint -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--data_dir", default="data/train")
    parser.add_argument("--output_dir", default="lora_output")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True,
                         help="Trade speed for VRAM by recomputing activations during backward instead "
                              "of storing them. Needed on small-VRAM cards (e.g. 4GB laptop GPUs); on a "
                              "16GB+ card you have VRAM to spare, so turn this OFF for a real speed-up: "
                              "--no-gradient_checkpointing")
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()
    main(args)