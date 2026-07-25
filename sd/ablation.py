"""
ablation.py — Noise-schedule ablation study for the Stable Diffusion pipeline.

Runs text-to-image and image-to-image generation for two noise schedules
(linear vs cosine) and saves side-by-side denoising progression plots.

Usage
-----
    python ablation.py

Outputs (written to  ablation_results/ )
-----------------------------------------
    schedule_comparison.png          — alpha-bar curves for both schedules
    t2i_<slug>_linear.png            — T2I denoising strip, linear schedule
    t2i_<slug>_cosine.png            — T2I denoising strip, cosine schedule
    i2i_<slug>_linear.png            — I2I forward-noise + denoising strip, linear
    i2i_<slug>_cosine.png            — I2I forward-noise + denoising strip, cosine
    t2i_comparison_<slug>.png        — linear vs cosine side-by-side final images
    i2i_comparison_<slug>.png        — linear vs cosine side-by-side final images
"""

import os
import sys
import math

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from transformers import CLIPTokenizer

# ── make sure the SD source files are on the path ────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import model_loader
import pipeline
from ddpm import DDPMSampler


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit these paths / settings to match your machine
# ═════════════════════════════════════════════════════════════════════════════

TOKENIZER_VOCAB   = r"E:\Stable_Diffusion\data\vocab.json"
TOKENIZER_MERGES  = r"E:\Stable_Diffusion\data\merges.txt"
MODEL_FILE        = r"E:\Stable_Diffusion\data\v1-5-pruned-emaonly.ckpt"
DOG_IMAGE_PATH    = r"E:\Stable_Diffusion\images\Dog.jpg"        # put your dog image here
OUTPUT_DIR        = "ablation_results"


DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
IDLE_DEVICE = "cpu"

SEED               = 42
NUM_INFERENCE_STEPS = 50
CFG_SCALE          = 7.5
I2I_STRENGTH       = 0.75          # how much to re-noise the dog image

# Timesteps at which we capture latent snapshots during denoising
# (indices into the reversed timestep list — 0 = first denoising step, 49 = final)
# Spread across the full 50 steps so the strip shows noise → structure → sharp image
SNAPSHOT_INDICES = [0, 8, 16, 24, 32, 40, 49]

# Text-to-image prompts for the ablation
T2I_PROMPTS = [
    "A serene mountain lake at sunset, photorealistic, 8K",
    "A futuristic city skyline at night with neon lights, cinematic",
    "A close-up portrait of a red fox in a snowy forest",
]

# Image-to-image: we use the dog and these prompts
I2I_PROMPTS = [
    "A golden retriever wearing a red scarf, studio lighting",
    "An oil painting of a dog in the style of Van Gogh",
]

UNCOND_PROMPT = ""        # negative prompt


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def slug(text: str) -> str:
    """Make a safe filename fragment from a prompt string."""
    return text[:40].lower().replace(" ", "_").replace(",", "").replace(".", "")


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert a (C, H, W) float tensor in [-1, 1] to a PIL image."""
    t = t.detach().cpu().float()
    t = (t + 1) / 2          # [-1,1] → [0,1]
    t = t.clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def latent_to_pil(latent: torch.Tensor, decoder, device,
                  decode_with_vae: bool = False) -> Image.Image:
    """
    Convert a (1, 4, H, W) latent tensor to a PIL image for visualisation.

    decode_with_vae=False  (default, used for mid-denoising snapshots)
        Normalises each of the 4 latent channels independently to [0,1],
        then composites channels 0-2 as RGB.  Gives a meaningful colour
        picture even when the latent is still noisy.

    decode_with_vae=True  (used only for the final image)
        Runs the full VAE decoder — correct for clean latents only.
    """
    if decode_with_vae:
        decoder.to(device)
        with torch.no_grad():
            img = decoder(latent.to(device))
        decoder.to(IDLE_DEVICE)
        img = pipeline.rescale(img, (-1, 1), (0, 255), clamp=True)
        img = img.permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
        return Image.fromarray(img[0])
    else:
        lat = latent.detach().cpu().float().squeeze(0)   # (4, H, W)
        def norm_ch(c):
            lo, hi = c.min(), c.max()
            return (c - lo) / (hi - lo + 1e-8)
        r = norm_ch(lat[0])
        g = norm_ch(lat[1])
        b = norm_ch(lat[2])
        rgb = torch.stack([r, g, b], dim=0)
        arr = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr).resize((512, 512), Image.NEAREST)


def plot_schedule_curves(num_steps=1000, out_path="ablation_results/schedule_comparison.png"):
    """Plot alpha-bar curves for linear vs cosine schedules."""
    # Linear
    beta_start, beta_end = 0.00085, 0.0120
    betas_lin = torch.linspace(beta_start**0.5, beta_end**0.5, num_steps)**2
    alphas_lin = torch.cumprod(1 - betas_lin, dim=0)

    # Cosine
    s = 0.008
    t_cos = torch.linspace(0, num_steps, num_steps + 1)
    f = torch.cos(((t_cos / num_steps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cos = (f / f[0])[1:]

    steps = np.arange(num_steps)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, alphas_lin.numpy(), label="Linear", linewidth=2)
    ax.plot(steps, alphas_cos.numpy(), label="Cosine", linewidth=2, linestyle="--")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel(r"$\bar{\alpha}_t$")
    ax.set_title("Noise Schedule Comparison: Linear vs Cosine")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [saved] {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  CORE GENERATION WITH SNAPSHOT CAPTURE
# ═════════════════════════════════════════════════════════════════════════════

def generate_with_snapshots(
    prompt, uncond_prompt, input_image, strength,
    do_cfg, cfg_scale, schedule,
    n_steps, seed, models, tokenizer, device,
    snapshot_indices,
):
    """
    Mirrors pipeline.generate() but collects decoded latent snapshots
    at the requested step indices.

    Returns
    -------
    final_image   : np.ndarray  (H, W, 3)  uint8
    snapshots     : list of PIL.Image
    """
    WIDTH, HEIGHT = 512, 512
    LATENTS_WIDTH  = WIDTH  // 8
    LATENTS_HEIGHT = HEIGHT // 8

    with torch.no_grad():
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        # ── CLIP context ──────────────────────────────────────────────────
        clip = models["clip"]
        clip.to(device)
        if do_cfg:
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            cond_tokens   = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            cond_context  = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens   = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context  = clip(uncond_tokens)
            context = torch.cat([cond_context, uncond_context])
        else:
            tokens  = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            tokens  = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)
        clip.to(IDLE_DEVICE)

        # ── Sampler ───────────────────────────────────────────────────────
        sampler = DDPMSampler(generator, schedule=schedule)
        sampler.set_inference_timesteps(n_steps)

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        # ── Latent init (text-to-image or image-to-image) ─────────────────
        if input_image is not None:
            encoder = models["encoder"]
            encoder.to(device)
            img_tensor = input_image.resize((WIDTH, HEIGHT))
            img_tensor = np.array(img_tensor)
            img_tensor = torch.tensor(img_tensor, dtype=torch.float32, device=device)
            img_tensor = pipeline.rescale(img_tensor, (0, 255), (-1, 1))
            img_tensor = img_tensor.unsqueeze(0).permute(0, 3, 1, 2)

            enc_noise = torch.randn(latents_shape, generator=generator, device=device)
            latents   = encoder(img_tensor, enc_noise)
            sampler.set_strength(strength=strength)
            latents   = sampler.add_noise(latents, sampler.timesteps[0])
            encoder.to(IDLE_DEVICE)
        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)

        # ── Denoising loop ────────────────────────────────────────────────
        diffusion = models["diffusion"]
        diffusion.to(device)

        snap_set = set(snapshot_indices)
        snapshots = []

        for i, timestep in enumerate(sampler.timesteps):
            time_embedding = pipeline.get_time_embedding(timestep).to(device)
            model_input    = latents.repeat(2, 1, 1, 1) if do_cfg else latents

            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                out_cond, out_uncond = model_output.chunk(2)
                model_output = cfg_scale * (out_cond - out_uncond) + out_uncond

            # step() returns the denoised latent for this timestep
            latents = sampler.step(timestep, latents, model_output)

            # capture AFTER the step.  Use normalised-latent visualisation for
            # intermediate frames (no VAE decode) so noisy latents render as
            # meaningful colour images rather than static.
            if i in snap_set:
                is_last = (i == len(sampler.timesteps) - 1)
                snapshots.append(
                    latent_to_pil(latents, models["decoder"], device,
                                  decode_with_vae=is_last)
                )

        diffusion.to(IDLE_DEVICE)

        # If the very last step wasn't in snap_set, append a proper final frame.
        if (len(sampler.timesteps) - 1) not in snap_set:
            snapshots.append(
                latent_to_pil(latents, models["decoder"], device,
                               decode_with_vae=True)
            )

        # ── Decode final image ────────────────────────────────────────────
        decoder = models["decoder"]
        decoder.to(device)
        images  = decoder(latents)
        decoder.to(IDLE_DEVICE)

        images = pipeline.rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
        return images[0], snapshots


# ═════════════════════════════════════════════════════════════════════════════
#  PLOT HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def save_denoising_strip(snapshots, labels, title, out_path):
    """Save a horizontal strip of denoising snapshots."""
    n = len(snapshots)
    px = 1 / 100  # 100 dpi below → each inch = 100px
    fig, axes = plt.subplots(1, n, figsize=(n * 512 * px, 512 * px + 0.7))
    fig.suptitle(title, fontsize=11, y=1.02)

    for ax, img, lbl in zip(axes, snapshots, labels):
        ax.imshow(img)
        ax.set_title(lbl, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path}")


def save_side_by_side(img_lin, img_cos, prompt, out_path):
    """Save linear vs cosine final images side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.24, 5.5))
    fig.suptitle(f'"{prompt[:60]}"', fontsize=9)

    ax1.imshow(img_lin)
    ax1.set_title("Linear schedule", fontsize=9)
    ax1.axis("off")

    ax2.imshow(img_cos)
    ax2.set_title("Cosine schedule", fontsize=9)
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path}")


def save_i2i_forward_strip(original_pil, schedule, out_path, num_steps=1000,
                            show_steps=None):
    """
    Visualise the *forward* noising process (q(x_t | x_0)) for a given
    noise schedule applied to the pixel image (not latent space).
    """
    if show_steps is None:
        show_steps = [0, 100, 250, 500, 750, 999]

    img_np  = np.array(original_pil.resize((512, 512))).astype(np.float32) / 255.0
    x0      = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0)   # (1,3,512,512)

    sampler = DDPMSampler(torch.Generator(), schedule=schedule)
    ac      = sampler.alphas_cumprod

    frames, labels = [], []
    for t in show_steps:
        sqrt_a  = ac[t].sqrt()
        sqrt_1a = (1 - ac[t]).sqrt()
        noise   = torch.randn_like(x0)
        xt      = (sqrt_a * x0 + sqrt_1a * noise).clamp(0, 1)
        arr     = (xt.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frames.append(Image.fromarray(arr))
        labels.append(f"t={t}")

    save_denoising_strip(
        frames, labels,
        title=f"Forward noise process — {schedule} schedule",
        out_path=out_path,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 0. Schedule comparison curve ─────────────────────────────────────────
    print("\n=== Schedule comparison curve ===")
    plot_schedule_curves(out_path=os.path.join(OUTPUT_DIR, "schedule_comparison.png"))

    # ── Load models once ─────────────────────────────────────────────────────
    print("\n=== Loading models ===")
    tokenizer = CLIPTokenizer(TOKENIZER_VOCAB, merges_file=TOKENIZER_MERGES)
    models    = model_loader.preload_models_from_standard_weights(MODEL_FILE, DEVICE)

    step_labels = [f"step {i}" for i in SNAPSHOT_INDICES]

    # ── 1. Text-to-image ablation ─────────────────────────────────────────────
    print("\n=== Text-to-Image Ablation ===")
    for prompt in T2I_PROMPTS:
        print(f"\n  Prompt: {prompt[:60]}")
        final_imgs = {}
        for sched in ("linear", "cosine"):
            print(f"    schedule={sched} …", end=" ", flush=True)
            final_img, snaps = generate_with_snapshots(
                prompt        = prompt,
                uncond_prompt = UNCOND_PROMPT,
                input_image   = None,
                strength      = 1.0,
                do_cfg        = True,
                cfg_scale     = CFG_SCALE,
                schedule      = sched,
                n_steps       = NUM_INFERENCE_STEPS,
                seed          = SEED,
                models        = models,
                tokenizer     = tokenizer,
                device        = DEVICE,
                snapshot_indices = SNAPSHOT_INDICES,
            )
            final_imgs[sched] = Image.fromarray(final_img)
            print("done")

            out = os.path.join(OUTPUT_DIR, f"t2i_{slug(prompt)}_{sched}.png")
            save_denoising_strip(
                snaps, step_labels,
                title=f"T2I | {sched} schedule | {prompt[:50]}",
                out_path=out,
            )

        # Side-by-side comparison
        save_side_by_side(
            final_imgs["linear"], final_imgs["cosine"], prompt,
            out_path=os.path.join(OUTPUT_DIR, f"t2i_comparison_{slug(prompt)}.png"),
        )

    # ── 2. Image-to-image ablation ────────────────────────────────────────────
    print("\n=== Image-to-Image Ablation ===")

    if not os.path.exists(DOG_IMAGE_PATH):
        print(f"  [!] Dog image not found at '{DOG_IMAGE_PATH}'. "
              "Creating a placeholder (grey square).")
        placeholder = Image.fromarray(
            (np.ones((512, 512, 3)) * 128).astype(np.uint8)
        )
        os.makedirs(os.path.dirname(DOG_IMAGE_PATH), exist_ok=True)
        placeholder.save(DOG_IMAGE_PATH)

    dog_image = Image.open(DOG_IMAGE_PATH).convert("RGB")

    # 2a. Forward noising strips for the dog
    print("\n  Forward noise strips for input image …")
    for sched in ("linear", "cosine"):
        out = os.path.join(OUTPUT_DIR, f"i2i_forward_noise_{sched}.png")
        save_i2i_forward_strip(dog_image, sched, out_path=out)

    # 2b. Denoising ablation for each I2I prompt
    for prompt in I2I_PROMPTS:
        print(f"\n  Prompt: {prompt[:60]}")
        final_imgs = {}
        for sched in ("linear", "cosine"):
            print(f"    schedule={sched} …", end=" ", flush=True)
            final_img, snaps = generate_with_snapshots(
                prompt        = prompt,
                uncond_prompt = UNCOND_PROMPT,
                input_image   = dog_image,
                strength      = I2I_STRENGTH,
                do_cfg        = True,
                cfg_scale     = CFG_SCALE,
                schedule      = sched,
                n_steps       = NUM_INFERENCE_STEPS,
                seed          = SEED,
                models        = models,
                tokenizer     = tokenizer,
                device        = DEVICE,
                snapshot_indices = SNAPSHOT_INDICES,
            )
            final_imgs[sched] = Image.fromarray(final_img)
            print("done")

            out = os.path.join(OUTPUT_DIR, f"i2i_{slug(prompt)}_{sched}.png")
            save_denoising_strip(
                snaps, step_labels,
                title=f"I2I | {sched} schedule | {prompt[:50]}",
                out_path=out,
            )

        save_side_by_side(
            final_imgs["linear"], final_imgs["cosine"], prompt,
            out_path=os.path.join(OUTPUT_DIR, f"i2i_comparison_{slug(prompt)}.png"),
        )

    print(f"\n✓ All ablation results saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()