# Stable Diffusion — From Scratch

A clean, fully explicit PyTorch implementation of Stable Diffusion v1.5. Every component — the VAE, the CLIP text encoder, the UNet denoiser, and the DDPM sampler — is written from first principles without hiding logic behind library abstractions. Built for learning, experimentation, and ablation studies.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Setup](#setup)
3. [Theoretical Background](#theoretical-background)
   - [Diffusion Models](#diffusion-models)
   - [The Forward Process](#the-forward-process)
   - [The Reverse Process](#the-reverse-process)
   - [Noise Schedules](#noise-schedules)
   - [Classifier-Free Guidance](#classifier-free-guidance)
   - [Latent Diffusion](#latent-diffusion)
4. [Architecture](#architecture)
   - [CLIP Text Encoder](#clip-text-encoder)
   - [VAE Encoder / Decoder](#vae-encoder--decoder)
   - [UNet Denoiser](#unet-denoiser)
   - [DDPM Sampler](#ddpm-sampler)
5. [Running Inference](#running-inference)
6. [Noise Schedule Ablation Study](#noise-schedule-ablation-study)
   - [What the Ablation Measures](#what-the-ablation-measures)
   - [Schedule Comparison Curve](#schedule-comparison-curve)
   - [Text-to-Image Results](#text-to-image-results)
   - [Image-to-Image Results](#image-to-image-results)
   - [Denoising Progression Strips](#denoising-progression-strips)
   - [Customising the Ablation](#customising-the-ablation)
7. [References](#references)

---

## Repository Structure

```
.
├── attention.py        # SelfAttention and CrossAttention modules
├── clip.py             # CLIP text encoder (embedding + 12 transformer layers)
├── ddpm.py             # DDPM sampler — linear and cosine noise schedules
├── decoder.py          # VAE decoder  (latent → pixel space)
├── encoder.py          # VAE encoder  (pixel space → latent)
├── diffusion.py        # UNet with time embedding, residual blocks, cross-attention
├── model_converter.py  # Load weights from CompVis / HuggingFace .ckpt files
├── model_loader.py     # Assemble all four models from a single checkpoint
├── pipeline.py         # End-to-end generate() function (text-to-image & image-to-image)
├── inference.py        # Script for a single generation run
├── ablation.py         # Noise-schedule ablation study (linear vs cosine)
│
├── data/
│   ├── vocab.json                   # CLIP BPE vocabulary (49 408 entries)
│   ├── merges.txt                   # BPE merge rules
│   └── v1-5-pruned-emaonly.ckpt     # SD 1.5 weights — download separately
│
├── images/
│   └── dog.jpg                      # Input image for image-to-image ablation
│
└── ablation_results/                # Created automatically by ablation.py
    ├── schedule_comparison.png
    ├── t2i_<prompt-slug>_linear.png
    ├── t2i_<prompt-slug>_cosine.png
    ├── t2i_comparison_<prompt-slug>.png
    ├── i2i_forward_noise_linear.png
    ├── i2i_forward_noise_cosine.png
    ├── i2i_<prompt-slug>_linear.png
    ├── i2i_<prompt-slug>_cosine.png
    └── i2i_comparison_<prompt-slug>.png
```

---

## Setup

```bash
pip install torch torchvision transformers tqdm matplotlib pillow numpy
```

Download the SD 1.5 EMA-only weights from [HuggingFace (runwayml/stable-diffusion-v1-5)](https://huggingface.co/runwayml/stable-diffusion-v1-5) and place `v1-5-pruned-emaonly.ckpt` inside `data/`.

Download the CLIP tokenizer files (`vocab.json`, `merges.txt`) from the same repo and place them in `data/`.

---

## Theoretical Background

### Diffusion Models

Diffusion models are a class of generative models trained to learn the reverse of a noise-addition process. The core idea, introduced by Ho et al. (2020), is to define a **forward process** that gradually corrupts data into pure Gaussian noise over $T$ timesteps, and to train a neural network to invert that process — predicting the noise at each step so the original signal can be recovered.

Formally, given a data sample $\mathbf{x}_0 \sim q(\mathbf{x})$, the forward process defines a Markov chain of increasingly noisy latents $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T$:

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t;\; \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\; \beta_t \mathbf{I})$$

where $\{\beta_t\}_{t=1}^{T}$ is a fixed noise schedule. The key insight is that this chain can be sampled *in closed form* for any arbitrary $t$ without simulating all intermediate steps.

### The Forward Process

Define the cumulative noise product:

$$\bar{\alpha}_t = \prod_{s=1}^{t}(1 - \beta_s) = \prod_{s=1}^{t}\alpha_s$$

The forward process in closed form is then:

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t;\; \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\; (1-\bar{\alpha}_t)\mathbf{I})$$

which means a noisy sample at time $t$ can be sampled directly:

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

As $t \to T$, the signal term $\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0$ vanishes and $\mathbf{x}_T \approx \mathcal{N}(\mathbf{0},\mathbf{I})$ — pure noise. This is implemented in `ddpm.py` as `add_noise()`.

### The Reverse Process

The reverse process denoises step by step, starting from $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0},\mathbf{I})$:

$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1};\; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\; \sigma_t^2 \mathbf{I})$$

The network $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ predicts the noise $\boldsymbol{\epsilon}$ added at step $t$. The posterior mean is derived analytically from Bayes' theorem (DDPM, Eq. 7):

$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\,\hat{\mathbf{x}}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,\mathbf{x}_t$$

where the predicted clean sample is:

$$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}$$

The posterior variance is:

$$\sigma_t^2 = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\,\beta_t$$

At each step, a small noise term is added (except at $t=1$) to match the stochastic nature of the true posterior. This entire step is `DDPMSampler.step()` in `ddpm.py`.

**Training objective.** The simplified DDPM loss is a reweighted ELBO that reduces to noise prediction:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}\!\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon},\; t)\|^2\right]$$

This is equivalent to training the network to denoise a randomly corrupted version of $\mathbf{x}_0$ at a uniformly sampled noise level.

### Noise Schedules

The noise schedule $\{\beta_t\}$ controls how fast signal is destroyed. This implementation supports two schedules:

#### Linear Schedule (SD 1.5 default)

Betas grow linearly in square-root space between `beta_start = 8.5e-4` and `beta_end = 0.012`:

```python
betas = torch.linspace(beta_start**0.5, beta_end**0.5, T)**2
```

This gives a roughly linear $\bar{\alpha}_t$ decay. The schedule tends to destroy most structure during the middle timesteps, which can cause the network to see very little clean signal at low noise levels.

#### Cosine Schedule (Nichol & Dhariwal, 2021)

Defines $\bar{\alpha}_t$ directly as a cosine curve:

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \qquad f(t) = \cos\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

with offset $s = 0.008$ to prevent $\beta_t$ from becoming too small near $t = 0$ (which would make the model waste capacity predicting essentially-clean data). Betas are then derived from the cumulative product and clipped at 0.999:

```python
betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
betas = betas.clamp(max=0.999)
```

The cosine schedule degrades structure more gently at both extremes — the signal survives longer at mid-noise levels compared to the linear schedule. This is especially important for complex high-frequency images.

Both schedules share identical API; swap by passing `schedule="linear"` or `schedule="cosine"` to `DDPMSampler`.

### Classifier-Free Guidance

Standard conditional diffusion trains $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c)$ where $c$ is the conditioning signal (here, CLIP text embeddings). Classifier-Free Guidance (CFG; Ho & Salimans, 2022) improves sample quality by jointly training a conditional and unconditional model with a single network. During training, the conditioning is randomly dropped with probability $p_{\text{uncond}}$, forcing the network to also learn $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset)$.

At inference, the conditional and unconditional noise predictions are combined:

$$\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, c) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset) + w\,\bigl[\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset)\bigr]$$

where $w$ is the **guidance scale** (`cfg_scale`). A higher $w$ pushes the sample further in the direction of the conditioning, increasing prompt adherence at the cost of diversity and sometimes image naturalness. Values in the range $7$–$12$ are typical for Stable Diffusion.

In practice the two forward passes are batched:

```python
# pipeline.py
model_input = latents.repeat(2, 1, 1, 1)          # [cond; uncond] batch
model_output = diffusion(model_input, context, time_embedding)
out_cond, out_uncond = model_output.chunk(2)
model_output = cfg_scale * (out_cond - out_uncond) + out_uncond
```

### Latent Diffusion

Running the diffusion process directly in pixel space is prohibitively expensive at high resolutions. Rombach et al. (2022) introduced **Latent Diffusion Models (LDM)**, which first encode an image into a compressed latent representation using a VAE, run diffusion in that lower-dimensional space, and decode back to pixels only at the very end.

For a 512×512 RGB image the spatial compression factor is 8×, so the UNet operates on 64×64×4 latents — roughly **48× fewer tokens** than pixel-space diffusion. This makes high-resolution generation tractable on consumer hardware.

The latent is scaled by the SD convention factor 0.18215 before being passed to the UNet:

```python
# encoder.py
x *= 0.18215
```

and divided again before decoding:

```python
# decoder.py
x /= 0.18215
```

This ensures the latent variance is roughly unit-scale, matching the Gaussian noise added during training.

---

## Architecture

### CLIP Text Encoder

The text encoder is a standard CLIP ViT-L/14 transformer (`clip.py`):

| Parameter | Value |
|---|---|
| Vocabulary size | 49 408 (BPE) |
| Sequence length | 77 tokens |
| Embedding dimension | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| Activation | QuickGELU: $x \cdot \sigma(1.702x)$ |
| Attention masking | Causal (autoregressive) |

Tokens are summed with learned positional embeddings, passed through 12 transformer layers with pre-normalisation and causal self-attention, and layer-normalised at the output. The resulting (batch, 77, 768) context tensor is passed to every cross-attention layer of the UNet.

During classifier-free guidance, both the conditional prompt and the unconditional (empty) prompt are encoded and concatenated into a single (2·batch, 77, 768) tensor for a single UNet forward pass.

### VAE Encoder / Decoder

The VAE maps 512×512×3 pixel images to 64×64×4 latents (`encoder.py`, `decoder.py`).

**Encoder** — a convolutional downsampling network:

```
(3, 512, 512) → Conv → ResBlocks → ↓2 → ResBlocks → ↓2 → ResBlocks → ↓2
→ ResBlocks → AttentionBlock → ResBlocks → GroupNorm → SiLU
→ Conv → (8, 64, 64) → chunk → mean (4,64,64), log_var (4,64,64)
```

The reparameterisation trick samples the latent: $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \cdot \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$, then scales by 0.18215.

**Decoder** — mirror of the encoder with transposed upsampling:

```
(4, 64, 64) → /0.18215 → Conv → ResBlocks → AttentionBlock → ResBlocks
→ ↑2 → ResBlocks → ↑2 → ResBlocks → ↑2 → GroupNorm → SiLU → Conv → (3, 512, 512)
```

**Residual Block** — GroupNorm(32) → SiLU → Conv3×3 → GroupNorm(32) → SiLU → Conv3×3, with an identity or 1×1 projection shortcut.

**Attention Block** — 1-head self-attention over the spatial tokens of a feature map. The (C, H, W) feature is reshaped to (H·W, C), attended over, and reshaped back.

### UNet Denoiser

The core of the diffusion model is a time-conditioned UNet (`diffusion.py`) operating in latent space:

```
Latent (4, 64, 64) + time embedding (320,) + CLIP context (77, 768)
  ↓
Encoder:  SwitchSequential blocks at resolutions 64, 32, 16, 8
  - ResidualBlock(time_emb) + AttentionBlock(cross_attn to context)
  - Downsampling Conv at each spatial transition
  ↓
Bottleneck: ResBlock + AttentionBlock + ResBlock
  ↓
Decoder:  SwitchSequential blocks at 8, 16, 32, 64 (with skip connections)
  - Concatenate skip from encoder before each block
  - ResidualBlock(time_emb) + AttentionBlock(cross_attn)
  - Upsampling at each transition
  ↓
Output Conv → predicted noise (4, 64, 64)
```

**Time embedding.** The integer timestep $t$ is encoded into sinusoidal features, then projected through two linear layers with SiLU:

$$\text{TE}(t) = \text{Linear}(\text{SiLU}(\text{Linear}(\text{sinusoidal}(t)))) \in \mathbb{R}^{1280}$$

Each residual block adds the time embedding to its hidden state after the first normalisation, conditioning the denoiser on the current noise level.

**Cross-attention conditioning.** Each attention block in the UNet attends over the CLIP context sequence using cross-attention (`attention.py`):

$$\text{CrossAttn}(\mathbf{x}, \mathbf{c}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

where $\mathbf{Q} = \mathbf{x}W_Q$ (from the UNet hidden state) and $\mathbf{K}, \mathbf{V} = \mathbf{c}W_K, \mathbf{c}W_V$ (from the CLIP text embedding). This is how the prompt semantics are injected at every layer and resolution of the UNet.

### DDPM Sampler

The sampler (`ddpm.py`) implements the standard DDPM reverse step (DDPM, Eq. 11):

```python
def step(self, timestep, latents, model_output):
    # Predicted clean latent
    pred_x0 = (latents - sqrt(beta_prod_t) * model_output) / sqrt(alpha_prod_t)
    # Posterior mean coefficients
    coeff_x0 = (sqrt(alpha_prod_t_prev) * beta_t) / beta_prod_t
    coeff_xt = sqrt(alpha_t) * beta_prod_t_prev / beta_prod_t
    # Posterior mean
    pred_prev = coeff_x0 * pred_x0 + coeff_xt * latents
    # Stochastic noise (skipped at t=0)
    if t > 0:
        variance = (sigma_t) * randn(...)
    return pred_prev + variance
```

**Inference timesteps.** Training uses $T=1000$ steps. At inference, a subset of $N$ steps (default 50) is selected by striding uniformly through the training timesteps. This sub-sampling does not require retraining.

---

## Running Inference

Edit the prompt and paths at the top of `inference.py`, then:

```bash
python inference.py
```

The generated image is saved to `images/generated_image.png`.

### Key Settings

| Variable | Description | Typical range |
|---|---|---|
| `prompt` | Text description of the image | — |
| `uncond_prompt` | Negative prompt (steer away from this) | `""` or descriptive |
| `cfg_scale` | Classifier-free guidance scale | 5–12 |
| `num_inference_steps` | Denoising steps | 20–50 |
| `strength` | Image-to-image re-noise amount (0 = no change, 1 = full T2I) | 0.5–0.9 |
| `seed` | Random seed for reproducibility | any integer |

For **image-to-image**, set `input_image = Image.open("path/to/image.jpg")` and adjust `strength`. A strength of 0.9 re-noises almost completely, giving the model near-full creative freedom; 0.3–0.5 preserves the original structure while adjusting style or content.

---

## Noise Schedule Ablation Study

`ablation.py` is a self-contained script that systematically compares the linear and cosine noise schedules across multiple prompts for both text-to-image (T2I) and image-to-image (I2I) generation.

```bash
python ablation.py
```

Place a dog image at `images/dog.jpg` before running (any JPEG/PNG works). If the file is missing, a grey placeholder is created automatically so the script still runs.

All output is written to `ablation_results/`.

### What the Ablation Measures

The two schedules differ only in the noise schedule $\{\beta_t\}$ — everything else (weights, prompts, seed, CFG scale) is held fixed. This isolates the effect of the noise schedule on:

- Final image quality and fidelity
- How structure emerges during the denoising process
- The shape of the $\bar{\alpha}_t$ curve (how aggressively the signal is destroyed at each noise level)
- The forward noising behaviour applied to a real image (I2I only)

### Schedule Comparison Curve

`ablation_results/schedule_comparison.png` — plots $\bar{\alpha}_t$ for both schedules over the full 1000 training steps.

The cosine curve decays more slowly through mid-noise levels and more steeply near $t=0$ and $t=T$ compared to the linear schedule. This means:

- At moderate noise levels the signal survives longer in the cosine schedule, giving the network more context to learn from.
- Near $t = 0$ (very clean images), the cosine schedule applies a small but non-trivial amount of noise (controlled by the offset $s=0.008$), preventing the network from wasting capacity on near-perfect images.

### Text-to-Image Results

For each prompt and each schedule, a horizontal denoising strip is saved alongside a side-by-side final image comparison.

**Denoising strip** (`t2i_<slug>_<schedule>.png`):

Seven snapshots are captured at steps 0, 8, 16, 24, 32, 40, and 49 of the 50-step inference chain. Intermediate snapshots (steps 0–40) show the raw latent channels normalised to [0,1] — this renders the latent content as a meaningful colour image even under high noise, making the progressive structure emergence visible. The final frame (step 49) is decoded through the full VAE for the true output image.

Example outputs for the prompt *"A futuristic city skyline at night with neon lights, cinematic"*:

```
ablation_results/
├── t2i_a_futuristic_city_skyline_at_night_with_neon_lights_linear.png
├── t2i_a_futuristic_city_skyline_at_night_with_neon_lights_cosine.png
└── t2i_comparison_a_futuristic_city_skyline_at_night_with_neon_lights.png
```

What to look for in the strips: the cosine schedule typically shows more legible structure earlier in the denoising chain (around steps 16–24), while the linear schedule tends to hold noise longer before resolving. Final image quality differences are usually subtle but observable in fine detail and high-frequency textures.

### Image-to-Image Results

**Forward noise strips** (`i2i_forward_noise_<schedule>.png`):

Shows the input dog image at increasing noise levels under $q(\mathbf{x}_t|\mathbf{x}_0)$: $t = 0, 100, 250, 500, 750, 999$. This directly visualises how each schedule destroys structure. The cosine schedule retains recognisable dog features further along the noise axis — you can typically still make out the shape at $t=250$–$t=500$, whereas the linear schedule produces near-pure noise earlier. This difference is what makes the cosine schedule preferable for image-to-image tasks with high strength, where the re-noised image needs to retain enough signal for the decoder to respect the original composition.

**Denoising strips** (`i2i_<slug>_<schedule>.png`):

Starting from the re-noised dog image (strength=0.75), the model denoises toward the conditioning prompt. The I2I strips show how well each schedule preserves the original image's compositional structure while transforming it according to the text.

Example for *"An oil painting of a dog in the style of Van Gogh"*:

```
ablation_results/
├── i2i_an_oil_painting_of_a_dog_in_the_style_of_van_gogh_linear.png
├── i2i_an_oil_painting_of_a_dog_in_the_style_of_van_gogh_cosine.png
└── i2i_comparison_an_oil_painting_of_a_dog_in_the_style_of_van_gogh.png
```

### Denoising Progression Strips

The strips are produced by `generate_with_snapshots()`, a mirrored version of `pipeline.generate()` that saves decoded latents at configurable step indices. Key implementation details:

**Why intermediate latents are not VAE-decoded.** The VAE decoder is trained to decode *clean* latents. Feeding a noisy intermediate latent produces visual garbage — pure colourful static — because the decoder has never seen noise-corrupted inputs during training. Instead, intermediate snapshots are visualised by normalising each of the 4 latent channels to [0,1] and treating channels 0–2 as RGB. This gives a meaningful colour picture showing structure emerge, with no decoder needed.

**Only the final step is VAE-decoded**, producing the true output image as the rightmost frame in each strip.

### Customising the Ablation

All settings live in the `CONFIG` block at the top of `ablation.py`:

| Variable | Purpose |
|---|---|
| `T2I_PROMPTS` | List of prompts for text-to-image ablation |
| `I2I_PROMPTS` | List of prompts for image-to-image ablation |
| `DOG_IMAGE_PATH` | Path to the input image for I2I |
| `SNAPSHOT_INDICES` | Which denoising step indices to capture (default: 0, 8, 16, 24, 32, 40, 49) |
| `I2I_STRENGTH` | How aggressively to re-noise the input image (0–1) |
| `CFG_SCALE` | Guidance scale applied during generation |
| `SEED` | Fixed seed — identical for both schedules, ensuring a fair comparison |
| `NUM_INFERENCE_STEPS` | Number of denoising steps (default: 50) |

### Full Output Listing

```
ablation_results/
├── schedule_comparison.png
│
├── t2i_a_serene_mountain_lake_at_sunset_photorealistic_linear.png
├── t2i_a_serene_mountain_lake_at_sunset_photorealistic_cosine.png
├── t2i_comparison_a_serene_mountain_lake_at_sunset_photorealistic.png
│
├── t2i_a_futuristic_city_skyline_at_night_with_neon_lights_linear.png
├── t2i_a_futuristic_city_skyline_at_night_with_neon_lights_cosine.png
├── t2i_comparison_a_futuristic_city_skyline_at_night_with_neon_lights.png
│
├── t2i_a_close-up_portrait_of_a_red_fox_in_a_snowy_forest_linear.png
├── t2i_a_close-up_portrait_of_a_red_fox_in_a_snowy_forest_cosine.png
├── t2i_comparison_a_close-up_portrait_of_a_red_fox_in_a_snowy_forest.png
│
├── i2i_forward_noise_linear.png
├── i2i_forward_noise_cosine.png
│
├── i2i_a_golden_retriever_wearing_a_red_scarf_linear.png
├── i2i_a_golden_retriever_wearing_a_red_scarf_cosine.png
├── i2i_comparison_a_golden_retriever_wearing_a_red_scarf.png
│
├── i2i_an_oil_painting_of_a_dog_in_the_style_of_van_gogh_linear.png
├── i2i_an_oil_painting_of_a_dog_in_the_style_of_van_gogh_cosine.png
└── i2i_comparison_an_oil_painting_of_a_dog_in_the_style_of_van_gogh.png
```

---

## References

- Ho, J., Jain, A., Abbeel, P. — *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.  
  The foundational DDPM paper. Derives the forward/reverse process, training objective, and the posterior mean formula implemented in `DDPMSampler.step()`.

- Nichol, A., Dhariwal, P. — *Improved Denoising Diffusion Probabilistic Models*, ICML 2021.  
  Introduces the cosine noise schedule and learned variance parameterisation. The cosine schedule in `ddpm.py` follows this paper exactly.

- Dhariwal, P., Nichol, A. — *Diffusion Models Beat GANs on Image Synthesis*, NeurIPS 2021.  
  Proposes classifier guidance; the CFG variant implemented here is the follow-up work.

- Ho, J., Salimans, T. — *Classifier-Free Diffusion Guidance*, NeurIPS Workshop 2021.  
  Introduces CFG — training a single network jointly for conditional and unconditional generation, and combining them at inference with a guidance scale $w$.

- Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B. — *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022.  
  Introduces LDM / Stable Diffusion. Moves the diffusion process into the compressed latent space of a pretrained VAE, enabling high-resolution synthesis at tractable cost.

- Radford, A. et al. — *Learning Transferable Visual Models from Natural Language Supervision*, ICML 2021.  
  The CLIP model used as the text encoder. The ViT-L/14 transformer in `clip.py` follows this architecture.