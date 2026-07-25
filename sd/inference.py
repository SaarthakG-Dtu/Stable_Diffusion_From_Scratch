import os
import model_loader
import pipeline
from PIL import Image
from transformers import CLIPTokenizer
import torch


# --------------------------------------------------
# Device Configuration
# --------------------------------------------------

DEVICE = "cpu"

ALLOW_CUDA = True
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif torch.backends.mps.is_available() and ALLOW_MPS:
    DEVICE = "mps"

print(f"Using Device: {DEVICE}")


# --------------------------------------------------
# Load Tokenizer and Model
# --------------------------------------------------

tokenizer = CLIPTokenizer(
    r"E:\Stable_Diffusion\data\vocab.json",
    merges_file=r"E:\Stable_Diffusion\data\merges.txt"
)

model_file = r"E:\Stable_Diffusion\data\v1-5-pruned-emaonly.ckpt"

models = model_loader.preload_models_from_standard_weights(
    model_file,
    DEVICE
)


# --------------------------------------------------
# Text-to-Image
# --------------------------------------------------

prompt = "A cinematic, ultra-detailed scene of a small orange tabby cat sitting on a wooden windowsill during a rainy evening. The cat is gently stretching its front paws forward while looking outside through a large glass window covered with tiny raindrops. cinematic color grading, 8K resolution."

uncond_prompt = "zoom out"

# Classifier-Free Guidance
do_cfg = True
cfg_scale = 7


# --------------------------------------------------
# Image-to-Image
# --------------------------------------------------

input_image = None

# Uncomment these lines if you want to use an input image
# image_path = "../images/input.jpg"
# input_image = Image.open(image_path)

strength = 0.9


# --------------------------------------------------
# Generation Settings
# --------------------------------------------------

sampler = "ddpm"
num_inference_steps = 50
seed = 42


# --------------------------------------------------
# Generate Image
# --------------------------------------------------

output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer
)


# --------------------------------------------------
# Save Output Image
# --------------------------------------------------

# Create images folder in E:\Stable_Diffusion
output_folder = r"E:\Stable_Diffusion\images"

os.makedirs(output_folder, exist_ok=True)

# Define output file
output_path = os.path.join(
    output_folder,
    "generated_image.png"
)

# Convert NumPy array to PIL Image and save
image = Image.fromarray(output_image)

image.save(output_path)

print(f"Image saved successfully at:")
print(output_path)
