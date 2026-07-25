from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion
import model_converter

def preload_models_from_standard_weights(ckpt_path, device):
    state_diict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_diict["encoder"], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_diict["decoder"], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_diict["diffusion"], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_diict["clip"], strict=True)

    return {
        "clip":clip,
        "encoder":encoder,
        "decoder":decoder,
        "diffusion":diffusion
    }

