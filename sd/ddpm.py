import torch
import numpy as np
import math


class DDPMSampler:

    def __init__(self, generator: torch.Generator, num_training_steps=1000,
                 beta_start: float = 0.00085, beta_end: float = 0.0120,
                 schedule: str = "linear"):
        """
        Args:
            generator:            torch.Generator for reproducibility
            num_training_steps:   total diffusion steps (T)
            beta_start / beta_end: only used for the linear schedule
            schedule:             "linear" or "cosine"
        """
        self.schedule = schedule

        if schedule == "linear":
            # Original SD-style linear schedule with sqrt-space betas
            self.betas = (
                torch.linspace(beta_start ** 0.5, beta_end ** 0.5,
                               num_training_steps, dtype=torch.float32) ** 2
            )
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, 0)

        elif schedule == "cosine":
            # Improved DDPM cosine schedule (Nichol & Dhariwal, 2021)
            # alpha_bar(t) = f(t)/f(0),  f(t) = cos((t/T + s) / (1 + s) * pi/2)^2
            s = 0.008
            steps = num_training_steps + 1
            t = torch.linspace(0, num_training_steps, steps, dtype=torch.float32)
            f = torch.cos(((t / num_training_steps) + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = f / f[0]
            # Derive betas from the cumprod
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = betas.clamp(max=0.999)
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = alphas_cumprod[1:]   # length == num_training_steps

        else:
            raise ValueError(f"Unknown schedule '{schedule}'. Choose 'linear' or 'cosine'.")

        self.one = torch.tensor(1.0)
        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(
            np.arange(0, num_training_steps)[::-1].copy()
        )

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = timestep - (self.num_training_steps // self.num_inference_steps)
        return prev_t

    def _get_variance(self, timestep: int) -> torch.Tensor:
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)
        return variance

    def set_strength(self, strength=1):
        """
        Set how much noise to add to the input image.
        More noise (strength ~ 1) means that the output will be further from the input image.
        Less noise (strength ~ 0) means that the output will be closer to the input image.
        """
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # Predicted x_0 — formula (15) from DDPM paper
        pred_original_sample = (latents - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # Coefficients for the posterior mean — formula (7)
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t

        # Posterior mean
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * latents
        )

        # Add stochastic noise (skip at t == 0)
        variance = 0
        if t > 0:
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            variance = (self._get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype,
        )
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples