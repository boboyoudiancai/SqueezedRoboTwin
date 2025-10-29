"""
Squeezed Diffusion Policy with PCA-based Noise Squeezing.

This policy extends the standard Diffusion Policy by optionally applying
PCA-based noise squeezing transformations during training and inference.

Usage:
    # Standard mode (identical to DiffusionUnetImagePolicy)
    policy = DiffusionUnetSqueezedImagePolicy(..., noise_squeezer=None)

    # Squeezed mode
    from diffusion_policy.model.diffusion.noise_squeezer import NoiseSqueezer
    squeezer = NoiseSqueezer(pca_dir='./pca_results', squeeze_strength=-0.8)
    policy = DiffusionUnetSqueezedImagePolicy(..., noise_squeezer=squeezer)
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.diffusion.noise_squeezer import NoiseSqueezer


class DiffusionUnetSqueezedImagePolicy(BaseImagePolicy):
    """
    Diffusion-based policy with optional noise squeezing.

    This policy extends the standard DDPM formulation by applying PCA-based
    noise squeezing transformations. When noise_squeezer is None, it behaves
    identically to DiffusionUnetImagePolicy.

    Args:
        shape_meta: Dictionary containing shape information for observations and actions.
        noise_scheduler: DDPMScheduler for the diffusion process.
        obs_encoder: Vision encoder for processing image observations.
        horizon: Prediction horizon length.
        n_action_steps: Number of action steps to execute.
        n_obs_steps: Number of observation steps to condition on.
        num_inference_steps: Number of denoising steps during inference.
        obs_as_global_cond: If True, use observations as global conditioning.
        diffusion_step_embed_dim: Dimension of diffusion timestep embedding.
        down_dims: Channel dimensions for U-Net downsampling blocks.
        kernel_size: Convolution kernel size.
        n_groups: Number of groups for GroupNorm.
        cond_predict_scale: If True, use FiLM with scale prediction.
        noise_squeezer: Optional NoiseSqueezer instance for PCA-based squeezing.
            If None, behaves as standard DDPM (default).
        **kwargs: Additional arguments passed to scheduler.step().

    Attributes:
        noise_squeezer: NoiseSqueezer instance or None.
    """

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        noise_squeezer: Optional[NoiseSqueezer] = None,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # Store noise squeezer for optional PCA-based squeezing
        self.noise_squeezer = noise_squeezer
        if noise_squeezer is not None:
            print(f"[DiffusionUnetSqueezedImagePolicy] Noise squeezing enabled")
            print(f"  Squeeze strength: {noise_squeezer.squeeze_strength}")
            print(f"  Quantum limited: {noise_squeezer.quantum_limited}")
        else:
            print(f"[DiffusionUnetSqueezedImagePolicy] Standard DDPM mode (no squeezing)")

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        """
        Conditional sampling with optional noise squeezing.

        If noise_squeezer is not None, applies PCA-based squeezing to the
        initial noise before denoising. This reduces variance along the
        principal component direction of the action space.

        Args:
            condition_data: Conditioning data (e.g., observed actions).
            condition_mask: Binary mask indicating which elements are conditioned.
            local_cond: Local conditioning features.
            global_cond: Global conditioning features (e.g., from observations).
            generator: Random generator for reproducibility.
            **kwargs: Additional arguments for scheduler.step().

        Returns:
            Denoised trajectory (action sequence).
        """
        model = self.model
        scheduler = self.noise_scheduler

        # Sample initial noise
        noise = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        # Apply noise squeezing if enabled
        if self.noise_squeezer is not None:
            # Get batch size and create timestep tensor at maximum timestep
            batch_size = noise.shape[0]
            # Initialize at t_max for maximum squeezing strength
            t_max = scheduler.config.num_train_timesteps - 1
            timesteps_init = torch.full(
                (batch_size,),
                t_max,
                dtype=torch.long,
                device=noise.device
            )

            # Apply squeezing transformation
            noise = self.noise_squeezer.squeeze_noise(
                noise, timesteps_init, scheduler
            )

        trajectory = noise

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator, **kwargs).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the trajectory
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=trajectory.device,
        ).long()

        # Apply noise squeezing if enabled
        # The model will learn to predict the squeezed noise
        if self.noise_squeezer is not None:
            noise_squeezed = self.noise_squeezer.squeeze_noise(
                noise, timesteps, self.noise_scheduler
            )
        else:
            noise_squeezed = noise

        # Add noise to the clean trajectory according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise_squeezed, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            # Target is the squeezed noise (if squeezing is enabled)
            # Model learns to predict S(t) @ ε instead of ε
            target = noise_squeezed
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss
