"""
Noise Squeezer for Squeezed Diffusion Policy.

This module implements PCA-based noise squeezing for diffusion models,
inspired by quantum squeezing in optics. The squeezer applies anisotropic
noise transformations along the principal component directions of the action space.

Reference:
    Squeezed Diffusion Models (arxiv.org/abs/2502.xxxxx)

Usage:
    squeezer = NoiseSqueezer(
        pca_dir='/path/to/pca/results',
        squeeze_strength=-0.8
    )
    noise_squeezed = squeezer.squeeze_noise(noise, timesteps, scheduler)
"""

import os
from typing import Optional, Tuple, Union
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn as nn


class NoiseSqueezer(nn.Module):
    """
    PCA-based noise squeezing for diffusion models in action space.

    This class loads pre-computed PCA components from action data and applies
    time-dependent squeezing transformations to noise during training and inference.

    The squeezing operation reduces variance along the principal component direction
    (typically the direction of maximum action variation) and optionally increases
    variance in orthogonal directions to preserve volume.

    Mathematical formulation:
        For action space noise ε ~ N(0, I) of dimension D:
        - Principal direction: v_max (eigenvector of largest eigenvalue)
        - Projection matrix: P = v_max ⊗ v_max
        - Squeeze matrix: S(t) = exp(-r(t)) * P + exp(q(t)) * (I - P)
        - Squeezed noise: ε_sq = S(t) @ ε

    where r(t) is time-dependent squeeze strength and q(t) = r(t)/(D-1) for
    volume-preserving squeezing.

    Attributes:
        squeeze_strength (float): Base squeezing strength. Default 0.0 (standard diffusion).
            Negative values squeeze along principal component.
            Typical range: [-1.5, 0.0]
        quantum_limited (bool): If True, use volume-preserving squeeze (det(S)=1).
        action_dim (int): Dimensionality of action space.
        eigenvalues (torch.Tensor): PCA eigenvalues [action_dim,], sorted ascending.
        eigenvectors (torch.Tensor): PCA eigenvectors [action_dim, action_dim].
        principal_direction (torch.Tensor): Largest eigenvector [action_dim,].
    """

    def __init__(
        self,
        pca_dir: str,
        squeeze_strength: float = 0.0,
        quantum_limited: bool = False,
    ):
        """
        Initialize NoiseSqueezer with PCA statistics from pre-computed results.

        Args:
            pca_dir: Directory containing PCA results from pca_analysis.py.
                Must contain:
                - pca_components_frames.npy: PCA components [action_dim, action_dim]
                - pca_variance_frames.npy: Explained variance ratios [action_dim,]
                Action dimension is automatically inferred from PCA data.
            squeeze_strength: Squeezing strength parameter.
                - 0.0: Standard DDPM (no squeezing)
                - Negative: Squeeze along principal component (typical)
                - Positive: Anti-squeeze (not recommended)
                Range: [-2.0, 2.0], typical: [-1.0, -0.5]
            quantum_limited: Whether to use volume-preserving squeezing.
                - False: Simple squeeze S = exp(-r)*P + (I-P)
                - True: Volume-preserving S = exp(-r)*P + exp(r/(D-1))*(I-P)

        Raises:
            FileNotFoundError: If PCA files are missing in pca_dir.
            ValueError: If PCA data dimensions are inconsistent.
            RuntimeError: If PCA data is corrupted or invalid.
        """
        super().__init__()

        self.squeeze_strength = squeeze_strength
        self.quantum_limited = quantum_limited

        # action_dim will be inferred from PCA data
        self.action_dim = None

        # Load and validate PCA results (sets self.action_dim)
        self._load_pca_statistics(pca_dir)

        # Precompute projection matrices
        self._precompute_projections()

    def _load_pca_statistics(self, pca_dir: str) -> None:
        """
        Load PCA components and variance from disk and infer action_dim.

        Expected file format (from pca_analysis.py):
            pca_components_frames.npy: [action_dim, action_dim] float32/float64
            pca_variance_frames.npy: [action_dim,] float32/float64

        Args:
            pca_dir: Directory path containing PCA results.

        Raises:
            FileNotFoundError: If required files are missing.
            ValueError: If dimensions are inconsistent or data is invalid.
        """
        pca_dir_path = Path(pca_dir)

        # Check directory existence
        if not pca_dir_path.exists():
            raise FileNotFoundError(
                f"PCA directory not found: {pca_dir}\n"
                f"Please run pca_analysis.py first to generate PCA statistics."
            )

        # File paths
        components_path = pca_dir_path / 'pca_components_frames.npy'
        variance_path = pca_dir_path / 'pca_variance_frames.npy'

        # Check file existence
        missing_files = []
        if not components_path.exists():
            missing_files.append('pca_components_frames.npy')
        if not variance_path.exists():
            missing_files.append('pca_variance_frames.npy')

        if missing_files:
            raise FileNotFoundError(
                f"Missing PCA files in {pca_dir}:\n" +
                "\n".join(f"  - {f}" for f in missing_files) +
                f"\n\nExpected files generated by pca_analysis.py:\n"
                f"  - pca_components_frames.npy: PCA components\n"
                f"  - pca_variance_frames.npy: Explained variance ratios"
            )

        # Load PCA components
        try:
            components = np.load(str(components_path))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load PCA components from {components_path}: {e}"
            )

        # Load variance
        try:
            variance = np.load(str(variance_path))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load PCA variance from {variance_path}: {e}"
            )

        # Validate dimensions
        if components.ndim != 2:
            raise ValueError(
                f"PCA components must be 2D, got shape {components.shape}"
            )

        if components.shape[0] != components.shape[1]:
            raise ValueError(
                f"PCA components must be square, got shape {components.shape}"
            )

        # Infer action_dim from PCA data
        inferred_action_dim = components.shape[0]
        self.action_dim = inferred_action_dim

        if variance.shape[0] != self.action_dim:
            raise ValueError(
                f"PCA variance shape mismatch: components suggest action_dim={self.action_dim}, "
                f"but variance has shape {variance.shape[0]}"
            )

        # Check for NaN or Inf
        if not np.isfinite(components).all():
            raise ValueError("PCA components contain NaN or Inf values")
        if not np.isfinite(variance).all():
            raise ValueError("PCA variance contains NaN or Inf values")

        # Check variance is non-negative
        if (variance < 0).any():
            raise ValueError(
                f"PCA variance must be non-negative, got min={variance.min()}"
            )

        # Convert to torch tensors (CPU first, will move to device during forward)
        self.register_buffer(
            'eigenvectors',
            torch.from_numpy(components.T).float()
        )
        self.register_buffer(
            'eigenvalues',
            torch.from_numpy(variance).float()
        )

        # Extract principal direction (largest eigenvalue)
        # sklearn.PCA stores components in descending order of explained variance
        # So components[0] is PC1 (largest eigenvalue)
        self.register_buffer(
            'principal_direction',
            torch.from_numpy(components[0]).float()
        )

        # Log statistics
        print(f"[NoiseSqueezer] Loaded PCA statistics from {pca_dir}")
        print(f"  Action dimension: {self.action_dim} (auto-inferred)")
        print(f"  PC1 explained variance: {variance[0]:.4f}")
        print(f"  Squeeze strength: {self.squeeze_strength}")
        print(f"  Quantum limited: {self.quantum_limited}")

        # Warn if PC1 doesn't explain much variance
        if variance[0] < 0.1:
            warnings.warn(
                f"PC1 explains only {variance[0]:.1%} of variance. "
                f"Squeezing may not be effective.",
                UserWarning
            )

    def _precompute_projections(self) -> None:
        """
        Precompute projection matrices for efficiency.

        Computes:
            - P: Projection onto principal direction (v_max ⊗ v_max)
            - I - P: Projection onto orthogonal complement

        These are stored as buffers and automatically moved to correct device.
        """
        # P = v_max ⊗ v_max (outer product)
        # Shape: [action_dim, action_dim]
        principal = self.principal_direction.view(-1, 1)  # [D, 1]
        self.register_buffer(
            'projection_principal',
            torch.mm(principal, principal.t())  # [D, D]
        )

        # I - P (orthogonal projection)
        identity = torch.eye(
            self.action_dim,
            dtype=torch.float32
        )
        self.register_buffer(
            'projection_orthogonal',
            identity - self.projection_principal
        )

    def get_squeeze_matrices(
        self,
        batch_size: int,
        timesteps: torch.Tensor,
        scheduler,
    ) -> torch.Tensor:
        """
        Compute time-dependent squeeze matrices S(t) for a batch.

        The squeeze strength scales with the noise schedule parameter β_t:
            r(t) = squeeze_strength * (β_t / β_max)

        This ensures:
            - Early timesteps (small β_t): weak squeezing
            - Late timesteps (large β_t): strong squeezing

        Args:
            batch_size: Number of samples in batch.
            timesteps: Timestep indices [batch_size,] in range [0, num_train_timesteps).
            scheduler: DDPMScheduler instance containing betas.

        Returns:
            Squeeze matrices [batch_size, action_dim, action_dim].

        Raises:
            RuntimeError: If timesteps are out of range.
        """
        device = timesteps.device

        # Validate timesteps
        if (timesteps < 0).any() or (timesteps >= len(scheduler.betas)).any():
            raise RuntimeError(
                f"Timesteps out of range: got [{timesteps.min()}, {timesteps.max()}], "
                f"expected [0, {len(scheduler.betas)})"
            )

        # Get betas for current timesteps
        betas = scheduler.betas.to(device)  # [num_train_timesteps,]
        beta_t = betas[timesteps]  # [batch_size,]
        beta_max = betas.max()

        # Time-dependent squeeze parameter
        # Scale by β_t / β_max so squeezing increases with noise level
        time_scale = beta_t / beta_max  # [batch_size,]
        squeeze_params = self.squeeze_strength * time_scale  # [batch_size,]

        # Move projection matrices to correct device
        P = self.projection_principal.to(device)  # [D, D]
        I_minus_P = self.projection_orthogonal.to(device)  # [D, D]

        if self.quantum_limited:
            # Volume-preserving squeeze: det(S) = 1
            # S = exp(-r) * P + exp(r/(D-1)) * (I-P)
            n = self.action_dim
            r = squeeze_params  # [batch_size,]
            q = r / (n - 1)  # [batch_size,]

            squeeze_factor = torch.exp(-r).view(-1, 1, 1)  # [batch_size, 1, 1]
            antisqueeze_factor = torch.exp(q).view(-1, 1, 1)  # [batch_size, 1, 1]

            # Broadcast: [batch_size, 1, 1] * [D, D] -> [batch_size, D, D]
            S = squeeze_factor * P.unsqueeze(0) + antisqueeze_factor * I_minus_P.unsqueeze(0)

        else:
            # Simple squeeze (not volume-preserving)
            # S = exp(-r) * P + (I-P)
            r = squeeze_params  # [batch_size,]
            squeeze_factor = torch.exp(-r).view(-1, 1, 1)  # [batch_size, 1, 1]

            S = squeeze_factor * P.unsqueeze(0) + I_minus_P.unsqueeze(0)

        return S  # [batch_size, action_dim, action_dim]

    def apply_squeeze(
        self,
        noise: torch.Tensor,
        squeeze_matrices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply squeeze transformation to noise.

        For noise ε with shape [B, T, D] or [B, D]:
            ε_squeezed = S @ ε

        Handles both temporal action sequences [B, T, D] and single actions [B, D].

        Args:
            noise: Input noise [batch_size, (horizon), action_dim].
            squeeze_matrices: Squeeze matrices [batch_size, action_dim, action_dim].

        Returns:
            Squeezed noise with same shape as input.

        Raises:
            ValueError: If shapes are incompatible.
        """
        original_shape = noise.shape
        batch_size = noise.shape[0]

        # Validate batch size
        if squeeze_matrices.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch: noise has {batch_size}, "
                f"squeeze_matrices has {squeeze_matrices.shape[0]}"
            )

        # Validate action dimension
        action_dim = original_shape[-1]
        if action_dim != self.action_dim:
            raise ValueError(
                f"Action dimension mismatch: expected {self.action_dim}, "
                f"got {action_dim}"
            )

        # Reshape noise to [batch_size, -1, action_dim]
        if noise.ndim == 2:
            # [B, D] -> [B, 1, D]
            noise_reshaped = noise.unsqueeze(1)
            squeeze_2d = True
        elif noise.ndim == 3:
            # [B, T, D] -> [B, T, D]
            noise_reshaped = noise
            squeeze_2d = False
        else:
            raise ValueError(
                f"Noise must be 2D [B, D] or 3D [B, T, D], got shape {original_shape}"
            )

        # Apply squeeze: S @ ε
        # noise_reshaped: [B, T, D]
        # squeeze_matrices: [B, D, D]
        # Result: [B, T, D]

        # Flatten temporal dimension for batch matrix multiply
        B, T, D = noise_reshaped.shape
        noise_flat = noise_reshaped.reshape(B, T, D)  # [B, T, D]

        # Transpose for bmm: [B, D, T]
        noise_flat_t = noise_flat.transpose(1, 2)  # [B, D, T]

        # S @ ε: [B, D, D] @ [B, D, T] -> [B, D, T]
        squeezed_flat_t = torch.bmm(squeeze_matrices, noise_flat_t)

        # Transpose back: [B, T, D]
        squeezed_flat = squeezed_flat_t.transpose(1, 2)

        # Reshape to original shape
        if squeeze_2d:
            # [B, 1, D] -> [B, D]
            squeezed_noise = squeezed_flat.squeeze(1)
        else:
            # [B, T, D] -> [B, T, D]
            squeezed_noise = squeezed_flat

        return squeezed_noise

    def squeeze_noise(
        self,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler,
    ) -> torch.Tensor:
        """
        Main interface: Generate and apply squeeze transformation to noise.

        This is the primary method to use during training and inference.
        If squeeze_strength is 0.0, returns original noise unchanged.

        Args:
            noise: Standard Gaussian noise [batch_size, (horizon), action_dim].
            timesteps: Diffusion timesteps [batch_size,].
            scheduler: DDPMScheduler instance.

        Returns:
            Squeezed noise with same shape as input.

        Example:
            >>> squeezer = NoiseSqueezer(pca_dir='./pca_results', squeeze_strength=-0.8)
            >>> noise = torch.randn(32, 8, 14)  # [batch, horizon, action_dim]
            >>> timesteps = torch.randint(0, 100, (32,))
            >>> noise_squeezed = squeezer.squeeze_noise(noise, timesteps, scheduler)
        """
        # No-op if squeeze_strength is zero
        if self.squeeze_strength == 0.0:
            return noise

        batch_size = noise.shape[0]

        # Compute squeeze matrices
        squeeze_matrices = self.get_squeeze_matrices(
            batch_size=batch_size,
            timesteps=timesteps,
            scheduler=scheduler,
        )

        # Apply squeeze transformation
        squeezed_noise = self.apply_squeeze(noise, squeeze_matrices)

        return squeezed_noise

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"action_dim={self.action_dim}, "
            f"squeeze_strength={self.squeeze_strength}, "
            f"quantum_limited={self.quantum_limited}"
        )
