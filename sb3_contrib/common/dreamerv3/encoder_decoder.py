"""
Encoder and Decoder networks for DreamerV3.

The Encoder converts observations to latent tokens that feed into the RSSM.
The Decoder reconstructs observations from latent features.
"""

from typing import Dict, Optional, Tuple

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces


class Encoder(nn.Module):
    """
    Encoder network that converts observations to latent tokens.
    
    Supports:
    - Vector observations: MLP encoder
    - Image observations: CNN encoder
    - Mixed observations: Separate encoders concatenated
    
    :param observation_space: Observation space
    :param hidden_dim: Hidden layer dimension (default: 1024)
    :param num_layers: Number of MLP layers for vectors (default: 3)
    :param activation: Activation function (default: nn.SiLU)
    :param symlog_transform: Whether to apply symlog transformation to inputs
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        activation: nn.Module = nn.SiLU,
        symlog_transform: bool = True,
    ):
        super().__init__()
        
        self.observation_space = observation_space
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.symlog_transform = symlog_transform
        
        # Determine observation type and build appropriate encoder
        if isinstance(observation_space, spaces.Box):
            if len(observation_space.shape) == 1:
                # Vector observation
                self.obs_type = 'vector'
                self.encoder = self._build_mlp_encoder(observation_space.shape[0])
            elif len(observation_space.shape) == 3:
                # Image observation
                self.obs_type = 'image'
                self.encoder = self._build_cnn_encoder(observation_space.shape)
            else:
                raise NotImplementedError(f"Observation shape {observation_space.shape} not supported")
        elif isinstance(observation_space, spaces.Dict):
            # Dictionary observation space
            self.obs_type = 'dict'
            self.encoder = self._build_dict_encoder(observation_space)
        else:
            raise NotImplementedError(f"Observation space {observation_space} not supported")
    
    def _build_mlp_encoder(self, input_dim: int) -> nn.Module:
        """Build MLP encoder for vector observations."""
        layers = []
        current_dim = input_dim
        
        for i in range(self.num_layers):
            layers.extend([
                nn.Linear(current_dim, self.hidden_dim),
                nn.RMSNorm(self.hidden_dim),
                self.activation(),
            ])
            current_dim = self.hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_cnn_encoder(self, input_shape: Tuple[int, int, int]) -> nn.Module:
        """
        Build CNN encoder for image observations.
        
        Architecture from DreamerV3:
        - 4 convolutional layers
        - Depths: [128, 192, 256, 256] (depth * [2, 3, 4, 4])
        - Kernel size: 5
        - Stride: 2
        - RMSNorm and GeLU activation
        """
        channels, height, width = input_shape
        
        # Define depths (base_depth=64, multipliers=[2,3,4,4])
        depths = [128, 192, 256, 256]
        
        layers = []
        in_channels = channels
        
        for depth in depths:
            layers.extend([
                nn.Conv2d(in_channels, depth, kernel_size=5, stride=2, padding=2),
                nn.GroupNorm(1, depth),  # Approximation of RMSNorm for conv
                self.activation(),
            ])
            in_channels = depth
        
        # Calculate output size after convolutions
        # Each layer reduces size by factor of 2 (stride=2)
        out_height = height // (2 ** len(depths))
        out_width = width // (2 ** len(depths))
        out_features = depths[-1] * out_height * out_width
        
        return nn.Sequential(
            *layers,
            nn.Flatten(),
            nn.Linear(out_features, self.hidden_dim),
        )
    
    def _build_dict_encoder(self, observation_space: spaces.Dict) -> nn.ModuleDict:
        """Build separate encoders for each key in dictionary observation space."""
        encoders = nn.ModuleDict()
        
        for key, space in observation_space.spaces.items():
            if isinstance(space, spaces.Box):
                if len(space.shape) == 1:
                    encoders[key] = self._build_mlp_encoder(space.shape[0])
                elif len(space.shape) == 3:
                    encoders[key] = self._build_cnn_encoder(space.shape)
        
        return encoders
    
    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Encode observations to tokens.
        
        :param obs: Observations (B, *obs_shape) or dict of observations
        :return: Tokens (B, hidden_dim)
        """
        if self.obs_type == 'vector':
            # Apply symlog if enabled
            if self.symlog_transform:
                obs = self.symlog(obs)
            tokens = self.encoder(obs)
            
        elif self.obs_type == 'image':
            # Normalize images from [0, 255] to [-0.5, 0.5]
            if obs.dtype == th.uint8:
                obs = obs.float() / 255.0 - 0.5
            tokens = self.encoder(obs)
            
        elif self.obs_type == 'dict':
            # Encode each modality and concatenate
            encoded = []
            for key, encoder in self.encoder.items():
                obs_key = obs[key]
                if self.symlog_transform and len(obs_key.shape) <= 2:
                    obs_key = self.symlog(obs_key)
                encoded.append(encoder(obs_key))
            tokens = th.cat(encoded, dim=-1)
        
        else:
            raise NotImplementedError(f"Observation type {self.obs_type} not supported")
        
        return tokens
    
    @staticmethod
    def symlog(x: th.Tensor) -> th.Tensor:
        """
        Symlog transformation: sign(x) * log(1 + |x|).
        
        Used to normalize inputs with varying scales.
        """
        return th.sign(x) * th.log(1 + th.abs(x))


class Decoder(nn.Module):
    """
    Decoder network that reconstructs observations from latent features.
    
    :param observation_space: Observation space
    :param feature_dim: Dimension of input features (deter_dim + stoch_size)
    :param hidden_dim: Hidden layer dimension (default: 1024)
    :param num_layers: Number of MLP layers for vectors (default: 3)
    :param activation: Activation function (default: nn.SiLU)
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        feature_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        activation: nn.Module = nn.SiLU,
    ):
        super().__init__()
        
        self.observation_space = observation_space
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        
        # Determine observation type and build appropriate decoder
        if isinstance(observation_space, spaces.Box):
            if len(observation_space.shape) == 1:
                # Vector observation
                self.obs_type = 'vector'
                self.decoder = self._build_mlp_decoder(observation_space.shape[0])
            elif len(observation_space.shape) == 3:
                # Image observation
                self.obs_type = 'image'
                self.decoder = self._build_cnn_decoder(observation_space.shape)
            else:
                raise NotImplementedError(f"Observation shape {observation_space.shape} not supported")
        elif isinstance(observation_space, spaces.Dict):
            # Dictionary observation space
            self.obs_type = 'dict'
            self.decoder = self._build_dict_decoder(observation_space)
        else:
            raise NotImplementedError(f"Observation space {observation_space} not supported")
    
    def _build_mlp_decoder(self, output_dim: int) -> nn.Module:
        """Build MLP decoder for vector observations."""
        layers = []
        current_dim = self.feature_dim
        
        for i in range(self.num_layers):
            layers.extend([
                nn.Linear(current_dim, self.hidden_dim),
                nn.RMSNorm(self.hidden_dim),
                self.activation(),
            ])
            current_dim = self.hidden_dim
        
        # Output layer (no activation - will use appropriate dist)
        layers.append(nn.Linear(current_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def _build_cnn_decoder(self, output_shape: Tuple[int, int, int]) -> nn.Module:
        """
        Build CNN decoder for image observations.
        
        Architecture (reverse of encoder):
        - 4 transposed convolutional layers
        - Depths: [256, 256, 192, 128]
        """
        channels, height, width = output_shape
        
        # Calculate starting size (after encoder convolutions)
        depths = [256, 256, 192, 128]
        start_height = height // (2 ** len(depths))
        start_width = width // (2 ** len(depths))
        start_features = depths[0] * start_height * start_width
        
        layers = [
            nn.Linear(self.feature_dim, start_features),
            nn.Unflatten(1, (depths[0], start_height, start_width)),
        ]
        
        for i, (in_depth, out_depth) in enumerate(zip(depths[:-1], depths[1:])):
            layers.extend([
                nn.ConvTranspose2d(in_depth, out_depth, kernel_size=5, stride=2, padding=2, output_padding=1),
                nn.GroupNorm(1, out_depth),
                self.activation(),
            ])
        
        # Final layer to match output channels
        layers.append(
            nn.ConvTranspose2d(depths[-1], channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        )
        
        return nn.Sequential(*layers)
    
    def _build_dict_decoder(self, observation_space: spaces.Dict) -> nn.ModuleDict:
        """Build separate decoders for each key in dictionary observation space."""
        decoders = nn.ModuleDict()
        
        for key, space in observation_space.spaces.items():
            if isinstance(space, spaces.Box):
                if len(space.shape) == 1:
                    decoders[key] = self._build_mlp_decoder(space.shape[0])
                elif len(space.shape) == 3:
                    decoders[key] = self._build_cnn_decoder(space.shape)
        
        return decoders
    
    def forward(self, features: th.Tensor) -> th.Tensor:
        """
        Decode features to reconstructed observations.
        
        :param features: Latent features (B, feature_dim)
        :return: Reconstructed observations or dict of reconstructions
        """
        if self.obs_type in ['vector', 'image']:
            recon = self.decoder(features)
            
        elif self.obs_type == 'dict':
            # Decode each modality
            recon = {}
            for key, decoder in self.decoder.items():
                recon[key] = decoder(features)
        
        else:
            raise NotImplementedError(f"Observation type {self.obs_type} not supported")
        
        return recon
    
    def reconstruction_loss(
        self,
        recon: th.Tensor,
        target: th.Tensor,
    ) -> th.Tensor:
        """
        Compute reconstruction loss.
        
        For images: Binary cross-entropy (assuming normalized [0, 1])
        For vectors: MSE loss
        
        :param recon: Reconstructed observations
        :param target: Target observations
        :return: Reconstruction loss
        """
        if self.obs_type == 'image':
            # Normalize target to [0, 1] if needed
            if target.dtype == th.uint8:
                target = target.float() / 255.0
            # Binary cross-entropy
            loss = F.binary_cross_entropy_with_logits(
                recon, target, reduction='none'
            ).sum(dim=[1, 2, 3]).mean()
            
        elif self.obs_type == 'vector':
            # MSE loss
            loss = F.mse_loss(recon, target, reduction='mean')
            
        elif self.obs_type == 'dict':
            # Sum losses from all modalities
            total_loss = 0
            for key in recon.keys():
                if len(target[key].shape) > 2:  # Image
                    if target[key].dtype == th.uint8:
                        target[key] = target[key].float() / 255.0
                    loss = F.binary_cross_entropy_with_logits(
                        recon[key], target[key], reduction='none'
                    ).sum(dim=[1, 2, 3]).mean()
                else:  # Vector
                    loss = F.mse_loss(recon[key], target[key], reduction='mean')
                total_loss += loss
            return total_loss
        
        else:
            raise NotImplementedError(f"Observation type {self.obs_type} not supported")
        
        return loss
