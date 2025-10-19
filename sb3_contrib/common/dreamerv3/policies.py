"""Policy classes for DreamerV3 implementation."""

import math
from collections import OrderedDict
from typing import Any, Optional, Union

import numpy as np
import torch
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    Distribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import zip_strict
from torch import nn

from sb3_contrib.common.dreamerv3.distributions import (
    BoundedDiagGaussianDistribution,
    SymexpTwoHotDistribution,
)
from sb3_contrib.common.recurrent.type_aliases import RNNStates


def _create_mlp(
    input_size: int,
    output_size: int,
    net_arch: list[int],
    activation_fn: type[nn.Module] = nn.SiLU,
    use_rms_norm: bool = True,
) -> nn.Sequential:
    """
    Create a multi-layer perceptron (MLP) network.

    :param input_size: Input dimension
    :param output_size: Output dimension
    :param net_arch: Architecture of the network (list of layer sizes)
    :param activation_fn: Activation function (default: SiLU for DreamerV3)
    :param use_rms_norm: Whether to use RMSNorm layers (default: True for DreamerV3)
    :return: Sequential network
    """
    if len(net_arch) == 0:
        return nn.Sequential(nn.Linear(input_size, output_size))

    layers = []
    current_size = input_size

    for i, hidden_size in enumerate(net_arch):
        layers.append(nn.Linear(current_size, hidden_size))
        if use_rms_norm:
            layers.append(nn.RMSNorm(hidden_size))
        layers.append(activation_fn())
        current_size = hidden_size

    layers.append(nn.Linear(current_size, output_size))

    return nn.Sequential(*layers)


class BlockDiagonalGRU(nn.Module):
    """
    A GRU with block-diagonal recurrent weights.

    This module breaks the hidden state into `num_blocks` chunks and applies
    separate linear transformations to each, creating a block-diagonal structure.
    This reduces parameters and computation compared to standard GRU.

    :param input_size: Size of input features
    :param hidden_size: Size of hidden state
    :param num_blocks: Number of diagonal blocks (default: 8 for DreamerV3)
    """

    def __init__(self, input_size: int, hidden_size: int, num_blocks: int = 8):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks

        if hidden_size % num_blocks != 0:
            raise ValueError("hidden_size must be divisible by num_blocks")

        self.block_size = hidden_size // num_blocks

        # Input-to-hidden weights (dense) for all 3 gates
        self.W_i = nn.Linear(input_size, 3 * hidden_size)

        # Hidden-to-hidden weights (block-diagonal) for all 3 gates
        self.W_h = nn.ModuleList([nn.Linear(self.block_size, 3 * self.block_size, bias=False) for _ in range(num_blocks)])

        # Hidden-to-hidden bias
        self.b_h = nn.Parameter(torch.zeros(3 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights for training stability."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x: torch.Tensor, h_0: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process input sequence through block-diagonal GRU.

        :param x: Input tensor of shape (seq_len, batch_size, input_size)
        :param h_0: Initial hidden state of shape (batch_size, hidden_size) or (1, batch_size, hidden_size)
        :return: Output sequence and final hidden state
        """
        seq_len, batch_size, _ = x.shape

        # Initialize hidden state if not provided
        if h_0 is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            # Handle both (B, H) and (1, B, H) shapes
            if h_0.dim() == 3:
                h_t = h_0.squeeze(0)
            else:
                h_t = h_0

        # Precompute input transformations
        x_gates = self.W_i(x)

        outputs = []

        for t in range(seq_len):
            # Apply block-diagonal transformation to hidden state
            h_t_chunks = h_t.chunk(self.num_blocks, dim=1)
            h_gates_chunks = [self.W_h[i](h_t_chunks[i]) for i in range(self.num_blocks)]
            h_gates = torch.cat(h_gates_chunks, dim=1) + self.b_h

            # Combine input and hidden transformations
            gates = x_gates[t] + h_gates

            # Split into reset, update, and new gates
            r_gate_raw, z_gate_raw, n_gate_raw = gates.chunk(3, dim=1)

            # Apply activations
            r_t = torch.sigmoid(r_gate_raw)
            z_t = torch.sigmoid(z_gate_raw)

            # Compute new gate with reset applied to hidden state
            h_r_gate, h_z_gate, h_n_gate = h_gates.chunk(3, dim=1)
            x_r_gate, x_z_gate, x_n_gate = x_gates[t].chunk(3, dim=1)
            n_t = torch.tanh(x_n_gate + r_t * h_n_gate)

            # Update hidden state
            h_t = (1 - z_t) * n_t + z_t * h_t

            outputs.append(h_t)

        output = torch.stack(outputs, dim=0)
        return output, h_t


class DreamerV3ActorCriticPolicy(ActorCriticPolicy):
    """
    Actor-Critic policy for DreamerV3 implementation.

    This policy implements the DreamerV3 architecture with:
    - World model with sequence (GRU) and dynamics models
    - Actor network for action prediction
    - Critic network for value prediction
    - Reward predictor
    - Continue predictor (episode termination)

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule
    :param net_arch: Network architecture specification
    :param activation_fn: Activation function (SiLU for DreamerV3)
    :param sequence_model_size: Size of sequence model hidden state (default: 512)
    :param sequence_model_blocks: Number of blocks in block-diagonal GRU (default: 8)
    :param latent_dim: Dimension of latent state (default: 256)
    :param actor_net_arch: Architecture for actor network (default: [1024, 1024, 1024])
    :param critic_net_arch: Architecture for critic network (default: [1024, 1024, 1024])
    :param reward_net_arch: Architecture for reward predictor (default: [1024])
    :param continue_net_arch: Architecture for continue predictor (default: [1024])
    :param unimix: Uniform mixing coefficient for discrete actions (default: 0.01)
    :param value_bins: Number of bins for value TwoHot encoding (default: 255)
    :param reward_bins: Number of bins for reward TwoHot encoding (default: 255)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.SiLU,
        ortho_init: bool = False,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        # DreamerV3-specific parameters
        sequence_model_size: int = 512,
        sequence_model_blocks: int = 8,
        latent_dim: int = 256,
        actor_net_arch: Optional[list[int]] = None,
        critic_net_arch: Optional[list[int]] = None,
        reward_net_arch: Optional[list[int]] = None,
        continue_net_arch: Optional[list[int]] = None,
        unimix: float = 0.01,
        value_bins: int = 255,
        reward_bins: int = 255,
        encoder_hidden_dim: int = 1024,
        encoder_num_layers: int = 3,
    ):
        # Set default architectures
        if actor_net_arch is None:
            actor_net_arch = [1024, 1024, 1024]
        if critic_net_arch is None:
            critic_net_arch = [1024, 1024, 1024]
        if reward_net_arch is None:
            reward_net_arch = [1024]
        if continue_net_arch is None:
            continue_net_arch = [1024]

        self.sequence_model_size = sequence_model_size
        self.sequence_model_blocks = sequence_model_blocks
        self.latent_dim = latent_dim
        self.full_state_size = sequence_model_size + latent_dim
        self.actor_net_arch = actor_net_arch
        self.critic_net_arch = critic_net_arch
        self.reward_net_arch = reward_net_arch
        self.continue_net_arch = continue_net_arch
        self.unimix = unimix
        self.value_bins = value_bins
        self.reward_bins = reward_bins
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_num_layers = encoder_num_layers

        # In DreamerV3, the policy operates on RSSM features, not raw observations
        # The Encoder is part of the world model, not the policy
        # We use FlattenExtractor as a simple pass-through for RSSM features
        # The actual observation encoding happens in the DreamerV3 algorithm via Encoder → RSSM
        
        # IMPORTANT: We do NOT override observation_space here.
        # The policy will be built with the original observation_space dimensions,
        # but during training, it will receive RSSM features instead of raw observations.
        # During rollouts, the DreamerV3 algorithm handles:
        # 1. Encode observations → tokens via Encoder
        # 2. Process tokens → RSSM features via RSSM.observe
        # 3. Pass RSSM features to world_model_actor_net (NOT this policy)
        
        # Use FlattenExtractor which will just pass through RSSM features during training
        features_extractor_class = FlattenExtractor
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        super().__init__(
            observation_space,  # Original observation_space (NOT overridden)
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            None, # features_extractor_kwargs
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """
        Build the MLP extractor and DreamerV3-specific networks.
        """
        # Get action dimension
        if isinstance(self.action_space, spaces.Discrete):
            action_dim = self.action_space.n
        elif isinstance(self.action_space, spaces.Box):
            action_dim = get_action_dim(self.action_space)
        else:
            raise NotImplementedError(f"Action space {self.action_space} not supported")

        # Get features dimension from feature extractor
        with th.no_grad():
            sample_obs = th.as_tensor(self.observation_space.sample()[None]).float()
            n_flatten = self.features_extractor(sample_obs).shape[1]

        # Build sequence model (block-diagonal GRU)
        # Note: In full DreamerV3, this would process sequences of latent states
        self.sequence_model = BlockDiagonalGRU(
            n_flatten + action_dim,
            self.sequence_model_size,
            self.sequence_model_blocks,
        )

        # Build actor network - takes extracted features
        self.actor_net = _create_mlp(
            n_flatten,
            action_dim,
            self.actor_net_arch,
            self.activation_fn,
        )

        # Build critic network (uses TwoHot encoding)
        self.critic_dist = SymexpTwoHotDistribution(
            action_dim=1,
            bins=self.value_bins,
        )
        self.critic_net = self.critic_dist.proba_distribution_net(n_flatten)

        # Build reward predictor (uses TwoHot encoding)
        self.reward_dist = SymexpTwoHotDistribution(
            action_dim=1,
            bins=self.reward_bins,
        )
        self.reward_net = self.reward_dist.proba_distribution_net(n_flatten)

        # Build continue predictor (Bernoulli for binary prediction)
        self.continue_dist = BernoulliDistribution(1)
        self.continue_net = self.continue_dist.proba_distribution_net(n_flatten)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Get the action distribution from the latent policy representation.

        :param latent_pi: Latent representation from which to generate actions
        :return: Action distribution
        """
        if isinstance(self.action_space, spaces.Box):
            # Continuous actions - use BoundedDiagGaussianDistribution
            mean_actions, log_std = self.action_net(latent_pi)
            return self.action_dist.proba_distribution(mean_actions, log_std)

        elif isinstance(self.action_space, spaces.Discrete):
            # Discrete actions - use CategoricalDistribution with unimix
            action_logits = self.actor_net(latent_pi)

            # Apply uniform mixing (unimix) if configured
            if self.unimix > 0.0:
                probs = th.softmax(action_logits, dim=-1)
                uniform = th.ones_like(probs) / probs.shape[-1]
                probs = (1 - self.unimix) * probs + self.unimix * uniform
                action_logits = th.log(probs + 1e-8)  # Add epsilon for numerical stability

            return self.action_dist.proba_distribution(action_logits)

        else:
            raise NotImplementedError(f"Action space {self.action_space} not supported")

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Build the policy networks and distributions.

        :param lr_schedule: Learning rate schedule
        """
        self._build_mlp_extractor()

        # Get features dimension
        with th.no_grad():
            sample_obs = th.as_tensor(self.observation_space.sample()[None]).float()
            n_flatten = self.features_extractor(sample_obs).shape[1]

        # Setup action distribution
        if isinstance(self.action_space, spaces.Box):
            action_dim = get_action_dim(self.action_space)
            self.action_dist = BoundedDiagGaussianDistribution(action_dim)

            # Action net outputs both mean and std
            class ActionNet(nn.Module):
                def __init__(self, actor_net: nn.Module):
                    super().__init__()
                    self.actor_net = actor_net

                def forward(self, x: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
                    # Split output into mean and std
                    out = self.actor_net(x)
                    mid = out.shape[-1] // 2
                    return out[..., :mid], out[..., mid:]

            # Modify actor_net to output 2*action_dim for mean and std
            self.actor_net = _create_mlp(
                n_flatten,
                2 * action_dim,
                self.actor_net_arch,
                self.activation_fn,
            )
            self.action_net = ActionNet(self.actor_net)

        elif isinstance(self.action_space, spaces.Discrete):
            self.action_dist = CategoricalDistribution(self.action_space.n)
        else:
            raise NotImplementedError(f"Action space {self.action_space} not supported")

        # Build optimizer
        if self.optimizer_class is not None:
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all networks (actor and critic).

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Extract features from observation
        features = self.extract_features(obs, self.pi_features_extractor)

        # For now, use features directly as latent representation
        # In full DreamerV3, this would go through the world model
        latent_pi = features

        # Get action distribution and sample/predict
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        # Get value from critic
        values = self.predict_values(obs)

        return actions, values, log_prob

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation: Observation
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        # Extract features
        features = self.extract_features(observation, self.pi_features_extractor)
        latent_pi = features

        # Get distribution and sample
        distribution = self._get_action_dist_from_latent(latent_pi)
        return distribution.get_actions(deterministic=deterministic)

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        IMPORTANT: This method should NOT be called for DreamerV3!
        
        In DreamerV3, inference must go through the world model:
          observation → Encoder → RSSM → features → policy
        
        Direct policy.predict(observation) bypasses the world model entirely,
        which defeats the purpose of DreamerV3.
        
        Instead, use DreamerV3.predict() which properly processes observations
        through the encoder and RSSM before calling the policy.
        
        :param observation: Observation (should be RSSM features, not raw observations)
        :param state: RNN state (not used)
        :param episode_start: Episode start flags (not used)
        :param deterministic: Whether to use deterministic actions
        :return: Actions and next state
        """
        # This should only be called with RSSM features (from training), not raw observations
        # If this is being called during rollouts, something is wrong!
        
        # For backward compatibility during training when RSSM features are passed,
        # we allow this to work, but add a warning if the input looks like raw observations
        import warnings
        if isinstance(observation, np.ndarray):
            # Check if this looks like raw observations (e.g., images or high-dim vectors)
            # RSSM features should be (N, 5120) typically
            if observation.ndim > 2 or (observation.ndim == 2 and observation.shape[1] > 6000):
                warnings.warn(
                    "DreamerV3ActorCriticPolicy.predict() called with what appears to be "
                    "raw observations instead of RSSM features. This will not work correctly. "
                    "Use DreamerV3.predict() instead, which processes through the world model.",
                    UserWarning,
                    stacklevel=2
                )
        
        # Call parent predict (for RSSM features only)
        return super().predict(observation, state, episode_start, deterministic)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: The estimated values
        """
        # Extract features
        features = self.extract_features(obs, self.vf_features_extractor)
        latent_vf = features

        # Use SymexpTwoHotDistribution for value prediction
        value_logits = self.critic_net(latent_vf)
        self.critic_dist.proba_distribution(value_logits)
        values = self.critic_dist.mode()

        return values

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions, and entropy
        """
        # Extract features
        features = self.extract_features(obs, self.pi_features_extractor)
        latent_pi = features

        # Get distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # Get values
        values = self.predict_values(obs)

        return values, log_prob, entropy


class DreamerV3CnnPolicy(DreamerV3ActorCriticPolicy):
    """
    DreamerV3 policy for image observations.
    
    Note: In DreamerV3, CNN encoding happens in the world model's Encoder,
    not in the policy's features_extractor. The policy operates on RSSM features
    (latent state), so this class uses the same FlattenExtractor as the parent.
    
    The actual CNN architecture from original DreamerV3 (4 conv layers with depths
    [128, 192, 256, 256], kernel_size=5, stride=2) is implemented in the world
    model's Encoder component, which is created by the DreamerV3 algorithm.

    :param observation_space: Observation space (should be an image space)
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule
    :param net_arch: Network architecture specification
    :param activation_fn: Activation function
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.SiLU,
        **kwargs: Any,
    ):
        # Do NOT override features_extractor_class - use parent's FlattenExtractor
        # The CNN encoding is handled by the world model's Encoder, not the policy
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            **kwargs,
        )


class DreamerV3MultiInputPolicy(DreamerV3ActorCriticPolicy):
    """
    DreamerV3 policy with multi-input feature extractor.

    :param observation_space: Observation space (dict space)
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule
    :param net_arch: Network architecture specification
    :param activation_fn: Activation function
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.SiLU,
        **kwargs: Any,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class=CombinedExtractor,
            **kwargs,
        )
