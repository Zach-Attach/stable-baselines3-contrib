"""
DreamerV3: Mastering Diverse Domains through World Models

Paper: https://arxiv.org/abs/2301.04104
"""

from typing import Any, Callable, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_schedule_fn, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from torch import nn

from sb3_contrib.common.dreamerv3.policies import DreamerV3ActorCriticPolicy
from sb3_contrib.common.dreamerv3.sequence_buffer import SequenceReplayBuffer
from sb3_contrib.dreamerV3.policies import CnnPolicy, MlpPolicy, MultiInputPolicy

SelfDreamerV3 = TypeVar("SelfDreamerV3", bound="DreamerV3")


class DreamerV3(OffPolicyAlgorithm):
    """
    DreamerV3: Mastering Diverse Domains through World Models

    DreamerV3 is a model-based reinforcement learning algorithm that learns
    a world model to predict future states and rewards, then trains an actor-critic
    policy within the learned model.

    Paper: https://arxiv.org/abs/2301.04104

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, MultiInputPolicy, ...)
    :param env: The environment to learn from
    :param learning_rate: Learning rate for the optimizer
    :param buffer_size: Size of the replay buffer
    :param learning_starts: How many steps before training starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: The soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: Discount factor
    :param train_freq: Update the model every ``train_freq`` steps
    :param gradient_steps: How many gradient steps to do after each rollout
    :param action_noise: The action noise type (None by default)
    :param replay_buffer_class: Replay buffer class to use
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
    :param n_steps: Number of steps for n-step returns
    :param batch_length: Length of sequences for training
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param imagination_horizon: Horizon for imagination rollouts (default: 15)
    :param model_lr: Learning rate for world model (default: 1e-4)
    :param actor_lr: Learning rate for actor (default: 3e-5)
    :param critic_lr: Learning rate for critic (default: 3e-5)
    :param target_update_interval: Update target network every N steps
    :param replay_context: Number of context frames to prepend for warm-starting recurrent state (default: 0)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling during warmup phase
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
    :param tensorboard_log: The log location for tensorboard
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device on which the code should be run
    :param _init_setup_model: Whether to build the network at creation
    """

    policy_aliases: ClassVar[Dict[str, Type[DreamerV3ActorCriticPolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[DreamerV3ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Callable] = 1e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 5000,
        batch_size: int = 16,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 4,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        batch_length: int = 50,
        gae_lambda: float = 0.95,
        imagination_horizon: int = 15,
        model_lr: float = 1e-4,
        actor_lr: float = 3e-5,
        critic_lr: float = 3e-5,
        target_update_interval: int = 100,
        replay_context: int = 0,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        # Set default replay buffer to SequenceReplayBuffer if not specified
        if replay_buffer_class is None:
            replay_buffer_class = SequenceReplayBuffer
            if replay_buffer_kwargs is None:
                replay_buffer_kwargs = {}
            # Set sequence_length parameter (same as batch_length)
            replay_buffer_kwargs.setdefault("sequence_length", batch_length)
            # Set replay_context parameter
            replay_buffer_kwargs.setdefault("replay_context", replay_context)

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            sde_support=False,  # DreamerV3 doesn't use SDE, so don't auto-add use_sde to policy_kwargs
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
            ),
            support_multi_env=True,
        )

        # DreamerV3-specific parameters
        self.batch_length = batch_length
        self.imagination_horizon = imagination_horizon
        self.model_lr = model_lr
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gae_lambda = gae_lambda
        self.target_update_interval = target_update_interval
        self.replay_context = replay_context

        self.model_optimizer = None
        self.actor_optimizer = None
        self.critic_optimizer = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        """Create networks and optimizers."""
        # Call parent _setup_model which handles:
        # - Learning rate schedule setup
        # - Random seed
        # - Replay buffer creation
        # - Policy creation
        # - Train frequency conversion
        super()._setup_model()

        # Setup DreamerV3-specific components (RSSM, Encoder, Decoder, etc.)
        # This must be called after policy creation since we use the policy's encoder
        self._setup_dreamerv3_components()

        # Setup optimizers for different components
        # (already done in _setup_dreamerv3_components)

    def _reinit_actor_critic(self) -> None:
        """Reinitialize actor and critic networks after NaN corruption."""
        from sb3_contrib.common.dreamerv3.distributions import SymexpTwoHotDistribution, BoundedDiagGaussianDistribution
        from stable_baselines3.common.distributions import CategoricalDistribution

        rssm_feature_dim = 4096 + 32 * 32  # 5120

        # Helper to create MLP
        def create_mlp(input_dim: int, output_dim: int, hidden_dim: int = 1024, num_layers: int = 3):
            """Create simple MLP without normalization for stability."""
            layers = [
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
            ]
            for _ in range(num_layers - 2):
                layers.extend(
                    [
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.SiLU(),
                    ]
                )
            layers.append(nn.Linear(hidden_dim, output_dim))
            return nn.Sequential(*layers)

        # Recreate actor network
        if isinstance(self.action_space, spaces.Box):
            action_dim = self.action_space.shape[0]
            self.world_model_actor_dist = BoundedDiagGaussianDistribution(action_dim)
            self.world_model_actor_net = create_mlp(rssm_feature_dim, 2 * action_dim).to(self.device)
        elif isinstance(self.action_space, spaces.Discrete):
            action_dim = self.action_space.n
            self.world_model_actor_dist = CategoricalDistribution(action_dim)
            self.world_model_actor_net = create_mlp(rssm_feature_dim, action_dim).to(self.device)

        # Initialize actor network weights
        for module in self.world_model_actor_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Recreate critic network
        self.world_model_critic_dist = SymexpTwoHotDistribution(action_dim=1, bins=255).to(self.device)
        self.world_model_critic_net = self.world_model_critic_dist.proba_distribution_net(rssm_feature_dim).to(self.device)

        # Recreate slow critic network
        self.slow_critic_dist = SymexpTwoHotDistribution(action_dim=1, bins=255).to(self.device)
        self.slow_critic_net = self.slow_critic_dist.proba_distribution_net(rssm_feature_dim).to(self.device)
        self.slow_critic_net.load_state_dict(self.world_model_critic_net.state_dict())

        # Recreate optimizers
        self.actor_optimizer = th.optim.Adam(self.world_model_actor_net.parameters(), lr=self.actor_lr)
        self.critic_optimizer = th.optim.Adam(self.world_model_critic_net.parameters(), lr=self.critic_lr)

    def _setup_dreamerv3_components(self) -> None:
        """Setup DreamerV3-specific components (RSSM, Encoder, Decoder, etc.)."""
        from sb3_contrib.common.dreamerv3 import (
            RSSM,
            Encoder,
            Decoder,
            ValueNormalizer,
            SlowValueNetwork,
        )
        from sb3_contrib.common.dreamerv3.distributions import SymexpTwoHotDistribution, BoundedDiagGaussianDistribution
        from stable_baselines3.common.distributions import BernoulliDistribution, CategoricalDistribution

        # Create RSSM (world model)
        self.rssm = RSSM(
            action_space=self.action_space,
            deter_dim=4096,
            stoch_dim=32,
            num_classes=32,
            hidden_dim=2048,
            num_blocks=8,
        ).to(self.device)

        # Create Encoder as part of the world model (NOT as policy's features_extractor)
        # In DreamerV3, the encoder is part of the world model and processes observations
        # into tokens that are fed to the RSSM. The policy never sees raw observations.
        self.encoder = Encoder(
            observation_space=self.observation_space,
            hidden_dim=1024,
            num_layers=3,
            activation=nn.SiLU,
            symlog_transform=True,
        ).to(self.device)

        # Create Decoder
        # Feature dim = deter_dim + stoch_dim * num_classes
        rssm_feature_dim = 4096 + 32 * 32  # 5120
        self.decoder = Decoder(
            observation_space=self.observation_space,
            feature_dim=rssm_feature_dim,
            hidden_dim=1024,
            num_layers=3,
        ).to(self.device)

        # Create world model reward and continue predictors (accept RSSM features)
        # These are separate from policy networks and are used during world model training
        # Note: SymexpTwoHotDistribution is an nn.Module with buffers, so it needs .to(device)
        self.world_model_reward_dist = SymexpTwoHotDistribution(action_dim=1, bins=255).to(self.device)
        self.world_model_reward_net = self.world_model_reward_dist.proba_distribution_net(rssm_feature_dim).to(self.device)

        # Note: BernoulliDistribution is not an nn.Module, so it doesn't need .to(device)
        self.world_model_continue_dist = BernoulliDistribution(1)
        self.world_model_continue_net = self.world_model_continue_dist.proba_distribution_net(rssm_feature_dim).to(self.device)

        # Create world model actor and critic networks (for imagination training)
        # These take RSSM features as input and are used during imagination rollouts

        # Helper to create MLP
        def create_mlp(input_dim: int, output_dim: int, hidden_dim: int = 1024, num_layers: int = 3):
            """Create simple MLP without normalization for stability."""
            layers = [
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
            ]
            for _ in range(num_layers - 2):
                layers.extend(
                    [
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.SiLU(),
                    ]
                )
            layers.append(nn.Linear(hidden_dim, output_dim))
            return nn.Sequential(*layers)

        # Actor network
        if isinstance(self.action_space, spaces.Box):
            # Continuous actions
            action_dim = self.action_space.shape[0]
            self.world_model_actor_dist = BoundedDiagGaussianDistribution(action_dim)
            self.world_model_actor_net = create_mlp(rssm_feature_dim, 2 * action_dim).to(self.device)
        elif isinstance(self.action_space, spaces.Discrete):
            # Discrete actions
            action_dim = self.action_space.n
            self.world_model_actor_dist = CategoricalDistribution(action_dim)
            self.world_model_actor_net = create_mlp(rssm_feature_dim, action_dim).to(self.device)
        else:
            raise NotImplementedError(f"Action space {self.action_space} not supported")

        # Initialize actor network weights with Xavier/Glorot initialization
        for module in self.world_model_actor_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)  # Small gain for stability
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Critic network (value function)
        self.world_model_critic_dist = SymexpTwoHotDistribution(action_dim=1, bins=255).to(self.device)
        self.world_model_critic_net = self.world_model_critic_dist.proba_distribution_net(rssm_feature_dim).to(self.device)

        # Slow critic network for stable targets
        self.slow_critic_dist = SymexpTwoHotDistribution(action_dim=1, bins=255).to(self.device)
        self.slow_critic_net = self.slow_critic_dist.proba_distribution_net(rssm_feature_dim).to(self.device)
        # Initialize slow critic with same weights as critic
        self.slow_critic_net.load_state_dict(self.world_model_critic_net.state_dict())
        self.slow_critic_tau = 0.98  # Slow update coefficient

        # Create normalizers
        self.value_normalizer = ValueNormalizer().to(self.device)
        self.return_normalizer = ValueNormalizer().to(self.device)
        self.advantage_normalizer = ValueNormalizer().to(self.device)

        # Setup optimizers with all world model components
        world_model_params = (
            list(self.rssm.parameters())
            + list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.world_model_reward_net.parameters())
            + list(self.world_model_continue_net.parameters())
        )

        self.model_optimizer = th.optim.Adam(world_model_params, lr=self.model_lr)

        # Actor optimizer (for imagination-trained actor)
        self.actor_optimizer = th.optim.Adam(self.world_model_actor_net.parameters(), lr=self.actor_lr)

        # Critic optimizer (for imagination-trained critic)
        self.critic_optimizer = th.optim.Adam(self.world_model_critic_net.parameters(), lr=self.critic_lr)

        # Initialize RSSM carry state for rollouts
        # This will be properly initialized when rollouts start
        self._rssm_carry = None
        self._last_actions = None
        self._last_episode_starts = None

    def _initialize_rssm_carry(self, n_envs: int) -> None:
        """
        Initialize RSSM carry state for environment rollouts.

        :param n_envs: Number of parallel environments
        """
        self._rssm_carry = self.rssm.initial(n_envs, self.device)
        if isinstance(self.action_space, spaces.Discrete):
            self._last_actions = np.zeros(n_envs, dtype=np.int64)
        else:
            self._last_actions = np.zeros((n_envs, self.action_space.shape[0]))
        self._last_episode_starts = np.zeros(n_envs, dtype=bool)

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample an action using the world model (Encoder → RSSM → Policy).

        This method implements the correct DreamerV3 inference flow:
        1. Encode observations to tokens
        2. Update RSSM state (observe)
        3. Get RSSM features
        4. Sample action from policy using features

        :param learning_starts: Number of steps before learning starts
        :param action_noise: Action noise
        :param n_envs: Number of environments
        :return: action to take and scaled action
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts:
            # Warmup phase - random actions
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Use world model for inference
            with th.no_grad():
                # Convert observations to tensor
                obs_tensor = obs_as_tensor(self._last_obs, self.device)

                # Encode observations to tokens
                tokens = self.encoder(obs_tensor)  # (n_envs, token_dim)
                tokens = tokens.unsqueeze(1)  # (n_envs, 1, token_dim)

                # Get previous actions (or zeros for first step)
                if hasattr(self, "_last_actions"):
                    prev_actions = th.as_tensor(self._last_actions, device=self.device).unsqueeze(1)
                else:
                    if isinstance(self.action_space, spaces.Discrete):
                        prev_actions = th.zeros(n_envs, 1, dtype=th.long, device=self.device)
                    else:
                        prev_actions = th.zeros(n_envs, 1, self.action_space.shape[0], device=self.device)

                # Check for episode resets
                if hasattr(self, "_last_episode_starts"):
                    resets = th.as_tensor(self._last_episode_starts, device=self.device).unsqueeze(1)
                else:
                    resets = th.zeros(n_envs, 1, dtype=th.bool, device=self.device)

                # RSSM observe to get features
                carry, _, features = self.rssm.observe(self._rssm_carry, tokens, prev_actions, resets, training=False)

                # Update RSSM carry for next step
                self._rssm_carry = carry

                # Reset carry for episodes that ended
                if resets.any():
                    for idx in range(n_envs):
                        if resets[idx, 0]:
                            init_state = self.rssm.initial(1, self.device)
                            self._rssm_carry["deter"][idx] = init_state["deter"][0]
                            self._rssm_carry["stoch"][idx] = init_state["stoch"][0]

                # Get RSSM features for policy
                deter = features["deter"].squeeze(1)  # (n_envs, deter_dim)
                stoch = features["stoch"].squeeze(1)  # (n_envs, stoch_dim, num_classes)
                feat = self.rssm.get_feat(deter, stoch)  # (n_envs, feat_dim)

                # Sample action from world model actor network
                actor_output = self.world_model_actor_net(feat)

                # Create distribution and sample
                if isinstance(self.action_space, spaces.Box):
                    # Continuous actions - split output into mean and log_std
                    action_dim = self.action_space.shape[0]
                    mean = actor_output[:, :action_dim]
                    log_std = actor_output[:, action_dim:]
                    action_dist = self.world_model_actor_dist.proba_distribution(mean, log_std)
                    unscaled_action = action_dist.sample().cpu().numpy()
                elif isinstance(self.action_space, spaces.Discrete):
                    # Discrete actions - output is already logits
                    action_dist = self.world_model_actor_dist.proba_distribution(actor_output)
                    unscaled_action = action_dist.sample().cpu().numpy()
                else:
                    raise NotImplementedError(f"Action space {self.action_space} not supported")

        # Rescale action from [-1, 1] to actual action space bounds
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)
            # Clip in case of numerical imprecision
            scaled_action = np.clip(scaled_action, self.action_space.low, self.action_space.high)
        else:
            scaled_action = unscaled_action

        # Store last actions for next step
        self._last_actions = unscaled_action

        return unscaled_action, scaled_action

    def _apply_replay_context(
        self,
        rssm_state: Dict[str, th.Tensor],
        obs: th.Tensor,
        actions: th.Tensor,
        consec: th.Tensor,
        entries: Optional[Dict[str, Dict[str, th.Tensor]]] = None,
    ) -> tuple[dict[str, th.Tensor], th.Tensor, th.Tensor]:
        """
        Apply replay context to warm-start RSSM state from stored entries.

        This allows the model to maintain temporal consistency across consecutive
        chunks of the same sequence by restoring the model state from a previous chunk.

        :param rssm_state: Current RSSM state (batch, *)
        :param obs: Observations (batch, time, *)
        :param actions: Actions (batch, time, *)
        :param consec: Consecutive chunk indices (batch,) - 0 for first chunk
        :param entries: Optional dict with stored model entries from previous chunks
        :return: Tuple of (updated_rssm_state, sliced_obs, sliced_actions)
        """
        if self.replay_context == 0 or entries is None:
            # No replay context, return as is
            return rssm_state, obs, actions

        K = self.replay_context
        B = obs.shape[0]

        # Check if this is the first chunk (consec == 0)
        first_chunk = consec == 0

        # For non-first chunks, restore state from entries
        if not first_chunk.all():
            # Extract context entries (first K timesteps)
            if "rssm" in entries:
                context_entries = {k: v[:, :K] for k, v in entries["rssm"].items()}
                # Truncate to get the state at the last context timestep
                restored_state = self.rssm.truncate(context_entries, rssm_state)

                # Use restored state for non-first chunks
                for key in rssm_state:
                    rssm_state[key] = th.where(
                        first_chunk.view(-1, *([1] * (rssm_state[key].ndim - 1))), rssm_state[key], restored_state[key]
                    )

        # Slice observations and actions to exclude context prefix for non-first chunks
        # For first chunks, we use all timesteps; for others, we skip the context prefix
        if not first_chunk.all():
            # Create masks for slicing
            obs_sliced = th.where(first_chunk.view(-1, *([1] * (obs.ndim - 1))), obs, obs[:, K:])
            actions_sliced = th.where(first_chunk.view(-1, *([1] * (actions.ndim - 1))), actions, actions[:, K:])
        else:
            obs_sliced = obs
            actions_sliced = actions

        return rssm_state, obs_sliced, actions_sliced

    def train(self, gradient_steps: int, batch_size: int = 16) -> None:
        """
        Train the DreamerV3 agent.

        Training consists of two phases:
        1. World model training: Learn to predict observations, rewards, and episode ends
        2. Actor-critic training: Learn policy and value function in imagination

        :param gradient_steps: Number of gradient steps
        :param batch_size: Number of sequences to sample
        """
        # Switch to train mode
        self.policy.set_training_mode(True)

        # Import required modules
        from sb3_contrib.common.dreamerv3 import (
            RSSM,
            Encoder,
            Decoder,
            ValueNormalizer,
            SlowValueNetwork,
            lambda_return,
            compute_advantages,
            compute_actor_loss,
            compute_value_loss,
            compute_episode_weights,
        )

        # Skip imagination training for the first N updates to let world model stabilize
        imagination_warmup_updates = 50  # Number of world model updates before starting imagination
        should_skip_imagination = self._n_updates < imagination_warmup_updates

        losses = {
            "world_model": [],
            "dyn_loss": [],
            "rep_loss": [],
            "rec_loss": [],
            "rew_loss": [],
            "con_loss": [],
            "actor_loss": [],
            "critic_loss": [],
        }

        for _ in range(gradient_steps):
            # Sample sequences from replay buffer
            replay_data = self.replay_buffer.sample(batch_size)

            # =================================================================
            # PHASE 1: WORLD MODEL TRAINING
            # =================================================================

            # Check if using SequenceReplayBuffer
            from sb3_contrib.common.dreamerv3.sequence_buffer import SequenceReplayBufferSamples

            is_sequence_buffer = isinstance(replay_data, SequenceReplayBufferSamples)

            # Prepare data
            observations = replay_data.observations
            actions = replay_data.actions
            rewards = replay_data.rewards
            dones = replay_data.dones

            if is_sequence_buffer:
                # Using SequenceReplayBuffer - data already in sequence format
                # Shape: (B, T, ...)
                B, T = observations.shape[0], observations.shape[1]
                mask = replay_data.mask  # (B, T) - mask for valid timesteps

                # Encode observations (handle sequences)
                # Flatten to (B*T, obs_shape) for encoding
                flat_obs = observations.reshape(B * T, *observations.shape[2:])
                flat_tokens = self.encoder(flat_obs)  # (B*T, token_dim)
                tokens = flat_tokens.reshape(B, T, -1)  # (B, T, token_dim)
            else:
                # Using standard ReplayBuffer - treat as sequences of length 1
                B = batch_size
                T = 1

                # Reshape to (B, T, ...)
                if len(observations.shape) == 2:
                    observations = observations.unsqueeze(1)  # (B, 1, obs_dim)
                elif len(observations.shape) == 4:  # Image observations
                    observations = observations.unsqueeze(1)  # (B, 1, C, H, W)

                if len(actions.shape) == 1:
                    actions = actions.unsqueeze(-1)  # (B, action_dim)
                actions = actions.unsqueeze(1)  # (B, 1, action_dim)
                rewards = rewards.unsqueeze(1)  # (B, 1, 1)
                dones = dones.unsqueeze(1)  # (B, 1, 1)

                # Create mask (all ones for standard buffer)
                mask = th.ones(B, T, device=self.device)

                # Encode observations to tokens
                # Ensure observations are properly shaped for encoder
                obs_for_encoder = observations.squeeze(1)  # Remove time dimension: (B, 1, ...) -> (B, ...)
                
                # Verify observation shape matches observation space
                if isinstance(self.observation_space, spaces.Box):
                    expected_shape = (B,) + self.observation_space.shape
                    if obs_for_encoder.shape != expected_shape:
                        raise ValueError(
                            f"Observation shape mismatch! obs_for_encoder.shape={obs_for_encoder.shape}, "
                            f"expected={expected_shape}, observation_space.shape={self.observation_space.shape}"
                        )
                
                tokens = self.encoder(obs_for_encoder)  # (B, token_dim)
                tokens = tokens.unsqueeze(1)  # (B, 1, token_dim)

            # Initialize or get RSSM state
            if not hasattr(self, "_rssm_state"):
                self._rssm_state = self.rssm.initial(B, self.device)

            # RSSM observe step (posterior path)
            resets = dones.squeeze(-1)  # (B, 1)
            carry, entries, features = self.rssm.observe(self._rssm_state, tokens, actions, resets, training=True)

            # Get features (deter + stoch)
            # Remove time dimension if T=1
            deter = features["deter"]  # (B, T, deter_dim)
            stoch = features["stoch"]  # (B, T, stoch_dim, num_classes)
            
            # Squeeze time dimension for single timestep
            if deter.shape[1] == 1:
                deter = deter.squeeze(1)  # (B, deter_dim)
                stoch = stoch.squeeze(1)  # (B, stoch_dim, num_classes)
            
            feat = self.rssm.get_feat(deter, stoch)  # (B, feat_dim) or (B, T, feat_dim)
            
            # Verify feature dimension matches what decoder expects
            expected_feat_dim = self.rssm.deter_dim + self.rssm.stoch_dim * self.rssm.num_classes
            if feat.shape[-1] != expected_feat_dim:
                raise ValueError(
                    f"Feature dimension mismatch! feat.shape={feat.shape}, expected last dim={expected_feat_dim}. "
                    f"deter.shape={deter.shape}, stoch.shape={stoch.shape}, "
                    f"RSSM config: deter_dim={self.rssm.deter_dim}, stoch_dim={self.rssm.stoch_dim}, num_classes={self.rssm.num_classes}"
                )

            # Decode observations
            # Handle both 2D (B, feat_dim) and 3D (B, T, feat_dim) features
            if feat.ndim == 3:
                # Has time dimension - flatten, decode, then unflatten
                B, T, feat_dim = feat.shape
                feat_flat = feat.reshape(B * T, feat_dim)  # (B*T, feat_dim)
                recon_obs_flat = self.decoder(feat_flat)  # (B*T, ...)
                
                # Reshape back to (B, T, ...)
                if isinstance(recon_obs_flat, dict):
                    # Dict observation space
                    recon_obs = {}
                    for key, val in recon_obs_flat.items():
                        if val.ndim == 2:  # Vector observation
                            recon_obs[key] = val.reshape(B, T, -1)
                        elif val.ndim == 4:  # Image observation (B*T, C, H, W)
                            recon_obs[key] = val.reshape(B, T, *val.shape[1:])
                        else:
                            raise ValueError(f"Unexpected recon shape for key {key}: {val.shape}")
                elif recon_obs_flat.ndim == 2:  # Vector observation
                    recon_obs = recon_obs_flat.reshape(B, T, -1)
                elif recon_obs_flat.ndim == 4:  # Image observation (B*T, C, H, W)
                    recon_obs = recon_obs_flat.reshape(B, T, *recon_obs_flat.shape[1:])
                else:
                    raise ValueError(f"Unexpected recon_obs_flat shape: {recon_obs_flat.shape}")
            else:
                # No time dimension - decode directly
                recon_obs = self.decoder(feat)

            # Compute reconstruction loss
            # Handle both single timestep and sequence cases
            if is_sequence_buffer and T > 1:
                # Sequence case: observations already have shape (B, T, ...)
                target_obs = observations
            else:
                # Single timestep case: remove time dimension
                target_obs = observations.squeeze(1)
            rec_loss = self.decoder.reconstruction_loss(recon_obs, target_obs)

            # Predict rewards and continues using world model networks
            # Handle both 2D and 3D features
            if feat.ndim == 3:
                # Sequence case: flatten, predict, then compute loss over all timesteps
                B, T, feat_dim = feat.shape
                feat_flat = feat.reshape(B * T, feat_dim)
                reward_logits = self.world_model_reward_net(feat_flat)  # (B*T, 255)
                continue_pred = self.world_model_continue_net(feat_flat)  # (B*T, 1)
                
                # Flatten target rewards and continues
                target_rewards_flat = rewards.reshape(B * T, -1)  # (B*T, 1)
                target_continues_flat = (1 - dones.reshape(B * T, -1)).float().squeeze(-1)  # (B*T,)
                
                # Compute losses
                self.world_model_reward_dist.proba_distribution(reward_logits)
                rew_loss = -self.world_model_reward_dist.log_prob(target_rewards_flat).mean()
                con_loss = F.binary_cross_entropy_with_logits(continue_pred.squeeze(), target_continues_flat)
            else:
                # Single timestep case
                reward_logits = self.world_model_reward_net(feat)  # (B, 255)
                continue_pred = self.world_model_continue_net(feat)  # (B, 1)
                
                target_rewards = rewards.squeeze(1)  # (B, 1)
                self.world_model_reward_dist.proba_distribution(reward_logits)
                rew_loss = -self.world_model_reward_dist.log_prob(target_rewards).mean()
                
                target_continues = (1 - dones.squeeze(1).squeeze(-1)).float()  # (B,)
                con_loss = F.binary_cross_entropy_with_logits(continue_pred.squeeze(), target_continues)

            # Compute KL losses
            posterior_logits = features["logits"]  # (B, T, stoch_dim, num_classes)
            
            # Compute prior - handle both 2D and 3D deter
            if deter.ndim == 3:
                # Has time dimension
                B, T, _ = deter.shape
                prior_logits_flat = self.rssm._prior(deter)  # Returns (B*T, stoch_dim, num_classes) due to reshape
                prior_logits = prior_logits_flat.reshape(B, T, self.rssm.stoch_dim, self.rssm.num_classes)
            else:
                # No time dimension
                prior_logits = self.rssm._prior(deter)  # (B, stoch_dim, num_classes)
                prior_logits = prior_logits.unsqueeze(1)  # (B, 1, stoch_dim, num_classes) to match posterior
            
            dyn_loss, rep_loss = self.rssm.kl_loss(posterior_logits, prior_logits)

            # Total world model loss (with loss scales from DreamerV3)
            world_model_loss = (
                0.5 * dyn_loss  # Dynamics KL
                + 0.1 * rep_loss  # Representation KL
                + 1.0 * rec_loss  # Reconstruction
                + 1.0 * rew_loss  # Reward prediction
                + 1.0 * con_loss  # Continue prediction
            )

            # Update world model
            self.model_optimizer.zero_grad()
            world_model_loss.backward()
            # Gradient clipping (optional but recommended)
            th.nn.utils.clip_grad_norm_(
                [p for group in self.model_optimizer.param_groups for p in group["params"]], max_norm=100.0
            )
            self.model_optimizer.step()

            # =================================================================
            # PHASE 2: ACTOR-CRITIC TRAINING (IN IMAGINATION)
            # =================================================================

            # Skip imagination during warmup period
            if should_skip_imagination:
                actor_loss_val = th.tensor(0.0, device=self.device)
                critic_loss_val = th.tensor(0.0, device=self.device)
                losses["actor_loss"].append(actor_loss_val.item())
                losses["critic_loss"].append(critic_loss_val.item())
                continue

            # Get starting states for imagination (use posterior states from replay)
            # In the full implementation, we'd select K starting points from the last K timesteps
            # For sequences, use the last timestep; for single timesteps, squeeze the time dimension
            if deter.ndim == 3:
                # Has time dimension - use last timestep
                start_deter = deter[:, -1].detach()  # (B, deter_dim)
                start_stoch = stoch[:, -1].detach()  # (B, stoch_dim, num_classes)
            else:
                # No time dimension (already squeezed)
                start_deter = deter.detach()  # (B, deter_dim)
                start_stoch = stoch.detach()  # (B, stoch_dim, num_classes)
            start_state = {"deter": start_deter, "stoch": start_stoch}

            # Check for NaN in start state or features
            if th.isnan(start_deter).any() or th.isnan(start_stoch).any():
                print(
                    f"Warning: NaN in start state - deter NaN: {th.isnan(start_deter).any()}, stoch NaN: {th.isnan(start_stoch).any()}"
                )
                # Skip imagination training this step
                actor_loss_val = th.tensor(0.0, device=self.device)
                critic_loss_val = th.tensor(0.0, device=self.device)
                losses["actor_loss"].append(actor_loss_val.item())
                losses["critic_loss"].append(critic_loss_val.item())
                continue

            # Check if actor network is producing valid outputs (early training may have numerical issues)
            test_feat = self.rssm.get_feat(start_deter[:1], start_stoch[:1])
            test_out = self.world_model_actor_net(th.clamp(test_feat, -10, 10))
            if th.isnan(test_out).any() or th.isinf(test_out).any():
                # Check if actor/critic weights are corrupted
                actor_params_ok = all(
                    not th.isnan(p).any() and not th.isinf(p).any() for p in self.world_model_actor_net.parameters()
                )
                critic_params_ok = all(
                    not th.isnan(p).any() and not th.isinf(p).any() for p in self.world_model_critic_net.parameters()
                )

                if not actor_params_ok or not critic_params_ok:
                    if self.verbose > 0:
                        print(f"DreamerV3: Reinitializing actor/critic (world model features not stable yet)")
                    # Reinitialize just the actor and critic networks
                    self._reinit_actor_critic()

                actor_loss_val = th.tensor(0.0, device=self.device)
                critic_loss_val = th.tensor(0.0, device=self.device)
                losses["actor_loss"].append(actor_loss_val.item())
                losses["critic_loss"].append(critic_loss_val.item())
                continue

            # Define policy function for imagination
            # This takes RSSM features and returns actions
            def imagination_policy(feat: th.Tensor) -> th.Tensor:
                """Policy for imagination rollouts."""
                # Clip features to prevent overflow
                feat = th.clamp(feat, -10, 10)

                # Check for NaN in features
                if th.isnan(feat).any():
                    print(f"Warning: NaN in RSSM features! feat shape: {feat.shape}, NaN count: {th.isnan(feat).sum().item()}")
                    # Use random actions if features are NaN
                    if isinstance(self.action_space, spaces.Discrete):
                        action_idx = th.randint(0, self.action_space.n, (feat.shape[0],), device=feat.device)
                        action = th.nn.functional.one_hot(action_idx.long(), num_classes=self.action_space.n).float()
                    else:
                        action = th.randn(feat.shape[0], self.action_space.shape[0], device=feat.device) * 0.1
                    return action

                # Get action distribution from world model actor
                if isinstance(self.action_space, spaces.Box):
                    # Continuous actions
                    actor_out = self.world_model_actor_net(feat)
                    mean, log_std = th.chunk(actor_out, 2, dim=-1)
                    self.world_model_actor_dist.proba_distribution(mean, log_std)
                    action = self.world_model_actor_dist.sample()
                else:
                    # Discrete actions
                    logits = self.world_model_actor_net(feat)
                    # Clip logits to prevent NaN
                    logits = th.clamp(logits, -20, 20)
                    # Check for NaN and clip if needed
                    if th.isnan(logits).any():
                        # If we get NaN, use uniform random actions
                        print(f"Warning: NaN in actor logits after clipping, using uniform random actions")
                        action_idx = th.randint(0, self.action_space.n, (logits.shape[0],), device=logits.device)
                    else:
                        self.world_model_actor_dist.proba_distribution(logits)
                        action_idx = self.world_model_actor_dist.sample()
                    # Convert to one-hot for RSSM (expects one-hot for discrete actions)
                    action = th.nn.functional.one_hot(
                        action_idx.long().squeeze(-1) if action_idx.dim() > 1 else action_idx.long(),
                        num_classes=self.action_space.n,
                    ).float()
                return action

            # Imagine trajectories
            H = self.imagination_horizon
            final_state, imag_features, imag_actions = self.rssm.imagine(
                start_state,
                imagination_policy,
                H,
                training=True,
            )

            # Get imagined features (B, H, feat_dim)
            imag_deter = imag_features["deter"]  # (B, H, deter_dim)
            imag_stoch = imag_features["stoch"]  # (B, H, stoch_dim, num_classes)
            imag_feat = self.rssm.get_feat(
                imag_deter.reshape(-1, imag_deter.shape[-1]), imag_stoch.reshape(-1, *imag_stoch.shape[-2:])
            ).reshape(
                B, H, -1
            )  # (B, H, feat_dim)

            # Predict rewards and continues for imagined states
            imag_feat_flat = imag_feat.reshape(-1, imag_feat.shape[-1])  # (B*H, feat_dim)
            imag_reward_logits = self.world_model_reward_net(imag_feat_flat)
            imag_continue_logits = self.world_model_continue_net(imag_feat_flat)

            # Get reward predictions
            self.world_model_reward_dist.proba_distribution(imag_reward_logits)
            imag_rewards = self.world_model_reward_dist.mode().reshape(B, H, 1)  # (B, H, 1)

            # Get continue predictions
            imag_continues = th.sigmoid(imag_continue_logits).reshape(B, H, 1)  # (B, H, 1)

            # Get value predictions from critic
            imag_value_logits = self.world_model_critic_net(imag_feat_flat)
            self.world_model_critic_dist.proba_distribution(imag_value_logits)
            imag_values = self.world_model_critic_dist.mode().reshape(B, H)  # (B, H)

            # Get slow value predictions for stable targets
            with th.no_grad():
                slow_value_logits = self.slow_critic_net(imag_feat_flat)
                self.slow_critic_dist.proba_distribution(slow_value_logits)
                slow_values = self.slow_critic_dist.mode().reshape(B, H)  # (B, H)

            # Compute bootstrap value (value at H+1, assumed to be last value)
            bootstrap = slow_values[:, -1]  # (B,)

            # Compute lambda returns
            # Squeeze continues to (B, H)
            continues_for_returns = imag_continues.squeeze(-1)  # (B, H)
            rewards_for_returns = imag_rewards.squeeze(-1)  # (B, H)

            # Add bootstrap to values for lambda return computation
            values_with_bootstrap = th.cat([slow_values, bootstrap.unsqueeze(1)], dim=1)  # (B, H+1)

            # Compute returns using lambda return
            from sb3_contrib.common.dreamerv3 import lambda_return

            returns = lambda_return(
                rewards=rewards_for_returns,
                values=values_with_bootstrap,
                continues=continues_for_returns,
                bootstrap=bootstrap,
                lambda_=self.gae_lambda,
                gamma=self.gamma,
            )  # (B, H)

            # Detach returns to prevent gradient flow through slow_values during actor training
            returns_detached = returns.detach()

            # Normalize returns for value training
            returns_normalized = self.return_normalizer.normalize(returns_detached, update=True)

            # Compute advantages (use detached returns)
            advantages = returns_detached - slow_values.detach()  # (B, H)

            # Normalize advantages for policy training
            advantages_normalized = self.advantage_normalizer.normalize(advantages, update=True)

            # Compute episode weights (for handling episode boundaries)
            from sb3_contrib.common.dreamerv3 import compute_episode_weights

            weights = compute_episode_weights(continues_for_returns.detach(), gamma=self.gamma)  # (B, H)

            # =================================================================
            # TRAIN ACTOR (POLICY)
            # =================================================================

            # Recompute action distribution for gradient
            # Use fresh forward pass to avoid shared computational graph with critic
            if isinstance(self.action_space, spaces.Box):
                actor_out = self.world_model_actor_net(imag_feat_flat.detach())  # Detach to separate graphs
                mean, log_std = th.chunk(actor_out, 2, dim=-1)
                self.world_model_actor_dist.proba_distribution(mean, log_std)
            else:
                logits = self.world_model_actor_net(imag_feat_flat.detach())  # Detach to separate graphs
                # Don't apply unimix in log space - just use raw logits
                self.world_model_actor_dist.proba_distribution(logits)

            # Compute log probabilities and entropy
            imag_actions_flat = imag_actions.reshape(-1, *imag_actions.shape[2:])  # (B*H, action_dim)

            # For discrete actions, convert one-hot back to indices for log_prob
            if isinstance(self.action_space, spaces.Discrete):
                imag_actions_indices = th.argmax(imag_actions_flat, dim=-1)  # (B*H,) - squeeze for CategoricalDistribution
                log_probs = self.world_model_actor_dist.log_prob(imag_actions_indices).reshape(B, H)  # (B, H)
            else:
                log_probs = self.world_model_actor_dist.log_prob(imag_actions_flat).reshape(B, H)  # (B, H)

            entropy = self.world_model_actor_dist.entropy().reshape(B, H)  # (B, H)

            # Actor loss: -weight * (log_prob * advantage + entropy_coef * entropy)
            entropy_coef = 3e-4

            # Use detached copies of advantages and weights to prevent graph sharing
            actor_loss_val = compute_actor_loss(
                log_probs=log_probs,
                advantages=advantages_normalized.detach(),
                entropies=entropy,
                weights=weights.detach(),
                entropy_coef=entropy_coef,
            )

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss_val.backward()
            th.nn.utils.clip_grad_norm_(self.world_model_actor_net.parameters(), max_norm=100.0)
            self.actor_optimizer.step()

            # Store actor loss value before clearing
            actor_loss_item = actor_loss_val.item()

            # Clear any lingering gradients/graphs
            del actor_loss_val, log_probs, entropy

            # =================================================================
            # TRAIN CRITIC (VALUE FUNCTION)
            # =================================================================

            # Recompute value predictions for gradient (separate forward pass)
            imag_value_logits_for_grad = self.world_model_critic_net(imag_feat_flat.detach())  # Detach to separate graphs
            self.world_model_critic_dist.proba_distribution(imag_value_logits_for_grad)
            imag_values_for_grad = self.world_model_critic_dist.mode().reshape(B, H)  # (B, H)

            # Normalize values using return normalizer
            values_normalized = self.value_normalizer.normalize(imag_values_for_grad, update=True)

            # Value targets (normalized returns)
            value_targets = returns_normalized.detach()

            # Compute value loss with slow regularization
            from sb3_contrib.common.dreamerv3 import compute_value_loss

            slow_values_normalized = self.value_normalizer.normalize(slow_values.detach(), update=False)
            critic_loss_val = compute_value_loss(
                value_pred=values_normalized,
                value_target=value_targets,
                weights=weights.detach(),
                slow_value_pred=slow_values_normalized.detach(),
                slow_reg=1.0,
            )

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss_val.backward()
            th.nn.utils.clip_grad_norm_(self.world_model_critic_net.parameters(), max_norm=100.0)
            self.critic_optimizer.step()

            # Update slow critic (EMA update)
            with th.no_grad():
                for slow_param, param in zip(self.slow_critic_net.parameters(), self.world_model_critic_net.parameters()):
                    slow_param.data.copy_(self.slow_critic_tau * slow_param.data + (1 - self.slow_critic_tau) * param.data)

            # Store losses
            losses["world_model"].append(world_model_loss.item())
            losses["dyn_loss"].append(dyn_loss.item())
            losses["rep_loss"].append(rep_loss.item())
            losses["rec_loss"].append(rec_loss.item())
            losses["rew_loss"].append(rew_loss.item())
            losses["con_loss"].append(con_loss.item())
            losses["actor_loss"].append(actor_loss_item)
            losses["critic_loss"].append(critic_loss_val.item())

        # Log training info
        self._n_updates += gradient_steps
        if hasattr(self, "_logger") and self._logger is not None:
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            # Only log if we have data (avoid "mean of empty slice" warning)
            if losses["world_model"]:
                self.logger.record("train/world_model_loss", np.mean(losses["world_model"]))
            if losses["dyn_loss"]:
                self.logger.record("train/dyn_loss", np.mean(losses["dyn_loss"]))
            if losses["rep_loss"]:
                self.logger.record("train/rep_loss", np.mean(losses["rep_loss"]))
            if losses["rec_loss"]:
                self.logger.record("train/rec_loss", np.mean(losses["rec_loss"]))
            if losses["rew_loss"]:
                self.logger.record("train/rew_loss", np.mean(losses["rew_loss"]))
            if losses["con_loss"]:
                self.logger.record("train/con_loss", np.mean(losses["con_loss"]))
            if losses["actor_loss"]:
                self.logger.record("train/actor_loss", np.mean(losses["actor_loss"]))
            if losses["critic_loss"]:
                self.logger.record("train/critic_loss", np.mean(losses["critic_loss"]))

    def learn(
        self: SelfDreamerV3,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DreamerV3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDreamerV3:
        """
        Learn according to the DreamerV3 algorithm.

        :param total_timesteps: Total number of samples to train on
        :param callback: Callback(s) called at every step
        :param log_interval: Log info every n steps
        :param tb_log_name: Name for TensorBoard log
        :param reset_num_timesteps: Whether to reset timesteps
        :param progress_bar: Display progress bar
        :return: Trained model
        """
        # Initialize RSSM carry state before training starts
        if self._rssm_carry is None and self.env is not None:
            n_envs = self.env.num_envs if isinstance(self.env, VecEnv) else 1
            self._initialize_rssm_carry(n_envs)

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        """
        Returns list of parameters that should not be saved.

        :return: List of parameter names
        """
        return super()._excluded_save_params() + ["replay_buffer"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        """
        Get parameters to save for TorchSave.

        :return: State dicts to save and optimizer state dicts
        """
        state_dicts = ["policy"]

        return state_dicts, []
