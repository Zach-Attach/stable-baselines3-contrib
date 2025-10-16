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

        # Setup optimizers for different components - will be updated in _setup_dreamerv3_components
        self.model_optimizer = None
        self.actor_optimizer = None
        self.critic_optimizer = None

    def _setup_dreamerv3_components(self) -> None:
        """Setup DreamerV3-specific components (RSSM, Encoder, Decoder, etc.)."""
        from sb3_contrib.common.dreamerv3 import (
            RSSM,
            Encoder,
            Decoder,
            ValueNormalizer,
            SlowValueNetwork,
        )
        from sb3_contrib.common.dreamerv3.distributions import SymexpTwoHotDistribution
        from stable_baselines3.common.distributions import BernoulliDistribution

        # Create RSSM (world model)
        self.rssm = RSSM(
            action_space=self.action_space,
            deter_dim=4096,
            stoch_dim=32,
            num_classes=32,
            hidden_dim=2048,
            num_blocks=8,
        ).to(self.device)

        # Create Encoder
        self.encoder = Encoder(
            observation_space=self.observation_space,
            hidden_dim=1024,
            num_layers=3,
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

        # Create normalizers
        self.value_normalizer = ValueNormalizer().to(self.device)
        self.return_normalizer = ValueNormalizer().to(self.device)
        self.advantage_normalizer = ValueNormalizer().to(self.device)

        # Create slow value network (if policy has critic_net)
        if hasattr(self.policy, "critic_net"):
            self.slow_value = SlowValueNetwork(self.policy.critic_net, tau=0.98).to(self.device)

        # Setup optimizers with all world model components
        world_model_params = (
            list(self.rssm.parameters()) + 
            list(self.encoder.parameters()) + 
            list(self.decoder.parameters()) +
            list(self.world_model_reward_net.parameters()) +
            list(self.world_model_continue_net.parameters())
        )

        self.model_optimizer = th.optim.Adam(world_model_params, lr=self.model_lr)

        # Actor optimizer
        if hasattr(self.policy, "actor_net"):
            self.actor_optimizer = th.optim.Adam(self.policy.actor_net.parameters(), lr=self.actor_lr)

        # Critic optimizer
        if hasattr(self.policy, "critic_net"):
            self.critic_optimizer = th.optim.Adam(self.policy.critic_net.parameters(), lr=self.critic_lr)

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the policy.

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
            # Note: no gradient computation needed here
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale action from [-1, 1] to actual action space bounds
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)
            # Clip in case of numerical imprecision
            scaled_action = np.clip(scaled_action, self.action_space.low, self.action_space.high)
        else:
            scaled_action = unscaled_action

        return unscaled_action, scaled_action

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

        # Initialize components if not already done
        if not hasattr(self, "rssm"):
            self._setup_dreamerv3_components()

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
            # For now, we'll use the standard replay buffer
            # In a full implementation, this should be a sequence buffer
            replay_data = self.replay_buffer.sample(batch_size)

            # =================================================================
            # PHASE 1: WORLD MODEL TRAINING
            # =================================================================

            # Prepare data
            observations = replay_data.observations
            actions = replay_data.actions
            rewards = replay_data.rewards
            dones = replay_data.dones

            # For simplicity, treat each sample as a sequence of length 1
            # In full implementation, we'd have proper sequences
            B = batch_size
            T = 1

            # Reshape to (B, T, ...)
            if len(observations.shape) == 2:
                observations = observations.unsqueeze(1)  # (B, 1, obs_dim)
            if len(actions.shape) == 1:
                actions = actions.unsqueeze(-1)  # (B, action_dim)
            actions = actions.unsqueeze(1)  # (B, 1, action_dim)
            rewards = rewards.unsqueeze(1)  # (B, 1, 1)
            dones = dones.unsqueeze(1)  # (B, 1, 1)

            # Encode observations to tokens
            tokens = self.encoder(observations.squeeze(1))  # (B, token_dim)
            tokens = tokens.unsqueeze(1)  # (B, 1, token_dim)

            # Initialize or get RSSM state
            if not hasattr(self, "_rssm_state"):
                self._rssm_state = self.rssm.initial(B, self.device)

            # RSSM observe step (posterior path)
            resets = dones.squeeze(-1)  # (B, 1)
            carry, entries, features = self.rssm.observe(self._rssm_state, tokens, actions, resets, training=True)

            # Get features (deter + stoch)
            deter = features["deter"]  # (B, T, deter_dim)
            stoch = features["stoch"]  # (B, T, stoch_dim, num_classes)
            feat = self.rssm.get_feat(deter.squeeze(1), stoch.squeeze(1))  # (B, feat_dim)

            # Decode observations
            recon_obs = self.decoder(feat)

            # Compute reconstruction loss
            target_obs = observations.squeeze(1)
            rec_loss = self.decoder.reconstruction_loss(recon_obs, target_obs)

            # Predict rewards and continues using world model networks
            reward_logits = self.world_model_reward_net(feat)  # (B, 255) - logits for TwoHot distribution
            continue_pred = self.world_model_continue_net(feat)  # (B, 1) - logits for Bernoulli

            # Compute reward loss using TwoHot distribution
            # The reward_dist expects logits and computes cross-entropy with soft targets
            target_rewards = rewards.squeeze(1)  # (B, 1)
            self.world_model_reward_dist.proba_distribution(reward_logits)
            rew_loss = -self.world_model_reward_dist.log_prob(target_rewards).mean()

            # Compute continue loss (BCE)
            target_continues = (1 - dones.squeeze(1).squeeze(-1)).float()  # (B,)
            con_loss = F.binary_cross_entropy_with_logits(continue_pred.squeeze(), target_continues)

            # Compute KL losses
            posterior_logits = features["logits"]  # (B, T, stoch_dim, num_classes)
            prior_logits = self.rssm._prior(deter)  # (B, T, stoch_dim, num_classes)
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
            
            # NOTE: Actor-critic training in imagination is currently disabled
            # because it requires either:
            # 1. Separate actor/critic networks that accept RSSM features, OR
            # 2. A proper architecture that maps RSSM features to policy inputs
            # For now, we only train the world model (Phase 1 above)
            
            # Placeholder losses for actor and critic
            actor_loss_val = th.tensor(0.0, device=self.device)
            critic_loss_val = th.tensor(0.0, device=self.device)
            
            # TODO: Implement proper imagination-based actor-critic training
            # This would involve:
            # - Creating world model actor/critic networks that accept RSSM features
            # - Imagining trajectories using the world model actor
            # - Computing returns and advantages
            # - Training the actor and critic on imagined data

            # Store losses
            losses["world_model"].append(world_model_loss.item())
            losses["dyn_loss"].append(dyn_loss.item())
            losses["rep_loss"].append(rep_loss.item())
            losses["rec_loss"].append(rec_loss.item())
            losses["rew_loss"].append(rew_loss.item())
            losses["con_loss"].append(con_loss.item())
            losses["actor_loss"].append(actor_loss_val.item())
            losses["critic_loss"].append(critic_loss_val.item())

        # Log training info
        self._n_updates += gradient_steps
        if hasattr(self, '_logger') and self._logger is not None:
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/world_model_loss", np.mean(losses["world_model"]))
            self.logger.record("train/dyn_loss", np.mean(losses["dyn_loss"]))
            self.logger.record("train/rep_loss", np.mean(losses["rep_loss"]))
            self.logger.record("train/rec_loss", np.mean(losses["rec_loss"]))
            self.logger.record("train/rew_loss", np.mean(losses["rew_loss"]))
            self.logger.record("train/con_loss", np.mean(losses["con_loss"]))
            self.logger.record("train/actor_loss", np.mean(losses["actor_loss"]))
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
