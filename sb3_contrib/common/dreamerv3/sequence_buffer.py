"""
Sequence Replay Buffer for DreamerV3

This buffer stores sequences of experiences instead of individual transitions,
which is critical for temporal credit assignment in model-based RL.
"""

from typing import Any, Dict, List, NamedTuple, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class SequenceReplayBufferSamples(NamedTuple):
    """
    Samples from sequence replay buffer.
    
    All tensors have shape (batch_size, sequence_length, *dims)
    """
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    # Mask indicating valid timesteps (1) vs padding (0)
    mask: th.Tensor
    # Sequence lengths (before padding)
    seq_lengths: th.Tensor
    # Consecutive chunk index (0 for first chunk, 1 for second, etc.)
    consec: Optional[th.Tensor] = None
    # Optional: entries for replay context (dict with 'rssm', 'encoder', 'decoder' keys)
    entries: Optional[Dict[str, Dict[str, th.Tensor]]] = None


class SequenceReplayBuffer(BaseBuffer):
    """
    Replay buffer that stores and samples sequences of experiences.
    
    This is essential for DreamerV3 which requires temporal sequences for:
    - Training the world model (RSSM) on sequential data
    - Proper temporal credit assignment
    - Maintaining consistency across time steps
    
    The buffer stores complete or partial episode sequences and samples
    random sequences during training.
    
    :param buffer_size: Maximum number of sequences to store
    :param observation_space: Observation space
    :param action_space: Action space
    :param sequence_length: Length of sequences to sample
    :param replay_context: Number of context frames for replay (default: 0 = disabled)
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Not used for sequence buffer
    :param handle_timeout_termination: Not used for sequence buffer
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    # Sequence metadata
    sequence_lengths: np.ndarray  # Actual length of each sequence
    episode_ids: np.ndarray  # Track which episode each sequence belongs to
    consec_indices: np.ndarray  # Track consecutive chunk index
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        sequence_length: int = 50,
        replay_context: int = 0,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        
        self.sequence_length = sequence_length
        self.replay_context = replay_context
        
        # Buffer for accumulating current episode data
        self._current_episode_obs: List[np.ndarray] = [[] for _ in range(n_envs)]
        self._current_episode_actions: List[np.ndarray] = [[] for _ in range(n_envs)]
        self._current_episode_rewards: List[np.ndarray] = [[] for _ in range(n_envs)]
        self._current_episode_dones: List[np.ndarray] = [[] for _ in range(n_envs)]
        self._current_episode_next_obs: List[np.ndarray] = [[] for _ in range(n_envs)]
        
        # Episode counter for tracking
        self._episode_counter = 0
        
        # Initialize storage arrays
        # We store sequences, not individual transitions
        self.observations = np.zeros(
            (self.buffer_size, self.sequence_length) + self.obs_shape, 
            dtype=observation_space.dtype
        )
        self.next_observations = np.zeros(
            (self.buffer_size, self.sequence_length) + self.obs_shape,
            dtype=observation_space.dtype
        )
        
        if isinstance(action_space, spaces.Discrete):
            self.actions = np.zeros((self.buffer_size, self.sequence_length), dtype=np.int64)
        else:
            self.actions = np.zeros(
                (self.buffer_size, self.sequence_length, self.action_dim),
                dtype=np.float32
            )
        
        self.rewards = np.zeros((self.buffer_size, self.sequence_length), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.sequence_length), dtype=np.float32)
        
        # Metadata
        self.sequence_lengths = np.zeros(self.buffer_size, dtype=np.int32)
        self.episode_ids = np.zeros(self.buffer_size, dtype=np.int64)
        self.consec_indices = np.zeros(self.buffer_size, dtype=np.int32)
        
        # Entry storage for replay context (if enabled)
        # These will store RSSM, encoder, and decoder states
        self.entries: Dict[str, Dict[str, np.ndarray]] = {}
        if self.replay_context > 0:
            # Will be populated when we know the entry shapes from first sample
            self.entries_initialized = False
        else:
            self.entries_initialized = True
        
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Add a transition to the buffer.
        
        Transitions are accumulated into episodes. When an episode ends,
        it is chunked into sequences and stored in the buffer.
        
        :param obs: Observation
        :param next_obs: Next observation
        :param action: Action
        :param reward: Reward
        :param done: End of episode signal
        :param infos: Additional information
        """
        # Handle multiple environments
        for env_idx in range(self.n_envs):
            # Extract data for this environment
            env_obs = obs[env_idx] if self.n_envs > 1 else obs
            env_next_obs = next_obs[env_idx] if self.n_envs > 1 else next_obs
            env_action = action[env_idx] if self.n_envs > 1 else action
            env_reward = reward[env_idx] if self.n_envs > 1 else reward
            env_done = done[env_idx] if self.n_envs > 1 else done
            
            # Ensure observations have correct shape (remove any extra dimensions)
            # Sometimes obs[env_idx] keeps an extra dimension (batch dim)
            if env_obs.ndim > len(self.obs_shape):
                env_obs = np.squeeze(env_obs, axis=0)
            if env_next_obs.ndim > len(self.obs_shape):
                env_next_obs = np.squeeze(env_next_obs, axis=0)
            
            # Ensure actions have correct shape
            # For continuous actions, might have shape (1, action_dim) instead of (action_dim,)
            if isinstance(self.action_space, spaces.Box):
                expected_action_shape = (self.action_dim,)
                if env_action.ndim > len(expected_action_shape):
                    env_action = np.squeeze(env_action, axis=0)
            elif isinstance(self.action_space, spaces.Discrete):
                # Discrete actions should be scalars
                if env_action.ndim > 0:
                    env_action = np.squeeze(env_action)
            
            # Ensure rewards and dones are scalars
            if isinstance(env_reward, np.ndarray):
                env_reward = float(np.squeeze(env_reward))
            if isinstance(env_done, np.ndarray):
                env_done = bool(np.squeeze(env_done))
            
            # Accumulate transition
            self._current_episode_obs[env_idx].append(env_obs.copy())
            self._current_episode_next_obs[env_idx].append(env_next_obs.copy())
            self._current_episode_actions[env_idx].append(env_action.copy())
            self._current_episode_rewards[env_idx].append(env_reward)
            self._current_episode_dones[env_idx].append(env_done)
            
            # If episode ended, store sequences
            if env_done:
                self._store_episode(env_idx)
                # Clear episode buffer
                self._current_episode_obs[env_idx] = []
                self._current_episode_actions[env_idx] = []
                self._current_episode_rewards[env_idx] = []
                self._current_episode_dones[env_idx] = []
                self._current_episode_next_obs[env_idx] = []
                self._episode_counter += 1
                
            # Also store sequences if episode gets too long
            elif len(self._current_episode_obs[env_idx]) >= self.sequence_length * 2:
                # Store overlapping sequences from long episode
                self._store_partial_episode(env_idx)
    
    def _store_episode(self, env_idx: int) -> None:
        """
        Store completed episode as sequences in the buffer.
        
        Long episodes are split into overlapping sequences.
        Short episodes are stored as single sequences (with padding if needed).
        
        :param env_idx: Environment index
        """
        episode_length = len(self._current_episode_obs[env_idx])
        
        if episode_length == 0:
            return
        
        # Convert lists to arrays
        obs_array = np.array(self._current_episode_obs[env_idx])
        next_obs_array = np.array(self._current_episode_next_obs[env_idx])
        actions_array = np.array(self._current_episode_actions[env_idx])
        rewards_array = np.array(self._current_episode_rewards[env_idx])
        dones_array = np.array(self._current_episode_dones[env_idx])
        
        # Split into sequences
        # Use overlapping sequences to maximize data efficiency
        stride = max(1, self.sequence_length // 2)  # 50% overlap
        
        for start_idx in range(0, episode_length, stride):
            end_idx = min(start_idx + self.sequence_length, episode_length)
            seq_len = end_idx - start_idx
            
            # Only store if sequence has meaningful length
            if seq_len < 2:
                continue
            
            # Store sequence
            pos = self.pos
            
            # Store data with padding if needed
            self.observations[pos, :seq_len] = obs_array[start_idx:end_idx]
            self.next_observations[pos, :seq_len] = next_obs_array[start_idx:end_idx]
            self.actions[pos, :seq_len] = actions_array[start_idx:end_idx]
            self.rewards[pos, :seq_len] = rewards_array[start_idx:end_idx]
            self.dones[pos, :seq_len] = dones_array[start_idx:end_idx]
            
            # Zero-pad remaining timesteps if sequence is shorter than max length
            if seq_len < self.sequence_length:
                self.observations[pos, seq_len:] = 0
                self.next_observations[pos, seq_len:] = 0
                self.actions[pos, seq_len:] = 0
                self.rewards[pos, seq_len:] = 0
                self.dones[pos, seq_len:] = 0
            
            # Store metadata
            self.sequence_lengths[pos] = seq_len
            self.episode_ids[pos] = self._episode_counter
            self.consec_indices[pos] = 0  # Will be set properly during sampling with replay context
            
            # Increment position
            self.pos += 1
            if self.pos == self.buffer_size:
                self.full = True
                self.pos = 0
            
            # Stop if we've reached the end of the episode
            if end_idx >= episode_length:
                break
    
    def _store_partial_episode(self, env_idx: int) -> None:
        """
        Store sequences from ongoing long episode and clear old data.
        
        This prevents memory issues with very long episodes.
        
        :param env_idx: Environment index
        """
        episode_length = len(self._current_episode_obs[env_idx])
        
        if episode_length < self.sequence_length:
            return
        
        # Store sequences up to episode_length - sequence_length
        # Keep the most recent sequence_length steps in buffer
        store_up_to = episode_length - self.sequence_length
        
        # Convert lists to arrays (only the part we're storing)
        obs_array = np.array(self._current_episode_obs[env_idx][:store_up_to])
        next_obs_array = np.array(self._current_episode_next_obs[env_idx][:store_up_to])
        actions_array = np.array(self._current_episode_actions[env_idx][:store_up_to])
        rewards_array = np.array(self._current_episode_rewards[env_idx][:store_up_to])
        dones_array = np.array(self._current_episode_dones[env_idx][:store_up_to])
        
        # Split into sequences with overlap
        stride = self.sequence_length // 2
        
        for start_idx in range(0, store_up_to, stride):
            end_idx = min(start_idx + self.sequence_length, store_up_to)
            seq_len = end_idx - start_idx
            
            if seq_len < self.sequence_length // 2:  # Skip very short sequences
                continue
            
            # Store sequence
            pos = self.pos
            
            self.observations[pos, :seq_len] = obs_array[start_idx:end_idx]
            self.next_observations[pos, :seq_len] = next_obs_array[start_idx:end_idx]
            self.actions[pos, :seq_len] = actions_array[start_idx:end_idx]
            self.rewards[pos, :seq_len] = rewards_array[start_idx:end_idx]
            self.dones[pos, :seq_len] = dones_array[start_idx:end_idx]
            
            # Pad if necessary
            if seq_len < self.sequence_length:
                self.observations[pos, seq_len:] = 0
                self.next_observations[pos, seq_len:] = 0
                self.actions[pos, seq_len:] = 0
                self.rewards[pos, seq_len:] = 0
                self.dones[pos, seq_len:] = 0
            
            self.sequence_lengths[pos] = seq_len
            self.episode_ids[pos] = self._episode_counter
            self.consec_indices[pos] = 0
            
            self.pos += 1
            if self.pos == self.buffer_size:
                self.full = True
                self.pos = 0
        
        # Keep only the most recent sequence_length steps
        self._current_episode_obs[env_idx] = self._current_episode_obs[env_idx][-self.sequence_length:]
        self._current_episode_actions[env_idx] = self._current_episode_actions[env_idx][-self.sequence_length:]
        self._current_episode_rewards[env_idx] = self._current_episode_rewards[env_idx][-self.sequence_length:]
        self._current_episode_dones[env_idx] = self._current_episode_dones[env_idx][-self.sequence_length:]
        self._current_episode_next_obs[env_idx] = self._current_episode_next_obs[env_idx][-self.sequence_length:]
    
    def sample(
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None,
    ) -> SequenceReplayBufferSamples:
        """
        Sample a batch of sequences from the buffer.
        
        :param batch_size: Number of sequences to sample
        :param env: Associated VecEnv for normalization (not used)
        :return: Batch of sequence samples
        """
        # Sample random sequence indices
        if self.full:
            indices = np.random.randint(0, self.buffer_size, size=batch_size)
        else:
            indices = np.random.randint(0, self.pos, size=batch_size)
        
        return self._get_samples(indices, env=env)
    
    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> SequenceReplayBufferSamples:
        """
        Get samples from buffer at specified indices.
        
        :param batch_inds: Indices of sequences to retrieve
        :param env: Associated VecEnv (not used)
        :return: Batch of sequence samples
        """
        # Get sequence lengths for this batch
        seq_lengths = self.sequence_lengths[batch_inds]
        consec = self.consec_indices[batch_inds]
        
        # Create mask for valid timesteps
        # Shape: (batch_size, sequence_length)
        mask = np.arange(self.sequence_length)[None, :] < seq_lengths[:, None]
        
        # Convert to tensors
        observations = self.to_torch(self.observations[batch_inds])
        next_observations = self.to_torch(self.next_observations[batch_inds])
        actions = self.to_torch(self.actions[batch_inds])
        rewards = self.to_torch(self.rewards[batch_inds])
        dones = self.to_torch(self.dones[batch_inds])
        mask_tensor = self.to_torch(mask.astype(np.float32))
        seq_lengths_tensor = self.to_torch(seq_lengths.astype(np.int64))
        consec_tensor = self.to_torch(consec.astype(np.int32))
        
        # Prepare entries if replay context is enabled
        entries = None
        if self.replay_context > 0 and self.entries_initialized:
            entries = {}
            for module_name, module_entries in self.entries.items():
                entries[module_name] = {
                    key: self.to_torch(arr[batch_inds])
                    for key, arr in module_entries.items()
                }
        
        return SequenceReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards.unsqueeze(-1),  # Add feature dimension
            mask=mask_tensor,
            seq_lengths=seq_lengths_tensor,
            consec=consec_tensor,
            entries=entries,
        )
    
    def size(self) -> int:
        """
        Get the current size of the buffer (number of sequences stored).
        
        :return: Number of sequences in buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos
    
    def reset(self) -> None:
        """
        Reset the buffer, clearing all stored sequences.
        """
        super().reset()
        self.sequence_lengths = np.zeros(self.buffer_size, dtype=np.int32)
        self.episode_ids = np.zeros(self.buffer_size, dtype=np.int64)
        self.consec_indices = np.zeros(self.buffer_size, dtype=np.int32)
        self._current_episode_obs = [[] for _ in range(self.n_envs)]
        self._current_episode_actions = [[] for _ in range(self.n_envs)]
        self._current_episode_rewards = [[] for _ in range(self.n_envs)]
        self._current_episode_dones = [[] for _ in range(self.n_envs)]
        self._current_episode_next_obs = [[] for _ in range(self.n_envs)]
        self._episode_counter = 0
        if self.replay_context > 0:
            self.entries = {}
            self.entries_initialized = False
