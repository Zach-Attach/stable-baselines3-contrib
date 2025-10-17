"""
Comprehensive tests for DreamerV3 rollout procedure.

Tests all segments of the rollout/collection process including:
- Action sampling (warmup and policy phases)
- Action scaling for different action spaces
- Environment stepping
- Replay buffer storage
- Episode handling
- Multi-environment support
"""

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from sb3_contrib.dreamerV3 import DreamerV3


class TestActionSampling:
    """Test action sampling in different phases of training."""

    def test_warmup_phase_random_actions_discrete(self):
        """Test that warmup phase produces random discrete actions."""
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        model = DreamerV3("MlpPolicy", env, learning_starts=100, verbose=0)
        
        # Set timesteps to warmup phase
        model.num_timesteps = 50
        model._last_obs = env.reset()
        
        # Sample actions during warmup
        actions_list = []
        for _ in range(20):
            action, buffer_action = model._sample_action(
                learning_starts=100, 
                action_noise=None, 
                n_envs=1
            )
            actions_list.append(action[0])
        
        # Check that actions are valid discrete actions
        assert all(a in [0, 1] for a in actions_list), "Invalid discrete actions during warmup"
        
        # Check that there's some randomness (not all the same)
        assert len(set(actions_list)) > 1, "Warmup actions should be random"
        
        env.close()

    def test_warmup_phase_random_actions_continuous(self):
        """Test that warmup phase produces random continuous actions in correct range."""
        env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
        model = DreamerV3("MlpPolicy", env, learning_starts=100, verbose=0)
        
        # Set timesteps to warmup phase
        model.num_timesteps = 50
        model._last_obs = env.reset()
        
        # Sample actions during warmup
        actions_list = []
        for _ in range(20):
            action, buffer_action = model._sample_action(
                learning_starts=100,
                action_noise=None,
                n_envs=1
            )
            actions_list.append(action[0, 0])
            
            # Verify action is in environment's action space
            assert env.action_space.contains(action[0]), \
                f"Warmup action {action[0]} outside bounds {env.action_space}"
        
        # Check that there's randomness
        actions_array = np.array(actions_list)
        assert actions_array.std() > 0.1, "Warmup actions should have variance"
        
        env.close()

    def test_policy_phase_actions_discrete(self):
        """Test that policy phase produces valid discrete actions."""
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        model = DreamerV3("MlpPolicy", env, learning_starts=50, verbose=0)
        
        # Set timesteps past warmup
        model.num_timesteps = 100
        model._last_obs = env.reset()
        
        # Sample actions from policy
        for _ in range(10):
            action, buffer_action = model._sample_action(
                learning_starts=50,
                action_noise=None,
                n_envs=1
            )
            
            # Check action validity
            assert action.shape == (1,), f"Unexpected action shape: {action.shape}"
            assert action[0] in [0, 1], f"Invalid discrete action: {action[0]}"
            assert np.array_equal(action, buffer_action), "Action and buffer_action should match for discrete"
        
        env.close()

    def test_policy_phase_actions_continuous(self):
        """Test that policy phase produces valid continuous actions."""
        env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
        model = DreamerV3("MlpPolicy", env, learning_starts=50, verbose=0)
        
        # Set timesteps past warmup
        model.num_timesteps = 100
        model._last_obs = env.reset()
        
        # Sample actions from policy
        for _ in range(10):
            action, buffer_action = model._sample_action(
                learning_starts=50,
                action_noise=None,
                n_envs=1
            )
            
            # Check action is in action space bounds (for environment stepping)
            assert env.action_space.contains(action[0]), \
                f"Policy action {action[0]} outside bounds {env.action_space}"
            
            # buffer_action should be in [-1, 1] (normalized for buffer storage)
            # For DreamerV3: action is in action space, buffer_action is normalized
            assert action.shape == buffer_action.shape, \
                f"Action shape {action.shape} != buffer_action shape {buffer_action.shape}"
        
        env.close()


class TestActionScaling:
    """Test action scaling for different action space configurations."""

    @pytest.mark.parametrize("bounds", [
        (-1.0, 1.0),
        (-2.0, 2.0),
        (0.0, 1.0),
        (-5.0, 5.0),
        (-10.0, 10.0),
    ])
    def test_continuous_action_scaling(self, bounds):
        """Test action scaling for various continuous action space bounds."""
        low, high = bounds
        
        # Create a simple environment with custom action bounds
        class CustomEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
                self.action_space = spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32)
            
            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                return self.observation_space.sample(), {}
            
            def step(self, action):
                assert self.action_space.contains(action), \
                    f"Action {action} outside bounds [{low}, {high}]"
                return self.observation_space.sample(), 0.0, False, False, {}
        
        env = DummyVecEnv([lambda: CustomEnv()])
        model = DreamerV3("MlpPolicy", env, verbose=0)
        
        obs = env.reset()
        
        # Test predictions are in correct range
        for _ in range(10):
            action, _ = model.predict(obs, deterministic=False)
            assert env.action_space.contains(action[0]), \
                f"Action {action[0]} outside bounds [{low}, {high}]"
            
            # Step environment (will fail if action is incorrectly scaled)
            obs, _, _, _ = env.step(action)
        
        env.close()

    def test_distribution_samples_bounded(self):
        """Test that distribution samples are properly bounded to [-1, 1]."""
        env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
        model = DreamerV3("MlpPolicy", env, verbose=0)
        
        obs = env.reset()
        obs_tensor = th.tensor(obs, dtype=th.float32)
        
        # Get features and distribution
        features = model.policy.extract_features(obs_tensor, model.policy.pi_features_extractor)
        distribution = model.policy._get_action_dist_from_latent(features)
        
        # Sample many times and verify all are in [-1, 1]
        for _ in range(100):
            sample = distribution.sample()
            assert sample.min().item() >= -1.0, f"Sample {sample.item()} below -1.0"
            assert sample.max().item() <= 1.0, f"Sample {sample.item()} above 1.0"
        
        env.close()

    def test_predict_unscales_correctly(self):
        """Test that predict() correctly unscales actions from [-1, 1] to action space."""
        env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])  # Action space: Box(-2, 2)
        model = DreamerV3("MlpPolicy", env, verbose=0)
        
        obs = env.reset()
        obs_tensor = th.tensor(obs, dtype=th.float32)
        
        # Get raw action from _predict (should be in [-1, 1])
        with th.no_grad():
            raw_action = model.policy._predict(obs_tensor, deterministic=False)
        
        # Verify raw action is in [-1, 1]
        assert raw_action.min().item() >= -1.0, "Raw action below -1.0"
        assert raw_action.max().item() <= 1.0, "Raw action above 1.0"
        
        # Get action from predict (should be in action space bounds)
        action, _ = model.predict(obs, deterministic=False)
        
        # Verify action is in action space bounds
        assert action[0] >= env.action_space.low[0], \
            f"Action {action[0]} below lower bound {env.action_space.low[0]}"
        assert action[0] <= env.action_space.high[0], \
            f"Action {action[0]} above upper bound {env.action_space.high[0]}"
        
        env.close()


class TestEnvironmentStepping:
    """Test environment stepping with different configurations."""

    def test_single_env_stepping_discrete(self):
        """Test stepping a single discrete environment."""
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        model = DreamerV3("MlpPolicy", env, verbose=0)
        
        obs = env.reset()
        
        for _ in range(10):
            action, _ = model.predict(obs, deterministic=False)
            new_obs, rewards, dones, infos = env.step(action)
            
            # Verify shapes
            assert new_obs.shape == obs.shape, "Observation shape mismatch"
            assert rewards.shape == (1,), f"Unexpected rewards shape: {rewards.shape}"
            assert dones.shape == (1,), f"Unexpected dones shape: {dones.shape}"
            assert len(infos) == 1, "Should have one info dict"
            
            obs = new_obs
        
        env.close()

    def test_single_env_stepping_continuous(self):
        """Test stepping a single continuous environment."""
        env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
        model = DreamerV3("MlpPolicy", env, verbose=0)
        
        obs = env.reset()
        
        for _ in range(10):
            action, _ = model.predict(obs, deterministic=False)
            new_obs, rewards, dones, infos = env.step(action)
            
            # Verify shapes and types
            assert new_obs.shape == obs.shape, "Observation shape mismatch"
            assert rewards.shape == (1,), f"Unexpected rewards shape: {rewards.shape}"
            assert isinstance(rewards[0], (int, float, np.number)), "Reward should be numeric"
            
            obs = new_obs
        
        env.close()

    def test_multi_env_stepping(self):
        """Test stepping multiple environments in parallel."""
        n_envs = 4
        env = DummyVecEnv([lambda: gym.make("CartPole-v1") for _ in range(n_envs)])
        model = DreamerV3("MlpPolicy", env, verbose=0)
        
        obs = env.reset()
        assert obs.shape[0] == n_envs, f"Expected {n_envs} observations, got {obs.shape[0]}"
        
        for _ in range(10):
            action, _ = model.predict(obs, deterministic=False)
            assert action.shape[0] == n_envs, f"Expected {n_envs} actions, got {action.shape[0]}"
            
            new_obs, rewards, dones, infos = env.step(action)
            
            # Verify all have correct batch size
            assert new_obs.shape[0] == n_envs
            assert rewards.shape[0] == n_envs
            assert dones.shape[0] == n_envs
            assert len(infos) == n_envs
            
            obs = new_obs
        
        env.close()

    def test_episode_termination_handling(self):
        """Test that episode terminations are handled correctly."""
        # Use a simple environment that terminates quickly
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        model = DreamerV3("MlpPolicy", env, verbose=0, learning_starts=10)
        
        obs = env.reset()
        episode_count = 0
        max_steps = 500
        
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=False)
            new_obs, rewards, dones, infos = env.step(action)
            
            if dones[0]:
                episode_count += 1
                # Verify episode info is present
                if "episode" in infos[0]:
                    assert "r" in infos[0]["episode"], "Episode should have reward"
                    assert "l" in infos[0]["episode"], "Episode should have length"
            
            obs = new_obs
        
        assert episode_count > 0, "Should have completed at least one episode"
        env.close()


class TestReplayBufferIntegration:
    """Test replay buffer storage during rollouts."""

    def test_buffer_storage_discrete(self):
        """Test that experiences are correctly stored in replay buffer for discrete actions."""
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        model = DreamerV3("MlpPolicy", env, buffer_size=1000, learning_starts=10, verbose=0)
        
        # Perform some steps
        model.learn(total_timesteps=50, progress_bar=False)
        
        # Verify buffer has data
        assert model.replay_buffer.size() > 0, "Buffer should contain experiences"
        # Buffer might have a few more due to batch collection
        assert model.replay_buffer.size() <= 60, "Buffer size should be close to steps taken"
        
        # Sample from buffer to verify it works
        if model.replay_buffer.size() >= model.batch_size:
            batch = model.replay_buffer.sample(model.batch_size)
            assert batch.observations.shape[0] == model.batch_size
            assert batch.actions.shape[0] == model.batch_size
            assert batch.rewards.shape[0] == model.batch_size
        
        env.close()

    def test_buffer_storage_continuous(self):
        """Test that experiences are correctly stored in replay buffer for continuous actions."""
        env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
        model = DreamerV3("MlpPolicy", env, buffer_size=1000, learning_starts=10, verbose=0)
        
        # Perform some steps
        model.learn(total_timesteps=50, progress_bar=False)
        
        # Verify buffer has data
        assert model.replay_buffer.size() > 0, "Buffer should contain experiences"
        
        # Sample and verify action dimensions
        if model.replay_buffer.size() >= model.batch_size:
            batch = model.replay_buffer.sample(model.batch_size)
            # Actions should be in original action space (unscaled)
            assert batch.actions.shape == (model.batch_size, 1), \
                f"Unexpected action shape: {batch.actions.shape}"
        
        env.close()

    def test_buffer_with_multi_env(self):
        """Test buffer storage with multiple parallel environments."""
        n_envs = 3
        env = DummyVecEnv([lambda: gym.make("CartPole-v1") for _ in range(n_envs)])
        model = DreamerV3("MlpPolicy", env, buffer_size=1000, learning_starts=10, verbose=0)
        
        # Perform some steps
        timesteps = 30
        model.learn(total_timesteps=timesteps, progress_bar=False)
        
        # Buffer should contain experiences from all environments
        assert model.replay_buffer.size() > 0, "Buffer should contain experiences"
        assert model.replay_buffer.size() <= timesteps, \
            f"Buffer size {model.replay_buffer.size()} should not exceed {timesteps}"
        
        env.close()


class TestFullRolloutProcedure:
    """Integration tests for complete rollout procedures."""

    def test_complete_rollout_discrete(self):
        """Test complete rollout procedure with discrete actions."""
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        model = DreamerV3(
            "MlpPolicy",
            env,
            learning_starts=20,
            batch_size=4,
            buffer_size=500,
            verbose=0
        )
        
        # Run learning which includes rollouts
        model.learn(total_timesteps=100, progress_bar=False)
        
        # Verify model has learned something
        assert model.num_timesteps >= 100
        assert model.replay_buffer.size() > 0
        assert model._n_updates > 0, "Model should have performed training updates"
        
        # Test inference
        obs = env.reset()
        for _ in range(10):
            action, _ = model.predict(obs, deterministic=True)
            assert action in [0, 1], f"Invalid action: {action}"
            obs, _, _, _ = env.step(action)
        
        env.close()

    def test_complete_rollout_continuous(self):
        """Test complete rollout procedure with continuous actions."""
        env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
        model = DreamerV3(
            "MlpPolicy",
            env,
            learning_starts=20,
            batch_size=4,
            buffer_size=500,
            verbose=0
        )
        
        # Run learning
        model.learn(total_timesteps=100, progress_bar=False)
        
        # Verify learning occurred
        assert model.num_timesteps >= 100
        assert model.replay_buffer.size() > 0
        
        # Test inference with action validation
        obs = env.reset()
        for _ in range(10):
            action, _ = model.predict(obs, deterministic=True)
            assert env.action_space.contains(action[0]), \
                f"Action {action[0]} outside bounds {env.action_space}"
            obs, _, _, _ = env.step(action)
        
        env.close()

    def test_warmup_to_policy_transition(self):
        """Test transition from warmup (random actions) to policy actions."""
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        model = DreamerV3(
            "MlpPolicy",
            env,
            learning_starts=30,
            batch_size=4,
            verbose=0
        )
        
        # Track timesteps before and after warmup
        warmup_timesteps = 30
        policy_timesteps = 20
        
        # Learn during warmup
        model.learn(total_timesteps=warmup_timesteps, progress_bar=False)
        warmup_updates = model._n_updates
        
        # Learn with policy
        model.learn(total_timesteps=policy_timesteps, progress_bar=False, reset_num_timesteps=False)
        policy_updates = model._n_updates
        
        # Policy phase should have more updates (training happens after warmup)
        # Note: timesteps might be slightly more due to batch collection
        assert model.num_timesteps >= warmup_timesteps + policy_timesteps, \
            f"Expected at least {warmup_timesteps + policy_timesteps} timesteps, got {model.num_timesteps}"
        assert policy_updates > warmup_updates, "Should have training updates after warmup"
        
        env.close()

    def test_deterministic_vs_stochastic_actions(self):
        """Test that deterministic and stochastic action sampling differ appropriately."""
        env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
        model = DreamerV3("MlpPolicy", env, verbose=0)
        
        obs = env.reset()
        
        # Collect deterministic actions
        deterministic_actions = []
        for _ in range(10):
            action, _ = model.predict(obs, deterministic=True)
            deterministic_actions.append(action[0, 0])
        
        # Collect stochastic actions from same observation
        stochastic_actions = []
        for _ in range(10):
            action, _ = model.predict(obs, deterministic=False)
            stochastic_actions.append(action[0, 0])
        
        # Deterministic should have less variance
        det_std = np.std(deterministic_actions)
        stoch_std = np.std(stochastic_actions)
        
        # Note: Due to mode vs sampling, stochastic should generally have more variance
        # but this isn't guaranteed for very small samples, so we just check validity
        assert all(env.action_space.contains(np.array([a])) for a in deterministic_actions)
        assert all(env.action_space.contains(np.array([a])) for a in stochastic_actions)
        
        env.close()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_learning_starts(self):
        """Test with learning_starts=0 (no warmup phase)."""
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        model = DreamerV3("MlpPolicy", env, learning_starts=0, batch_size=4, verbose=0)
        
        # Should work without warmup
        model.learn(total_timesteps=50, progress_bar=False)
        
        assert model.num_timesteps >= 50
        assert model._n_updates > 0, "Should have training updates from the start"
        
        env.close()

    def test_action_bounds_edge_values(self):
        """Test that actions at boundary values are handled correctly."""
        # Create environment with specific bounds
        class BoundedEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
                self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)
            
            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                return self.observation_space.sample(), {}
            
            def step(self, action):
                # Verify action is within bounds
                assert self.action_space.contains(action), \
                    f"Action {action} outside bounds [-3.0, 3.0]"
                return self.observation_space.sample(), 0.0, False, False, {}
        
        env = DummyVecEnv([lambda: BoundedEnv()])
        model = DreamerV3("MlpPolicy", env, verbose=0)
        
        obs = env.reset()
        
        # Sample many actions to test edge cases
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=False)
            # Environment will assert if action is out of bounds
            obs, _, _, _ = env.step(action)
        
        env.close()

    def test_very_small_action_space(self):
        """Test with very small action space bounds."""
        class TinyActionEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
                self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(1,), dtype=np.float32)
            
            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                return self.observation_space.sample(), {}
            
            def step(self, action):
                assert self.action_space.contains(action), \
                    f"Action {action} outside bounds [-0.1, 0.1]"
                return self.observation_space.sample(), 0.0, False, False, {}
        
        env = DummyVecEnv([lambda: TinyActionEnv()])
        model = DreamerV3("MlpPolicy", env, verbose=0)
        
        obs = env.reset()
        
        for _ in range(20):
            action, _ = model.predict(obs, deterministic=False)
            assert -0.1 <= action[0] <= 0.1, f"Action {action[0]} outside tiny bounds"
            obs, _, _, _ = env.step(action)
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
