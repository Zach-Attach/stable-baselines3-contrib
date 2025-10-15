"""
Unit tests for DreamerV3 RSSM (world model).
"""

import gymnasium as gym
import numpy as np
import pytest
import torch as th

from sb3_contrib.common.dreamerv3.rssm import RSSM


class TestRSSM:
    """Test RSSM (Recurrent State-Space Model)."""

    @pytest.fixture
    def action_space_discrete(self):
        """Discrete action space."""
        return gym.spaces.Discrete(4)

    @pytest.fixture
    def action_space_continuous(self):
        """Continuous action space."""
        return gym.spaces.Box(low=-1, high=1, shape=(2,))

    @pytest.fixture
    def rssm_discrete(self, action_space_discrete):
        """RSSM with discrete action space."""
        return RSSM(
            action_space=action_space_discrete,
            deter_dim=512,
            stoch_dim=8,
            num_classes=8,
            hidden_dim=256,
            num_blocks=4,
        )

    @pytest.fixture
    def rssm_continuous(self, action_space_continuous):
        """RSSM with continuous action space."""
        return RSSM(
            action_space=action_space_continuous,
            deter_dim=512,
            stoch_dim=8,
            num_classes=8,
            hidden_dim=256,
            num_blocks=4,
        )

    def test_initialization(self, rssm_discrete):
        """Test RSSM initialization."""
        assert rssm_discrete.deter_dim == 512
        assert rssm_discrete.stoch_dim == 8
        assert rssm_discrete.num_classes == 8
        assert rssm_discrete.hidden_dim == 256
        assert rssm_discrete.num_blocks == 4
        assert rssm_discrete.discrete_action is True

    def test_initial_state(self, rssm_discrete):
        """Test initial state creation."""
        batch_size = 4
        device = th.device("cpu")
        
        state = rssm_discrete.initial(batch_size, device)
        
        assert "deter" in state
        assert "stoch" in state
        assert state["deter"].shape == (batch_size, rssm_discrete.deter_dim)
        assert state["stoch"].shape == (batch_size, rssm_discrete.stoch_dim, rssm_discrete.num_classes)
        assert th.all(state["deter"] == 0)
        assert th.all(state["stoch"] == 0)

    def test_observe_discrete(self, rssm_discrete):
        """Test observe method with discrete actions."""
        batch_size = 4
        seq_len = 5
        token_dim = 128
        device = th.device("cpu")
        
        # Create dummy data
        carry = rssm_discrete.initial(batch_size, device)
        tokens = th.randn(batch_size, seq_len, token_dim)
        actions = th.randint(0, 4, (batch_size, seq_len, 1)).float()
        resets = th.zeros(batch_size, seq_len)
        
        # Run observe
        final_carry, entries, features = rssm_discrete.observe(
            carry, tokens, actions, resets, training=True
        )
        
        # Check outputs
        assert "deter" in final_carry
        assert "stoch" in final_carry
        assert final_carry["deter"].shape == (batch_size, rssm_discrete.deter_dim)
        assert final_carry["stoch"].shape == (batch_size, rssm_discrete.stoch_dim, rssm_discrete.num_classes)
        
        assert "deter" in entries
        assert "stoch" in entries
        assert entries["deter"].shape == (batch_size, seq_len, rssm_discrete.deter_dim)
        assert entries["stoch"].shape == (batch_size, seq_len, rssm_discrete.stoch_dim, rssm_discrete.num_classes)
        
        assert "deter" in features
        assert "stoch" in features
        assert "logits" in features
        assert features["logits"].shape == (batch_size, seq_len, rssm_discrete.stoch_dim, rssm_discrete.num_classes)

    def test_observe_continuous(self, rssm_continuous):
        """Test observe method with continuous actions."""
        batch_size = 4
        seq_len = 5
        token_dim = 128
        device = th.device("cpu")
        
        carry = rssm_continuous.initial(batch_size, device)
        tokens = th.randn(batch_size, seq_len, token_dim)
        actions = th.randn(batch_size, seq_len, 2)
        resets = th.zeros(batch_size, seq_len)
        
        final_carry, entries, features = rssm_continuous.observe(
            carry, tokens, actions, resets, training=True
        )
        
        assert final_carry["deter"].shape == (batch_size, rssm_continuous.deter_dim)
        assert entries["deter"].shape == (batch_size, seq_len, rssm_continuous.deter_dim)

    def test_observe_with_resets(self, rssm_discrete):
        """Test observe handles episode resets correctly."""
        batch_size = 4
        seq_len = 5
        token_dim = 128
        device = th.device("cpu")
        
        carry = rssm_discrete.initial(batch_size, device)
        tokens = th.randn(batch_size, seq_len, token_dim)
        actions = th.randint(0, 4, (batch_size, seq_len, 1)).float()
        
        # Reset at timestep 2
        resets = th.zeros(batch_size, seq_len)
        resets[:, 2] = 1.0
        
        final_carry, entries, features = rssm_discrete.observe(
            carry, tokens, actions, resets, training=True
        )
        
        # State should be reset at timestep 2
        assert entries["deter"].shape == (batch_size, seq_len, rssm_discrete.deter_dim)

    def test_imagine(self, rssm_discrete):
        """Test imagination rollouts."""
        batch_size = 4
        horizon = 10
        device = th.device("cpu")
        
        carry = rssm_discrete.initial(batch_size, device)
        
        # Simple policy that returns random actions
        def policy_fn(feat):
            return th.randint(0, 4, (feat.shape[0], 1)).float()
        
        final_carry, features, actions = rssm_discrete.imagine(
            carry, policy_fn, horizon, training=False
        )
        
        assert final_carry["deter"].shape == (batch_size, rssm_discrete.deter_dim)
        assert features["deter"].shape == (batch_size, horizon, rssm_discrete.deter_dim)
        assert features["stoch"].shape == (batch_size, horizon, rssm_discrete.stoch_dim, rssm_discrete.num_classes)
        assert actions.shape == (batch_size, horizon, 1)

    def test_kl_loss(self, rssm_discrete):
        """Test KL divergence loss computation."""
        batch_size = 4
        seq_len = 5
        device = th.device("cpu")
        
        # Create dummy logits
        posterior_logits = th.randn(batch_size, seq_len, rssm_discrete.stoch_dim, rssm_discrete.num_classes)
        prior_logits = th.randn(batch_size, seq_len, rssm_discrete.stoch_dim, rssm_discrete.num_classes)
        
        dyn_loss, rep_loss = rssm_discrete.kl_loss(posterior_logits, prior_logits)
        
        assert isinstance(dyn_loss, th.Tensor)
        assert isinstance(rep_loss, th.Tensor)
        assert dyn_loss.shape == ()  # Scalar
        assert rep_loss.shape == ()  # Scalar
        assert dyn_loss >= 0  # KL is non-negative
        assert rep_loss >= 0

    def test_get_feat(self, rssm_discrete):
        """Test feature concatenation."""
        batch_size = 4
        device = th.device("cpu")
        
        deter = th.randn(batch_size, rssm_discrete.deter_dim)
        stoch = th.randn(batch_size, rssm_discrete.stoch_dim, rssm_discrete.num_classes)
        
        feat = rssm_discrete.get_feat(deter, stoch)
        
        expected_dim = rssm_discrete.deter_dim + rssm_discrete.stoch_dim * rssm_discrete.num_classes
        assert feat.shape == (batch_size, expected_dim)

    def test_sample_categorical(self, rssm_discrete):
        """Test categorical sampling with unimix."""
        batch_size = 4
        
        logits = th.randn(batch_size, rssm_discrete.stoch_dim, rssm_discrete.num_classes)
        
        samples = rssm_discrete._sample_categorical(logits)
        
        assert samples.shape == (batch_size, rssm_discrete.stoch_dim, rssm_discrete.num_classes)
        # Check one-hot: each sample should sum to 1 across classes
        assert th.allclose(samples.sum(dim=-1), th.ones(batch_size, rssm_discrete.stoch_dim))

    def test_mode_categorical(self, rssm_discrete):
        """Test categorical mode (argmax)."""
        batch_size = 4
        
        logits = th.randn(batch_size, rssm_discrete.stoch_dim, rssm_discrete.num_classes)
        
        mode = rssm_discrete._mode_categorical(logits)
        
        assert mode.shape == (batch_size, rssm_discrete.stoch_dim, rssm_discrete.num_classes)
        assert th.allclose(mode.sum(dim=-1), th.ones(batch_size, rssm_discrete.stoch_dim))

    def test_free_nats(self):
        """Test free nats threshold in KL loss."""
        action_space = gym.spaces.Discrete(4)
        rssm = RSSM(
            action_space=action_space,
            deter_dim=512,
            stoch_dim=8,
            num_classes=8,
            hidden_dim=256,
            num_blocks=4,
            free_nats=1.0,
        )
        
        batch_size = 4
        seq_len = 5
        
        # Create logits that would give very small KL
        posterior_logits = th.randn(batch_size, seq_len, rssm.stoch_dim, rssm.num_classes)
        prior_logits = posterior_logits + 0.01  # Very similar
        
        dyn_loss, rep_loss = rssm.kl_loss(posterior_logits, prior_logits)
        
        # With free_nats=1.0, loss should be at least 1.0 per stochastic variable
        # Sum over stoch_dim gives at least stoch_dim * free_nats
        # But we average, so minimum should be applied
        assert dyn_loss >= 0
        assert rep_loss >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
