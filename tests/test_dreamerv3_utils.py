"""
Unit tests for DreamerV3 normalizers and utilities.
"""

import pytest
import torch as th

from sb3_contrib.common.dreamerv3.normalizers import (
    RunningNormalizer,
    SlowValueNetwork,
    ValueNormalizer,
)
from sb3_contrib.common.dreamerv3.utils import (
    compute_actor_loss,
    compute_advantages,
    compute_episode_weights,
    compute_value_loss,
    lambda_return,
    symexp,
    symlog,
)


class TestRunningNormalizer:
    """Test RunningNormalizer."""

    @pytest.fixture
    def normalizer(self):
        """Create a running normalizer."""
        return RunningNormalizer(momentum=0.99, epsilon=1e-8)

    def test_initialization(self, normalizer):
        """Test normalizer initialization."""
        assert normalizer.momentum == 0.99
        assert normalizer.epsilon == 1e-8
        assert normalizer.running_mean is None
        assert normalizer.running_var is None

    def test_first_forward(self, normalizer):
        """Test first forward pass initializes statistics."""
        x = th.randn(100)
        
        normalized, mean, std = normalizer(x, update=True)
        
        assert normalizer.running_mean is not None
        assert normalizer.running_var is not None
        assert normalized.shape == x.shape

    def test_normalization(self, normalizer):
        """Test that normalization produces zero mean, unit variance."""
        # Initialize with some data
        x1 = th.randn(1000) * 10 + 5
        normalizer(x1, update=True)
        
        # Normalize new data
        x2 = th.randn(100) * 10 + 5
        normalized, mean, std = normalizer(x2, update=False)
        
        # Normalized data should have approximately zero mean
        assert th.abs(normalized.mean()) < 1.0

    def test_ema_update(self, normalizer):
        """Test exponential moving average updates."""
        x1 = th.randn(100)
        x2 = th.randn(100) + 10  # Different distribution
        
        normalizer(x1, update=True)
        mean1 = normalizer.running_mean.clone()
        
        normalizer(x2, update=True)
        mean2 = normalizer.running_mean.clone()
        
        # Mean should have moved towards x2's mean
        assert mean2 > mean1


class TestValueNormalizer:
    """Test ValueNormalizer."""

    @pytest.fixture
    def value_normalizer(self):
        """Create a value normalizer."""
        return ValueNormalizer(momentum=0.99)

    def test_normalize(self, value_normalizer):
        """Test value normalization."""
        values = th.randn(100)
        
        normalized = value_normalizer.normalize(values, update=True)
        
        assert normalized.shape == values.shape

    def test_denormalize(self, value_normalizer):
        """Test value denormalization."""
        values = th.randn(100)
        
        normalized = value_normalizer.normalize(values, update=True)
        denormalized = value_normalizer.denormalize(normalized)
        
        # Should approximately recover original values
        assert th.allclose(denormalized, values, atol=0.5)


class TestSlowValueNetwork:
    """Test SlowValueNetwork."""

    @pytest.fixture
    def value_network(self):
        """Create a simple value network."""
        return th.nn.Sequential(
            th.nn.Linear(10, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, 1),
        )

    @pytest.fixture
    def slow_value(self, value_network):
        """Create a slow value network."""
        return SlowValueNetwork(value_network, tau=0.9)

    def test_initialization(self, slow_value, value_network):
        """Test slow value network initialization."""
        assert slow_value.tau == 0.9
        
        # Target network should be a copy
        for p in slow_value.target_network.parameters():
            assert not p.requires_grad

    def test_forward(self, slow_value):
        """Test forward pass through target network."""
        x = th.randn(4, 10)
        
        output = slow_value(x)
        
        assert output.shape == (4, 1)

    def test_update(self, slow_value, value_network):
        """Test EMA weight update."""
        # Get initial target weights
        target_params_before = [p.clone() for p in slow_value.target_network.parameters()]
        
        # Change source network
        for p in value_network.parameters():
            p.data += 1.0
        
        # Update target
        slow_value.update()
        
        # Target should have moved towards source
        target_params_after = list(slow_value.target_network.parameters())
        
        for before, after in zip(target_params_before, target_params_after):
            assert not th.allclose(before, after)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_symlog(self):
        """Test symlog transformation."""
        x = th.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        
        y = symlog(x)
        
        # Preserves sign
        assert th.all((x >= 0) == (y >= 0))
        # Zero maps to zero
        assert y[2] == 0
        # Compresses large values
        assert th.abs(y[0]) < th.abs(x[0])

    def test_symexp(self):
        """Test symexp transformation."""
        x = th.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        
        y = symexp(x)
        
        # Preserves sign
        assert th.all((x >= 0) == (y >= 0))
        # Zero maps to zero
        assert y[2] == 0

    def test_symlog_symexp_inverse(self):
        """Test that symexp is inverse of symlog."""
        x = th.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
        
        y = symlog(x)
        x_recovered = symexp(y)
        
        assert th.allclose(x, x_recovered, atol=1e-5)

    def test_lambda_return(self):
        """Test lambda return computation."""
        batch_size = 4
        horizon = 10
        
        rewards = th.randn(batch_size, horizon)
        values = th.randn(batch_size, horizon + 1)
        continues = th.ones(batch_size, horizon)
        bootstrap = values[:, -1]
        
        returns = lambda_return(
            rewards, values, continues, bootstrap,
            lambda_=0.95, gamma=0.99
        )
        
        assert returns.shape == (batch_size, horizon)
        # Returns should be finite
        assert th.isfinite(returns).all()

    def test_compute_advantages(self):
        """Test advantage computation."""
        batch_size = 4
        horizon = 10
        
        returns = th.randn(batch_size, horizon)
        values = th.randn(batch_size, horizon)
        
        advantages = compute_advantages(returns, values)
        
        assert advantages.shape == (batch_size, horizon)
        assert th.allclose(advantages, returns - values)

    def test_compute_actor_loss(self):
        """Test actor loss computation."""
        batch_size = 4
        horizon = 10
        
        log_probs = th.randn(batch_size, horizon)
        advantages = th.randn(batch_size, horizon)
        entropies = th.rand(batch_size, horizon)
        weights = th.ones(batch_size, horizon)
        
        loss = compute_actor_loss(
            log_probs, advantages, entropies, weights,
            entropy_coef=3e-4
        )
        
        assert isinstance(loss, th.Tensor)
        assert loss.shape == ()
        assert th.isfinite(loss)

    def test_compute_value_loss(self):
        """Test value loss computation."""
        batch_size = 4
        horizon = 10
        
        value_pred = th.randn(batch_size, horizon)
        value_target = th.randn(batch_size, horizon)
        weights = th.ones(batch_size, horizon)
        
        loss = compute_value_loss(value_pred, value_target, weights)
        
        assert isinstance(loss, th.Tensor)
        assert loss.shape == ()
        assert loss >= 0  # MSE is non-negative

    def test_compute_value_loss_with_slow_value(self):
        """Test value loss with slow value regularization."""
        batch_size = 4
        horizon = 10
        
        value_pred = th.randn(batch_size, horizon)
        value_target = th.randn(batch_size, horizon)
        slow_value_pred = th.randn(batch_size, horizon)
        weights = th.ones(batch_size, horizon)
        
        loss = compute_value_loss(
            value_pred, value_target, weights,
            slow_value_pred=slow_value_pred,
            slow_reg=1.0
        )
        
        assert loss >= 0

    def test_compute_episode_weights(self):
        """Test episode weight computation."""
        batch_size = 4
        horizon = 10
        
        continues = th.ones(batch_size, horizon)
        continues[:, 5] = 0  # Episode ends at timestep 5
        
        weights = compute_episode_weights(continues, gamma=0.99)
        
        assert weights.shape == (batch_size, horizon)
        # Weights should decrease over time (before episode end)
        assert th.all(weights[:, 1:5] <= weights[:, :4] + 1e-6)
        # After episode end (continue=0), cumulative product includes 0
        # So weights become 0
        assert th.all(weights[:, 6:] == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
