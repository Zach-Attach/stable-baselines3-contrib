"""
Normalization utilities for DreamerV3.

Implements running statistics normalization for values, returns, and advantages.
"""

import torch
import torch as th
import torch.nn as nn


class RunningNormalizer(nn.Module):
    """
    Running normalization using exponential moving average (EMA).
    
    Tracks mean and standard deviation of inputs and normalizes them.
    Used for normalizing returns, values, and advantages in DreamerV3.
    
    :param momentum: EMA momentum (default: 0.99)
    :param epsilon: Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(
        self,
        momentum: float = 0.99,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        
        self.momentum = momentum
        self.epsilon = epsilon
        
        # Running statistics (initialized to None, created on first forward)
        self.register_buffer('running_mean', None)
        self.register_buffer('running_var', None)
        self.register_buffer('count', th.tensor(0, dtype=th.long))
    
    def forward(
        self,
        x: th.Tensor,
        update: bool = True,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Normalize input and optionally update running statistics.
        
        :param x: Input tensor
        :param update: Whether to update running statistics
        :return: Tuple of (normalized_x, mean, std)
        """
        # Initialize running statistics on first forward
        if self.running_mean is None:
            self.running_mean = th.zeros(1, device=x.device)
            self.running_var = th.ones(1, device=x.device)
        
        # Compute batch statistics
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        
        # Update running statistics if in training mode and update=True
        if update and self.training:
            if self.count == 0:
                # First batch: initialize with batch statistics
                self.running_mean.copy_(batch_mean)
                self.running_var.copy_(batch_var)
            else:
                # EMA update
                self.running_mean.mul_(self.momentum).add_(
                    (1 - self.momentum) * batch_mean
                )
                self.running_var.mul_(self.momentum).add_(
                    (1 - self.momentum) * batch_var
                )
            self.count += 1
        
        # Use running statistics for normalization
        mean = self.running_mean
        std = th.sqrt(self.running_var + self.epsilon)
        
        # Normalize
        normalized = (x - mean) / std
        
        return normalized, mean, std
    
    def get_stats(self) -> tuple[th.Tensor, th.Tensor]:
        """Get current running mean and std."""
        if self.running_mean is None:
            # Not initialized yet
            return th.tensor(0.0), th.tensor(1.0)
        
        mean = self.running_mean
        std = th.sqrt(self.running_var + self.epsilon)
        return mean, std


class ValueNormalizer(nn.Module):
    """
    Specialized normalizer for value functions in DreamerV3.
    
    Normalizes target values and predictions to improve learning stability.
    """
    
    def __init__(
        self,
        momentum: float = 0.99,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.normalizer = RunningNormalizer(momentum=momentum, epsilon=epsilon)
    
    def normalize(
        self,
        values: th.Tensor,
        update: bool = True,
    ) -> th.Tensor:
        """
        Normalize values.
        
        :param values: Value tensor
        :param update: Whether to update statistics
        :return: Normalized values
        """
        normalized, _, _ = self.normalizer(values, update=update)
        return normalized
    
    def denormalize(self, normalized_values: th.Tensor) -> th.Tensor:
        """
        Denormalize values back to original scale.
        
        :param normalized_values: Normalized value tensor
        :return: Denormalized values
        """
        mean, std = self.normalizer.get_stats()
        return normalized_values * std + mean
    
    def get_stats(self) -> tuple[th.Tensor, th.Tensor]:
        """Get current mean and std."""
        return self.normalizer.get_stats()


class SlowValueNetwork(nn.Module):
    """
    Slow-updating target network for value function.
    
    Maintains a slowly updating copy of the value network for stable targets.
    Uses exponential moving average (EMA) to update weights.
    
    :param value_network: The main value network to track
    :param tau: EMA coefficient for weight updates (default: 0.98)
    """
    
    def __init__(
        self,
        value_network: nn.Module,
        tau: float = 0.98,
    ):
        super().__init__()
        
        self.value_network = value_network
        self.tau = tau
        
        # Create a copy of the value network
        self.target_network = self._create_copy(value_network)
        
        # Freeze target network (no gradient computation)
        for param in self.target_network.parameters():
            param.requires_grad = False
    
    def _create_copy(self, network: nn.Module) -> nn.Module:
        """Create a deep copy of the network."""
        import copy
        return copy.deepcopy(network)
    
    def forward(self, *args, **kwargs):
        """Forward pass through target network."""
        return self.target_network(*args, **kwargs)
    
    def update(self):
        """
        Update target network weights using EMA.
        
        target = tau * target + (1 - tau) * source
        """
        with th.no_grad():
            for target_param, source_param in zip(
                self.target_network.parameters(),
                self.value_network.parameters()
            ):
                target_param.data.mul_(self.tau).add_(
                    (1 - self.tau) * source_param.data
                )
