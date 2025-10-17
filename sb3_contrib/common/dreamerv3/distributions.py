"""Custom distributions for DreamerV3 implementation."""
from typing import Optional

import torch
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.distributions import DiagGaussianDistribution, Distribution
from torch import nn

class BoundedDiagGaussianDistribution(Distribution):
    """
    A custom DiagGaussian distribution that applies a tanh bounding
    to the mean and uses the specific stddev scaling from DreamerV3.
    
    This distribution is used for continuous action spaces in DreamerV3.
    It applies tanh to bound the mean and uses a sigmoid-based transformation
    for the standard deviation.
    
    :param action_dim: Dimension of the action space
    :param min_std: Minimum standard deviation (default: 1e-6)
    :param max_std: Maximum standard deviation (default: 1.0)
    """
    def __init__(self, action_dim: int, min_std: float = 1e-6, max_std: float = 1.0):
        super().__init__()
        self.action_dim = action_dim
        self.distribution = None
        self.min_std = min_std
        self.max_std = max_std
        self.gaussian_actions = None


    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Creates the network head that outputs mean and std parameters.
        
        :param latent_dim: Dimension of the input latent vector
        :return: Module that outputs mean and raw std parameters
        """
        class ActionNet(nn.Module):
            def __init__(self, latent_dim: int, action_dim: int):
                super().__init__()
                self.mean_net = nn.Linear(latent_dim, action_dim)
                self.std_net = nn.Linear(latent_dim, action_dim)
            
            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return self.mean_net(x), self.std_net(x)
        
        return ActionNet(latent_dim, self.action_dim)
    
    def proba_distribution(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> "BoundedDiagGaussianDistribution":
        """
        Set the parameters of the distribution and create the internal Gaussian.
        
        :param mean_actions: Raw mean outputs from the network (pre-tanh)
        :param log_std: Raw std outputs from the network (pre-transformation)
        :return: self
        """
        # Transform std: stddev = (max_std - min_std) * sigmoid(raw_std + 2.0) + min_std
        sigmoid_transformed = torch.sigmoid(log_std + 2.0)
        stddev = (self.max_std - self.min_std) * sigmoid_transformed + self.min_std
        
        # Bound mean with tanh
        bounded_mean = torch.tanh(mean_actions)
        
        # Store for later use
        self.gaussian_actions = bounded_mean
        
        # Create internal distribution
        self.distribution = torch.distributions.Normal(bounded_mean, stddev)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log probability of actions under the distribution.
        
        :param actions: Actions to evaluate
        :return: Log probabilities
        """
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call proba_distribution first.")
        
        # Sum over action dimensions
        return self.distribution.log_prob(actions).sum(dim=-1)

    def entropy(self) -> Optional[torch.Tensor]:
        """
        Calculates the entropy of the distribution.
        
        :return: Entropy of the distribution
        """
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call proba_distribution first.")
        
        return self.distribution.entropy().sum(dim=-1)

    def sample(self) -> torch.Tensor:
        """
        Sample an action from the distribution.
        
        :return: Sampled action
        """
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call proba_distribution first.")
        
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        """
        Returns the mode of the distribution (the mean).
        
        :return: Mode of the distribution
        """
        if self.gaussian_actions is None:
            raise RuntimeError("Distribution not initialized. Call proba_distribution first.")
        
        return self.gaussian_actions

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """
        Return actions according to the probability distribution.
        
        :param deterministic: Whether to use deterministic (mode) or stochastic sampling
        :return: Actions
        """
        if deterministic:
            return self.mode()
        return self.sample()
    
    def actions_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get actions from distribution parameters (required by SB3 Distribution interface).
        
        :param mean_actions: Raw mean outputs from network
        :param log_std: Raw std outputs from network
        :param deterministic: Whether to sample or use mode
        :return: Actions
        """
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)
    
    def log_prob_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get log probability and actions from distribution parameters (required by SB3 Distribution interface).
        
        :param mean_actions: Raw mean outputs from network
        :param log_std: Raw std outputs from network
        :return: Tuple of (actions, log_prob)
        """
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Applies the symlog function: sign(x) * log(|x| + 1)."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Applies the symexp function: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class SymexpTwoHotDistribution(Distribution, nn.Module):
    """
    Distribution for continuous values using DreamerV3's TwoHot encoding.
    Models a continuous value as a categorical distribution over symmetrically
    spaced bins in symlog space.
    
    This is typically used for reward and value prediction in DreamerV3.
    
    :param action_dim: Dimension of the output. Must be 1.
    :param bins: Number of bins to use. Must be odd. (default=255)
    :param low: Lower bound for symlog-spaced bins. (default=-20.0)
    :param high: Upper bound for symlog-spaced bins. (default=20.0)
    """

    def __init__(
        self,
        action_dim: int,
        bins: int = 255,
        low: float = -20.0,
        high: float = 20.0
    ):
        # Initialize both parent classes
        Distribution.__init__(self)
        nn.Module.__init__(self)
        
        assert action_dim == 1, "SymexpTwoHotDistribution only supports 1D outputs."
        assert bins % 2 == 1, "Number of bins must be odd for symmetry."

        self.action_dim = action_dim
        self.bins_count = bins

        # Create bins linearly spaced in symlog space (do not transform with symexp here)
        bins_tensor = torch.linspace(low, high, self.bins_count)

        self.register_buffer("bins", bins_tensor)
        self.logits = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Creates the network head that outputs logits for bins.
        
        :param latent_dim: Dimension of the input latent vector
        :return: Linear layer that outputs logits
        """
        return nn.Linear(latent_dim, self.action_dim * self.bins_count)

    def proba_distribution(self, latent_pi: torch.Tensor) -> "SymexpTwoHotDistribution":
        """
        Set the logits for the distribution.
        
        :param latent_pi: Output from the network head
        :return: self
        """
        self.logits = latent_pi
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Calculate log probability by transforming continuous values to soft targets.
        
        :param actions: Continuous values to evaluate
        :return: Log probabilities
        """
        # Transform to symlog space
        target = symlog(actions)

        # Find two nearest bins
        target = target.squeeze(dim=-1)
        above = torch.searchsorted(self.bins, target, right=True)
        below = above - 1
        
        # Clamp to valid range
        below = torch.clamp(below, 0, self.bins_count - 1)
        above = torch.clamp(above, 0, self.bins_count - 1)

        # Calculate interpolation weights
        dist_to_below = torch.abs(self.bins[below] - target)
        dist_to_above = torch.abs(self.bins[above] - target)

        total_dist = dist_to_below + dist_to_above
        total_dist = torch.where(total_dist == 0, torch.ones_like(total_dist), total_dist)

        weight_below = dist_to_above / total_dist
        weight_above = dist_to_below / total_dist

        # Create soft target distribution
        target_dist = (
            F.one_hot(below, self.bins_count) * weight_below.unsqueeze(-1) +
            F.one_hot(above, self.bins_count) * weight_above.unsqueeze(-1)
        )
        
        # Calculate cross-entropy (negative log likelihood)
        log_pred = F.log_softmax(self.logits, dim=-1)
        log_prob = torch.sum(target_dist * log_pred, dim=-1)
        return log_prob.unsqueeze(-1)

    def mode(self) -> torch.Tensor:
        """
        Return the expected value (mode) of the distribution.
        Uses symmetric sum for numerical stability.
        
        :return: Mode of the distribution
        """
        probs = F.softmax(self.logits, dim=-1)
        
        # Symmetric sum for stability
        m = (self.bins_count - 1) // 2
        p1, p2, p3 = probs[:, :m], probs[:, m:m+1], probs[:, m+1:]
        b1, b2, b3 = self.bins[:m], self.bins[m:m+1], self.bins[m+1:]
        
        wavg = torch.sum(p2 * b2, dim=-1) + \
               torch.sum(torch.flip(p1, dims=[-1]) * b1 + p3 * b3, dim=-1)
        
        # Transform back from symlog space
        mode_val = symexp(wavg)
        return mode_val.unsqueeze(-1)

    def sample(self) -> torch.Tensor:
        """
        Sample from the categorical distribution over bins.
        
        :return: Sampled value
        """
        dist = torch.distributions.Categorical(logits=self.logits)
        indices = dist.sample()
        sampled_bins = self.bins[indices]
        
        # Transform back from symlog space
        actions = symexp(sampled_bins)
        return actions.unsqueeze(-1)

    def entropy(self) -> torch.Tensor:
        """
        Calculate entropy of the categorical distribution.
        
        :return: Entropy
        """
        return torch.distributions.Categorical(logits=self.logits).entropy()
    
    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """
        Return actions according to the probability distribution.
        
        :param deterministic: Whether to use mode or sample
        :return: Actions
        """
        if deterministic:
            return self.mode()
        return self.sample()
    
    def actions_from_params(self, latent_pi: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get actions from distribution parameters (required by SB3 Distribution interface).
        
        :param latent_pi: Latent representation (logits)
        :param deterministic: Whether to sample or use mode
        :return: Actions
        """
        self.proba_distribution(latent_pi)
        return self.get_actions(deterministic=deterministic)
    
    def log_prob_from_params(self, latent_pi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get log probability and actions from distribution parameters (required by SB3 Distribution interface).
        
        :param latent_pi: Latent representation (logits)
        :return: Tuple of (actions, log_prob)
        """
        actions = self.actions_from_params(latent_pi)
        log_prob = self.log_prob(actions)
        return actions, log_prob

