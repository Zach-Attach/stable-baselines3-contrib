from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution, BernoulliDistribution, CategoricalDistribution
from torch import nn
from torch.distributions import Categorical, Normal
from torch.distributions.utils import logits_to_probs
import torch.nn.functional as F

##################### CONTINUE PREDICTOR #########################
fullState = th.randn(1, 512) # example input to continue predictor net
fullStateSize = fullState.shape[-1]

# init
contPredDistributionObject = BernoulliDistribution(1) # Create Bernoulli distribution object for cont space
contPredDistributionNet = contPredDistributionObject.proba_distribution_net(fullStateSize) # Create NN layer (head) that outputs the distribution parameters (logits)

# forward pass
cont_logit = contPredDistributionNet(fullState) # raw output from continue predictor net
contPredDistributionObject.proba_distribution(cont_logit) # prepare PyTorch Bernoulli object for sampling/log_prob calculation

stoch_cont_prediction = contPredDistributionObject.sample()  # Returns a tensor of binary continuations (0 or 1) for the current fullState(s)
cont_log_prob = contPredDistributionObject.log_prob(stoch_cont_prediction) # Calculates log(P(action | state)) for the sampled action

# return stoch_cont_prediction, cont_log_prob

# evaluation or deterministic prediction
cont_logit = contPredDistributionNet(fullState) # raw output from continue predictor net
contPredDistributionObject.proba_distribution(cont_logit) # prepare PyTorch Bernoulli object for sampling/log_prob calculation

continuation_prediction = contPredDistributionObject.mode() # Returns the most likely continuation (0 or 1) for the current fullState(s)

####################### CONTINUE PREDICTOR #########################

####################### CATEGORICAL DISTRIBUTION (DISCRETE ACTION) #########################
unimix = 0.01 # example unimix value

action_space = spaces.Discrete(3) # example discrete action space
fullState = th.randn(1, 512) # example input to continue predictor net
fullStateSize = fullState.shape[-1]

# init
actionDistributionObject = CategoricalDistribution(action_space.shape[-1]) # Create Bernoulli distribution object for cont space
actionDistributionNet = actionDistributionObject.proba_distribution_net(fullStateSize) # Create NN layer (head) that outputs the distribution parameters (logits)

# forward pass
action_logits = actionDistributionNet(fullState) # raw output from continue predictor net

# Apply unimix
if unimix > 0.0:
    # Calculate probabilities from logits
    probs = th.softmax(action_logits, dim=-1)
    # Create uniform distribution tensor
    uniform = th.ones_like(probs) / probs.shape[-1]
    # Mix the distributions
    probs = (1 - unimix) * probs + unimix * uniform
    # Convert back to logits (for numerical stability)
    logits = th.log(probs)

actionDistributionObject.proba_distribution(action_logits) # prepare PyTorch Bernoulli object for sampling/log_prob calculation

stoch_action = actionDistributionObject.sample()  # Returns a tensor of binary continuations (0 or 1) for the current fullState(s)
action_log_prob = actionDistributionObject.log_prob(stoch_action) # Calculates log(P(action | state)) for the sampled action

# return stoch_cont_prediction, cont_log_prob

# evaluation or deterministic prediction
action_logit = actionDistributionNet(fullState) # raw output from continue predictor net
actionDistributionObject.proba_distribution(action_logit) # prepare PyTorch Bernoulli object for sampling/log_prob calculation

action_prediction = actionDistributionObject.mode() # Returns the most likely continuation (0 or 1) for the current fullState(s)

####################### CATEGORICAL DISTRIBUTION (DISCRETE ACTION) #########################

####################### BOUNDED NORMAL DISTRIBUTION (CONTINUOUS ACTION) #########################
minstd = 0.1
maxstd = 1.0

import numpy as np
import torch
from torch import nn
from stable_baselines3.common.distributions import DiagGaussianDistribution, Distribution
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.preprocessing import get_action_dim

# Constants from your JAX implementation
MIN_STD = 1e-6  # Equivalent to self.minstd
MAX_STD = 1.0   # Equivalent to self.maxstd

# --- 1. Custom Bounded Distribution Class (SB3 Style) ---

class BoundedDiagGaussianDistribution(Distribution):
    """
    A custom DiagGaussian distribution that applies a tanh bounding
    to the mean and uses the specific stddev scaling from the original code.

    This class mimics the functionality of the original 'Normal' class combined
    with the 'bounded_normal' method's logic for stddev and mean.
    """
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        # Initialize internal DiagGaussian to handle log_prob, entropy, etc.
        # This will be constructed in the 'actions_from_params' method.
        self.distribution = None
        self.action_net = None # Placeholder for mean/std-producing layers

        # Constants for stddev scaling (Matching 'lo, hi = self.minstd, self.maxstd')
        self.min_std = MIN_STD
        self.max_std = MAX_STD

    def actions_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Calculates the actions and constructs the internal distribution.
        This method combines the mean/std calculation logic from the original 'bounded_normal' method.
        """
        
        # 1. Stddev calculation (Matching 'stddev = (hi - lo) * jax.nn.sigmoid(stddev + 2.0) + lo')
        # In SB3, the network usually outputs 'log_std'. We'll assume 'log_std' here is 
        # the raw output that corresponds to 'stddev' in the original code *before* sigmoid.
        # Original: stddev = self.sub('stddev', nets.Linear, ...)(x)
        
        # NOTE: We assume 'log_std' here is the raw linear output (pre-sigmoid/log/exp)
        raw_std_output = log_std # Rename for clarity
        
        # Equivalent to jax.nn.sigmoid(stddev + 2.0)
        sigmoid_transformed = torch.sigmoid(raw_std_output + 2.0)
        
        # Equivalent to (hi - lo) * ... + lo
        stddev = (self.max_std - self.min_std) * sigmoid_transformed + self.min_std
        
        # log_std needed for DiagGaussianDistribution constructor
        log_std_for_sb3 = torch.log(stddev) 

        # 2. Mean bounding (Matching 'output = Normal(jnp.tanh(mean), stddev)')
        # Original: mean = self.sub('mean', nets.Linear, ...)(x)
        # Bounded mean
        bounded_mean = torch.tanh(mean_actions)

        # 3. Construct the SB3 DiagGaussianDistribution
        # This internal distribution handles log_prob, entropy, and KL (kl_div)
        self.distribution = DiagGaussianDistribution(self.action_dim)
        # The internal DiagGaussian needs its 'log_std' set to handle the custom stddev.
        self.distribution.log_std = log_std_for_sb3 
        
        # 'actions' are the mean for a continuous distribution
        return bounded_mean

    def sample(self) -> torch.Tensor:
        """
        Samples an action from the distribution.
        """
        # We need the mean from the original network output (pre-tanh) to sample
        # *and then* apply tanh to the sample, but SB3's DiagGaussian just samples 
        # and returns the sample. For a strictly equivalent conversion, the *sample*
        # should be derived from the *bounded* mean/std, but the original code
        # applies tanh only to the *mean* before creating the Normal.
        
        # Based on the original: 'output = Normal(jnp.tanh(mean), stddev)'
        # and 'sample = ... * self.stddev + self.mean'
        # The sample should use the Tanh-bounded mean.
        
        # The internal SB3 distribution sample uses its 'mean' and 'log_std'
        if self.distribution is None:
             raise RuntimeError("Distribution not initialized. Call actions_from_params first.")

        # Get the sample from the internal distribution (which uses the custom stddev)
        unbounded_sample = self.distribution.sample()
        
        # Since the original code used tanh(mean) as the mean of the Normal, 
        # the SB3 equivalent sample *is* the bounded sample.
        return unbounded_sample


    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log probability of the actions.
        Matches 'logp(self, event)' from the original code.
        """
        if self.distribution is None:
             raise RuntimeError("Distribution not initialized. Call actions_from_params first.")
        
        # log_prob of the event (action) using the mean/std that was set up.
        return self.distribution.log_prob(actions)


    def entropy(self) -> torch.Tensor:
        """
        Calculates the entropy of the distribution.
        Matches 'entropy()' from the original code.
        """
        if self.distribution is None:
             raise RuntimeError("Distribution not initialized. Call actions_from_params first.")
             
        # SB3 DiagGaussian's entropy is equivalent to the original's
        return self.distribution.entropy()

    def kl_div(self, other: "CustomBoundedDiagGaussianDistribution") -> torch.Tensor:
        """
        Calculates the KL divergence between two distributions.
        Matches 'kl(self, other)' from the original code.
        """
        if self.distribution is None or other.distribution is None:
             raise RuntimeError("Distributions not initialized. Call actions_from_params first.")

        # SB3's DiagGaussian's kl_div is equivalent to the original's 'kl(self, other)'
        return self.distribution.kl_div(other.distribution)

    def mode(self) -> torch.Tensor:
        """
        Returns the mode of the distribution (the mean).
        Matches 'pred()' from the original code.
        """
        if self.distribution is None:
             raise RuntimeError("Distribution not initialized. Call actions_from_params first.")
        
        # The mean of the distribution (which is tanh-bounded)
        return self.distribution.mean


    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """
        Returns the actions based on the sampling mode.
        """
        if deterministic:
            return self.mode()
        return self.sample()

    def __getitem__(self, val: torch.Tensor) -> torch.Tensor:
        """
        Equivalent to log_prob.
        """
        return self.log_prob(val)
    
    # NOTE on minent/maxent: 
    # The original minent/maxent were properties of the 'Normal' object *after* construction,
    # used for entropy regularization. In SB3, these would be calculated separately 
    # within the policy's learning/loss function, not as part of the distribution object itself.

# CLASS CustomActorCriticPolicy INHERITS ActorCriticPolicy:
    
    # FUNCTION _build(features_extractor, observation_space, action_space, net_arch, ...):        
        # Use the custom distribution class
        self.action_dist = BoundedDiagGaussianDistribution(get_action_dim(action_space))

        # Create the mean and stddev linear layers (nets.Linear from original JAX)
        self.mean_net = Linear(last_layer_dim, action_dim)
        self.log_std_net = Linear(last_layer_dim, action_dim)

    # FUNCTION forward(obs, deterministic=FALSE):
        # 1. Pass observation through feature and MLP layers
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)

        # 2. Get raw mean and log_std outputs (from original JAX)
        mean_actions_raw = self.mean_net(latent_pi)
        log_std_raw = self.log_std_net(latent_pi)

        # 3. Construct and set parameters on the custom distribution
        # This calculates the bounded mean and custom log_std
        mean_actions_bounded = self.action_dist.actions_from_params(mean_actions_raw, log_std_raw)

        # 4. Get the action from the distribution
        # IF deterministic:
            actions = mean_actions_bounded # Mode of the distribution
        # ELSE:
            actions = self.action_dist.sample() # Sample from the distribution

        # 5. Get value prediction (standard SB3)
        latent_vf = self.mlp_extractor.forward_critic(features)
        values = self.value_net(latent_vf)

        # 4. Get the action from the distribution
        # IF deterministic:
            actions = mean_actions_bounded # Mode of the distribution
        # ELSE:
            actions = self.action_dist.sample() # Sample from the distribution

        # 5. Get value prediction (standard SB3)
        latent_vf = self.mlp_extractor.forward_critic(features)
        values = self.value_net(latent_vf)

        # RETURN actions, values, self.log_std_net # log_std_net is often returned for convenience

    # FUNCTION evaluate_actions(obs, actions):
        # Standard SB3 logic, but uses the custom distribution's methods
        # ... (call forward to populate this.action_dist) ...
        
        log_prob = self.action_dist.log_prob(actions)
        entropy = self.action_dist.entropy()
        
        # Add min/max entropy for use in loss calculation (entropy regularization)
        min_ent = self.action_dist.min_entropy()
        max_ent = self.action_dist.max_entropy()

        # return log_prob, entropy, values, min_ent, max_ent

####################### BOUNDED NORMAL DISTRIBUTION (CONTINUOUS ACTION) #########################

####################### SYMEXP_TWOHOT (reward & critic) #########################

def symlog(x: torch.Tensor) -> torch.Tensor:
    """Applies the symlog function."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

def symexp(x: torch.Tensor) -> torch.Tensor:
    """Applies the symexp function."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

class SymexpTwoHotDistribution(Distribution):
    """
    Distribution for continuous actions inspired by the DreamerV2 'TwoHot'
    representation. It models a continuous action as a categorical
    distribution over a set of symmetrically spaced bins.

    :param action_dim: Dimension of the action space. Must be 1.
    :param bins: The number of bins to use. Must be an odd number. (default=255)
    :param low: The lower bound for the symlog-spaced bins before symexp. (default=-20.0)
    :param high: The upper bound for the symlog-spaced bins before symexp. (default=0.0)
    """

    def __init__(
        self,
        action_dim: int,
        bins: int = 255,
        low: float = -20.0,
        high: float = 0.0
    ):
        super().__init__()
        assert action_dim == 1, "SymexpTwoHotDistribution only supports 1D actions."
        assert bins % 2 == 1, "Number of bins must be odd for symmetry."

        self.action_dim = action_dim
        self.bins_count = bins

        # Create the symmetric bins
        half = torch.linspace(low, high, (self.bins_count - 1) // 2 + 1)
        half = symexp(half)
        bins_tensor = torch.cat([half, -torch.flip(half[:-1], dims=[0])], 0)

        # Register bins as a buffer to ensure it's moved to the correct device
        self.register_buffer("bins", bins_tensor)
        self.logits = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Creates the network head that outputs the distribution parameters
        (in this case, the logits for the bins).
        """
        return nn.Linear(latent_dim, self.action_dim * self.bins_count)

    def proba_distribution(self, latent_pi: torch.Tensor) -> "SymexpTwoHotDistribution":
        """
        Takes the output of the network head and stores it as the distribution's
        logits.
        """
        self.logits = latent_pi
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log probability of a batch of actions. This is done by
        transforming the continuous action into a "soft" target distribution
        over the bins and computing the cross-entropy.
        """
        # Squash the continuous actions into the same space as the bins
        target = symlog(actions)

        # Find the two nearest bins for each target value
        target = target.squeeze(dim=-1)
        above = torch.searchsorted(self.bins, target, right=True)
        below = above - 1
        
        # Clamp indices to be within the valid range
        below = torch.clamp(below, 0, self.bins_count - 1)
        above = torch.clamp(above, 0, self.bins_count - 1)

        # Calculate the weights for each of the two bins
        dist_to_below = torch.abs(self.bins[below] - target)
        dist_to_above = torch.abs(self.bins[above] - target)

        total_dist = dist_to_below + dist_to_above
        # Avoid division by zero when target is exactly on a bin
        total_dist[total_dist == 0] = 1.0

        weight_below = dist_to_above / total_dist
        weight_above = dist_to_below / total_dist

        # Create the soft target distribution
        target_dist = (
            F.one_hot(below, self.bins_count) * weight_below.unsqueeze(-1) +
            F.one_hot(above, self.bins_count) * weight_above.unsqueeze(-1)
        )
        
        # Calculate the log probability (cross-entropy)
        log_pred = F.log_softmax(self.logits, dim=-1)
        log_prob = torch.sum(target_dist * log_pred, dim=-1)
        return log_prob.unsqueeze(-1)


    def mode(self) -> torch.Tensor:
        """
        Returns the mode of the distribution (the expected value).
        This is calculated using a numerically stable symmetric sum.
        """
        probs = F.softmax(self.logits, dim=-1)
        
        # Symmetric sum to calculate the weighted average for numerical stability
        m = (self.bins_count - 1) // 2
        p1, p2, p3 = probs[:, :m], probs[:, m:m+1], probs[:, m+1:]
        b1, b2, b3 = self.bins[:m], self.bins[m:m+1], self.bins[m+1:]
        
        wavg = torch.sum(p2 * b2, dim=-1) + \
               torch.sum(torch.flip(p1 * b1, dims=[-1]) + p3 * b3, dim=-1)
        
        # Unsquash the result back to the original action space
        mode_val = symexp(wavg)
        return mode_val.unsqueeze(-1)

    def sample(self) -> torch.Tensor:
        """
        Samples an action from the distribution by treating it as a categorical
        distribution over the bins.
        """
        dist = torch.distributions.Categorical(logits=self.logits)
        indices = dist.sample()
        sampled_bins = self.bins[indices]
        
        # Unsquash the sampled bin value
        actions = symexp(sampled_bins)
        return actions.unsqueeze(-1)

    def entropy(self) -> torch.Tensor:
        """
        Returns the entropy of the distribution.
        """
        return torch.distributions.Categorical(logits=self.logits).entropy()

####################### SYMEXP_TWOHOT (reward & critic) #########################
