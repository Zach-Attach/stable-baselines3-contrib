"""
Utility functions for DreamerV3.

Includes lambda returns, reward/value predictions, and other helper functions.
"""

from typing import Optional

import torch
import torch as th


def lambda_return(
    rewards: th.Tensor,
    values: th.Tensor,
    continues: th.Tensor,
    bootstrap: th.Tensor,
    lambda_: float = 0.95,
    gamma: float = 0.997,
) -> th.Tensor:
    """
    Compute lambda returns for advantage estimation.
    
    Lambda return is a mixture of n-step returns that balances bias and variance.
    Used in DreamerV3 for computing target values.
    
    :param rewards: Rewards (B, T)
    :param values: Value predictions (B, T+1)
    :param continues: Continue flags (B, T) - 1 if episode continues, 0 if terminal
    :param bootstrap: Bootstrap value (B,) - value at T+1
    :param lambda_: Lambda coefficient for mixing returns (default: 0.95)
    :param gamma: Discount factor (default: 0.997)
    :return: Lambda returns (B, T)
    """
    B, T = rewards.shape
    
    # Prepare values
    # values is (B, T+1) where values[:, -1] is the bootstrap
    # We want values[:, 1:] as the next values
    next_values = values[:, 1:]
    current_values = values[:, :-1]
    
    # Compute TD targets
    # td_target = reward + gamma * continue * next_value
    td_targets = rewards + gamma * continues * next_values
    
    # Compute lambda returns recursively (from last to first)
    returns = th.zeros_like(rewards)
    last_return = bootstrap
    
    for t in reversed(range(T)):
        # Lambda return: mix between TD target and bootstrapped return
        # ret[t] = td_target[t] + gamma * continue[t] * lambda * (last_return - next_value[t])
        # This simplifies to: ret[t] = (1 - lambda) * td_target[t] + lambda * (reward[t] + gamma * continue[t] * last_return)
        
        # Simpler formulation:
        # ret[t] = reward[t] + gamma * continue[t] * ((1 - lambda) * next_value[t] + lambda * last_return)
        returns[:, t] = rewards[:, t] + gamma * continues[:, t] * (
            (1 - lambda_) * next_values[:, t] + lambda_ * last_return
        )
        last_return = returns[:, t]
    
    return returns


def symlog(x: th.Tensor) -> th.Tensor:
    """
    Symmetric logarithm transformation: sign(x) * log(1 + |x|).
    
    Used to normalize inputs with varying scales while preserving sign.
    """
    return th.sign(x) * th.log(1 + th.abs(x))


def symexp(x: th.Tensor) -> th.Tensor:
    """
    Symmetric exponential transformation: sign(x) * (exp(|x|) - 1).
    
    Inverse of symlog.
    """
    return th.sign(x) * (th.exp(th.abs(x)) - 1)


def compute_advantages(
    returns: th.Tensor,
    values: th.Tensor,
) -> th.Tensor:
    """
    Compute advantages as returns - values.
    
    :param returns: Lambda returns (B, T)
    :param values: Value predictions (B, T)
    :return: Advantages (B, T)
    """
    return returns - values


def compute_actor_loss(
    log_probs: th.Tensor,
    advantages: th.Tensor,
    entropies: th.Tensor,
    weights: th.Tensor,
    entropy_coef: float = 3e-4,
) -> th.Tensor:
    """
    Compute policy gradient loss for actor.
    
    Loss = -weight * (log_prob * advantage + entropy_coef * entropy)
    
    :param log_probs: Log probabilities of actions (B, T)
    :param advantages: Advantages (B, T)
    :param entropies: Action entropies (B, T)
    :param weights: Episode weights for handling variable length (B, T)
    :param entropy_coef: Coefficient for entropy bonus (default: 3e-4)
    :return: Actor loss
    """
    # Policy gradient with entropy bonus
    policy_loss = -(log_probs * advantages.detach() + entropy_coef * entropies)
    
    # Apply weights and reduce
    weighted_loss = weights * policy_loss
    
    return weighted_loss.mean()


def compute_value_loss(
    value_pred: th.Tensor,
    value_target: th.Tensor,
    weights: th.Tensor,
    slow_value_pred: Optional[th.Tensor] = None,
    slow_reg: float = 1.0,
) -> th.Tensor:
    """
    Compute value function loss.
    
    Loss = weight * (value_pred - value_target)^2
    If slow_value provided, add regularization: slow_reg * (value_pred - slow_value_pred)^2
    
    :param value_pred: Value predictions (B, T)
    :param value_target: Target values (B, T)
    :param weights: Episode weights (B, T)
    :param slow_value_pred: Predictions from slow value network (B, T)
    :param slow_reg: Coefficient for slow value regularization (default: 1.0)
    :return: Value loss
    """
    # Main value loss (regression to targets)
    value_loss = (value_pred - value_target.detach()) ** 2
    
    # Add slow value regularization if provided
    if slow_value_pred is not None:
        value_loss = value_loss + slow_reg * (value_pred - slow_value_pred.detach()) ** 2
    
    # Apply weights and reduce
    weighted_loss = weights * value_loss
    
    return weighted_loss.mean()


def compute_episode_weights(
    continues: th.Tensor,
    gamma: float = 0.997,
) -> th.Tensor:
    """
    Compute episode weights for handling variable-length episodes.
    
    Weight at time t is the cumulative product of (gamma * continue) up to t.
    This down-weights later timesteps and handles episode boundaries.
    
    :param continues: Continue flags (B, T)
    :param gamma: Discount factor (default: 0.997)
    :return: Weights (B, T)
    """
    # Compute cumulative product of (gamma * continue)
    # weight[t] = product_{i=0}^{t} (gamma * continue[i])
    discounts = gamma * continues
    
    # Cumulative product along time dimension
    weights = th.cumprod(discounts, dim=1) / gamma  # Divide by gamma to start with weight=1
    
    return weights
