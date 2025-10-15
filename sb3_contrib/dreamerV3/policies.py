"""
Policies for DreamerV3.

This module re-exports policies from sb3_contrib.common.dreamerv3.policies
for convenience and compatibility with SB3 conventions.
"""

from sb3_contrib.common.dreamerv3.policies import (
    BlockDiagonalGRU,
    DreamerV3ActorCriticPolicy,
    DreamerV3CnnPolicy,
    DreamerV3MultiInputPolicy,
)

# Alias for consistency with SB3 naming
DreamerV3Policy = DreamerV3ActorCriticPolicy
MlpPolicy = DreamerV3ActorCriticPolicy
CnnPolicy = DreamerV3CnnPolicy
MultiInputPolicy = DreamerV3MultiInputPolicy

__all__ = [
    "DreamerV3Policy",
    "DreamerV3ActorCriticPolicy",
    "DreamerV3CnnPolicy",
    "DreamerV3MultiInputPolicy",
    "MlpPolicy",
    "CnnPolicy",
    "MultiInputPolicy",
    "BlockDiagonalGRU",
]
