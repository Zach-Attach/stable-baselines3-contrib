"""DreamerV3 common components."""

from sb3_contrib.common.dreamerv3.distributions import (
    BoundedDiagGaussianDistribution,
    SymexpTwoHotDistribution,
)
from sb3_contrib.common.dreamerv3.encoder_decoder import Decoder, Encoder
from sb3_contrib.common.dreamerv3.normalizers import (
    RunningNormalizer,
    SlowValueNetwork,
    ValueNormalizer,
)
from sb3_contrib.common.dreamerv3.policies import (
    BlockDiagonalGRU,
    DreamerV3ActorCriticPolicy,
)
from sb3_contrib.common.dreamerv3.rssm import RSSM
from sb3_contrib.common.dreamerv3.utils import (
    compute_actor_loss,
    compute_advantages,
    compute_episode_weights,
    compute_value_loss,
    lambda_return,
    symexp,
    symlog,
)

__all__ = [
    "RSSM",
    "Encoder",
    "Decoder",
    "BlockDiagonalGRU",
    "DreamerV3ActorCriticPolicy",
    "BoundedDiagGaussianDistribution",
    "SymexpTwoHotDistribution",
    "RunningNormalizer",
    "ValueNormalizer",
    "SlowValueNetwork",
    "lambda_return",
    "compute_advantages",
    "compute_actor_loss",
    "compute_value_loss",
    "compute_episode_weights",
    "symlog",
    "symexp",
]
