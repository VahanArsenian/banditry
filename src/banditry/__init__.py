"""banditry: contextual bandit agents with GP surrogates.

GP-UCB (OFU) and Thompson-sampling agents over mixed design spaces,
with evolutionary acquisition optimisation and Langevin/NUTS samplers.
"""

from banditry.agents.factory import (
    AgentConfig,
    OFUGPConfig,
    TSConfig,
    build_agent,
)
from banditry.variable_domains.design_space import DesignSpace

__version__ = "0.1.0"

__all__ = [
    "AgentConfig",
    "OFUGPConfig",
    "TSConfig",
    "build_agent",
    "DesignSpace",
    "__version__",
]
