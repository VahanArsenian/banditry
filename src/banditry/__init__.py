"""banditry: contextual bandit agents with GP surrogates.

GP-UCB (OFU) and Thompson-sampling agents over mixed design spaces,
with evolutionary acquisition optimisation and Langevin/NUTS samplers.
"""

from banditry.agents.agent import AbstractAgent
from banditry.agents.factory import (
    AgentConfig,
    OFUGPConfig,
    TSConfig,
    build_agent,
)
from banditry.agents.ofugpagent import OFUGPAgent
from banditry.agents.tsagent import TSAgent
from banditry.variable_domains.design_space import DesignSpace

__version__ = "0.2.0"

__all__ = [
    "AbstractAgent",
    "AgentConfig",
    "OFUGPAgent",
    "OFUGPConfig",
    "TSAgent",
    "TSConfig",
    "build_agent",
    "DesignSpace",
    "__version__",
]
