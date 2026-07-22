"""Surrogate models: exact and sparse variational GPs, and the TS value function."""

from banditry.surrogates.gp import GP
from banditry.surrogates.svgp import SVGP, BaseModel
from banditry.surrogates.tsmodel import ValueFunction

__all__ = [
    "GP",
    "SVGP",
    "BaseModel",
    "ValueFunction",
]
