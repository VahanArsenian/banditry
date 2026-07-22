"""Deprecated alias for :mod:`banditry.optimisation_subroutines.contextual_problem`."""

import warnings

from banditry.optimisation_subroutines.contextual_problem import ContextualProblem

warnings.warn(
    "banditry.optimisation_subroutines.contextal_problem is deprecated; "
    "import from banditry.optimisation_subroutines.contextual_problem instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ContextualProblem"]
