from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Normal

from banditry.constants import HALF_LOG_2PI
from banditry.surrogates.svgp import BaseModel
from banditry.surrogates.tsmodel import ValueFunction


class Objective(ABC):
    """Base class for acquisition objectives evaluated by the evolutionary optimiser.

    An objective wraps a surrogate model and maps transformed design points to one or more
    columns that the optimiser MINIMISES. Subclasses declare how many objective columns
    (``num_obj``) and constraint columns (``num_constr``) their :meth:`eval` output
    contains; ``ContextualProblem`` splits the returned tensor accordingly (the first
    ``num_obj`` columns become objectives ``F``, the rest constraints ``G``).

    Args:
        model: Surrogate model the objective queries.
        **conf: Subclass-specific options.
    """

    def __init__(self, model, **conf):
        self.model = model

    @property
    @abstractmethod
    def num_obj(self):
        """Number of objective columns produced by :meth:`eval`."""

    @property
    @abstractmethod
    def num_constr(self):
        """Number of constraint columns produced by :meth:`eval`."""

    @abstractmethod
    def eval(self, x: Tensor, xe: Tensor) -> Tensor:
        """Evaluate the objective at a batch of transformed design points.

        Args:
            x: Continuous features of shape ``(n, num_cont)``.
            xe: Categorical (enum) features of shape ``(n, num_enum)``.

        Returns:
            Tensor whose first ``num_obj`` columns are objective values to MINIMISE and
            whose remaining ``num_constr`` columns are constraint values.
        """

    def __call__(self, x: Tensor, xe: Tensor):
        """Alias for :meth:`eval`."""
        return self.eval(x, xe)


class SingleObjective(Objective):
    """Convenience base for unconstrained, single-column objectives.

    Fixes ``num_obj = 1`` and ``num_constr = 0``; subclasses only implement ``eval``.
    """

    def __init__(self, model: BaseModel, **conf):
        super().__init__(model, **conf)

    @property
    def num_obj(self):
        return 1

    @property
    def num_constr(self):
        return 0


class Mean(SingleObjective):
    """Posterior predictive mean of a single-output surrogate.

    Minimising it drives the search towards the lowest predicted value — pure exploitation
    with no uncertainty bonus.
    """

    def __init__(self, model: BaseModel, **conf):
        super().__init__(model, **conf)
        assert model.num_out == 1

    def eval(self, x: Tensor, xe: Tensor) -> Tensor:
        py, _ = self.model.predict(x, xe)
        return py


class Sigma(SingleObjective):
    """Posterior predictive standard deviation of a single-output surrogate.

    Note that objectives are minimised, so this column favours well-explored (low-sigma)
    points; use its negation (minimising ``-sigma`` maximises uncertainty) to reward
    exploration.
    """

    def __init__(self, model: BaseModel, **conf):
        super().__init__(model, **conf)
        assert model.num_out == 1

    def eval(self, x: Tensor, xe: Tensor) -> Tensor:
        _, ps2 = self.model.predict(x, xe)
        return ps2.sqrt()


class LCB(SingleObjective):
    """Lower confidence bound acquisition: ``py - kappa * ps``.

    Minimising the LCB balances exploitation (low predicted mean ``py``) with optimism
    about uncertain points (high predicted standard deviation ``ps``).

    Args:
        model: Surrogate with a ``predict`` method returning mean and variance.
        best_y: Accepted for interface compatibility; unused.
        **conf: ``kappa`` — exploration weight on the standard deviation (default 2.0).
    """

    def __init__(self, model: BaseModel, best_y=None, **conf):
        super().__init__(model, **conf)
        self.kappa = conf.get("kappa", 2.0)

    def eval(self, x: torch.FloatTensor, xe: torch.LongTensor) -> torch.FloatTensor:
        with torch.no_grad():
            py, ps2 = self.model.predict(x, xe)
            ps = ps2.sqrt().clamp(min=torch.finfo(ps2.dtype).eps).reshape(-1)
            py = py.reshape(-1)
            # noisy variant:
            # noise = np.sqrt(2.0) * self.model.noise.sqrt()
            # lcb = (py + noise * torch.randn(py.shape)) - self.kappa * ps
            lcb = py - self.kappa * ps

            return lcb.reshape(-1, 1)


class MACE(Objective):
    """Multi-objective ACquisition function Ensemble (Lyu et al., ICML 2018), as in HEBO.

    Exposes three acquisition functions as one simultaneous multi-objective problem —
    column 0: LCB ``py - kappa * ps``; column 1: negated log expected improvement; column
    2: negated log probability of improvement — so the evolutionary optimiser returns a
    Pareto front trading them off rather than committing to a single acquisition. EI and
    PI are computed in log space against the jittered incumbent ``best_y - eps`` (with
    observation noise added to the improvement), switching to Mills-ratio asymptotic
    approximations where the exact values are numerically unstable.

    Args:
        model: Surrogate with a ``predict`` method returning mean and variance and a
            ``noise`` attribute.
        best_y: Incumbent (best observed) value the improvement is measured against.
        **conf: ``kappa`` — LCB exploration weight (default 2.0); ``eps`` — improvement
            jitter subtracted from the incumbent (default 1e-4).
    """

    def __init__(self, model, best_y, **conf):
        super().__init__(model, **conf)
        self.kappa = conf.get("kappa", 2.0)
        self.eps = conf.get("eps", 1e-4)
        self.tau = best_y

    @property
    def num_constr(self):
        return 0

    @property
    def num_obj(self):
        return 3

    @staticmethod
    def Mills_ratio_approximations(ps, normalised_improvement) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Mills-ratio asymptotic approximations of ``(log PI, log EI)`` for large negative improvement."""
        logEIapp = ps.log() - 0.5 * normalised_improvement**2 - (normalised_improvement**2 - 1).log()
        logPIapp = -0.5 * normalised_improvement**2 - torch.log(-1 * normalised_improvement) - HALF_LOG_2PI
        return logPIapp, logEIapp

    @staticmethod
    def exact_value(ps, normalised_improvement) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Exact ``(log PI, log EI)`` computed from the standard normal pdf/cdf."""
        dist = Normal(0.0, 1.0)
        log_phi = dist.log_prob(normalised_improvement)
        cdf_at_imp = dist.cdf(normalised_improvement)
        return cdf_at_imp.log(), (ps * (cdf_at_imp * normalised_improvement + log_phi.exp())).log()

    def eval(self, x: torch.FloatTensor, xe: torch.LongTensor) -> torch.FloatTensor:
        with torch.no_grad():
            py, ps2 = self.model.predict(x, xe)
            noise = np.sqrt(2.0) * self.model.noise.sqrt()
            ps = ps2.sqrt().clamp(min=torch.finfo(ps2.dtype).eps).reshape(-1)
            py = py.reshape(-1)
            # Wenlong Lyu et al. ICML 2018 https://proceedings.mlr.press/v80/lyu18a.html
            # Note that the direction of normalised, jittered improvment is flipped per Lyu et al
            # This is a slight deviation from common EI/PI for maximisiation,
            # but it is correct, as if the current solution is smaller than the best so far
            # the "normed" is positive, this is why EI and PI are still maximised
            norm_imp = (self.tau - self.eps - py - noise * torch.randn(py.shape)) / ps

            probability_of_imp, expected_imp = self.exact_value(ps, norm_imp)
            approximated_PI, approximated_EI = self.Mills_ratio_approximations(ps, norm_imp)

            use_approx = ~((norm_imp > -6) & torch.isfinite(expected_imp) & torch.isfinite(probability_of_imp))
            out = torch.zeros(x.shape[0], 3)

            out[:, 1] = torch.where(use_approx, approximated_EI, expected_imp)
            out[:, 2] = torch.where(use_approx, approximated_PI, probability_of_imp)

            out *= -1

            # lcb = (py + noise * torch.randn(py.shape)) - self.kappa * ps
            lcb = py - self.kappa * ps

            out[:, 0] = lcb.reshape(-1)

            return out


class ThompsonObjective(Objective):
    """Adapter: exposes a sampled ValueFunction as an Acquisition.

    Thompson-sampling acquisition — the objective is simply the sampled model's prediction
    at the candidate point, reshaped to one column per output. Minimising it selects the
    action the current posterior draw believes best; ``num_obj`` equals the model's
    ``num_out`` and there are no constraints.

    Args:
        model: A ``ValueFunction`` whose weights are a posterior draw (see ``Sampler``).
    """

    def __init__(self, model: ValueFunction):
        self.model = model

    def eval(self, x: Tensor, xe: Tensor) -> Tensor:
        pred = self.model(x, xe)
        if pred.dim() == 1:
            pred = pred.view(-1, 1)
        return pred

    @property
    def num_obj(self):
        return self.model.num_out

    @property
    def num_constr(self):
        return 0
