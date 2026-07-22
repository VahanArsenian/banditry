import logging
from abc import ABC, abstractmethod
from collections.abc import Callable

import pandas as pd
import torch
from torch import FloatTensor, LongTensor

from banditry.constants import HALF_LOG_2PI
from banditry.surrogates.tsmodel import ValueFunction
from banditry.variable_domains.transforms import TorchStandardScaler

logger = logging.getLogger(__name__)


def _gaussian_nll(pred: FloatTensor, target: FloatTensor, obs_std: FloatTensor, **kwargs) -> FloatTensor:
    """Element-wise Gaussian negative log-likelihood of ``target`` under ``N(pred, obs_std**2)``."""
    return 0.5 * ((target - pred) / obs_std) ** 2 + torch.log(obs_std) + HALF_LOG_2PI


class _ModelWrapper:
    """Thin proxy that optionally evaluates a ValueFunction via functional_call.

    When ``params`` is given, calls are routed through ``torch.func.functional_call`` so the
    model is evaluated with the supplied parameter tensors (e.g. an in-flight posterior draw)
    instead of its registered parameters; attribute access is forwarded to the wrapped model.
    """

    def __init__(self, base_model, params=None):
        self._base = base_model
        self._params = params

    def __call__(self, Xc, Xe, **kwargs):
        if self._params is not None:
            from torch.func import functional_call

            return functional_call(self._base, self._params, (Xc, Xe), kwargs)
        return self._base(Xc, Xe, **kwargs)

    def __getattr__(self, name):
        return getattr(self._base, name)


class NLL(ABC):
    """Base class for NLL factories. Subclasses bind context and return a callable.

    A concrete ``NLL`` is invoked once per suggestion round with the current optimisation
    context. It binds whatever state it needs (e.g. the contexts observed so far) and
    returns the actual loss callable ``(pred, target, obs_std, **kwargs) -> FloatTensor``
    that a :class:`Sampler` evaluates during posterior sampling.
    """

    @abstractmethod
    def __call__(self, fix_input: dict, space, X: pd.DataFrame) -> Callable[..., FloatTensor]:
        """Bind the optimisation context and return the loss callable.

        Args:
            fix_input: Mapping of context-variable names to their fixed values for this round.
            space: ``DesignSpace`` describing all decision and context variables.
            X: Raw (untransformed) observations collected so far, one row per data point.

        Returns:
            A callable ``(pred, target, obs_std, **kwargs) -> FloatTensor`` computing the
            negative log-likelihood used by a :class:`Sampler`.
        """


class GaussianNLL(NLL):
    """Factory for the plain per-element Gaussian negative log-likelihood.

    Ignores the binding context entirely and always returns :func:`_gaussian_nll`.
    """

    def __call__(self, fix_input: dict, space, X: pd.DataFrame) -> Callable[..., FloatTensor]:
        return _gaussian_nll


class FeelGoodNLL(NLL):
    """Feel-Good Thompson sampling loss factory (Zhang, 2021).

    Implements the exploration reweighting of Zhang (2021), "Feel-Good Thompson Sampling
    for Contextual Bandits and Reinforcement Learning". The returned loss augments the
    Gaussian NLL with an optimism ("feel-good") bonus: for every distinct context observed
    so far, a short evolutionary search over the free (non-context) variables finds the
    current weight sample's maximum predicted value; that value is capped at ``fg_bound``,
    weighted by the number of observations sharing the context, and ``fg_lambda`` times the
    total is subtracted from the NLL. Posterior draws are thereby tilted towards weights
    that believe a high value is attainable, adding optimism to Thompson sampling.

    Args:
        fg_lambda: Weight of the feel-good term. ``0`` recovers the plain Gaussian NLL;
            larger values add more optimism.
        fg_bound: Upper cap applied to each context's best predicted value before weighting,
            preventing the bonus from dominating the likelihood. Must be non-negative.

    Raises:
        ValueError: If ``fg_bound`` is negative.
    """

    def __init__(self, fg_lambda: float = 1.0, fg_bound: float = 10.0):
        if fg_bound < 0:
            raise ValueError(f"fg_bound must be non-negative, got {fg_bound}")
        self.fg_lambda = fg_lambda
        self.fg_bound = fg_bound

    def __call__(self, fix_input: dict, space, X: pd.DataFrame) -> Callable[..., FloatTensor]:
        """Bind the observed contexts and return the feel-good loss callable.

        Groups ``X`` by the context columns named in ``fix_input`` to enumerate the distinct
        contexts and their observation counts. The returned callable computes a scalar total
        loss: the Gaussian NLL summed over the dataset (rescaled to ``num_data``) minus the
        feel-good bonus. It expects the sampler to pass ``model`` (and, for functional
        evaluation, ``model_params``) as keyword arguments; without a model, or when
        ``fg_lambda == 0``, it falls back to the per-element Gaussian NLL.

        Args:
            fix_input: Context-variable mapping; only its keys are used, to group ``X``.
            space: ``DesignSpace`` used by the inner acquisition optimisation.
            X: Raw observations collected so far; grouped by the context columns.

        Returns:
            A callable ``(pred, target, obs_std, **kwargs) -> FloatTensor`` implementing
            the feel-good-adjusted negative log-likelihood.
        """
        from banditry.optimisation_oracles.gen_alg import EvolutionOpt
        from banditry.optimisation_subroutines.contextual_problem import ContextualProblem
        from banditry.optimisation_subroutines.objectives import ThompsonObjective

        context_names = list(fix_input.keys())

        if not context_names:
            contexts = [{}]
            counts = [len(X)]
        else:
            grouped = X.groupby(context_names).size().reset_index(name="count")
            contexts = grouped[context_names].to_dict("records")
            counts = grouped["count"].tolist()

        fg_lambda = self.fg_lambda
        fg_bound = self.fg_bound

        def nll_fn(pred: FloatTensor, target: FloatTensor, obs_std: FloatTensor, **kwargs) -> FloatTensor:
            base_nll = _gaussian_nll(pred, target, obs_std)

            model = kwargs.get("model")
            model_params = kwargs.get("model_params")

            if model is None or fg_lambda == 0:
                return base_nll

            eval_model = _ModelWrapper(model, model_params)
            num_data = kwargs.get("num_data", pred.shape[0])

            fg_values = []
            for ctx, count in zip(contexts, counts, strict=True):
                objective = ThompsonObjective(eval_model)
                problem = ContextualProblem(objective, space, ctx)
                opt = EvolutionOpt(space, pop=5, max_iters=2, verbose=False)
                rec = opt.optimise(problem)

                Xc_opt, Xe_opt = space.transform(rec)
                f_val = -eval_model(Xc_opt, Xe_opt)
                f_clipped = torch.clamp(f_val, max=fg_bound)
                fg_values.append(count * f_clipped.sum())

            fg_term = torch.stack(fg_values).sum()

            per_dp = base_nll.reshape(base_nll.shape[0], -1).sum(dim=1)
            total_base = per_dp.mean() * num_data

            return total_base - fg_lambda * fg_term

        return nll_fn


class Sampler(ABC):
    """Posterior-sampling contract used by ``TSAgent`` for Thompson sampling.

    A ``Sampler`` takes a template ``ValueFunction`` together with the data observed so far
    and returns a copy of the model whose weights are a single draw from the (approximate)
    posterior over weights; ``TSAgent`` then optimises the sampled model's predictions to
    pick the next action. Instances are constructed from ``TSConfig.sampler_config`` keyword
    arguments (see :class:`~banditry.sampling_oracles.langevin_sampler.LangevinSampler` and
    :class:`~banditry.sampling_oracles.nuts_sampler.NUTSSampler` for the accepted keys).

    Attributes:
        generator: Optional ``torch.Generator`` driving all stochastic choices, for
            reproducible sampling.
    """

    generator: torch.Generator | None

    @abstractmethod
    def sample(
        self,
        model: ValueFunction,
        Xc: FloatTensor | None,
        Xe: LongTensor | None,
        y: FloatTensor,
        nll: Callable[..., FloatTensor] | None = None,
    ) -> ValueFunction:
        """Draw one posterior sample of the model weights.

        Args:
            model: Template ``ValueFunction``; implementations work on a deep copy and
                leave the original unmodified.
            Xc: Continuous features of shape ``(n, model.num_cont)``, or ``None`` when the
                model has no continuous inputs.
            Xe: Categorical (enum) features of shape ``(n, model.num_enum)``, or ``None``
                when the model has no categorical inputs.
            y: Observed targets of shape ``(n, model.num_out)``.
            nll: Optional negative-log-likelihood callable
                ``(pred, target, obs_std, **kwargs) -> FloatTensor``; when ``None`` the
                Gaussian NLL is used.

        Returns:
            A new ``ValueFunction`` with weights set to a single posterior draw, in eval
            mode and with its y-scaler fitted to ``y``.
        """

    def __call__(
        self,
        model: ValueFunction,
        Xc: FloatTensor | None,
        Xe: LongTensor | None,
        y: FloatTensor,
        nll: Callable[..., FloatTensor] | None = None,
    ) -> ValueFunction:
        """Alias for :meth:`sample`."""
        return self.sample(model, Xc, Xe, y, nll=nll)

    @staticmethod
    def _prepare_xy(
        model: ValueFunction,
        Xc: FloatTensor | None,
        Xe: LongTensor | None,
        y: FloatTensor,
    ) -> tuple[FloatTensor, LongTensor, FloatTensor]:
        """Validate shapes, coerce dtypes, and drop rows with non-finite ``y``/``Xc`` values.

        Args:
            model: Model whose ``num_cont``/``num_enum``/``num_out`` define the expected shapes.
            Xc: Continuous features or ``None``.
            Xe: Categorical features or ``None``.
            y: Targets of shape ``(n, model.num_out)``.

        Returns:
            Tuple ``(Xc, Xe, y)`` of filtered tensors with matching row counts; absent
            feature blocks are replaced by zero-width tensors.

        Raises:
            ValueError: If shapes are inconsistent or no finite rows remain.
        """
        y_t = y.float()
        if y_t.dim() != 2:
            raise ValueError("y must be a 2D tensor of shape (n, num_out).")
        if y_t.shape[1] != model.num_out:
            raise ValueError(f"Expected y shape (n, {model.num_out}), got {tuple(y_t.shape)}.")

        n = y_t.shape[0]
        device = y_t.device

        if model.num_cont > 0:
            Xc_t = Xc.float()
            if Xc_t.shape[0] != n or Xc_t.shape[1] != model.num_cont:
                raise ValueError(f"Expected Xc shape ({n}, {model.num_cont}), got {tuple(Xc_t.shape)}.")
        else:
            Xc_t = torch.zeros(n, 0, device=device, dtype=y_t.dtype)

        if model.num_enum > 0:
            Xe_t = Xe.long()
            if Xe_t.shape[0] != n or Xe_t.shape[1] != model.num_enum:
                raise ValueError(f"Expected Xe shape ({n}, {model.num_enum}), got {tuple(Xe_t.shape)}.")
        else:
            Xe_t = torch.zeros(n, 0, device=device, dtype=torch.long)

        valid = torch.isfinite(y_t).all(dim=1)
        if Xc_t.shape[1] > 0:
            valid = valid & torch.isfinite(Xc_t).all(dim=1)
        dropped = n - int(valid.sum().item())
        if dropped > 0:
            dropped_indices = torch.nonzero(~valid, as_tuple=False).view(-1).tolist()
            logger.warning(
                "Dropping %d/%d rows due to non-finite y/Xc values; first dropped indices: %s",
                dropped,
                n,
                dropped_indices[:5],
            )
        if not valid.any():
            raise ValueError("No finite rows available for posterior sampling.")

        return Xc_t[valid], Xe_t[valid], y_t[valid]

    @staticmethod
    def _fit_y_scaler(y: FloatTensor) -> tuple[TorchStandardScaler, FloatTensor]:
        """Fit a ``TorchStandardScaler`` on ``y`` and return ``(scaler, standardised y)``."""
        yscaler = TorchStandardScaler()
        yscaler.fit(y.detach().cpu())
        return yscaler, yscaler.transform(y)

    def _draw_sample_id(self, num_samples: int) -> int:
        """Pick a uniform index in ``[0, num_samples)`` using ``self.generator`` (0 if <= 1)."""
        if num_samples <= 1:
            return 0
        return int(torch.randint(0, num_samples, (1,), generator=self.generator).item())

    def total_variation(self, **kwargs):
        """Diagnostic hook for sampler-quality metrics; the base implementation returns 0."""
        return 0
