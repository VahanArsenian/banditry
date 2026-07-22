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
    return 0.5 * ((target - pred) / obs_std) ** 2 + torch.log(obs_std) + HALF_LOG_2PI


class _ModelWrapper:
    """Thin proxy that optionally evaluates a ValueFunction via functional_call."""

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
    """Base class for NLL factories. Subclasses bind context and return a callable."""

    @abstractmethod
    def __call__(self, fix_input: dict, space, X: pd.DataFrame) -> Callable[..., FloatTensor]:
        pass


class GaussianNLL(NLL):
    def __call__(self, fix_input: dict, space, X: pd.DataFrame) -> Callable[..., FloatTensor]:
        return _gaussian_nll


class FeelGoodNLL(NLL):
    def __init__(self, fg_lambda: float = 1.0, fg_bound: float = 10.0):
        if fg_bound < 0:
            raise ValueError(f"fg_bound must be non-negative, got {fg_bound}")
        self.fg_lambda = fg_lambda
        self.fg_bound = fg_bound

    def __call__(self, fix_input: dict, space, X: pd.DataFrame) -> Callable[..., FloatTensor]:
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
        pass

    def __call__(
        self,
        model: ValueFunction,
        Xc: FloatTensor | None,
        Xe: LongTensor | None,
        y: FloatTensor,
        nll: Callable[..., FloatTensor] | None = None,
    ) -> ValueFunction:
        return self.sample(model, Xc, Xe, y, nll=nll)

    @staticmethod
    def _prepare_xy(
        model: ValueFunction,
        Xc: FloatTensor | None,
        Xe: LongTensor | None,
        y: FloatTensor,
    ) -> tuple[FloatTensor, LongTensor, FloatTensor]:
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
        yscaler = TorchStandardScaler()
        yscaler.fit(y.detach().cpu())
        return yscaler, yscaler.transform(y)

    def _draw_sample_id(self, num_samples: int) -> int:
        if num_samples <= 1:
            return 0
        return int(torch.randint(0, num_samples, (1,), generator=self.generator).item())

    def total_variation(self, **kwargs):
        return 0
