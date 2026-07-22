"""GP-UCB / optimism-in-the-face-of-uncertainty agent and its surrogate registry."""

import enum
from collections.abc import Callable

import numpy as np
import torch

from banditry.agents.agent import AbstractAgent
from banditry.constants import PI_SQUARED
from banditry.optimisation_oracles.gen_alg import EvolutionOpt
from banditry.optimisation_subroutines.contextual_problem import ContextualProblem
from banditry.optimisation_subroutines.objectives import LCB, MACE
from banditry.surrogates.gp import GP
from banditry.surrogates.svgp import SVGP
from banditry.variable_domains.design_space import DesignSpace


class ModelEnum(enum.Enum):
    """GP surrogate registry for `OFUGPAgent`.

    Members map surrogate names to classes: ``gp`` is an exact Gaussian
    process (accurate, but cubic cost in the number of observations) and
    ``svgp`` is a sparse variational GP (inducing-point approximation that
    scales to larger datasets). `model_config` builds the per-surrogate
    default configuration.
    """

    svgp = SVGP
    gp = GP

    @property
    def default_scaling(self):
        if self == ModelEnum.svgp:
            return 0

    @property
    def is_multi_objective(self):
        if self == ModelEnum.svgp:
            return False

    def model_config(self, space: DesignSpace, configs_to_override=None):
        """Build the default constructor config for this surrogate.

        Args:
            space: Design space (used to derive categorical cardinalities).
            configs_to_override: Optional overrides merged on top of the defaults.

        Returns:
            Keyword-argument dict for the surrogate class constructor. For
            ``svgp``: ``batch_size``, ``num_inducing``, ``use_ngd``. For ``gp``:
            ``lr``, ``num_epochs``, ``verbose``, ``noise_lb``, ``pred_likeli``,
            ``optimizer``. When the space has categorical parameters,
            ``num_uniqs`` (per-parameter cardinalities) is added.
        """
        cfg = {}
        if self == ModelEnum.svgp:
            cfg = {"batch_size": 128, "num_inducing": 256, "use_ngd": False}
        if self == ModelEnum.gp:
            cfg = {
                "lr": 0.01,
                "num_epochs": 100,
                "verbose": False,
                "noise_lb": 8e-4,
                "pred_likeli": True,
                "optimizer": "adam",
            }
        cfg.update(configs_to_override or {})
        if space.num_categorical > 0:
            cfg["num_uniqs"] = [len(space.paras[name].categories) for name in space.enum_names]
        return cfg


class OFUGPAgent(AbstractAgent):
    """Optimism-in-the-face-of-uncertainty (GP-UCB) agent.

    Fits a GP surrogate to the observations and suggests the minimiser of a
    lower-confidence-bound acquisition ``mu - kappa * sigma`` (or of the MACE
    multi-objective ensemble), optimised with an evolutionary optimiser over
    the design space, with context parameters pinned via ``fix_input``.

    The confidence multiplier `kappa` is Bayesian by default; with
    ``frequentist=True`` it becomes the Chowdhury-Gopalan (2017, "On
    Kernelized Multi-armed Bandits") ``beta_t``, which requires the RKHS-norm
    bound ``B`` (``rkhs_norm``) and the sub-Gaussian noise scale ``R``
    (``noise_std_proxy``).

    Args:
        space: The design space to optimise over.
        noise_std_proxy: Sub-Gaussian / GP observation-noise scale ``R``.
            Required — the constructor raises ``ValueError`` if ``None``. Used
            by the frequentist ``beta_t`` and by the realised information gain
            accumulated on every `observe`.
        surrogate: `ModelEnum` member selecting the GP surrogate
            (default: ``ModelEnum.svgp``).
        rand_sample: Sobol warmup budget; see `AbstractAgent`.
        acq_cls: Acquisition class — ``LCB`` (default) or ``MACE`` (MACE is
            required for ``n_suggestions > 1``).
        model_config: Overrides merged into the surrogate's default config
            (see `ModelEnum.model_config`).
        frequentist: Use the Chowdhury-Gopalan ``beta_t`` instead of the
            Bayesian confidence width (default ``False``).
        delta: Confidence level ``delta`` appearing in both width formulas
            (default ``0.01``).
        kappa_fn: Optional callable ``(agent, n_suggestions) -> float``
            overriding `kappa` entirely.
        rkhs_norm: RKHS-norm bound ``B``; required when ``frequentist=True``.
    """

    support_parallel_opt = True
    support_combinatorial = True
    support_contextual = True

    def __init__(
        self,
        space: DesignSpace,
        noise_std_proxy: float,
        surrogate=ModelEnum.svgp,
        rand_sample=None,
        acq_cls=LCB,
        model_config=None,
        frequentist: bool = False,
        delta=0.01,
        kappa_fn: Callable[["OFUGPAgent", int], float] | None = None,
        rkhs_norm=None,
    ):
        super().__init__(space, rand_sample=rand_sample)
        if noise_std_proxy is None:
            raise ValueError(
                "noise_std_proxy is required (sub-Gaussian / GP noise scale used by "
                "the frequentist β_t and the realised information gain)"
            )
        self.surrogate = surrogate
        self.acq_cls = acq_cls
        self.delta = delta
        self.model_config = model_config
        self.frequentist = frequentist
        self._kappa_fn = kappa_fn
        self._rkhs_norm = rkhs_norm
        self.noise_std_proxy = noise_std_proxy
        self._realised_information_gain = 0.0
        self._pending_var_t: np.ndarray | None = None

    def get_model(self, X, Xe, y):
        """Instantiate the configured GP surrogate and fit it to the data."""
        model = self.surrogate.value(
            self.space.num_numeric,
            self.space.num_categorical,
            1,
            **self.surrogate.model_config(self.space, self.model_config),
        )
        model.fit(X, Xe, y)
        return model

    def kappa(self, n_suggestions):
        """Confidence multiplier for the LCB acquisition ``mu - kappa * sigma``.

        If a ``kappa_fn`` override was supplied, it is delegated to. Otherwise,
        with ``t = max(1, n_plays() // n_suggestions)`` rounds played and input
        dimension ``d``:

        - Bayesian (default): ``beta_t = (2 + d/2) * ln(t) + ln(pi^2 / delta)``
          and ``sqrt(beta_t)`` is returned.
        - Frequentist (``frequentist=True``): the Chowdhury-Gopalan width
          ``beta_t = B + 4 * R * sqrt(gamma_t + 1 + ln(1/delta))``, where
          ``gamma_t`` is the realised information gain accumulated across
          `observe` calls; returned as is (already on the sqrt scale).

        Args:
            n_suggestions: Batch size of the current round (scales the round counter).

        Returns:
            The multiplier applied to the predictive standard deviation.

        Raises:
            ValueError: If ``frequentist=True`` and no ``rkhs_norm`` was provided.
        """
        # TODO: Rework
        if self._kappa_fn is not None:
            return self._kappa_fn(self, n_suggestions)

        # The benefit of this is arguable
        t = max(1, self.n_plays() // n_suggestions)
        d = self.X.shape[1]
        delta = self.delta

        if self.frequentist:
            if self._rkhs_norm is None:
                raise ValueError("rkhs_norm must be provided for the frequentist setting")
            beta_t = self._rkhs_norm + 4 * self.noise_std_proxy * np.sqrt(
                self._realised_information_gain + 1 + np.log(1 / delta)
            )
            # Already square rooted
            return beta_t
        else:
            beta_t = (2.0 + d / 2.0) * np.log(t) + np.log(PI_SQUARED / delta)
            return np.sqrt(beta_t)

    def pick_action(self, model, fix_input, n_suggestions=1):
        """Optimise the acquisition and select ``n_suggestions`` candidates.

        Runs an evolutionary optimiser seeded at the incumbent, pads with
        quasi-random samples if needed, and — for batches larger than 2 —
        reserves slots for the most uncertain and the best-predicted
        candidates. The selected candidates' predictive variances are cached
        for the information-gain update in `observe`.

        Raises:
            RuntimeError: If ``n_suggestions > 1`` with a non-MACE acquisition.
        """
        if self.acq_cls != MACE and n_suggestions != 1:
            raise RuntimeError("Parallel optimization is supported only for MACE acquisition")

        best_id = self.get_best_id(fix_input)
        best_x = self.X.iloc[[best_id]]

        py_best, _ = model.predict(*self.space.transform(best_x))
        py_best = py_best.detach().numpy().squeeze()

        kappa = self.kappa(n_suggestions)

        acq = self.acq_cls(model, best_y=py_best, kappa=kappa)
        opt = EvolutionOpt(self.space, pop=100, max_iters=100, verbose=False)
        prob = ContextualProblem(acq, self.space, fix_input)
        rec = opt.optimise(prob, initial_suggest=best_x)
        rec = self._fill_suggestions(rec, n_suggestions, fix_input, max_retries=4)
        select_id = np.random.choice(rec.shape[0], n_suggestions, replace=False).tolist()

        prev_pred_likeli = model.pred_likeli
        model.pred_likeli = False
        with torch.no_grad():
            py_t, ps2_t = model.predict(*self.space.transform(rec))
            py_all = py_t.reshape(-1).cpu().numpy()
            ps2_all = ps2_t.reshape(-1).cpu().numpy()
            best_pred_id = int(np.argmin(py_all))
            best_unce_id = int(np.argmax(ps2_all))
            if best_unce_id not in select_id and n_suggestions > 2:
                select_id[0] = best_unce_id
            if best_pred_id not in select_id and n_suggestions > 2:
                select_id[1] = best_pred_id
            rec_selected = rec.iloc[select_id].copy()
            self._pending_var_t = ps2_all[select_id]
        model.pred_likeli = prev_pred_likeli
        return rec_selected

    def observe(self, X, y):
        """Record observations and update the realised information gain.

        Uses the predictive variances cached by `pick_action` when available
        (otherwise refits the GP on the past data, falling back to the prior
        variance on failure) to accumulate ``0.5 * sum(log(1 + var / R^2))``,
        then delegates to `AbstractAgent.observe`.
        """
        if self._pending_var_t is not None:
            var_t = self._pending_var_t
            self._pending_var_t = None
        elif len(self.y) > 0:
            try:
                Xc_prev, Xe_prev, y_prev = self.prepare_data()
                model = self.get_model(Xc_prev, Xe_prev, y_prev)
                model.pred_likeli = False  # latent σ_f² for the info gain
                xc_new, xe_new = self.space.transform(X)
                with torch.no_grad():
                    _, var_new = model.predict(xc_new, xe_new)
                var_t = var_new.detach().cpu().numpy().reshape(-1)
            except Exception:
                import banditry.logging_utils as log

                log.debug("OFUGPAgent.observe: GP fit failed; falling back to prior variance")
                var_t = np.ones(len(X))
        else:
            var_t = np.ones(len(X))
        self._realised_information_gain += 0.5 * float(np.log1p(var_t / self.noise_std_proxy**2).sum())
        super().observe(X, y)

    def label_params(self) -> dict:
        return {
            "frequentist": bool(self.frequentist),
            "surrogate_name": getattr(self.surrogate, "name", None),
            "rkhs_norm": (float(self._rkhs_norm) if self._rkhs_norm is not None else None),
        }
