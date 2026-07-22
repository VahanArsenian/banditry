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
        model = self.surrogate.value(
            self.space.num_numeric,
            self.space.num_categorical,
            1,
            **self.surrogate.model_config(self.space, self.model_config),
        )
        model.fit(X, Xe, y)
        return model

    def kappa(self, n_suggestions):
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
