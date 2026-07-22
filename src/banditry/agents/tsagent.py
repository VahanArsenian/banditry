from typing import Optional

import numpy as np

from banditry.agents.agent import AbstractAgent
from banditry.optimisation_oracles.gen_alg import EvolutionOpt
from banditry.optimisation_subroutines.objectives import ThompsonObjective
from banditry.optimisation_subroutines.contextal_problem import ContextualProblem
from banditry.sampling_oracles.langevin_sampler import LangevinSampler
from banditry.sampling_oracles.sampler import NLL, Sampler
from banditry.surrogates.tsmodel import ValueFunction
from banditry.variable_domains.design_space import DesignSpace

import banditry.logging_utils as log


class TSAgent(AbstractAgent):
    support_parallel_opt = True
    support_combinatorial = True
    support_contextual = True

    def __init__(
        self,
        space: DesignSpace,
        rand_sample=None,
        model_config: Optional[dict] = None,
        sampler_cls: type[Sampler] = LangevinSampler,
        sampler_config: Optional[dict] = None,
        nll: NLL | None = None,
        should_warm_start: bool = False,
        latent_dimension: Optional[float] = None,
    ):
        super().__init__(space, rand_sample=rand_sample)
        self.model_config = {} if model_config is None else dict(model_config)

        self.sampler_cls = sampler_cls
        self.sampler = self.sampler_cls(**({} if sampler_config is None else dict(sampler_config)))
        self.nll = nll
        self.should_warm_start = should_warm_start
        self.warm_start_model: ValueFunction | None = None
        self._num_samplable_params: Optional[int] = None
        self.latent_dimension = latent_dimension if latent_dimension is not None else self.num_samplable_params()

    def _value_function_config(self) -> dict:
        cfg = dict(self.model_config)
        if self.space.num_categorical > 0 and "num_uniqs" not in cfg:
            cfg["num_uniqs"] = [self.space.paras[name].num_uniqs for name in self.space.enum_names]
        return cfg


    def get_model(self, Xc, Xe, y, nll=None) -> ValueFunction:
        if self.should_warm_start and self.warm_start_model is not None:
            initial_model = self.warm_start_model
        else:
            initial_model = ValueFunction(
            self.space.num_numeric,
            self.space.num_categorical,
            1,
            **self._value_function_config(),
        )
        sampled_model = self.sampler.sample(initial_model, Xc, Xe, y, nll=nll)
        if self.should_warm_start:
            self.warm_start_model = sampled_model
        return sampled_model

    def suggest(self, n_suggestions=1, fix_input=None):
        fix_input = fix_input or {}
        if self.X.shape[0] < self.rand_sample:
            return self.quasi_sample(n_suggestions, fix_input)

        Xc, Xe, y = self.prepare_data()
        nll = self.nll(fix_input, self.space, self.X) if self.nll is not None else None
        model = self.get_model(Xc, Xe, y, nll=nll)
        return self.pick_action(model, fix_input, n_suggestions)

    def label_params(self) -> dict:
        return {
            "sampler_name": self.sampler_cls.__name__,
            "nll_name": type(self.nll).__name__ if self.nll is not None else None,
            "latent_dimension": (float(self.latent_dimension)
                                 if self.latent_dimension is not None else None),
        }

    def pick_action(self, model, fix_input, n_suggestions=1):
        best_id = self.get_best_id(fix_input)
        best_x = self.X.iloc[[best_id]]

        opt = EvolutionOpt(self.space, pop=100, max_iters=100, verbose=False)
        prob = ContextualProblem(ThompsonObjective(model), self.space, fix_input)
        pop_rec = opt.optimise(
            prob,
            initial_suggest=best_x,
            return_pop=True,
        )
        mask = self.check_unique(pop_rec)
        log.debug(f"Leftover solutions after filtration (in parts): {np.mean(mask)}")
        rec = pop_rec[mask].reset_index(drop=True)
        rec = self._fill_suggestions(rec, n_suggestions, fix_input, max_retries=5)

        return rec.head(n_suggestions).copy()

    def num_samplable_params(self) -> int:
        """Count of `ValueFunction` parameters with `requires_grad=True`
        — the exact set both `LangevinSampler` and `NUTSSampler` iterate
        over. Determined by architecture alone (network builder +
        feature extractor + scalers), so a throwaway `ValueFunction`
        instance suffices to read it off. Cached after the first call."""
        if self._num_samplable_params is None:
            vf = ValueFunction(
                self.space.num_numeric,
                self.space.num_categorical,
                1,
                **self._value_function_config(),
            )
            self._num_samplable_params = sum(
                p.numel() for p in vf.parameters() if p.requires_grad
            )
        return self._num_samplable_params

