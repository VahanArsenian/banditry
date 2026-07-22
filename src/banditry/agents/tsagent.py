"""Thompson-sampling agent backed by a neural value function and MCMC posterior sampling."""

import numpy as np

import banditry.logging_utils as log
from banditry.agents.agent import AbstractAgent
from banditry.optimisation_oracles.gen_alg import EvolutionOpt
from banditry.optimisation_subroutines.contextual_problem import ContextualProblem
from banditry.optimisation_subroutines.objectives import ThompsonObjective
from banditry.sampling_oracles.langevin_sampler import LangevinSampler
from banditry.sampling_oracles.sampler import NLL, Sampler
from banditry.surrogates.tsmodel import ValueFunction
from banditry.variable_domains.design_space import DesignSpace


class TSAgent(AbstractAgent):
    """Thompson-sampling agent with a neural value function.

    Each round, the weights of a neural ``ValueFunction`` are sampled from
    their posterior by an MCMC sampling oracle (`LangevinSampler` / SGLD, or
    ``NUTSSampler`` / Pyro NUTS), and the sampled network is minimised over
    the design space with an evolutionary optimiser (context parameters pinned
    via ``fix_input``) — i.e. the agent plays the argmin of one posterior draw.

    Optional Feel-Good Thompson sampling (Zhang, 2021, "Feel-Good Thompson
    Sampling for Contextual Bandits and Reinforcement Learning") reweights the
    likelihood with an exploration-boosting term; enable it by passing
    ``nll=FeelGoodNLL(...)``.

    Args:
        space: The design space to optimise over.
        rand_sample: Sobol warmup budget; see `AbstractAgent`.
        model_config: Keyword arguments for the ``ValueFunction`` surrogate
            (e.g. ``num_uniqs``, ``emb_sizes``, a custom feature extractor
            ``fe``). ``num_uniqs`` is filled in automatically for spaces with
            categorical parameters.
        sampler_cls: Sampling-oracle class (default: `LangevinSampler`).
        sampler_config: Keyword arguments splatted into ``sampler_cls``;
            ``None`` uses the sampler's own constructor defaults.
        nll: Optional `NLL` factory replacing the plain Gaussian likelihood,
            e.g. ``FeelGoodNLL`` for Feel-Good Thompson sampling. It is bound
            to the current context and observation history on every `suggest`.
        should_warm_start: If ``True``, reuse the previous round's sampled
            model as the MCMC initialisation of the next round (default
            ``False`` here; the factory ``TSConfig`` enables it).
        latent_dimension: Latent dimension of the sampled parameter vector,
            recorded for labelling/diagnostics; ``None`` resolves to
            `num_samplable_params`.
    """

    support_parallel_opt = True
    support_combinatorial = True
    support_contextual = True

    def __init__(
        self,
        space: DesignSpace,
        rand_sample=None,
        model_config: dict | None = None,
        sampler_cls: type[Sampler] = LangevinSampler,
        sampler_config: dict | None = None,
        nll: NLL | None = None,
        should_warm_start: bool = False,
        latent_dimension: float | None = None,
    ):
        super().__init__(space, rand_sample=rand_sample)
        self.model_config = {} if model_config is None else dict(model_config)

        self.sampler_cls = sampler_cls
        self.sampler = self.sampler_cls(**({} if sampler_config is None else dict(sampler_config)))
        self.nll = nll
        self.should_warm_start = should_warm_start
        self.warm_start_model: ValueFunction | None = None
        self._num_samplable_params: int | None = None
        self.latent_dimension = latent_dimension if latent_dimension is not None else self.num_samplable_params()

    def _value_function_config(self) -> dict:
        cfg = dict(self.model_config)
        if self.space.num_categorical > 0 and "num_uniqs" not in cfg:
            cfg["num_uniqs"] = [self.space.paras[name].num_uniqs for name in self.space.enum_names]
        return cfg

    def get_model(self, Xc, Xe, y, nll=None) -> ValueFunction:
        """Draw one posterior sample of the value function.

        Starts from the previous round's sampled model when warm-starting is
        enabled (and available), otherwise from a freshly initialised
        ``ValueFunction``, and runs the sampling oracle on the data.

        Args:
            Xc: Transformed numeric features.
            Xe: Transformed categorical features.
            y: Observed values as a float tensor.
            nll: Optional bound NLL callable overriding the Gaussian likelihood.

        Returns:
            The sampled ``ValueFunction`` (also stored for warm-starting).
        """
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
        """Propose the next configurations to evaluate (see `AbstractAgent.suggest`).

        Identical to the base implementation, except that the ``nll`` factory,
        when present, is bound to the current context (``fix_input``) and the
        observation history before posterior sampling.
        """
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
            "latent_dimension": (float(self.latent_dimension) if self.latent_dimension is not None else None),
        }

    def pick_action(self, model, fix_input, n_suggestions=1):
        """Minimise the sampled value function over the design space.

        Runs an evolutionary optimiser seeded at the incumbent, keeps the
        final population, filters out already-observed rows, pads with
        quasi-random samples if needed, and returns the top ``n_suggestions``
        rows.
        """
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
        """Count the ``ValueFunction`` parameters with ``requires_grad=True``.

        This is the exact set both `LangevinSampler` and ``NUTSSampler``
        iterate over. It is determined by architecture alone (network builder
        + feature extractor + scalers), so a throwaway ``ValueFunction``
        instance suffices to read it off. Cached after the first call.

        Returns:
            Total number of samplable scalar parameters.
        """
        if self._num_samplable_params is None:
            vf = ValueFunction(
                self.space.num_numeric,
                self.space.num_categorical,
                1,
                **self._value_function_config(),
            )
            self._num_samplable_params = sum(p.numel() for p in vf.parameters() if p.requires_grad)
        return self._num_samplable_params
