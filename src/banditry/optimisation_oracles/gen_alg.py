from enum import Enum

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.config import Config
from pymoo.core.algorithm import Algorithm, default_termination
from pymoo.core.mixed import MixedVariableDuplicateElimination, MixedVariableGA, MixedVariableMating
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.termination.collection import TerminationCollection
from pymoo.util.ref_dirs import get_reference_directions

from banditry.variable_domains.design_space import DesignSpace

Config.show_compile_hint = True


def get_init_pop(
    space: DesignSpace,
    population_size: int,
    initial_suggest: pd.DataFrame | None = None,
    selected_para_names: list[str] | None = None,
) -> Population:
    """Build an initial pymoo ``Population`` in the optimiser's transformed space.

    Samples ``population_size`` points from the design space, optionally placing rows from
    ``initial_suggest`` at the head of the population (e.g. to warm-start from the
    incumbent), transforms everything into the numeric optimiser space, and keeps only the
    requested variables.

    Args:
        space: Design space to sample from and transform with.
        population_size: Number of individuals in the returned population.
        initial_suggest: Optional raw-space DataFrame whose rows are prepended before the
            population is truncated to ``population_size``.
        selected_para_names: Names of the variables to keep in each individual (defaults
            to all of ``space.para_names``); used when context variables are fixed and
            excluded from optimisation.

    Returns:
        A pymoo ``Population`` of dict-encoded individuals, with enum variables stored as
        ``int`` and numeric variables as ``float``.

    Raises:
        ValueError: If ``selected_para_names`` contains names unknown to ``space``.
    """
    active_para_names = list(space.para_names if selected_para_names is None else selected_para_names)
    unknown_names = [name for name in active_para_names if name not in space.para_names]
    if unknown_names:
        raise ValueError(f"Unknown optimisation parameters: {unknown_names}")

    init_pop = space.sample(population_size)

    # Useful for preconditioning, especially for inducing points.
    if initial_suggest is not None:
        init_pop = pd.concat([initial_suggest[space.para_names], init_pop], axis=0).head(population_size)

    x_num, x_cat = space.transform(init_pop[space.para_names])
    init_pop = np.hstack([x_num.numpy(), x_cat.numpy().astype(float)])
    name_to_idx = {name: i for i, name in enumerate(space.para_names)}
    pop_lst = []
    for item in init_pop:
        pop_item = {}
        for name in active_para_names:
            i = name_to_idx[name]
            if name in space.enum_names:
                pop_item[name] = int(item[i])
            else:
                pop_item[name] = item[i]
        pop_lst.append(pop_item)
    pop = Population.new(X=pop_lst)
    return pop


class GAEnum(Enum):
    """Registry of the pymoo genetic algorithms :class:`EvolutionOpt` can dispatch to.

    Members map to pymoo algorithm classes: ``nsga3`` (NSGA-III), ``unsga3`` (U-NSGA-III,
    NSGA-III with a pressure-based rather than random tournament), ``msga`` (pymoo's
    mixed-variable GA) and ``nsga2`` (NSGA-II). :meth:`determine` picks a member from the
    objective count and compute budget; :meth:`auto` instantiates the chosen algorithm with
    mixed-variable mating and duplicate elimination, adding energy-based reference
    directions for the reference-direction ("compute intense") members.
    """

    nsga3 = NSGA3
    unsga3 = UNSGA3  # Different tournament mechanism (computes pressure potential rather than random)
    msga = MixedVariableGA
    nsga2 = NSGA2

    @classmethod
    def compute_intense(cls):
        """Members requiring reference directions, which are costly to generate."""
        return [GAEnum.nsga3, GAEnum.unsga3]

    def is_compute_intense(self):
        """Whether this member needs reference directions (see :meth:`compute_intense`)."""
        return self in self.compute_intense()

    @classmethod
    def determine(cls, n_obj: int, compute_budget_high=True):
        """Select an algorithm: U-NSGA-III (<= 2 objectives) or NSGA-III (> 2) on a high
        compute budget, otherwise the mixed-variable GA or NSGA-II respectively."""
        if n_obj <= 2:
            return cls.unsga3 if compute_budget_high else cls.msga
        else:
            return cls.nsga3 if compute_budget_high else cls.nsga2

    def auto(self, n_dim=1, n_points=128, **kwargs) -> Algorithm:
        """Instantiate the algorithm with mixed-variable mating and duplicate elimination.

        Args:
            n_dim: Number of objectives; sets the dimensionality of the energy-based
                reference directions for the compute-intense members.
            n_points: Number of reference directions to generate.
            **kwargs: Forwarded to the pymoo algorithm constructor (e.g. ``pop_size``,
                ``repair``, ``sampling``).

        Returns:
            A configured pymoo ``Algorithm`` instance.
        """
        if self.is_compute_intense():
            ref_dir = get_reference_directions("energy", n_dim=n_dim, n_points=n_points)
            # Repair for NSGA3 may not work properly, so we may need to move to value conditioning
            return self.value(
                ref_dirs=ref_dir,
                mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                eliminate_duplicates=MixedVariableDuplicateElimination(),
                **kwargs,
            )
        else:
            return self.value(
                MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                eliminate_duplicates=MixedVariableDuplicateElimination(),
                **kwargs,
            )


class EvolutionOpt:
    """Evolutionary acquisition optimiser wrapping pymoo's mixed-variable GAs.

    Minimises an acquisition ``Problem`` (typically a ``ContextualProblem``) over the
    design space with a genetic algorithm chosen via :class:`GAEnum`: unless a method is
    passed explicitly, ``GAEnum.determine(problem.n_obj, compute_budget_high=True)`` picks
    U-NSGA-III for one or two objectives and NSGA-III for three or more. All algorithms run
    with mixed-variable mating and duplicate elimination, so continuous, integer and
    categorical variables are handled natively.

    Args:
        design_space: The search space to optimise over.
        **conf: Optional settings — ``pop`` (population size, default 100), ``max_iters``
            (generation cap, default 500), ``verbose`` (pymoo verbosity, default False),
            ``repair`` (optional pymoo repair operator, default None) and ``sobol_init``
            (default True; stored but currently unused).
    """

    def __init__(self, design_space: DesignSpace, **conf):
        self.space = design_space
        self.pop = conf.get("pop", 100)
        self.max_iters = conf.get("max_iters", 500)
        self.verbose = conf.get("verbose", False)
        self.repair = conf.get("repair", None)
        self.sobol_init = conf.get("sobol_init", True)

    def termnation_condition(self, problem: Problem):
        """Combine pymoo's default termination with an ``n_gen`` cap of ``max_iters``."""
        def_term = default_termination(problem)
        max_iter_term = get_termination("n_gen", self.max_iters)
        return TerminationCollection(def_term, max_iter_term)

    @staticmethod
    def _as_candidate_list(candidates) -> list:
        """Normalise pymoo's ``res.X`` output (dict, ndarray, or sequence) to a list."""
        if candidates is None:
            return []
        if isinstance(candidates, dict):
            return [candidates]
        if isinstance(candidates, np.ndarray):
            arr = np.asarray(candidates)
            if arr.ndim == 0:
                return [arr.item()]
            if arr.dtype == object:
                return arr.reshape(-1).tolist()
            if arr.ndim == 1:
                return [arr]
            return [row for row in arr]
        return list(candidates)

    @staticmethod
    def _candidate_frame(candidates: list, active_para_names: list[str]) -> pd.DataFrame:
        """Arrange candidate dicts/arrays into a DataFrame with the active variable columns."""
        if len(candidates) == 0:
            return pd.DataFrame(columns=active_para_names)

        first = candidates[0]
        if isinstance(first, dict):
            df_candidates = pd.DataFrame(candidates)
        else:
            arr = np.asarray(candidates, dtype=float)
            if arr.ndim == 1:
                if len(active_para_names) == 1:
                    arr = arr.reshape(-1, 1)
                else:
                    arr = arr.reshape(1, -1)
            df_candidates = pd.DataFrame(arr, columns=active_para_names)

        return df_candidates[active_para_names]

    def _decode_candidates(
        self,
        candidates: list,
        active_para_names: list[str],
        fix_input: dict,
    ) -> pd.DataFrame:
        """Inverse-transform optimiser candidates to raw values and re-attach fixed columns."""
        df_candidates = self._candidate_frame(candidates, active_para_names)
        n_rows = df_candidates.shape[0]
        df_out = pd.DataFrame(index=range(n_rows), columns=self.space.para_names)

        active_set = set(active_para_names)
        for name in self.space.para_names:
            if name in active_set:
                transformed = pd.to_numeric(df_candidates[name], errors="raise").to_numpy(dtype=float)
                df_out[name] = self.space.paras[name].inverse_transform(transformed)
            elif name in fix_input:
                df_out[name] = fix_input[name]
            else:
                raise ValueError(f"Cannot reconstruct parameter '{name}' from optimiser output.")
        return df_out[self.space.para_names]

    def _fixed_only_recommendation(self, fix_input: dict, initial_suggest: pd.DataFrame | None) -> pd.DataFrame:
        """Build the single recommendation when every variable is fixed (nothing to optimise)."""
        seed_row = None
        if initial_suggest is not None and not initial_suggest.empty:
            seed_row = initial_suggest.iloc[0]

        row = {}
        missing = []
        for name in self.space.para_names:
            if name in fix_input:
                row[name] = fix_input[name]
            elif seed_row is not None and name in seed_row:
                row[name] = seed_row[name]
            else:
                missing.append(name)

        if missing:
            raise ValueError(f"No free variables to optimise but missing fixed values for: {missing}")
        return pd.DataFrame([row], columns=self.space.para_names)

    def optimise(
        self,
        problem: Problem,
        initial_suggest: pd.DataFrame = None,
        return_pop=False,
        method: GAEnum = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Run the evolutionary search and decode the results to raw design-space values.

        Reads the fixed context values (``problem.fix``) and free variables
        (``problem.vars``) off the problem; when every variable is fixed the fixed values
        are returned directly without running the GA. Otherwise an initial population is
        built via :func:`get_init_pop`, the algorithm selected by ``method`` (or
        :meth:`GAEnum.determine`) is run under the combined default/``n_gen`` termination,
        and the resulting candidates are inverse-transformed back to raw parameter values
        with fixed context columns re-attached.

        Args:
            problem: pymoo ``Problem`` to minimise; typically a ``ContextualProblem``.
            initial_suggest: Optional raw-space DataFrame used to seed the initial
                population (and to fill free values when everything is fixed).
            return_pop: If ``True``, return the whole final population (sorted by the
                first objective) instead of the optimiser's solution set.
            method: Explicit :class:`GAEnum` member overriding the automatic selection.
            seed: Seed passed to ``pymoo.minimize``; when ``None`` one is drawn from the
                globally seeded numpy RNG so the run stays reproducible.

        Returns:
            DataFrame of recommended configurations in raw design-space values, one row
            per candidate, with columns ordered as ``space.para_names``. The raw pymoo
            result is stored on ``self.res``.
        """
        fix_input = getattr(problem, "fix", None) or {}

        problem_vars = getattr(problem, "vars", None)
        active_para_names = list(self.space.para_names) if problem_vars is None else list(problem_vars.keys())

        if len(active_para_names) == 0:
            self.res = None
            return self._fixed_only_recommendation(fix_input, initial_suggest)

        init_pop = get_init_pop(self.space, self.pop, initial_suggest, active_para_names)

        if method is None:
            method = GAEnum.determine(problem.n_obj, compute_budget_high=True)

        algo = method.auto(
            n_dim=problem.n_obj, n_points=self.pop, pop_size=self.pop, repair=self.repair, sampling=init_pop
        )

        # pymoo.minimize(seed=None) draws from a non-deterministic source
        # rather than the globally-seeded numpy RNG, so reproducibility
        # requires an explicit seed. We derive one from np.random — which
        # IS seeded by Experiment.run's seed_everything — so the chain
        # stays deterministic without any per-call seed plumbing upstream.
        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1))

        res = minimize(problem, algo, termination=self.termnation_condition(problem), seed=seed, verbose=self.verbose)

        if res.X is None:
            import banditry.logging_utils as log

            log.debug("Optimisation terminated with no solutions found")

        if res.X is not None and not return_pop:
            candidates = self._as_candidate_list(res.X)
        else:
            candidates = [p for p in res.pop]
            if problem.n_obj == 1 and not return_pop and len(candidates) > 0:
                candidates = [candidates[np.random.choice(len(candidates))]]
            candidates = sorted(candidates, key=lambda x: x.F[0])
            candidates = [c.X for c in candidates]

        self.res = res
        return self._decode_candidates(candidates, active_para_names, fix_input)
