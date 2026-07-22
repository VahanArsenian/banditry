"""Abstract base class shared by all bandit agents."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine

import banditry.logging_utils as log
from banditry.labels import agent_display_name
from banditry.variable_domains.design_space import DesignSpace


class AbstractAgent(ABC):
    """Base class for bandit agents driving a suggest -> evaluate -> observe loop.

    Usage contract:

    - Call `suggest` to obtain a batch of candidate configurations (a DataFrame
      with one column per parameter of the design space), evaluate them
      externally, then feed the results back with `observe`. Repeat.
    - The library **minimises** the observed values ``y``; negate rewards if
      you need maximisation.
    - The first `rand_sample` suggestions are quasi-random (scrambled Sobol)
      warmup draws; once at least ``rand_sample`` observations have been
      recorded, suggestions come from the subclass's model
      (`get_model` + `pick_action`).
    - ``fix_input`` turns the loop into a contextual bandit: any subset of
      parameters can be pinned to observed context values each round, and the
      remaining parameters are optimised conditionally.

    Args:
        space: The design space to optimise over.
        rand_sample: Sobol warmup budget. ``None`` (default) resolves to
            ``1 + space.num_paras``; explicit values are floored at 2.

    Attributes:
        space: The design space being optimised.
        X: DataFrame of all observed inputs, one column per parameter.
        y: ``(n, 1)`` float array of observed objective values (lower is better).
        rand_sample: Resolved warmup budget (see above).
        support_parallel_opt: Whether the agent supports ``n_suggestions > 1``.
        support_constraint: Whether constrained optimisation is supported.
        support_multi_objective: Whether multi-objective observations are supported.
        support_combinatorial: Whether categorical/integer parameters are supported.
        support_contextual: Whether ``fix_input`` contexts are supported.
    """

    support_parallel_opt = False
    support_constraint = False
    support_multi_objective = False
    support_combinatorial = False
    support_contextual = False

    def __init__(
        self,
        space: DesignSpace,
        rand_sample: int | None = None,
    ):
        self.space = space
        self.X = pd.DataFrame(columns=self.space.para_names)
        self.y = np.zeros((0, 1))
        self.rand_sample = 1 + self.space.num_paras if rand_sample is None else max(2, rand_sample)
        # SobolEngine is constructed lazily on first use so its scramble
        # is captured from torch's *post-seed* global state (set by
        # Experiment.run's seed_everything), not whatever torch state
        # happens to be live at factory time.
        self._sobol: SobolEngine | None = None

    @property
    def sobol(self) -> SobolEngine:
        if self._sobol is None:
            self._sobol = SobolEngine(self.space.num_paras, scramble=True)
        return self._sobol

    def observe(self, X: pd.DataFrame, y: np.ndarray) -> None:
        """Record evaluated suggestions in the agent's history.

        Rows whose ``y`` is non-finite (NaN or infinite) are dropped; the
        remaining rows are appended to `X` and `y`.

        Args:
            X: Evaluated inputs, one column per parameter (as returned by `suggest`).
            y: Observed objective values, one per row of ``X`` (reshaped to
                ``(n, 1)``). Lower is better — the agent minimises.
        """
        valid_id = np.where(np.isfinite(y.reshape(-1)))[0].tolist()
        XX = X.iloc[valid_id]
        yy = y[valid_id].reshape(-1, 1)
        self.X = pd.concat([self.X, XX], axis=0, ignore_index=True)
        self.y = np.vstack([self.y, yy])

    def quasi_sample(self, n: int, fix_input: dict[str, Any] | None = None) -> pd.DataFrame:
        """Draw quasi-random samples from the design space.

        Samples come from a scrambled Sobol sequence in the transformed space
        and are mapped back to the original parameter domain; numeric
        parameters that are discrete after transformation are rounded. Used
        for warmup rounds and to pad incomplete suggestion batches.

        Args:
            n: Number of samples to draw.
            fix_input: Optional mapping of parameter name to pinned value; the
                corresponding columns are overwritten after sampling.

        Returns:
            DataFrame with ``n`` rows, one column per parameter, in the
            original (untransformed) domain.
        """
        fix_input = fix_input or {}
        samp = self.sobol.draw(n)
        samp = samp * (self.space.opt_ub - self.space.opt_lb) + self.space.opt_lb
        x = samp[:, : self.space.num_numeric]
        xe = samp[:, self.space.num_numeric :]
        for i, name in enumerate(self.space.numeric_names):
            if self.space.paras[name].is_discrete_after_transform:
                x[:, i] = x[:, i].round()
        df_samp = self.space.inverse_transform(x, xe)
        for k, v in fix_input.items():
            df_samp[k] = v
        return df_samp

    def get_best_id(self, fix_input: dict[str, Any] | None = None) -> int:
        """Return the positional index of the best (lowest-``y``) observation.

        Args:
            fix_input: Optional context filter. When given, only observations
                whose pinned columns match the given values are considered
                (fully numeric columns are compared with a tight absolute
                tolerance, other columns by equality). If no row matches, the
                method falls back to the global argmin.

        Returns:
            Positional index into `X`/`y` of the best matching observation.
        """
        if not fix_input:
            return int(np.argmin(self.y.reshape(-1)))
        X = self.X.copy()
        y = self.y.copy()
        for k, v in fix_input.items():
            col = X[k]
            col_numeric = pd.to_numeric(col, errors="coerce")
            if col_numeric.notna().all() and np.isscalar(v):
                crit = ~np.isclose(
                    col_numeric.to_numpy(dtype=float),
                    float(v),
                    rtol=0.0,
                    atol=np.finfo(np.float64).eps,
                )
            else:
                crit = (col != v).to_numpy()
            y[crit] = np.inf
        if np.isfinite(y).any():
            return int(np.argmin(y.reshape(-1)))
        return int(np.argmin(self.y.reshape(-1)))

    def check_unique(self, rec: pd.DataFrame) -> list[bool]:
        """Flag which rows of ``rec`` are new.

        Args:
            rec: Candidate suggestions to check.

        Returns:
            One boolean per row of ``rec``: ``True`` if the row duplicates
            neither a past observation nor an earlier row of ``rec`` itself.
        """
        return (~pd.concat([self.X, rec], axis=0).duplicated().tail(rec.shape[0]).values).tolist()

    def _fill_suggestions(self, rec, n_suggestions, fix_input, max_retries=4):
        """Try to fill *rec* to *n_suggestions* rows with unique quasi-random
        samples, falling back to non-unique samples as a last resort."""
        for _ in range(max_retries):
            if rec.shape[0] >= n_suggestions:
                break
            log.debug(f"Quasi-sampling {n_suggestions - rec.shape[0]} more (fix_input={fix_input})")
            rand_rec = self.quasi_sample(n_suggestions - rec.shape[0], fix_input)
            rand_rec = rand_rec[self.check_unique(rand_rec)]
            rec = pd.concat([rec, rand_rec], axis=0, ignore_index=True)
        if rec.shape[0] < n_suggestions:
            rand_rec = self.quasi_sample(n_suggestions - rec.shape[0], fix_input)
            rec = pd.concat([rec, rand_rec], axis=0, ignore_index=True)
        return rec

    def prepare_data(self):
        """Transform the observation history into model-ready tensors.

        Returns:
            Tuple ``(Xc, Xe, y)``: transformed numeric features, transformed
            categorical features, and a float tensor copy of the observed values.
        """
        X, Xe = self.space.transform(self.X)
        y = torch.FloatTensor(self.y).clone()
        return X, Xe, y

    @abstractmethod
    def get_model(self, Xc, Xe, y):
        """Build the decision model from transformed data.

        Subclasses must fit (or posterior-sample) their surrogate here.

        Args:
            Xc: Transformed numeric features.
            Xe: Transformed categorical features.
            y: Observed values as a float tensor.

        Returns:
            A model consumable by `pick_action`.
        """
        pass

    @abstractmethod
    def pick_action(self, model, fix_input, n_suggestions=1):
        """Choose the next suggestions given a fitted or sampled model.

        Subclasses must optimise their acquisition here, honouring ``fix_input``.

        Args:
            model: The model returned by `get_model`.
            fix_input: Mapping of parameter name to pinned context value (may be empty).
            n_suggestions: Number of suggestions to return.

        Returns:
            DataFrame of ``n_suggestions`` suggested configurations.
        """
        pass

    def suggest(self, n_suggestions: int = 1, fix_input: dict[str, Any] | None = None) -> pd.DataFrame:
        """Propose the next configurations to evaluate.

        While fewer than `rand_sample` observations have been recorded, this
        returns quasi-random Sobol samples; afterwards it fits or samples the
        model (`get_model`) and optimises the acquisition (`pick_action`).

        Args:
            n_suggestions: Number of configurations to return.
            fix_input: Optional context — parameter values to pin this round
                (contextual bandit).

        Returns:
            DataFrame with ``n_suggestions`` rows, one column per parameter.
        """
        fix_input = fix_input or {}
        if self.X.shape[0] < self.rand_sample:
            return self.quasi_sample(n_suggestions, fix_input)

        Xc, Xe, y = self.prepare_data()

        model = self.get_model(Xc, Xe, y)

        return self.pick_action(model, fix_input, n_suggestions)

    def n_plays(self) -> int:
        """Return the number of observations recorded so far."""
        return len(self.y)

    def label_params(self) -> dict:
        return {}

    @property
    def display_name(self) -> str:
        return agent_display_name(type(self).__name__, self.label_params())
