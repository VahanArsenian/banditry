from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine
from pybandits.variable_domains.design_space import DesignSpace

import pybandits.logging_utils as log
from pybandits.labels import agent_display_name


class AbstractAgent(ABC):
    support_parallel_opt = False
    support_constraint = False
    support_multi_objective = False
    support_combinatorial = False
    support_contextual = False

    def __init__(
        self,
        space: DesignSpace,
        rand_sample: Optional[int] = None,
    ):
        self.space = space
        self.X = pd.DataFrame(columns=self.space.para_names)
        self.y = np.zeros((0, 1))
        self.rand_sample = (
            1 + self.space.num_paras if rand_sample is None else max(2, rand_sample)
        )
        # SobolEngine is constructed lazily on first use so its scramble
        # is captured from torch's *post-seed* global state (set by
        # Experiment.run's seed_everything), not whatever torch state
        # happens to be live at factory time.
        self._sobol: Optional[SobolEngine] = None

    @property
    def sobol(self) -> SobolEngine:
        if self._sobol is None:
            self._sobol = SobolEngine(self.space.num_paras, scramble=True)
        return self._sobol

    def observe(self, X: pd.DataFrame, y: np.ndarray):
        valid_id = np.where(np.isfinite(y.reshape(-1)))[0].tolist()
        XX = X.iloc[valid_id]
        yy = y[valid_id].reshape(-1, 1)
        self.X = pd.concat([self.X, XX], axis=0, ignore_index=True)
        self.y = np.vstack([self.y, yy])

    def quasi_sample(self, n, fix_input=None):
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

    def get_best_id(self, fix_input: dict = None) -> int:
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
        X, Xe = self.space.transform(self.X)
        y = torch.FloatTensor(self.y).clone()
        return X, Xe, y

    @abstractmethod
    def get_model(self, Xc, Xe, y):
        pass

    @abstractmethod
    def pick_action(self, model, fix_input, n_suggestions=1):
        pass

    def suggest(self, n_suggestions=1, fix_input=None):
        fix_input = fix_input or {}
        if self.X.shape[0] < self.rand_sample:
            return self.quasi_sample(n_suggestions, fix_input)

        Xc, Xe, y = self.prepare_data()

        model = self.get_model(Xc, Xe, y)

        return self.pick_action(model, fix_input, n_suggestions)

    def n_plays(self):
        return len(self.y)

    def label_params(self) -> dict:
        return {}

    @property
    def display_name(self) -> str:
        return agent_display_name(type(self).__name__, self.label_params())