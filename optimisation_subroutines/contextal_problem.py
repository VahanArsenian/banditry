from typing import Any

import numpy as np
import torch
from pymoo.core.problem import Problem

from variable_domains.design_space import DesignSpace
from optimisation_subroutines.objectives import Objective


class ContextualProblem(Problem):
    def __init__(self,
            objective : Objective,
            space : DesignSpace, 
            fix   : dict = None 
            ):
        self.objective = objective
        self.space = space
        self.fix = fix or {}
        self.fixed_transformed = self._transform_fix_values(self.space, self.fix)

        self.free_para_names = [name for name in self.space.para_names if name not in self.fix]
        all_vars = self.space.to_pymoo_vars()
        vars = {name: all_vars[name] for name in self.free_para_names}
        super().__init__(vars = vars, n_obj = self.objective.num_obj, n_constr = self.objective.num_constr)

    @staticmethod
    def _transform_fix_values(
        space: DesignSpace,
        fix_input: dict[str, Any],
    ) -> dict[str, float]:
        """Transform already-validated fix values into the optimiser's internal space."""
        transformed: dict[str, float] = {}
        for name, value in fix_input.items():
            para = space.paras[name]
            transformed[name] = float(np.asarray(para.transform(np.array([value], dtype=object))).flat[0])
        return transformed

    @staticmethod
    def _as_candidate_list(para: np.ndarray) -> list[dict]:
        if isinstance(para, dict):
            return [para]
        if isinstance(para, np.ndarray):
            return np.asarray(para, dtype=object).reshape(-1).tolist()
        return list(para)

    def _evaluate(self, para : np.ndarray, out : dict, *args, **kwargs):
        candidates = self._as_candidate_list(para)
        num_x = len(candidates)
        x_cols = []
        xe_cols = []

        for name in self.space.numeric_names:
            if name in self.fixed_transformed:
                col = np.full(num_x, self.fixed_transformed[name], dtype=float)
            else:
                col = np.array([item[name] for item in candidates], dtype=float)
            x_cols.append(col)

        for name in self.space.enum_names:
            if name in self.fixed_transformed:
                col = np.full(num_x, self.fixed_transformed[name], dtype=float)
            else:
                col = np.array([item[name] for item in candidates], dtype=float)
            xe_cols.append(col)

        if x_cols:
            x = torch.FloatTensor(np.stack(x_cols, axis=1))
        else:
            x = torch.empty((num_x, 0), dtype=torch.float32)

        if xe_cols:
            xe = torch.LongTensor(np.stack(xe_cols, axis=1).astype(int))
        else:
            xe = torch.empty((num_x, 0), dtype=torch.long)

        with torch.no_grad():   
            obj_v = self.objective(x, xe)
        out['F'] = obj_v[:, :self.objective.num_obj].detach().cpu().numpy()
        out['G'] = obj_v[:, self.objective.num_obj:].detach().cpu().numpy()
