import warnings

import pandas as pd
import torch
from pymoo.core.variable import Choice, Integer, Real
from torch import Tensor

from banditry.variable_domains.parameter_types import (
    BoolParameter,
    CategoricalParameter,
    IntParameter,
    NumericParameter,
    Parameter,
)


class DesignSpace:
    parameter_types = {"num": NumericParameter, "cat": CategoricalParameter, "bool": BoolParameter, "int": IntParameter}

    def __init__(self):
        self.paras = {}
        self.para_names = []
        self.numeric_names = []
        self.enum_names = []

    @property
    def num_paras(self):
        return len(self.para_names)

    @property
    def num_numeric(self):
        return len(self.numeric_names)

    @property
    def num_categorical(self):
        return len(self.enum_names)

    @classmethod
    def parse(cls, rec):
        object = cls()
        object.para_config = rec
        object.paras = {}
        object.para_names = []
        for item in rec:
            assert item["type"] in DesignSpace.parameter_types
            param = DesignSpace.parameter_types[item["type"]](item)
            object.paras[param.name] = param
            if param.is_categorical:
                object.enum_names.append(param.name)
            else:
                object.numeric_names.append(param.name)
        object.para_names = object.numeric_names + object.enum_names
        assert len(object.para_names) == len(set(object.para_names))
        return object

    @staticmethod
    def register_parameter_type(type_name: str, para_class: type[Parameter]):
        DesignSpace.parameter_types[type_name] = para_class

    @staticmethod
    def register_parmeter_type(type_name: str, para_class: type[Parameter]):
        """Deprecated alias for :meth:`register_parameter_type`."""
        warnings.warn(
            "DesignSpace.register_parmeter_type is deprecated; use DesignSpace.register_parameter_type instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        DesignSpace.register_parameter_type(type_name, para_class)

    def transform(self, data: pd.DataFrame) -> tuple[Tensor, Tensor]:
        x_numerical = data[self.numeric_names].values.astype(float).copy()
        x_categorical = data[self.enum_names].values.copy()
        for i, name in enumerate(self.numeric_names):
            x_numerical[:, i] = self.paras[name].transform(x_numerical[:, i])
        for i, name in enumerate(self.enum_names):
            x_categorical[:, i] = self.paras[name].transform(x_categorical[:, i])
        return torch.FloatTensor(x_numerical), torch.LongTensor(x_categorical.astype(int))

    def inverse_transform(self, x_numerical: Tensor, x_categorical: Tensor) -> pd.DataFrame:
        with torch.no_grad():
            inv_dict = {}
            for i, name in enumerate(self.numeric_names):
                inv_dict[name] = self.paras[name].inverse_transform(x_numerical.detach().double().numpy()[:, i])
            for i, name in enumerate(self.enum_names):
                inv_dict[name] = self.paras[name].inverse_transform(x_categorical.detach().numpy()[:, i])
            return pd.DataFrame(inv_dict)

    @property
    def opt_lb(self):
        lb_numeric = [self.paras[p].opt_lb for p in self.numeric_names]
        lb_enum = [self.paras[p].opt_lb for p in self.enum_names]
        return torch.tensor(lb_numeric + lb_enum)

    @property
    def opt_ub(self):
        ub_numeric = [self.paras[p].opt_ub for p in self.numeric_names]
        ub_enum = [self.paras[p].opt_ub for p in self.enum_names]
        return torch.tensor(ub_numeric + ub_enum)

    def to_pymoo_vars(self):
        vars = {}
        for i, p_name in enumerate(self.para_names):
            p = self.paras[p_name]
            lb = self.opt_lb[i].item()
            ub = self.opt_ub[i].item()
            if p.is_numeric:
                if not p.is_discrete_after_transform:
                    vars[p_name] = Real(bounds=(lb, ub))
                else:
                    # WARNING: Only for GenAlg type methods, others may require non-linear transformations
                    vars[p_name] = Integer(bounds=(lb, ub))
            elif p.is_categorical:
                vars[p_name] = Choice(options=list(range(int(lb), int(ub + 1))))
            else:
                raise ValueError("Unknown parameter type: {p_name}")
        return vars

    def sample(self, num_samples: int = 1):
        df = pd.DataFrame(columns=self.para_names)
        for c in df.columns:
            df[c] = self.paras[c].sample(num_samples)
        return df
