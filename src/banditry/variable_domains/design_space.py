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
    """Search-space definition for a contextual-bandit / optimisation problem.

    A design space is built from a list of spec dicts via :meth:`parse`, one dict per
    parameter. The built-in parameter types (see :attr:`parameter_types`) accept:

    - ``{"name": ..., "type": "num", "lb": float, "ub": float}`` -- continuous parameter.
    - ``{"name": ..., "type": "int", "lb": int, "ub": int}`` -- integer parameter (inclusive bounds).
    - ``{"name": ..., "type": "bool"}`` -- boolean parameter.
    - ``{"name": ..., "type": "cat", "categories": [...]}`` -- categorical parameter.

    Custom types can be added with :meth:`register_parameter_type`.

    Note on ordering: :attr:`para_names` lists the numeric-ish parameters
    (``num``/``int``/``bool``) first, followed by the categorical ones, regardless of
    their order in the spec list.

    Example:
        >>> space = DesignSpace.parse(
        ...     [
        ...         {"name": "lr", "type": "num", "lb": 1e-4, "ub": 1e-1},
        ...         {"name": "num_layers", "type": "int", "lb": 1, "ub": 8},
        ...         {"name": "use_bias", "type": "bool"},
        ...         {"name": "activation", "type": "cat", "categories": ["relu", "tanh", "gelu"]},
        ...     ]
        ... )
        >>> df = space.sample(5)
        >>> x_num, x_cat = space.transform(df)  # FloatTensor (5, 3), LongTensor (5, 1)
        >>> space.inverse_transform(x_num, x_cat).columns.tolist()
        ['lr', 'num_layers', 'use_bias', 'activation']
    """

    parameter_types = {"num": NumericParameter, "cat": CategoricalParameter, "bool": BoolParameter, "int": IntParameter}

    def __init__(self):
        self.paras = {}
        self.para_names = []
        self.numeric_names = []
        self.enum_names = []

    @property
    def num_paras(self):
        """Total number of parameters in the design space."""
        return len(self.para_names)

    @property
    def num_numeric(self):
        """Number of numeric-ish parameters (``num``/``int``/``bool``)."""
        return len(self.numeric_names)

    @property
    def num_categorical(self):
        """Number of categorical parameters."""
        return len(self.enum_names)

    @classmethod
    def parse(cls, rec: list[dict]) -> "DesignSpace":
        """Build a :class:`DesignSpace` from a list of parameter spec dicts.

        Args:
            rec: List of spec dicts, one per parameter. Each dict must have a unique
                ``"name"`` and a ``"type"`` registered in :attr:`parameter_types`
                (built-ins: ``"num"``, ``"int"``, ``"bool"``, ``"cat"``); the remaining
                keys are type-specific (e.g. ``"lb"``/``"ub"`` or ``"categories"``).
                See the class docstring for the full format.

        Returns:
            A new :class:`DesignSpace` whose ``para_names`` are ordered numeric-first,
            then categorical.
        """
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
        """Register a custom parameter type.

        After registration, ``type_name`` can be used as the ``"type"`` value in the
        spec dicts passed to :meth:`parse`.

        Args:
            type_name: Identifier to use as the ``"type"`` value in spec dicts.
            para_class: A :class:`Parameter` subclass; it is constructed with the
                spec dict of each parameter declared with this type.
        """
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
        """Map a DataFrame of raw parameter values into the model/optimiser domain.

        Args:
            data: DataFrame with one column per parameter holding raw values, as
                produced by :meth:`sample` or supplied by the user.

        Returns:
            Tuple ``(x_numerical, x_categorical)``: ``x_numerical`` is a ``FloatTensor``
            of shape ``(n, num_numeric)`` with the transformed ``num``/``int``/``bool``
            columns, and ``x_categorical`` is a ``LongTensor`` of shape
            ``(n, num_categorical)`` holding category indices.
        """
        x_numerical = data[self.numeric_names].values.astype(float).copy()
        x_categorical = data[self.enum_names].values.copy()
        for i, name in enumerate(self.numeric_names):
            x_numerical[:, i] = self.paras[name].transform(x_numerical[:, i])
        for i, name in enumerate(self.enum_names):
            x_categorical[:, i] = self.paras[name].transform(x_categorical[:, i])
        return torch.FloatTensor(x_numerical), torch.LongTensor(x_categorical.astype(int))

    def inverse_transform(self, x_numerical: Tensor, x_categorical: Tensor) -> pd.DataFrame:
        """Map transformed tensors back to a DataFrame of raw parameter values.

        Args:
            x_numerical: Tensor of shape ``(n, num_numeric)`` in the transformed domain.
            x_categorical: Tensor of shape ``(n, num_categorical)`` holding category indices.

        Returns:
            DataFrame with one column per parameter holding raw values (floats, ints,
            bools and the original category labels).
        """
        with torch.no_grad():
            inv_dict = {}
            for i, name in enumerate(self.numeric_names):
                inv_dict[name] = self.paras[name].inverse_transform(x_numerical.detach().double().numpy()[:, i])
            for i, name in enumerate(self.enum_names):
                inv_dict[name] = self.paras[name].inverse_transform(x_categorical.detach().numpy()[:, i])
            return pd.DataFrame(inv_dict)

    @property
    def opt_lb(self):
        """Per-parameter lower bounds in the transformed/optimiser domain (numeric first, then categorical)."""
        lb_numeric = [self.paras[p].opt_lb for p in self.numeric_names]
        lb_enum = [self.paras[p].opt_lb for p in self.enum_names]
        return torch.tensor(lb_numeric + lb_enum)

    @property
    def opt_ub(self):
        """Per-parameter upper bounds in the transformed/optimiser domain (numeric first, then categorical)."""
        ub_numeric = [self.paras[p].opt_ub for p in self.numeric_names]
        ub_enum = [self.paras[p].opt_ub for p in self.enum_names]
        return torch.tensor(ub_numeric + ub_enum)

    def to_pymoo_vars(self):
        """Express the design space as a dict of pymoo decision variables.

        Returns:
            Dict mapping each parameter name to a pymoo variable: ``Real`` for
            continuous parameters, ``Integer`` for numeric parameters that stay
            discrete after transform (``int``/``bool``), and ``Choice`` over the
            category indices for categorical parameters.

        Raises:
            ValueError: If a parameter is neither numeric nor categorical.
        """
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

    def sample(self, num_samples: int = 1) -> pd.DataFrame:
        """Draw uniform random samples of the raw parameter values.

        Args:
            num_samples: Number of rows to draw.

        Returns:
            DataFrame of shape ``(num_samples, num_paras)`` with columns ordered as
            :attr:`para_names` and raw (untransformed) values.
        """
        df = pd.DataFrame(columns=self.para_names)
        for c in df.columns:
            df[c] = self.paras[c].sample(num_samples)
        return df
