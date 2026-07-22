from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Parameter(ABC):
    """Abstract base class for a single design-space parameter.

    A parameter converts values between two representations:

    - the *raw* domain, as produced by :meth:`sample` and seen by the user
      (floats, ints, bools, category labels), and
    - the *transformed / optimiser* domain, a numeric encoding bounded by
      :attr:`opt_lb` and :attr:`opt_ub` that surrogate models and acquisition
      optimisers operate on.

    Subclasses implement :meth:`sample`, :meth:`transform` (raw to transformed) and
    :meth:`inverse_transform` (transformed back to raw), plus the ``is_numeric``,
    ``is_discrete`` and ``is_discrete_after_transform`` flags that determine how the
    optimiser treats the parameter. ``is_categorical`` is simply ``not is_numeric``.

    Args:
        param_dict: Spec dict for this parameter; must contain at least ``"name"``.
    """

    def __init__(self, param_dict):
        self.param_dict = param_dict
        self.name = param_dict["name"]
        pass

    @abstractmethod
    def sample(self, num=1) -> pd.DataFrame:
        pass

    @abstractmethod
    def transform(self, x: np.array) -> np.array:
        pass

    @abstractmethod
    def inverse_transform(self, x: np.array) -> np.array:
        pass

    @property
    @abstractmethod
    def is_numeric(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_discrete(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_discrete_after_transform(self) -> bool:
        pass

    @property
    def is_categorical(self) -> bool:
        return not self.is_numeric

    @property
    @abstractmethod
    def opt_lb(self) -> float:
        pass

    @property
    @abstractmethod
    def opt_ub(self) -> float:
        pass


class NumericParameter(Parameter):
    """Continuous parameter on ``[lb, ub]`` (spec type ``"num"``).

    Reads the spec keys ``"name"``, ``"lb"`` and ``"ub"``. The transform is the
    identity, so the optimiser domain equals the raw domain with bounds
    ``opt_lb = lb`` and ``opt_ub = ub``. Sampling is uniform on ``[lb, ub)``.
    """

    def __init__(self, param_dict):
        super().__init__(param_dict)
        self.lb = param_dict["lb"]
        self.ub = param_dict["ub"]

    def sample(self, num=1):
        assert num > 0
        return np.random.uniform(self.lb, self.ub, num)

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x

    @property
    def is_numeric(self):
        return True

    @property
    def opt_lb(self):
        return self.lb

    @property
    def opt_ub(self):
        return self.ub

    @property
    def is_discrete(self):
        return False

    @property
    def is_discrete_after_transform(self):
        return False


class CategoricalParameter(Parameter):
    """Categorical parameter over a fixed set of choices (spec type ``"cat"``).

    Reads the spec keys ``"name"`` and ``"categories"``. ``transform`` maps each
    category label to its (float) index in ``categories``; ``inverse_transform``
    rounds to the nearest index and returns the corresponding label. Optimiser
    bounds are ``[0, len(categories) - 1]`` and sampling is uniform over the
    categories.
    """

    def __init__(self, param):
        super().__init__(param)
        self.categories = list(param["categories"])
        try:
            self._categories_dict = {k: v for v, k in enumerate(self.categories)}
        except TypeError:
            self._categories_dict = None
        self.lb = 0
        self.ub = len(self.categories) - 1

    def sample(self, num=1):
        assert num > 0
        return np.random.choice(self.categories, num, replace=True)

    def transform(self, x: np.ndarray):
        if self._categories_dict:
            ret = np.array(list(map(lambda a: self._categories_dict[a], x)))
        else:
            ret = np.array(list(map(lambda a: np.where(self.categories == a)[0][0], x)))
        return ret.astype(float)

    def inverse_transform(self, x):
        return np.array([self.categories[x_] for x_ in x.round().astype(int)])

    @property
    def is_numeric(self):
        return False

    @property
    def is_discrete(self):
        return True

    @property
    def is_discrete_after_transform(self):
        return True

    @property
    def opt_lb(self):
        return self.lb

    @property
    def opt_ub(self):
        return self.ub

    @property
    def num_uniqs(self):
        return len(self.categories)


class BoolParameter(Parameter):
    """Boolean parameter (spec type ``"bool"``).

    Reads only the spec key ``"name"``. ``transform`` casts to float (0.0/1.0);
    ``inverse_transform`` thresholds at 0.5 (``x > 0.5``), so any optimiser value in
    ``[0, 1]`` maps back to a bool. Sampling is a fair coin flip.
    """

    def __init__(self, param):
        super().__init__(param)
        self.lb = 0
        self.ub = 1

    def sample(self, num=1):
        assert num > 0
        return np.random.choice([True, False], num, replace=True)

    def transform(self, x):
        return x.astype(float)

    def inverse_transform(self, x):
        return x > 0.5

    @property
    def is_numeric(self):
        return True

    @property
    def is_discrete(self):
        return True

    @property
    def is_discrete_after_transform(self):
        return True

    @property
    def opt_lb(self):
        return self.lb

    @property
    def opt_ub(self):
        return self.ub


class IntParameter(Parameter):
    """Integer parameter on the inclusive range ``[lb, ub]`` (spec type ``"int"``).

    Reads the spec keys ``"name"``, ``"lb"`` and ``"ub"`` (rounded to the nearest
    integer). ``transform`` casts to float; ``inverse_transform`` rounds back to the
    nearest integer. Sampling is uniform over the integers in ``[lb, ub]``.
    """

    def __init__(self, param_dict):
        super().__init__(param_dict)
        self.lb = round(param_dict["lb"])
        self.ub = round(param_dict["ub"])

    def sample(self, num=1):
        assert num > 0
        return np.random.randint(self.lb, self.ub + 1, num)

    def transform(self, x):
        return x.astype(float)

    def inverse_transform(self, x):
        return x.round().astype(int)

    @property
    def is_numeric(self):
        return True

    @property
    def opt_lb(self):
        return float(self.lb)

    @property
    def opt_ub(self):
        return float(self.ub)

    @property
    def is_discrete(self):
        return True

    @property
    def is_discrete_after_transform(self):
        return True
