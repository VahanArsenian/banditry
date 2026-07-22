from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Parameter(ABC):
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
