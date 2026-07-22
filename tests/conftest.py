import random

import numpy as np
import pytest
import torch

from banditry import DesignSpace


@pytest.fixture(autouse=True)
def seed_everything():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture
def numeric_space():
    return DesignSpace.parse(
        [
            {"name": "x0", "type": "num", "lb": -1, "ub": 1},
            {"name": "x1", "type": "num", "lb": -1, "ub": 1},
        ]
    )


@pytest.fixture
def mixed_space():
    return DesignSpace.parse(
        [
            {"name": "lr", "type": "num", "lb": 1e-4, "ub": 1e-1},
            {"name": "layers", "type": "int", "lb": 1, "ub": 8},
            {"name": "dropout", "type": "bool"},
            {"name": "opt", "type": "cat", "categories": ["adam", "sgd", "rmsprop"]},
        ]
    )
