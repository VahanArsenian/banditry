import numpy as np

from banditry.variable_domains import (
    BoolParameter,
    CategoricalParameter,
    IntParameter,
    NumericParameter,
)


def test_numeric_parameter_identity_transform():
    p = NumericParameter({"name": "x", "lb": -2.0, "ub": 3.0})
    x = p.sample(100)
    assert ((x >= -2.0) & (x <= 3.0)).all()
    assert np.allclose(p.inverse_transform(p.transform(x)), x)
    assert p.is_numeric and not p.is_discrete
    assert (p.opt_lb, p.opt_ub) == (-2.0, 3.0)


def test_int_parameter_rounds_back():
    p = IntParameter({"name": "n", "lb": 1, "ub": 8})
    x = p.sample(100)
    assert ((x >= 1) & (x <= 8)).all()
    back = p.inverse_transform(p.transform(x) + 0.4)
    assert (back == x).all()
    assert p.is_numeric and p.is_discrete and p.is_discrete_after_transform


def test_bool_parameter_thresholds():
    p = BoolParameter({"name": "b"})
    x = p.sample(100)
    assert np.isin(x, [True, False]).all()
    assert (p.inverse_transform(p.transform(x)) == x).all()
    assert (p.inverse_transform(np.array([0.4, 0.6])) == np.array([False, True])).all()
    assert p.is_numeric and p.is_discrete


def test_categorical_parameter_index_mapping():
    p = CategoricalParameter({"name": "c", "categories": ["a", "b", "c"]})
    x = p.sample(100)
    idx = p.transform(x)
    assert idx.dtype == float
    assert ((idx >= 0) & (idx <= 2)).all()
    assert (p.inverse_transform(idx) == x).all()
    assert p.is_categorical and not p.is_numeric
    assert p.num_uniqs == 3
    assert (p.opt_lb, p.opt_ub) == (0, 2)
