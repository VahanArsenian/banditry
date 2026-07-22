import numpy as np
import pytest
import torch
from pymoo.core.variable import Choice, Integer, Real

from banditry import DesignSpace
from banditry.variable_domains import NumericParameter


def test_parse_orders_numeric_before_enum(mixed_space):
    assert mixed_space.numeric_names == ["lr", "layers", "dropout"]
    assert mixed_space.enum_names == ["opt"]
    assert mixed_space.para_names == ["lr", "layers", "dropout", "opt"]
    assert mixed_space.num_paras == 4
    assert mixed_space.num_numeric == 3
    assert mixed_space.num_categorical == 1


def test_parse_rejects_unknown_type():
    with pytest.raises(AssertionError):
        DesignSpace.parse([{"name": "x", "type": "nope", "lb": 0, "ub": 1}])


def test_sample_shapes_and_bounds(mixed_space):
    df = mixed_space.sample(16)
    assert df.shape == (16, 4)
    assert list(df.columns) == mixed_space.para_names
    assert df["lr"].between(1e-4, 1e-1).all()
    assert df["layers"].between(1, 8).all()
    assert df["dropout"].isin([True, False]).all()
    assert df["opt"].isin(["adam", "sgd", "rmsprop"]).all()


def test_transform_round_trip(mixed_space):
    df = mixed_space.sample(32)
    x_num, x_cat = mixed_space.transform(df)
    assert isinstance(x_num, torch.FloatTensor)
    assert isinstance(x_cat, torch.LongTensor)
    assert x_num.shape == (32, 3)
    assert x_cat.shape == (32, 1)

    back = mixed_space.inverse_transform(x_num, x_cat)
    assert np.allclose(back["lr"].to_numpy(dtype=float), df["lr"].to_numpy(dtype=float))
    assert (back["layers"].to_numpy() == df["layers"].to_numpy()).all()
    assert (back["dropout"].to_numpy() == df["dropout"].to_numpy()).all()
    assert (back["opt"].to_numpy() == df["opt"].to_numpy()).all()


def test_opt_bounds(mixed_space):
    assert torch.allclose(mixed_space.opt_lb.double(), torch.tensor([1e-4, 1.0, 0.0, 0.0]).double())
    assert torch.allclose(mixed_space.opt_ub.double(), torch.tensor([1e-1, 8.0, 1.0, 2.0]).double())


def test_to_pymoo_vars(mixed_space):
    vars = mixed_space.to_pymoo_vars()
    assert isinstance(vars["lr"], Real)
    assert isinstance(vars["layers"], Integer)
    assert isinstance(vars["dropout"], Integer)
    assert isinstance(vars["opt"], Choice)
    assert vars["opt"].options == [0, 1, 2]


def test_register_parameter_type():
    class MyParam(NumericParameter):
        pass

    try:
        DesignSpace.register_parameter_type("mynum", MyParam)
        space = DesignSpace.parse([{"name": "x", "type": "mynum", "lb": 0, "ub": 1}])
        assert isinstance(space.paras["x"], MyParam)
    finally:
        DesignSpace.parameter_types.pop("mynum", None)


def test_register_parmeter_type_deprecated_alias():
    class MyParam(NumericParameter):
        pass

    try:
        with pytest.warns(DeprecationWarning, match="register_parameter_type"):
            DesignSpace.register_parmeter_type("mynum2", MyParam)
        assert DesignSpace.parameter_types["mynum2"] is MyParam
    finally:
        DesignSpace.parameter_types.pop("mynum2", None)


def test_deprecated_contextal_problem_module_warns():
    import importlib
    import sys

    sys.modules.pop("banditry.optimisation_subroutines.contextal_problem", None)
    with pytest.warns(DeprecationWarning, match="contextual_problem"):
        shim = importlib.import_module("banditry.optimisation_subroutines.contextal_problem")
    from banditry.optimisation_subroutines.contextual_problem import ContextualProblem

    assert shim.ContextualProblem is ContextualProblem
