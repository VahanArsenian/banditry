import numpy as np
import pandas as pd

from banditry import AbstractAgent


class DummyAgent(AbstractAgent):
    def get_model(self, Xc, Xe, y):
        return None

    def pick_action(self, model, fix_input, n_suggestions=1):
        return self.quasi_sample(n_suggestions, fix_input)


def observe_grid(agent, values):
    n = len(values)
    X = pd.DataFrame({"x0": np.linspace(-0.5, 0.5, n), "x1": np.zeros(n)})
    agent.observe(X, np.asarray(values, dtype=float))
    return X


def test_rand_sample_defaults(numeric_space):
    assert DummyAgent(numeric_space).rand_sample == 1 + numeric_space.num_paras
    assert DummyAgent(numeric_space, rand_sample=1).rand_sample == 2
    assert DummyAgent(numeric_space, rand_sample=7).rand_sample == 7


def test_observe_drops_non_finite(numeric_space):
    agent = DummyAgent(numeric_space)
    observe_grid(agent, [1.0, np.nan, np.inf, 2.0])
    assert agent.n_plays() == 2
    assert np.isfinite(agent.y).all()


def test_quasi_sample_bounds_and_fix_input(mixed_space):
    agent = DummyAgent(mixed_space)
    df = agent.quasi_sample(16)
    assert df.shape == (16, mixed_space.num_paras)
    assert list(df.columns) == mixed_space.para_names
    assert df["lr"].between(1e-4, 1e-1).all()
    assert df["layers"].between(1, 8).all()
    assert df["opt"].isin(["adam", "sgd", "rmsprop"]).all()

    pinned = agent.quasi_sample(8, fix_input={"lr": 0.05, "opt": "sgd"})
    assert (pinned["lr"] == 0.05).all()
    assert (pinned["opt"] == "sgd").all()


def test_get_best_id_global(numeric_space):
    agent = DummyAgent(numeric_space)
    observe_grid(agent, [3.0, 1.0, 2.0])
    assert agent.get_best_id() == 1


def test_get_best_id_with_fix_input(numeric_space):
    agent = DummyAgent(numeric_space)
    X = pd.DataFrame({"x0": [0.1, 0.2, 0.2], "x1": [0.0, 0.0, 0.0]})
    agent.observe(X, np.array([0.5, 3.0, 2.0]))
    # best among rows where x0 == 0.2, even though row 0 is globally better
    assert agent.get_best_id(fix_input={"x0": 0.2}) == 2
    # no row matches -> falls back to the global best
    assert agent.get_best_id(fix_input={"x0": 0.9}) == 0


def test_check_unique(numeric_space):
    agent = DummyAgent(numeric_space)
    X = observe_grid(agent, [1.0, 2.0])
    seen = X.iloc[[0]]
    fresh = pd.DataFrame({"x0": [0.42], "x1": [0.42]})
    assert agent.check_unique(pd.concat([seen, fresh], ignore_index=True)) == [False, True]


def test_suggest_uses_quasi_sample_during_warmup(numeric_space):
    agent = DummyAgent(numeric_space, rand_sample=5)
    rec = agent.suggest(3)
    assert rec.shape == (3, numeric_space.num_paras)
    assert rec["x0"].between(-1, 1).all()
    assert rec["x1"].between(-1, 1).all()
