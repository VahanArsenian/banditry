"""End-to-end suggest/observe smoke tests with tiny budgets.

These assert plumbing invariants only (shapes, bounds, bookkeeping) — never
solution quality, which is meaningless after a handful of iterations.
"""

import numpy as np
import pytest

from banditry import DesignSpace, OFUGPConfig, TSConfig, build_agent
from banditry.benchmark import VALLEY_SPACE, gaussian_valley

N_ITER = 5
RAND_SAMPLE = 3

TINY_LANGEVIN = {
    "batch_size": 16,
    "num_epochs": 4,
    "burn_in": 2,
    "temperature": 0.2,
    "max_obs_noise": 10.0,
    "precondition": True,
}
TINY_NUTS = {
    "num_samples": 4,
    "warmup_steps": 8,
    "max_tree_depth": 3,
    "disable_progbar": True,
}

CASES = [
    pytest.param(
        OFUGPConfig(
            rand_sample=RAND_SAMPLE, surrogate="gp", noise_std_proxy=1.0, model_config_overrides={"num_epochs": 20}
        ),
        id="ofugp-gp",
    ),
    pytest.param(
        OFUGPConfig(
            rand_sample=RAND_SAMPLE,
            surrogate="svgp",
            noise_std_proxy=1.0,
            model_config_overrides={"num_epochs": 10, "num_inducing": 8, "batch_size": 8},
        ),
        id="ofugp-svgp",
    ),
    pytest.param(
        TSConfig(rand_sample=RAND_SAMPLE, sampler="langevin", sampler_config=TINY_LANGEVIN),
        id="ts-langevin",
    ),
    pytest.param(
        TSConfig(rand_sample=RAND_SAMPLE, sampler="nuts", sampler_config=TINY_NUTS),
        id="ts-nuts",
        marks=pytest.mark.slow,
    ),
]


def run_loop(agent, space, n_iter, fix_input=None):
    for _ in range(n_iter):
        rec = agent.suggest(1, fix_input)
        assert list(rec.columns) == space.para_names
        assert rec.shape[0] == 1
        for name in space.para_names:
            p = space.paras[name]
            v = float(rec[name].iloc[0])
            assert np.isfinite(v)
            assert p.opt_lb - 1e-9 <= v <= p.opt_ub + 1e-9
        y = np.asarray(gaussian_valley(rec), dtype=float).reshape(-1)
        agent.observe(rec, y)


@pytest.mark.parametrize("config", CASES)
def test_agent_suggest_observe_loop(config):
    if isinstance(config, TSConfig) and config.sampler == "nuts":
        pytest.importorskip("pyro")
    space = DesignSpace.parse(VALLEY_SPACE)
    agent = build_agent(config, space)
    run_loop(agent, space, N_ITER)
    assert agent.n_plays() == N_ITER
    assert np.isfinite(agent.y).all()


def test_contextual_suggestions_pin_fix_input():
    space = DesignSpace.parse(VALLEY_SPACE)
    config = OFUGPConfig(
        rand_sample=RAND_SAMPLE, surrogate="gp", noise_std_proxy=1.0, model_config_overrides={"num_epochs": 20}
    )
    agent = build_agent(config, space)
    for _ in range(N_ITER):  # covers both warmup and model-based rounds
        rec = agent.suggest(1, {"x1": 0.3})
        assert float(rec["x1"].iloc[0]) == pytest.approx(0.3)
        y = np.asarray(gaussian_valley(rec), dtype=float).reshape(-1)
        agent.observe(rec, y)
    assert agent.get_best_id(fix_input={"x1": 0.3}) >= 0
