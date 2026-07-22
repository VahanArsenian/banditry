"""Contextual bandit: one coordinate is observed each round, not chosen.

Each round the environment reveals x1; the agent picks x0 conditionally on it
by pinning x1 via the second argument of `suggest`. The surrogate is shared
across contexts, so observations under one context inform the others.

Run:  python examples/03_contextual_bandit.py
"""

import numpy as np

from banditry import DesignSpace, OFUGPConfig, build_agent
from banditry.benchmark import VALLEY_SPACE, gaussian_valley, seed_everything

N_ITER = 25

seed_everything(7)
space = DesignSpace.parse(VALLEY_SPACE)
agent = build_agent(OFUGPConfig(rand_sample=4, surrogate="gp", noise_std_proxy=1.0), space)

best_y = np.inf
for i in range(N_ITER):
    context = {"x1": float(space.paras["x1"].sample(1)[0])}  # observed, not chosen
    rec = agent.suggest(1, context)
    assert float(rec["x1"].iloc[0]) == context["x1"]  # pinned
    y = np.asarray(gaussian_valley(rec), dtype=float).reshape(-1)
    agent.observe(rec, y)
    best_y = min(best_y, float(y[0]))
    print(f"iter {i:>2d}  ctx x1={context['x1']:>7.4f}  chose x0={float(rec['x0'].iloc[0]):>7.4f}  y={y[0]:>8.4f}")

ctx = {"x1": 0.5}
print(f"\nbest observed row for context {ctx}: index {agent.get_best_id(fix_input=ctx)}")
