"""Quickstart: GP-UCB (OFU) agent minimising the Branin function.

Run:  python examples/01_quickstart_branin.py
"""

import numpy as np

from banditry import DesignSpace, OFUGPConfig, build_agent
from banditry.benchmark import BRANIN_SPACE, branin, seed_everything

N_ITER = 25

seed_everything(42)
space = DesignSpace.parse(BRANIN_SPACE)
agent = build_agent(OFUGPConfig(rand_sample=4, surrogate="gp", noise_std_proxy=1.0), space)

best_y = np.inf
for i in range(N_ITER):
    rec = agent.suggest(1)
    y = np.asarray(branin(rec), dtype=float).reshape(-1)
    agent.observe(rec, y)
    best_y = min(best_y, float(y[0]))
    print(f"iter {i:>2d}  y={y[0]:>9.4f}  best={best_y:>9.4f}")

best = agent.get_best_id()
print(f"\nbest observation: {agent.X.iloc[best].to_dict()}  y={float(agent.y[best]):.4f}")
