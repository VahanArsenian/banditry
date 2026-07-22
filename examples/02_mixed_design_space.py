"""Mixed design space: numeric + integer + boolean + categorical parameters.

A toy "hyper-parameter tuning" objective is minimised over all four parameter
types at once. Suggestions come back as a DataFrame in the original domain
(e.g. `optimiser` is the string "adam", not an index).

Run:  python examples/02_mixed_design_space.py
"""

import numpy as np

from banditry import DesignSpace, OFUGPConfig, build_agent
from banditry.benchmark import seed_everything

N_ITER = 15

space = DesignSpace.parse(
    [
        {"name": "learning_rate", "type": "num", "lb": 1e-4, "ub": 1e-1},
        {"name": "num_layers", "type": "int", "lb": 1, "ub": 8},
        {"name": "use_dropout", "type": "bool"},
        {"name": "optimiser", "type": "cat", "categories": ["adam", "sgd", "rmsprop"]},
    ]
)


def objective(df):
    """Pretend validation loss: best at lr=1e-2, 4 layers, dropout on, adam."""
    lr = df["learning_rate"].to_numpy(dtype=float)
    layers = df["num_layers"].to_numpy(dtype=float)
    dropout = df["use_dropout"].to_numpy(dtype=bool)
    opt_penalty = df["optimiser"].map({"adam": 0.0, "sgd": 0.15, "rmsprop": 0.05}).to_numpy(dtype=float)
    return (np.log10(lr) + 2) ** 2 + 0.05 * (layers - 4) ** 2 + 0.1 * (~dropout) + opt_penalty


seed_everything(0)
agent = build_agent(OFUGPConfig(rand_sample=5, surrogate="gp", noise_std_proxy=1.0), space)

best_y = np.inf
for i in range(N_ITER):
    rec = agent.suggest(1)
    y = np.asarray(objective(rec), dtype=float).reshape(-1)
    agent.observe(rec, y)
    best_y = min(best_y, float(y[0]))
    row = rec.iloc[0]
    print(
        f"iter {i:>2d}  lr={row['learning_rate']:.5f} layers={row['num_layers']} "
        f"dropout={row['use_dropout']!s:<5} opt={row['optimiser']:<7}  y={y[0]:.4f}  best={best_y:.4f}"
    )

best = agent.get_best_id()
print(f"\nbest configuration: {agent.X.iloc[best].to_dict()}  y={float(agent.y[best]):.4f}")
