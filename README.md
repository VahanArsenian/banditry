# banditry

Contextual bandit agents for black-box optimisation over mixed design spaces.

`banditry` provides GP-UCB (optimism-in-the-face-of-uncertainty) and Thompson-sampling
agents built on Gaussian-process surrogates (exact GP or sparse variational GP),
with acquisition optimisation via evolutionary algorithms (pymoo) and posterior
sampling via Langevin dynamics (SGLD) or NUTS (pyro).

## Installation

```bash
pip install banditry
```

The NUTS sampler (used by the `ts-nuts` agent) needs pyro, which is an optional extra:

```bash
pip install "banditry[nuts]"
```

Requires Python 3.10+.

## Quickstart

```python
import numpy as np

from banditry import DesignSpace, OFUGPConfig, build_agent

# Objective: minimise a 2-D function over a box.
def objective(df):
    x0 = df["x0"].to_numpy(dtype=float)
    x1 = df["x1"].to_numpy(dtype=float)
    return (x0 - 0.3) ** 2 + (x1 + 0.2) ** 2

space = DesignSpace.parse([
    {"name": "x0", "type": "num", "lb": -1, "ub": 1},
    {"name": "x1", "type": "num", "lb": -1, "ub": 1},
])

agent = build_agent(OFUGPConfig(rand_sample=4, surrogate="gp"), space)

for _ in range(20):
    rec = agent.suggest(1)                                # DataFrame of suggestions
    y = np.asarray(objective(rec), dtype=float).reshape(-1)
    agent.observe(rec, y)                                 # agents minimise y

best = agent.get_best_id()
```

Contexts (variables fixed per round, e.g. observed environment state) are passed
as `agent.suggest(1, {"x0": 0.5})`.

## Agents

| Agent | Config | Surrogate / sampler |
|---|---|---|
| GP-UCB (MACE acquisition) | `OFUGPConfig(surrogate="gp")` | Exact GP (gpytorch) |
| GP-UCB, sparse variational | `OFUGPConfig(surrogate="svgp")` | SVGP (gpytorch) |
| Thompson sampling, Langevin | `TSConfig(sampler="langevin")` | Neural value function + SGLD |
| Thompson sampling, NUTS | `TSConfig(sampler="nuts")` | Neural value function + NUTS (needs `[nuts]`) |

Design spaces support numeric, integer, boolean, and categorical parameters
(`type`: `"num"`, `"int"`, `"bool"`, `"cat"`).

A runnable benchmark script lives in the repository:

```bash
python main.py --agent ofugp-gp --benchmark branin --n-iter 30 --seed 42
```

## License

[CC BY-NC-SA 4.0](LICENSE) — free for non-commercial use with attribution;
derivatives must be shared under the same terms.
