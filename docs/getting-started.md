# Getting started

## Install

Requires **Python 3.10+**.

```bash
pip install banditry
```

The NUTS sampler (used by `TSConfig(sampler="nuts")`) needs pyro, an optional
extra:

```bash
pip install "banditry[nuts]"
```

For development:

```bash
git clone https://github.com/VahanArsenian/banditry.git
cd banditry
pip install -e ".[dev,nuts]"
```

## The loop

Every agent implements the same contract: ask for suggestions, evaluate them
yourself, report the results back. Agents **minimise** the observed values.

```python
import numpy as np
from banditry import DesignSpace, OFUGPConfig, build_agent

# 1. Describe the search space.
space = DesignSpace.parse([
    {"name": "x0", "type": "num", "lb": -1, "ub": 1},
    {"name": "x1", "type": "num", "lb": -1, "ub": 1},
])

# 2. Build an agent from a config.
config = OFUGPConfig(rand_sample=4, surrogate="gp", noise_std_proxy=1.0)
agent = build_agent(config, space)

# 3. Run the suggest -> evaluate -> observe loop.
def objective(df):
    return (df["x0"].to_numpy(float) - 0.3) ** 2 + (df["x1"].to_numpy(float) + 0.2) ** 2

for _ in range(20):
    rec = agent.suggest(1)                                # pandas DataFrame, one row per suggestion
    y = np.asarray(objective(rec), dtype=float).reshape(-1)
    agent.observe(rec, y)

best_row = agent.get_best_id()                            # index of the best observation so far
print(agent.X.iloc[best_row], agent.y[best_row])
```

The first `rand_sample` suggestions are quasi-random (Sobol) warmup; after
that the agent fits its surrogate and optimises an acquisition function.

## Thompson-sampling agents

```python
from banditry import TSConfig, build_agent

agent = build_agent(TSConfig(rand_sample=4, sampler="langevin"), space)   # no extra deps
agent = build_agent(TSConfig(rand_sample=4, sampler="nuts"), space)       # needs banditry[nuts]

# Feel-Good Thompson sampling
agent = build_agent(TSConfig(sampler="langevin", feel_good=True, fg_lambda=1.0, fg_bound=1.0), space)
```

See [Configuring samplers](guides/configuring-samplers.md) for tuning the
MCMC behaviour, and [Choosing an agent](guides/choosing-an-agent.md) for
which agent fits which problem.

## Next steps

- [Design spaces](guides/design-spaces.md) — mixed numeric / integer /
  boolean / categorical parameters.
- [Contextual bandits](guides/contextual-bandits.md) — pinning observed
  context each round.
- Runnable scripts live in
  [`examples/`](https://github.com/VahanArsenian/banditry/tree/main/examples).
