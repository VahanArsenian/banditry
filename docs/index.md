# banditry

Contextual bandit agents for black-box optimisation over mixed design spaces.

`banditry` runs a **suggest → evaluate → observe** loop against an expensive
black-box objective (it *minimises* the observed values) and provides the
pieces of that loop as composable modules:

- **Agents** — the decision-making policies: a **GP-UCB / OFU agent**
  (`OFUGPAgent`) with Bayesian or frequentist (Chowdhury–Gopalan) confidence
  widths, and a **Thompson-sampling agent** (`TSAgent`) whose neural value
  function is sampled from the posterior by an MCMC oracle, with optional
  Feel-Good reweighting.
- **Surrogates** — exact Gaussian processes and sparse variational GPs
  (gpytorch), plus the neural `ValueFunction` used by TS.
- **Optimisation oracles** — evolutionary acquisition optimisers wrapping
  pymoo (NSGA-II / NSGA-III / U-NSGA-III) and an SGLD optimiser.
- **Sampling oracles** — posterior samplers for the TS agent: Langevin
  dynamics and NUTS (via pyro).
- **Variable domains** — mixed design spaces with numeric, integer, boolean,
  and categorical parameters.

Contexts are supported throughout: any subset of parameters can be pinned to
observed environment values each round, turning the loop into a **contextual
bandit**.

## Installation

```bash
pip install banditry            # core
pip install "banditry[nuts]"    # + NUTS sampler (pyro)
```

## At a glance

```python
import numpy as np
from banditry import DesignSpace, OFUGPConfig, build_agent

space = DesignSpace.parse([
    {"name": "x0", "type": "num", "lb": -1, "ub": 1},
    {"name": "x1", "type": "num", "lb": -1, "ub": 1},
])
agent = build_agent(OFUGPConfig(rand_sample=4, surrogate="gp", noise_std_proxy=1.0), space)

for _ in range(20):
    rec = agent.suggest(1)
    y = np.asarray((rec["x0"] - 0.3) ** 2 + (rec["x1"] + 0.2) ** 2, dtype=float).reshape(-1)
    agent.observe(rec, y)
```

Continue with [Getting started](getting-started.md), or jump to the
[API reference](api/agents.md).

## Origins

`banditry` isolates the bandit routines from
[MetaFrieren](https://github.com/VahanArsenian/MetaFrieren) into a standalone
library. Its design-space interface and the MACE acquisition ensemble are
partially inspired by [HEBO](https://github.com/huawei-noah/HEBO)
(Huawei Noah's Ark Lab). MIT licensed — see
[NOTICE](https://github.com/VahanArsenian/banditry/blob/main/NOTICE).
