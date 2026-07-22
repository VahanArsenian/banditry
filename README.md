# banditry

[![CI](https://github.com/VahanArsenian/banditry/actions/workflows/ci.yml/badge.svg)](https://github.com/VahanArsenian/banditry/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/banditry)](https://pypi.org/project/banditry/)
[![Python](https://img.shields.io/pypi/pyversions/banditry)](https://pypi.org/project/banditry/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Contextual bandit agents for black-box optimisation over mixed design spaces.

`banditry` isolates the bandit routines from
[MetaFrieren](https://github.com/VahanArsenian/MetaFrieren/tree/main) into a
standalone, reusable library. Its design-space interface and the MACE
acquisition strategy are partially inspired by the
[HEBO](https://github.com/huawei-noah/HEBO/tree/master/HEBO) repository from
Huawei Noah's Ark Lab.

## What it does

`banditry` runs a suggest → evaluate → observe loop against an expensive
black-box objective (it **minimises** the observed values) and provides the
pieces of that loop as composable modules:

- **Agents** (`banditry.agents`) — the decision-making policies:
  - **GP-UCB / OFU agent** (`OFUGPAgent`): optimism-in-the-face-of-uncertainty
    with either Bayesian or frequentist (Chowdhury–Gopalan) confidence widths,
    optimising the MACE multi-objective acquisition ensemble (mean, sigma, LCB).
  - **Thompson-sampling agent** (`TSAgent`): a neural value function whose
    posterior is sampled by an MCMC oracle, with optional Feel-Good Thompson
    sampling reweighting.
- **Surrogates** (`banditry.surrogates`) — exact Gaussian processes and sparse
  variational GPs (gpytorch), plus the neural `ValueFunction` used by TS.
- **Optimisation oracles** (`banditry.optimisation_oracles`) — evolutionary
  acquisition optimisers wrapping pymoo (NSGA-II / NSGA-III / U-NSGA-III) and an
  SGLD optimiser (stochastic gradient Langevin dynamics).
- **Optimisation subroutines** (`banditry.optimisation_subroutines`) —
  acquisition objectives (mean, sigma, LCB, MACE, Thompson objective) and the
  pymoo problem wrapper that handles contexts.
- **Sampling oracles** (`banditry.sampling_oracles`) — posterior samplers for
  the TS agent: Langevin dynamics (SGLD with a Welling–Teh step-size schedule)
  and NUTS (via pyro).
- **Variable domains** (`banditry.variable_domains`) — mixed design spaces with
  numeric, integer, boolean, and categorical parameters, plus the transforms
  (scaling, one-hot/embedding) used to feed them to the models.

Contexts are supported throughout: any subset of parameters can be pinned to
observed environment values each round, turning the loop into a contextual
bandit.

## Installation

Requires **Python 3.10+**.

From PyPI:

```bash
pip install banditry
```

The NUTS sampler (used by `TSConfig(sampler="nuts")`) needs pyro, which is an
optional extra:

```bash
pip install "banditry[nuts]"
```

From source, for development:

```bash
git clone https://github.com/VahanArsenian/banditry.git
cd banditry
pip install -e ".[nuts]"
```

Conda users can `pip install banditry` inside any conda environment; the core
dependencies (numpy, pandas, torch, gpytorch, pymoo, rich) are resolved by pip.

## Usage

### Quickstart

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

config = OFUGPConfig(rand_sample=4, surrogate="gp", noise_std_proxy=1.0)
agent = build_agent(config, space)

for _ in range(20):
    rec = agent.suggest(1)                                # DataFrame of suggestions
    y = np.asarray(objective(rec), dtype=float).reshape(-1)
    agent.observe(rec, y)                                 # agents minimise y

best_row = agent.get_best_id()                            # index of the best observation
```

The agent starts with `rand_sample` quasi-random (Sobol) warmup suggestions,
then switches to model-based suggestions.

### Mixed design spaces

All four parameter types can be combined freely:

```python
space = DesignSpace.parse([
    {"name": "learning_rate", "type": "num",  "lb": 1e-4, "ub": 1e-1},
    {"name": "num_layers",    "type": "int",  "lb": 1,    "ub": 8},
    {"name": "use_dropout",   "type": "bool"},
    {"name": "optimiser",     "type": "cat",  "categories": ["adam", "sgd", "rmsprop"]},
])
```

Suggestions come back as a pandas DataFrame with one column per parameter, in
the original (untransformed) domain.

### Contextual bandits

Pass the observed context each round via the second argument of `suggest`; the
pinned parameters are fixed while the rest are optimised conditionally:

```python
context = {"x1": 0.7}            # observed environment state this round
rec = agent.suggest(1, context)  # x1 is pinned to 0.7 in the suggestion
y = np.asarray(objective(rec), dtype=float).reshape(-1)
agent.observe(rec, y)

best_for_context = agent.get_best_id(fix_input=context)
```

### Thompson-sampling agents

```python
from banditry import TSConfig, build_agent

# Langevin posterior sampling (no extra dependencies)
agent = build_agent(TSConfig(rand_sample=4, sampler="langevin"), space)

# NUTS posterior sampling (requires banditry[nuts])
agent = build_agent(TSConfig(rand_sample=4, sampler="nuts"), space)

# Feel-Good Thompson sampling
agent = build_agent(
    TSConfig(sampler="langevin", feel_good=True, fg_lambda=1.0, fg_bound=1.0),
    space,
)
```

Sampler behaviour is controlled through `sampler_config`; the defaults live in
`banditry.agents.factory.DEFAULT_LANGEVIN_CONFIG` and `DEFAULT_NUTS_CONFIG` and
can be copied and overridden:

```python
from banditry.agents.factory import DEFAULT_LANGEVIN_CONFIG

config = TSConfig(
    sampler="langevin",
    sampler_config={**DEFAULT_LANGEVIN_CONFIG, "num_epochs": 256, "temperature": 0.1},
)
```

### Agent configuration reference

`OFUGPConfig` (GP-UCB agents):

| Field | Default | Meaning |
|---|---|---|
| `rand_sample` | `4` | Number of Sobol warmup suggestions |
| `surrogate` | `"svgp"` | `"gp"` (exact) or `"svgp"` (sparse variational) |
| `frequentist` | `False` | Chowdhury–Gopalan frequentist confidence widths instead of Bayesian |
| `rkhs_norm` | `None` | RKHS norm bound B (frequentist widths) |
| `noise_std_proxy` | `None` | **Required.** Sub-Gaussian / GP noise scale R used by the confidence widths |
| `model_config_overrides` | `{}` | Passed through to the surrogate |

`TSConfig` (Thompson-sampling agents):

| Field | Default | Meaning |
|---|---|---|
| `rand_sample` | `4` | Number of Sobol warmup suggestions |
| `sampler` | `"nuts"` | `"langevin"` or `"nuts"` |
| `feel_good` | `False` | Enable Feel-Good Thompson sampling |
| `fg_lambda`, `fg_bound` | `1.0` | Feel-Good reweighting strength and cap |
| `model_config` | `{}` | Value-function network configuration |
| `sampler_config` | `None` | Sampler overrides (`None` → per-sampler defaults) |
| `should_warm_start` | `True` | Warm-start MCMC from the previous round |
| `latent_dimension` | `None` | Latent dimension of the sampled parameter vector |

### Benchmark runner

Installing the package also installs a `banditry-bench` CLI that exercises
every agent on synthetic benchmarks:

```bash
banditry-bench --agent ofugp-gp    --benchmark branin          --n-iter 30 --seed 42
banditry-bench --agent ts-langevin --benchmark gaussian_valley --n-iter 30 --seed 42
banditry-bench --agent ofugp-svgp  --benchmark contextual_branin --n-iter 30 --seed 42
```

Agents: `ofugp-gp`, `ofugp-svgp`, `ts-langevin`, `ts-nuts`. Benchmarks:
`branin`, `contextual_branin`, `gaussian_valley`, `contextual_gaussian_valley`.

## Origins & acknowledgements

- Partially inspired by [HEBO](https://github.com/huawei-noah/HEBO/tree/master/HEBO)
  (Huawei Noah's Ark Lab) — in particular the design-space interface and the
  MACE acquisition ensemble.
- Isolates the bandit routines from
  [MetaFrieren](https://github.com/VahanArsenian/MetaFrieren/tree/main) into a
  standalone package.

## Please cite

If you use `banditry` in your research, please cite:

```bibtex
@software{banditry,
  author  = {Arsenyan, Vahan},
  title   = {banditry: contextual bandit agents with Gaussian-process surrogates},
  year    = {2026},
  url     = {https://github.com/VahanArsenian/banditry},
  version = {0.2.0}
}
```

## License

[MIT](LICENSE). Portions derive from [HEBO](https://github.com/huawei-noah/HEBO)
(Huawei Noah's Ark Lab), also MIT-licensed — see [NOTICE](NOTICE).
