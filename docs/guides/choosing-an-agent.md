# Choosing an agent

`banditry` ships two agent families, each with two variants. All of them are
built through `build_agent(config, space)`.

## OFU (GP-UCB) — `OFUGPConfig`

Optimism-in-the-face-of-uncertainty with a Gaussian-process surrogate: fit a
GP, optimise a lower-confidence-bound acquisition.

| Variant | When |
|---|---|
| `surrogate="gp"` (exact GP) | Small budgets (up to a few hundred observations). Exact inference, best sample-efficiency. |
| `surrogate="svgp"` (sparse variational GP) | Larger budgets where exact GP inference gets slow. Minibatch ELBO training with inducing points. |

`noise_std_proxy` (the assumed observation-noise scale) is required. Setting
`frequentist=True` switches the confidence width to the Chowdhury–Gopalan
β_t, which additionally needs `rkhs_norm`.

```python
from banditry import OFUGPConfig, build_agent
agent = build_agent(OFUGPConfig(surrogate="gp", noise_std_proxy=1.0), space)
```

## Thompson sampling — `TSConfig`

A neural value function whose weights are drawn from the posterior by an MCMC
oracle; each round acts greedily w.r.t. one posterior sample.

| Variant | When |
|---|---|
| `sampler="langevin"` (SGLD) | Default choice: no extra dependencies, fast, scales with data via minibatching. Approximate posterior. |
| `sampler="nuts"` (pyro) | Higher-quality posterior samples at (much) higher cost per round. Needs `banditry[nuts]`. |

`feel_good=True` enables Feel-Good Thompson sampling (Zhang, 2021), which
reweights the posterior toward optimistic value functions — helpful in
contextual settings where plain TS can be insufficiently exploratory.

```python
from banditry import TSConfig, build_agent
agent = build_agent(TSConfig(sampler="langevin", feel_good=True), space)
```

## Rules of thumb

- **Start with OFU-GP** (`surrogate="gp"`): strongest baseline at small
  budgets, no MCMC tuning surface.
- **Switch to `svgp`** when rounds get slow from data volume.
- **Reach for TS** when you want posterior-sampling behaviour (e.g. batch
  diversity, richer exploration in contextual problems) or a non-GP value
  function; prefer `langevin`, and treat `nuts` as the high-fidelity
  reference.
- All agents handle mixed spaces and contexts; see
  [Design spaces](design-spaces.md) and
  [Contextual bandits](contextual-bandits.md).
