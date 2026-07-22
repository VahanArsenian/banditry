# Configuring samplers

The TS agent's MCMC behaviour is controlled by `TSConfig.sampler_config`, a
dict splatted into the sampler's constructor — so **the valid keys are exactly
the constructor keyword arguments** of
[`LangevinSampler`](../api/sampling-oracles.md) or
[`NUTSSampler`](../api/sampling-oracles.md).

When `sampler_config=None`, per-sampler defaults are used:
`banditry.agents.factory.DEFAULT_LANGEVIN_CONFIG` and `DEFAULT_NUTS_CONFIG`.
Copy and override rather than starting from scratch:

```python
from banditry import TSConfig
from banditry.agents import DEFAULT_LANGEVIN_CONFIG

config = TSConfig(
    sampler="langevin",
    sampler_config={**DEFAULT_LANGEVIN_CONFIG, "num_epochs": 256, "temperature": 0.1},
)
```

## Langevin (SGLD) knobs that matter most

- `num_epochs` / `burn_in` — chain length and how much of it is discarded.
  More epochs → better posterior approximation, slower rounds.
- `step_size` — a float or a schedule; `welling_teh_schedule(a, b, gamma,
  lr_floor)` gives the classic polynomially decaying ε_t = a·(b + t)^−γ.
- `temperature` — <1 sharpens the posterior (less exploration), >1 flattens it.
- `precondition` — RMSProp-style preconditioning; usually helps neural value
  functions.
- `max_obs_noise` — upper bound on the learned observation-noise parameter.

## NUTS knobs that matter most

- `num_samples` / `warmup_steps` — posterior samples kept and adaptation
  steps. The dominant cost factors.
- `max_tree_depth` — caps trajectory length per sample (cost grows as 2^depth).
- `prior_std` — prior scale over network weights.
- `obs_noise_prior_loc` / `obs_noise_prior_scale`, `max_obs_noise` — the
  observation-noise prior (log-normal) and its cap.
- `disable_progbar` — set `True` in scripts and tests.

## Warm starting

`TSConfig(should_warm_start=True)` (the default) initialises each round's
chain from the previous round's sampled model, which shortens burn-in
substantially once the loop is underway.

Full parameter documentation:
[Sampling oracles reference](../api/sampling-oracles.md).
