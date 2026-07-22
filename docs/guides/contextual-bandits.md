# Contextual bandits

In a contextual bandit, part of the input is *observed*, not chosen: each
round the environment hands you a context, and the agent optimises the
remaining parameters **conditionally on it**.

In `banditry`, contexts are ordinary parameters of the design space that you
pin each round via the second argument of `suggest`:

```python
import numpy as np
from banditry import DesignSpace, OFUGPConfig, build_agent

space = DesignSpace.parse([
    {"name": "x0", "type": "num", "lb": -1, "ub": 1},   # decision variable
    {"name": "x1", "type": "num", "lb": -1, "ub": 1},   # observed context
])
agent = build_agent(OFUGPConfig(rand_sample=4, surrogate="gp", noise_std_proxy=1.0), space)

for _ in range(30):
    context = {"x1": observe_environment()}   # whatever the world gives you this round
    rec = agent.suggest(1, context)           # x1 is pinned; x0 is optimised given x1
    y = np.asarray(objective(rec), dtype=float).reshape(-1)
    agent.observe(rec, y)
```

The pinned columns are fixed both during quasi-random warmup and during
model-based optimisation (the acquisition is optimised only over the free
parameters).

## Best observation for a context

`get_best_id` accepts the same `fix_input` dict and returns the best
*matching* observation (falling back to the global best if nothing matches):

```python
best_for_context = agent.get_best_id(fix_input={"x1": 0.7})
```

## Notes

- Any subset of parameters can serve as context, including categorical ones.
- The surrogate is trained on the full parameter vector (decisions +
  contexts), so information is shared across contexts — nearby contexts
  inform each other.
- Both agent families support contexts (`support_contextual = True`).
