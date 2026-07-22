# Benchmark runner

Installing the package installs a `banditry-bench` CLI that exercises every
agent on synthetic benchmarks:

```bash
banditry-bench --agent ofugp-gp    --benchmark branin            --n-iter 30 --seed 42
banditry-bench --agent ts-langevin --benchmark gaussian_valley   --n-iter 30 --seed 42
banditry-bench --agent ofugp-svgp  --benchmark contextual_branin --n-iter 30 --seed 42
```

## Options

| Flag | Values | Default |
|---|---|---|
| `--agent` | `ofugp-gp`, `ofugp-svgp`, `ts-langevin`, `ts-nuts` | `ofugp-gp` |
| `--benchmark` | `branin`, `contextual_branin`, `gaussian_valley`, `contextual_gaussian_valley` | `gaussian_valley` |
| `--n-iter` | int | 30 |
| `--seed` | int | 42 |
| `--rand-sample` | warmup rounds | 4 |
| `--noise-std-proxy` | float (OFUGP agents only) | 1.0 |
| `--verbose` | flag | off |

The `contextual_*` variants sample one coordinate uniformly each round and
pin it via `fix_input`, exercising the contextual path.

The benchmark functions themselves (`branin`, `gaussian_valley`,
`BENCHMARKS`) are importable from `banditry.benchmark` for reuse in your own
scripts and tests.
