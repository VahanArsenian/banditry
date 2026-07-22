"""Run a single agent against a synthetic benchmark.

banditry-bench --agent ofugp-gp --benchmark branin --n-iter 30 --seed 42
"""

import argparse
import random
import time

import numpy as np
import torch

import banditry.logging_utils as log
from banditry.agents.factory import OFUGPConfig, TSConfig, build_agent
from banditry.variable_domains.design_space import DesignSpace


def branin(df):
    x0 = df["x0"].to_numpy(dtype=float)
    x1 = df["x1"].to_numpy(dtype=float)
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
    return a * (x1 - b * x0**2 + c * x0 - r) ** 2 + s * (1 - t) * np.cos(x0) + s - 2


def gaussian_valley(df, lengthscale=0.125**0.25):
    x0 = df["x0"].to_numpy(dtype=float)
    x1 = df["x1"].to_numpy(dtype=float)
    return -((x0 - x1) ** 2) * np.exp(-(x0**2 + x1**2) / (2 * lengthscale**2))


BRANIN_SPACE = [
    {"name": "x0", "type": "num", "lb": -5, "ub": 10},
    {"name": "x1", "type": "num", "lb": 0, "ub": 15},
]
VALLEY_SPACE = [
    {"name": "x0", "type": "num", "lb": -1, "ub": 1},
    {"name": "x1", "type": "num", "lb": -1, "ub": 1},
]

# name -> (function over the suggestion DataFrame, space spec, context names
# sampled uniformly each round and pinned via fix_input)
BENCHMARKS = {
    "branin": (branin, BRANIN_SPACE, []),
    "contextual_branin": (branin, BRANIN_SPACE, ["x0"]),
    "gaussian_valley": (gaussian_valley, VALLEY_SPACE, []),
    "contextual_gaussian_valley": (gaussian_valley, VALLEY_SPACE, ["x1"]),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single agent against a benchmark.")
    parser.add_argument("--agent", choices=["ofugp-gp", "ofugp-svgp", "ts-nuts", "ts-langevin"], default="ofugp-gp")
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), default="gaussian_valley")
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rand-sample", type=int, default=4)
    parser.add_argument("--noise-std-proxy", type=float, default=1.0, help="OFUGP agents only")
    parser.add_argument("--verbose", action="store_true")
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_agent(name: str, space: DesignSpace, rand_sample: int, noise_std_proxy: float):
    kind, variant = name.split("-")
    if kind == "ofugp":
        config = OFUGPConfig(rand_sample=rand_sample, surrogate=variant, noise_std_proxy=noise_std_proxy)
    else:
        config = TSConfig(rand_sample=rand_sample, sampler=variant)
    return build_agent(config, space)


def run(args: argparse.Namespace) -> float:
    log.set_verbose(args.verbose)
    seed_everything(args.seed)

    fn, spec, context_names = BENCHMARKS[args.benchmark]
    space = DesignSpace.parse(spec)
    agent = make_agent(args.agent, space, args.rand_sample, args.noise_std_proxy)
    log.console.print(
        f"▶ {agent.display_name} on {args.benchmark} (n_iter={args.n_iter}, seed={args.seed})",
        style="bold green",
    )

    best_y = np.inf
    start = time.time()
    for i in range(args.n_iter):
        warmup = agent.n_plays() < agent.rand_sample
        context = {n: float(space.paras[n].sample(1)[0]) for n in context_names} or None
        rec = agent.suggest(1, context)
        y = np.asarray(fn(rec), dtype=float).reshape(-1)
        agent.observe(rec, y)

        best_y = min(best_y, float(y.min()))
        ctx = "  ctx={" + ", ".join(f"{k}={v:.4f}" for k, v in context.items()) + "}" if context else ""
        x = ", ".join(f"{k}={float(v):.4f}" for k, v in rec.iloc[0].items())
        log.console.print(
            f"iter {i:>3d} [{'warmup' if warmup else 'model '}]{ctx}  x={{{x}}}  y={y[0]:>9.4f}  best={best_y:>9.4f}",
            markup=False,
        )

    log.console.print(f"✓ done  best_y={best_y:.4f}  t={time.time() - start:.1f}s", style="bold green")
    return best_y


def main(argv=None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
