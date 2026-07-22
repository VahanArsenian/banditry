"""Thompson sampling vs Feel-Good Thompson sampling: best-so-far comparison.

Runs both TS variants (Langevin posterior sampling) on the gaussian_valley
benchmark and plots the best observed value per iteration. Requires
matplotlib (`pip install matplotlib`); with `--save PATH` it writes the
figure instead of showing it.

Run:  python examples/04_feel_good_ts.py [--save docs/assets/feel-good-ts.png]
"""

import argparse

import numpy as np

from banditry import DesignSpace, TSConfig, build_agent
from banditry.benchmark import VALLEY_SPACE, gaussian_valley, seed_everything

N_ITER = 30
SEED = 42


def run_agent(config: TSConfig) -> list[float]:
    seed_everything(SEED)
    space = DesignSpace.parse(VALLEY_SPACE)
    agent = build_agent(config, space)
    best, trace = np.inf, []
    for _ in range(N_ITER):
        rec = agent.suggest(1)
        y = np.asarray(gaussian_valley(rec), dtype=float).reshape(-1)
        agent.observe(rec, y)
        best = min(best, float(y[0]))
        trace.append(best)
    return trace


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", default=None, help="write the figure to this path instead of showing it")
    args = parser.parse_args(argv)

    print("running plain TS (langevin)...")
    plain = run_agent(TSConfig(rand_sample=4, sampler="langevin"))
    print("running Feel-Good TS (langevin)...")
    feel_good = run_agent(TSConfig(rand_sample=4, sampler="langevin", feel_good=True, fg_lambda=1.0, fg_bound=1.0))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed (`pip install matplotlib`); printing traces instead:")
        print("plain     :", " ".join(f"{v:.3f}" for v in plain))
        print("feel-good :", " ".join(f"{v:.3f}" for v in feel_good))
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    iters = np.arange(1, N_ITER + 1)
    ax.plot(iters, plain, label="Thompson sampling", linewidth=2)
    ax.plot(iters, feel_good, label="Feel-Good Thompson sampling", linewidth=2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("best observed value")
    ax.set_title(f"gaussian_valley, Langevin posterior sampling (seed {SEED})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
