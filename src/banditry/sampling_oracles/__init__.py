"""Posterior samplers for the TS agent: Langevin dynamics and NUTS (pyro)."""

from banditry.sampling_oracles.langevin_sampler import LangevinSampler, welling_teh_schedule
from banditry.sampling_oracles.sampler import NLL, FeelGoodNLL, GaussianNLL, Sampler

__all__ = [
    "NLL",
    "FeelGoodNLL",
    "GaussianNLL",
    "LangevinSampler",
    "NUTSSampler",
    "Sampler",
    "welling_teh_schedule",
]


def __getattr__(name: str):
    # NUTSSampler needs pyro (the optional banditry[nuts] extra), so it is
    # loaded lazily rather than at package import.
    if name == "NUTSSampler":
        from banditry.sampling_oracles.nuts_sampler import NUTSSampler

        return NUTSSampler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
