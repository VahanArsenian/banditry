"""Declarative agent configuration and the `build_agent` factory.

This module is the main entry point of `banditry`: pick a config dataclass
(`OFUGPConfig` for GP-UCB / optimism-in-the-face-of-uncertainty agents,
`TSConfig` for Thompson-sampling agents), fill in its fields, and pass it to
`build_agent` together with a `DesignSpace` to obtain a ready-to-use agent that
drives the suggest -> evaluate -> observe loop.

Sampler behaviour for Thompson-sampling agents is controlled through
`TSConfig.sampler_config`; the defaults live in `DEFAULT_NUTS_CONFIG` and
`DEFAULT_LANGEVIN_CONFIG` and can be copied and overridden key by key.
"""

from dataclasses import dataclass, field

from banditry.agents.agent import AbstractAgent
from banditry.agents.ofugpagent import ModelEnum, OFUGPAgent
from banditry.agents.tsagent import TSAgent
from banditry.sampling_oracles.langevin_sampler import LangevinSampler, welling_teh_schedule
from banditry.sampling_oracles.sampler import FeelGoodNLL
from banditry.variable_domains.design_space import DesignSpace


@dataclass
class AgentConfig:
    """Base configuration shared by every agent built by `build_agent`.

    Concrete agents are configured through the subclasses `OFUGPConfig` and
    `TSConfig`; passing a plain `AgentConfig` to `build_agent` raises
    `TypeError`.
    """

    rand_sample: int = 4
    """Number of quasi-random (scrambled Sobol) warmup suggestions served before the agent switches to
    model-based suggestions. Values below 2 are floored to 2 by the agent. Default: ``4``."""


@dataclass
class OFUGPConfig(AgentConfig):
    """Configuration for the GP-UCB / optimism-in-the-face-of-uncertainty agent (`OFUGPAgent`).

    Although `noise_std_proxy` defaults to ``None`` (to keep the dataclass
    keyword-friendly), it is effectively required: `build_agent` forwards it to
    `OFUGPAgent`, whose constructor raises `ValueError` when it is ``None``.
    """

    surrogate: str = "svgp"  # "gp" | "svgp"
    """Surrogate model family: ``"gp"`` (exact Gaussian process; accurate but cubic in the number of
    observations) or ``"svgp"`` (sparse variational GP; inducing-point approximation that scales to larger
    datasets). Must be a `ModelEnum` member name. Default: ``"svgp"``."""

    frequentist: bool = False
    """If ``True``, use the frequentist Chowdhury-Gopalan (2017) confidence width ``beta_t`` (requires
    `rkhs_norm`) instead of the default Bayesian GP-UCB width. Default: ``False``."""

    rkhs_norm: float | None = None  # B in Chowdhury-Gopalan β_t
    """Bound ``B`` on the RKHS norm of the reward function, used by the frequentist ``beta_t``. Required
    when ``frequentist=True`` (the agent raises `ValueError` at suggestion time otherwise); ignored in the
    Bayesian setting. Default: ``None``."""

    noise_std_proxy: float | None = None  # R in Chowdhury-Gopalan β_t (sub-Gaussian noise scale)
    """Sub-Gaussian / GP observation-noise scale ``R``. **Effectively required**: building an OFU-GP agent
    raises `ValueError` if this is left as ``None``. Used by the frequentist ``beta_t`` and by the realised
    information gain accumulated on every ``observe``. Default: ``None``."""

    model_config_overrides: dict = field(default_factory=dict)
    """Overrides merged on top of the per-surrogate default model config (see `ModelEnum.model_config`)
    and passed through to the surrogate constructor. Default: ``{}``."""


@dataclass
class TSConfig(AgentConfig):
    """Configuration for the Thompson-sampling agent (`TSAgent`)."""

    sampler: str = "nuts"  # "langevin" | "nuts"
    """Posterior-sampling oracle for the value-function weights: ``"nuts"`` (Pyro NUTS; requires the
    optional ``banditry[nuts]`` extra) or ``"langevin"`` (SGLD; no extra dependencies).
    Default: ``"nuts"``."""

    feel_good: bool = False
    """If ``True``, enable Feel-Good Thompson sampling (Zhang, 2021): the Gaussian likelihood is reweighted
    with an exploration-boosting term built from `fg_lambda` and `fg_bound`. Default: ``False``."""

    fg_lambda: float = 1.0
    """Strength ``lambda`` of the Feel-Good reweighting term (only used when ``feel_good=True``); ``0``
    disables the term. Default: ``1.0``."""

    fg_bound: float = 1.0
    """Upper truncation of the Feel-Good optimistic-value term (the cap ``b`` in ``min(b, f*)``); must be
    non-negative. Only used when ``feel_good=True``. Default: ``1.0``."""

    model_config: dict = field(default_factory=dict)
    """Keyword arguments forwarded to the neural ``ValueFunction`` surrogate (e.g. ``num_uniqs``,
    ``emb_sizes``, a custom feature extractor ``fe``). ``num_uniqs`` is filled in automatically for spaces
    with categorical parameters. Default: ``{}``."""

    sampler_config: dict | None = None  # None -> per-sampler default below
    """Keyword arguments splatted into the sampler constructor (``NUTSSampler(**sampler_config)`` or
    ``LangevinSampler(**sampler_config)``), so any constructor kwarg is a valid key. ``None`` (default)
    selects `DEFAULT_NUTS_CONFIG` or `DEFAULT_LANGEVIN_CONFIG` according to `sampler`. An explicit dict
    *replaces* the default rather than being merged into it, so copy the default and override keys."""

    should_warm_start: bool = True
    """If ``True`` (default), initialise each round's MCMC run from the previous round's sampled model
    instead of a freshly initialised network."""

    latent_dimension: float | None = None  # None -> TSAgent.num_samplable_params() at build time
    """Latent dimension of the sampled parameter vector, recorded for labelling/diagnostics. ``None``
    (default) resolves to ``TSAgent.num_samplable_params()`` at build time."""


DEFAULT_NUTS_CONFIG: dict = {
    "num_samples": 16,
    "warmup_steps": 64,
    "target_accept_prob": 0.6,
    "prior_std": 0.5,
    "obs_noise_prior_loc": -3.0,
    "obs_noise_prior_scale": 0.5,
    "max_obs_noise": 1.0,
    "max_tree_depth": 6,
    "disable_progbar": False,
}
"""Default `TSConfig.sampler_config` for ``sampler="nuts"``.

The dict is splatted into the ``NUTSSampler`` constructor (``NUTSSampler(**sampler_config)``), so any
constructor keyword argument is a valid key. Keys set here:

- ``num_samples`` (``16``): posterior draws kept after warmup; one draw is then selected uniformly at
  random as the sampled value function.
- ``warmup_steps`` (``64``): NUTS adaptation iterations, discarded before sampling.
- ``target_accept_prob`` (``0.6``): target acceptance probability for NUTS step-size adaptation.
- ``prior_std`` (``0.5``): standard deviation of the zero-mean Gaussian prior placed on every network
  weight.
- ``obs_noise_prior_loc`` (``-3.0``): location of the prior over the observation-noise std, interpreted in
  log-noise space (clamped to ``[log(min_obs_noise), log(max_obs_noise)]``).
- ``obs_noise_prior_scale`` (``0.5``): scale of that prior in the unconstrained (pre-sigmoid)
  parameterisation of the noise.
- ``max_obs_noise`` (``1.0``): upper bound of the observation-noise std (the noise is a sigmoid-squashed
  parameter living in ``[min_obs_noise, max_obs_noise]``).
- ``max_tree_depth`` (``6``): maximum doubling depth of the NUTS trajectory tree.
- ``disable_progbar`` (``False``): show Pyro's MCMC progress bar (the constructor default hides it).

Other accepted keys include ``adapt_step_size``, ``adapt_mass_matrix``, ``use_multinomial_sampling``,
``init_obs_noise``, ``min_obs_noise``, ``jit_compile``, ``ignore_jit_warnings``, and ``generator``.
"""


DEFAULT_LANGEVIN_CONFIG: dict = {
    "batch_size": 512,
    "num_epochs": 128,
    "burn_in": 64,
    "temperature": 0.2,
    "step_size": welling_teh_schedule(a=5e-3, b=1.0, gamma=0.55, lr_floor=1e-3),
    "max_obs_noise": 10.0,
    "precondition": True,
}
"""Default `TSConfig.sampler_config` for ``sampler="langevin"``.

The dict is splatted into the `LangevinSampler` constructor (``LangevinSampler(**sampler_config)``), so
any constructor keyword argument is a valid key. Keys set here:

- ``batch_size`` (``512``): SGLD minibatch size (capped at the dataset size).
- ``num_epochs`` (``128``): passes over the data; the total number of updates is at least
  ``burn_in + 1`` epochs and ``min_batches`` updates, whichever is larger.
- ``burn_in`` (``64``): epochs discarded before a post-burn-in state is reservoir-sampled uniformly as
  the posterior draw.
- ``temperature`` (``0.2``): SGLD noise temperature; ``1.0`` targets the exact posterior, smaller values
  sample a colder (sharper) posterior.
- ``step_size``: constant float or callable schedule ``(t, n_burn_in) -> lr``; here a Welling-Teh
  polynomial decay ``a * (b + t) ** -gamma`` that is frozen after burn-in and floored at ``lr_floor``.
- ``max_obs_noise`` (``10.0``): upper clamp of the learned observation-noise std.
- ``precondition`` (``True``): apply RMSProp-style diagonal preconditioning to the SGLD updates.

Other accepted keys include ``prior_precision``, ``precond_alpha``, ``precond_eps``, ``min_batches``,
``init_obs_noise``, ``min_obs_noise``, and ``generator``.
"""


def build_agent(config: AgentConfig, space: DesignSpace) -> AbstractAgent:
    """Build a ready-to-use agent from a declarative config.

    Args:
        config: An `OFUGPConfig` (GP-UCB / OFU agent) or a `TSConfig`
            (Thompson-sampling agent).
        space: The design space the agent optimises over.

    Returns:
        A fully constructed `OFUGPAgent` or `TSAgent`.

    Raises:
        TypeError: If ``config`` is neither an `OFUGPConfig` nor a `TSConfig`.
        ValueError: If ``TSConfig.sampler`` is not ``"nuts"`` or ``"langevin"``, or if
            ``OFUGPConfig.noise_std_proxy`` is ``None`` (raised by `OFUGPAgent`).
        KeyError: If ``OFUGPConfig.surrogate`` is not a `ModelEnum` member name
            (``"gp"`` or ``"svgp"``).
        ImportError: If ``sampler="nuts"`` and the optional pyro dependency
            (``pip install "banditry[nuts]"``) is not installed.

    Example:
        Build an OFU-GP agent on a two-parameter space and run one round:

        ```python
        import numpy as np
        from banditry import DesignSpace, OFUGPConfig, build_agent

        space = DesignSpace.parse([
            {"name": "x0", "type": "num", "lb": -1, "ub": 1},
            {"name": "x1", "type": "num", "lb": -1, "ub": 1},
        ])
        agent = build_agent(OFUGPConfig(rand_sample=4, surrogate="gp", noise_std_proxy=1.0), space)

        rec = agent.suggest(1)  # DataFrame of suggestions
        y = ((rec["x0"] - 0.3) ** 2 + (rec["x1"] + 0.2) ** 2).to_numpy()
        agent.observe(rec, y)  # agents minimise y
        ```
    """
    if isinstance(config, OFUGPConfig):
        model = ModelEnum[config.surrogate]
        return OFUGPAgent(
            space,
            surrogate=model,
            rand_sample=config.rand_sample,
            model_config=model.model_config(space, config.model_config_overrides),
            frequentist=config.frequentist,
            rkhs_norm=config.rkhs_norm,
            noise_std_proxy=config.noise_std_proxy,
        )
    if isinstance(config, TSConfig):
        nll = FeelGoodNLL(fg_lambda=config.fg_lambda, fg_bound=config.fg_bound) if config.feel_good else None
        if config.sampler == "nuts":
            from banditry.sampling_oracles.nuts_sampler import NUTSSampler

            sampler_cls = NUTSSampler
            sampler_config = config.sampler_config if config.sampler_config is not None else DEFAULT_NUTS_CONFIG
        elif config.sampler == "langevin":
            sampler_cls = LangevinSampler
            sampler_config = config.sampler_config if config.sampler_config is not None else DEFAULT_LANGEVIN_CONFIG
        else:
            raise ValueError(f"Unknown sampler: {config.sampler!r}")
        return TSAgent(
            space,
            rand_sample=config.rand_sample,
            sampler_cls=sampler_cls,
            sampler_config=sampler_config,
            nll=nll,
            model_config=config.model_config,
            should_warm_start=config.should_warm_start,
            latent_dimension=config.latent_dimension,
        )
    raise TypeError(f"Unknown config type: {type(config).__name__}")
