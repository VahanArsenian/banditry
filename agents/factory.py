from dataclasses import dataclass, field
from typing import Optional

from agents.agent import AbstractAgent
from agents.ofugpagent import OFUGPAgent, ModelEnum
from agents.tsagent import TSAgent
from sampling_oracles.langevin_sampler import LangevinSampler, welling_teh_schedule
from sampling_oracles.sampler import FeelGoodNLL
from variable_domains.design_space import DesignSpace


@dataclass
class AgentConfig:
    rand_sample: int = 4


@dataclass
class OFUGPConfig(AgentConfig):
    surrogate: str = "svgp"                              # "gp" | "svgp"
    frequentist: bool = False
    rkhs_norm: Optional[float] = None                    # B in Chowdhury-Gopalan β_t
    noise_std_proxy: Optional[float] = None              # R in Chowdhury-Gopalan β_t (sub-Gaussian noise scale)
    model_config_overrides: dict = field(default_factory=dict)


@dataclass
class TSConfig(AgentConfig):
    sampler: str = "nuts"                                # "langevin" | "nuts"
    feel_good: bool = False
    fg_lambda: float = 1.0
    fg_bound: float = 1.0
    model_config: dict = field(default_factory=dict)
    sampler_config: Optional[dict] = None                # None -> per-sampler default below
    should_warm_start: bool = True
    regret_ub_coeff: float = 1.0                         # carried in custom_score_info as B_T
    function_scale: float = 1.0                          # value returned by function_class_context_projected_width()
    latent_dimension: Optional[float] = None             # None -> TSAgent.num_samplable_params() at build time
    delta: float = 0.01                                  # confidence parameter in the regret bound's log term
    paremeter_space_l1_diameter: float = 1.0             # B (parameter-space L1 diameter) in the regret bound


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


DEFAULT_LANGEVIN_CONFIG: dict = {
    "batch_size": 512,
    "num_epochs": 128,
    "burn_in": 64,
    "temperature": 0.2,
    "step_size": welling_teh_schedule(a=5e-3, b=1.0, gamma=0.55, lr_floor=1e-3),
    "max_obs_noise": 10.0,
    "precondition": True,
}


def build_agent(config: AgentConfig, space: DesignSpace) -> AbstractAgent:
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
            from sampling_oracles.nuts_sampler import NUTSSampler
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
            regret_ub_coeff=config.regret_ub_coeff,
            function_scale=config.function_scale,
            latent_dimension=config.latent_dimension,
            delta=config.delta,
            paremeter_space_l1_diameter=config.paremeter_space_l1_diameter,
        )
    raise TypeError(f"Unknown config type: {type(config).__name__}")

