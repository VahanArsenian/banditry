from collections.abc import Callable
from copy import deepcopy
import math

import torch
from torch import FloatTensor, LongTensor
from torch.func import functional_call

from banditry.sampling_oracles.sampler import Sampler, _gaussian_nll
from banditry.surrogates.tsmodel import ValueFunction
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide.initialization import init_to_value


class NUTSSampler(Sampler):
    """Sample TS model weights from a posterior using Pyro NUTS."""

    def __init__(
        self,
        num_samples: int = 64,
        warmup_steps: int = 256,
        target_accept_prob: float = 0.8,
        max_tree_depth: int = 10,
        adapt_step_size: bool = True,
        adapt_mass_matrix: bool = True,
        use_multinomial_sampling: bool = True,
        prior_std: float = 1.0,
        obs_noise_prior_loc: float = 0.0,
        obs_noise_prior_scale: float = 1.0,
        init_obs_noise: float = 1.0,
        min_obs_noise: float = 1e-6,
        max_obs_noise: float = 1e3,
        jit_compile: bool = False,
        ignore_jit_warnings: bool = True,
        disable_progbar: bool = True,
        generator: torch.Generator | None = None,
    ):

        self.num_samples = int(num_samples)
        self.warmup_steps = int(warmup_steps)
        self.target_accept_prob = float(target_accept_prob)
        self.max_tree_depth = int(max_tree_depth)
        self.adapt_step_size = bool(adapt_step_size)
        self.adapt_mass_matrix = bool(adapt_mass_matrix)
        self.use_multinomial_sampling = bool(use_multinomial_sampling)
        self.prior_std = float(prior_std)
        self.obs_noise_prior_loc = float(obs_noise_prior_loc)
        self.obs_noise_prior_scale = float(obs_noise_prior_scale)
        self.init_obs_noise = float(init_obs_noise)
        self.min_obs_noise = float(min_obs_noise)
        self.max_obs_noise = float(max_obs_noise)
        self.jit_compile = bool(jit_compile)
        self.ignore_jit_warnings = bool(ignore_jit_warnings)
        self.disable_progbar = bool(disable_progbar)
        self.generator = generator

    @staticmethod
    def _site_name(index: int) -> str:
        return f"param_{index}"

    def _noise_to_raw(self, noise: float) -> float:
        span = self.max_obs_noise - self.min_obs_noise
        p = (noise - self.min_obs_noise) / span
        p = max(1e-7, min(1.0 - 1e-7, p))
        return math.log(p) - math.log1p(-p)

    def _raw_to_noise(self, raw_noise: torch.Tensor) -> torch.Tensor:
        return self.min_obs_noise + (
            self.max_obs_noise - self.min_obs_noise
        ) * torch.sigmoid(raw_noise)

    def _raw_noise_prior_loc(self) -> float:
        prior_log_noise = min(
            max(self.obs_noise_prior_loc, math.log(self.min_obs_noise)),
            math.log(self.max_obs_noise),
        )
        return self._noise_to_raw(math.exp(prior_log_noise))

    def sample(
        self,
        model: ValueFunction,
        Xc: FloatTensor | None,
        Xe: LongTensor | None,
        y: FloatTensor,
        nll: Callable[..., FloatTensor] | None = None,
    ) -> ValueFunction:

        nll_fn = _gaussian_nll if nll is None else nll

        Xc_t, Xe_t, y_t = self._prepare_xy(model, Xc, Xe, y)
        yscaler, y_scaled = self._fit_y_scaler(y_t)

        working_model = deepcopy(model)
        if working_model.num_cont > 0:
            working_model.fit_x_scaler(Xc_t)
        Xc_prescaled, Xe_prescaled = working_model.xtrans(Xc_t, Xe_t)
        working_model.clear_y_scaler()
        working_model.eval()

        param_specs: list[tuple[str, str, torch.Tensor]] = []
        init_values: dict[str, torch.Tensor] = {}
        for idx, (param_name, param_value) in enumerate(working_model.named_parameters()):
            if not param_value.requires_grad:
                continue
            site_name = self._site_name(idx)
            detached = param_value.detach()
            param_specs.append((site_name, param_name, detached))
            init_values[site_name] = detached.clone()
        if len(param_specs) == 0:
            raise ValueError("No trainable parameters found for posterior sampling.")

        num_data = y_scaled.shape[0]

        init_noise = min(max(self.init_obs_noise, self.min_obs_noise), self.max_obs_noise)
        init_values["raw_obs_noise"] = torch.full(
            (working_model.num_out,),
            self._noise_to_raw(init_noise),
            device=y_scaled.device,
            dtype=y_scaled.dtype,
        )

        pyro.clear_param_store()

        def _model():
            sampled_params: dict[str, torch.Tensor] = {}
            for site_name, param_name, param_value in param_specs:
                sampled_params[param_name] = pyro.sample(
                    site_name,
                    dist.Normal(
                        torch.zeros_like(param_value),
                        torch.full_like(param_value, self.prior_std),
                    ).to_event(param_value.dim()),
                )

            raw_obs_noise = pyro.sample(
                "raw_obs_noise",
                dist.Normal(
                    torch.full(
                        (working_model.num_out,),
                        self._raw_noise_prior_loc(),
                        device=y_scaled.device,
                        dtype=y_scaled.dtype,
                    ),
                    torch.full(
                        (working_model.num_out,),
                        self.obs_noise_prior_scale,
                        device=y_scaled.device,
                        dtype=y_scaled.dtype,
                    ),
                ).to_event(1),
            )
            obs_noise = self._raw_to_noise(raw_obs_noise).view(1, -1)

            pred = functional_call(
                working_model,
                sampled_params,
                (Xc_prescaled, Xe_prescaled),
                {"_pre_scaled": True},
            )
            nll_value = nll_fn(
                pred,
                y_scaled,
                obs_noise,
                num_data=num_data,
                model=working_model,
                model_params=sampled_params,
            )
            if not torch.is_tensor(nll_value):
                nll_value = torch.as_tensor(nll_value, device=pred.device, dtype=pred.dtype)
            else:
                nll_value = nll_value.to(device=pred.device, dtype=pred.dtype)
            total_nll = nll_value.sum() if nll_value.dim() > 0 else nll_value
            pyro.factor("likelihood", -total_nll)

        if self.generator is not None:
            seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self.generator).item())
            pyro.set_rng_seed(seed)

        kernel = NUTS(
            _model,
            target_accept_prob=self.target_accept_prob,
            max_tree_depth=self.max_tree_depth,
            adapt_step_size=self.adapt_step_size,
            adapt_mass_matrix=self.adapt_mass_matrix,
            use_multinomial_sampling=self.use_multinomial_sampling,
            jit_compile=self.jit_compile,
            ignore_jit_warnings=self.ignore_jit_warnings,
            init_strategy=init_to_value(values=init_values),
        )
        mcmc = MCMC(
            kernel,
            num_samples=self.num_samples,
            warmup_steps=self.warmup_steps,
            disable_progbar=self.disable_progbar,
        )
        mcmc.run()

        samples = mcmc.get_samples(group_by_chain=False)
        if len(samples) == 0:
            raise RuntimeError("Pyro MCMC did not return any posterior samples.")

        first_site = next(iter(samples))
        posterior_count = int(samples[first_site].shape[0])
        if posterior_count <= 0:
            raise RuntimeError("Pyro MCMC returned zero posterior draws.")

        sample_id = self._draw_sample_id(posterior_count)
        sampled_state = working_model.state_dict()
        for site_name, param_name, param_value in param_specs:
            sampled_state[param_name] = samples[site_name][sample_id].to(
                device=param_value.device,
                dtype=param_value.dtype,
            )
        working_model.load_state_dict(sampled_state, strict=True)
        working_model.set_y_scaler(yscaler)
        working_model.eval()
        return working_model
