import math
from collections.abc import Callable
from copy import deepcopy

import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor
from torch.utils.data import DataLoader, TensorDataset

from banditry.optimisation_oracles.sgld import SGLD
from banditry.sampling_oracles.sampler import Sampler, _gaussian_nll
from banditry.surrogates.tsmodel import ValueFunction


def welling_teh_schedule(
    a: float, b: float = 1.0, gamma: float = 0.55, lr_floor: float = 1e-5
) -> Callable[[int], float]:
    """Build the polynomial step-size schedule of Welling & Teh (2011).

    Implements the decaying schedule from Welling & Teh (2011), "Bayesian Learning via
    Stochastic Gradient Langevin Dynamics": ``eps_t = a * (b + t) ** (-gamma)`` during
    burn-in. After burn-in the rate is frozen at its final burn-in value and clamped from
    below by ``lr_floor``, i.e. ``max(a * (b + n_burn_in) ** (-gamma), lr_floor)``.

    Args:
        a: Initial scale of the schedule; must be positive.
        b: Offset delaying the start of the decay; must be non-negative.
        gamma: Decay exponent; must lie in ``(0.5, 1]`` for the SGLD convergence
            conditions of Welling & Teh to hold.
        lr_floor: Lower bound applied to the post-burn-in learning rate.

    Returns:
        A callable ``(t, n_burn_in) -> lr`` mapping the update step ``t`` (given the total
        number of burn-in updates ``n_burn_in``) to a learning rate.

    Raises:
        ValueError: If ``a <= 0``, ``b < 0``, or ``gamma`` is outside ``(0.5, 1]``.
    """
    # Polynomial step-size schedule from Welling & Teh (2011).

    if a <= 0:
        raise ValueError(f"a must be positive, got {a}")
    if b < 0:
        raise ValueError(f"b must be non-negative, got {b}")
    if not (0.5 < gamma <= 1.0):
        raise ValueError(f"gamma must be in (0.5, 1], got {gamma}")

    def _schedule(t: int, n_burn_in: bool) -> float:
        if t < n_burn_in:
            return a * (b + t) ** (-gamma)
        else:
            return max(a * (b + n_burn_in) ** (-gamma), lr_floor)

    return _schedule


class LangevinSampler(Sampler):
    """Sample TS model weights from an approximate posterior using SGLD.

    Runs stochastic gradient Langevin dynamics (Welling & Teh, 2011) on a deep copy of the
    model: minibatch gradient steps on the negative log posterior with Gaussian noise
    injected by the :class:`~banditry.optimisation_oracles.sgld.SGLD` optimizer. The
    observation-noise standard deviation is learned jointly as a per-output parameter
    (excluded from the weight prior). The total number of updates is the maximum of
    ``num_epochs`` epochs, ``min_batches`` updates, and ``burn_in + 1`` epochs, so at least
    one post-burn-in epoch always runs; one post-burn-in state is kept via streaming
    uniform (reservoir) selection and returned as the posterior draw.

    The constructor keyword arguments below are exactly the valid keys of
    ``TSConfig.sampler_config`` when ``sampler="langevin"``.

    Args:
        step_size: Learning-rate schedule: either a constant ``float`` or a callable
            ``(t, n_burn_in) -> lr`` such as the one returned by
            :func:`welling_teh_schedule` (the default).
        prior_precision: Precision of the isotropic zero-mean Gaussian prior over model
            weights, applied as weight decay on the gradients.
        temperature: Scales the variance of the injected Langevin noise; ``1.0`` targets
            the true posterior, smaller values sharpen it, and ``0`` reduces the update to
            plain SGD.
        precondition: If ``True``, use RMSProp-style diagonal preconditioning (pSGLD):
            gradient step and noise are rescaled by an EMA of squared gradients.
        precond_alpha: EMA decay rate of the squared-gradient accumulator used by the
            preconditioner.
        precond_eps: Numerical-stability constant added to the preconditioner denominator.
        batch_size: Minibatch size (capped at the dataset size).
        min_batches: Minimum total number of gradient updates to perform.
        num_epochs: Target number of passes over the data.
        burn_in: Number of initial epochs discarded before model states become eligible
            for selection as the returned sample.
        init_obs_noise: Initial observation-noise standard deviation (in standardised-y
            units).
        min_obs_noise: Lower clamp for the learned observation noise.
        max_obs_noise: Upper clamp for the learned observation noise.
        generator: Optional ``torch.Generator`` making noise injection and sample
            selection reproducible.
    """

    def __init__(
        self,
        step_size: float | Callable[[int], float] = welling_teh_schedule(a=1e-3),  # noqa: B008 — deterministic closure
        prior_precision: float = 1e-4,
        temperature: float = 1.0,
        precondition: bool = False,
        precond_alpha: float = 0.99,
        precond_eps: float = 1e-5,
        batch_size: int = 64,
        min_batches: int = 10,
        num_epochs: int = 10,
        burn_in: int = 10,
        init_obs_noise: float = 1.0,
        min_obs_noise: float = 1e-6,
        max_obs_noise: float = 1.0,
        generator: torch.Generator | None = None,
    ):

        if callable(step_size):
            self._step_size_fn: Callable[[int], float] = step_size
        else:
            _c = float(step_size)

            def _f(*args, **kwargs):
                return _c

            self._step_size_fn = _f
        self.prior_precision = float(prior_precision)
        self.temperature = float(temperature)
        self.precondition = bool(precondition)
        self.precond_alpha = float(precond_alpha)
        self.precond_eps = float(precond_eps)
        self.batch_size = int(batch_size)
        self.min_batches = int(min_batches)
        self.num_epochs = int(num_epochs)
        self.burn_in = int(burn_in)
        self.init_obs_noise = float(init_obs_noise)
        self.min_obs_noise = float(min_obs_noise)
        self.max_obs_noise = float(max_obs_noise)
        self.generator = generator

    def sample(
        self,
        model: ValueFunction,
        Xc: FloatTensor | None,
        Xe: LongTensor | None,
        y: FloatTensor,
        nll: Callable[..., FloatTensor] | None = None,
    ) -> ValueFunction:
        """Draw one posterior weight sample via SGLD.

        Standardises ``y`` (and fits the model's x-scaler), then runs SGLD on a deep copy
        of ``model`` for the configured number of updates, keeping one post-burn-in state
        chosen uniformly at random as the posterior draw.

        Args:
            model: Template ``ValueFunction``; left unmodified.
            Xc: Continuous features of shape ``(n, model.num_cont)``, or ``None``.
            Xe: Categorical features of shape ``(n, model.num_enum)``, or ``None``.
            y: Observed targets of shape ``(n, model.num_out)``.
            nll: Optional NLL callable ``(pred, target, obs_std, **kwargs)``; it may
                return per-element values (reduced internally to a dataset total) or a
                scalar total loss. Defaults to the Gaussian NLL.

        Returns:
            A new ``ValueFunction`` in eval mode carrying the sampled weights and the
            fitted y-scaler.
        """

        nll_fn = _gaussian_nll if nll is None else nll

        Xc_t, Xe_t, y_t = self._prepare_xy(model, Xc, Xe, y)
        yscaler, y_scaled = self._fit_y_scaler(y_t)

        working_model = deepcopy(model)
        working_model.clear_y_scaler()
        if working_model.num_cont > 0:
            working_model.fit_x_scaler(Xc_t)
        working_model.train()

        Xc_scaled, Xe_ready = working_model.xtrans(Xc_t, Xe_t)

        num_data = y_scaled.shape[0]
        batch_size = min(self.batch_size, num_data)
        batch_size = max(1, batch_size)

        ds = TensorDataset(Xc_scaled, Xe_ready, y_scaled)
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        if len(dl) == 0:
            dl = DataLoader(ds, batch_size=num_data, shuffle=True, drop_last=False)

        updates_per_epoch = max(1, len(dl))
        min_updates_for_sampling = (self.burn_in + 1) * updates_per_epoch
        target_updates = max(
            self.num_epochs * updates_per_epoch,
            self.min_batches,
            min_updates_for_sampling,
        )
        burn_in_updates = self.burn_in * updates_per_epoch
        import banditry.logging_utils as log

        log.debug(f"Learning rate after burn in: {self._step_size_fn(burn_in_updates, n_burn_in=burn_in_updates)}")

        model_params = [p for p in working_model.parameters() if p.requires_grad]

        log_obs_noise = nn.Parameter(
            torch.full(
                (working_model.num_out,),
                math.log(self.init_obs_noise),
                dtype=y_scaled.dtype,
                device=y_scaled.device,
            )
        )
        optimizer = SGLD(
            [
                {"params": model_params, "prior_precision": self.prior_precision},
                {"params": [log_obs_noise], "prior_precision": 0.0},
            ],
            lr=self._step_size_fn(0, n_burn_in=burn_in_updates),
            temperature=self.temperature,
            precondition=self.precondition,
            precond_alpha=self.precond_alpha,
            precond_eps=self.precond_eps,
            generator=self.generator,
        )

        selected_state: dict[str, torch.Tensor] | None = None
        post_burn_seen = 0
        updates = 0

        while updates < target_updates:
            for bxc, bxe, by in dl:
                optimizer.zero_grad()

                pred = working_model.forward(bxc, bxe, _pre_scaled=True)

                obs_noise = torch.exp(log_obs_noise).clamp(min=self.min_obs_noise, max=self.max_obs_noise).view(1, -1)

                nll_value = nll_fn(
                    pred,
                    by,
                    obs_noise,
                    num_data=num_data,
                    model=working_model,
                )
                if not torch.is_tensor(nll_value):
                    nll_value = torch.as_tensor(nll_value, device=pred.device, dtype=pred.dtype)

                if nll_value.dim() == 0:
                    loss = nll_value
                else:
                    per_datapoint = nll_value.reshape(nll_value.shape[0], -1).sum(dim=1)
                    loss = per_datapoint.mean() * num_data

                loss.backward()

                lr = self._step_size_fn(updates, n_burn_in=burn_in_updates)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                optimizer.step(noise=True)

                with torch.no_grad():
                    log_obs_noise.clamp_(math.log(self.min_obs_noise), math.log(self.max_obs_noise))
                updates += 1

            if updates >= burn_in_updates:
                post_burn_seen += 1
                if selected_state is None:
                    selected_state = deepcopy(working_model.state_dict())
                else:
                    # replace with prob 1/post_burn_seen, stream uniform
                    if self._draw_sample_id(post_burn_seen) == 0:
                        selected_state = deepcopy(working_model.state_dict())

        sampled_model = working_model
        if selected_state is not None:
            sampled_model.load_state_dict(selected_state, strict=True)

        sampled_model.set_y_scaler(yscaler)
        sampled_model.eval()
        return sampled_model
