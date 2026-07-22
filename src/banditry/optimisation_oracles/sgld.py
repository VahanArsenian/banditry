import math

import torch


class SGLD(torch.optim.Optimizer):
    """Stochastic gradient Langevin dynamics optimizer (Welling & Teh, 2011).

    A ``torch.optim.Optimizer`` whose update is a gradient step on the negative log
    posterior — the loss gradient plus ``prior_precision * param`` from an isotropic
    zero-mean Gaussian prior — followed by injected Gaussian noise with standard deviation
    ``sqrt(2 * lr * temperature)``, so iterates approximately sample from the posterior
    rather than converge to a mode. With ``precondition=True`` an RMSProp-style diagonal
    preconditioner (an EMA of squared gradients) rescales both the gradient step and the
    noise, as in pSGLD (Li et al., 2016). All hyperparameters are per parameter group, so
    the prior can be disabled for selected parameters (e.g. the learned observation noise).
    Used by ``LangevinSampler`` to draw approximate posterior samples of ``ValueFunction``
    weights.

    Args:
        params: Iterable of parameters or parameter-group dicts; per-group overrides of the
            defaults below are honoured.
        lr: Learning rate (the SGLD step size ``eps_t``).
        prior_precision: Precision of the Gaussian prior, applied as weight decay; ``0``
            disables the prior term.
        temperature: Scales the injected-noise variance; ``0`` reduces the update to plain
            (optionally preconditioned) SGD.
        precondition: Enable RMSProp-style diagonal preconditioning.
        precond_alpha: EMA decay rate of the squared-gradient accumulator.
        precond_eps: Small constant added to the preconditioner denominator for stability.
        generator: Optional ``torch.Generator`` used for the injected noise.

    Raises:
        ValueError: If ``lr`` or ``temperature`` is negative.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        prior_precision: float = 1e-4,
        temperature: float = 1.0,
        precondition: bool = False,
        precond_alpha: float = 0.99,
        precond_eps: float = 1e-8,
        generator: torch.Generator | None = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if temperature < 0.0:
            raise ValueError(f"Invalid temperature: {temperature}")

        defaults = dict(
            lr=lr,
            prior_precision=prior_precision,
            temperature=temperature,
            precondition=precondition,
            precond_alpha=precond_alpha,
            precond_eps=precond_eps,
        )
        super().__init__(params, defaults)
        self.generator = generator

    def _randn_like(self, x: torch.Tensor) -> torch.Tensor:
        """Standard-normal noise shaped like ``x``, drawn from ``self.generator`` if set."""
        if self.generator is None:
            return torch.randn_like(x)
        return torch.randn(x.shape, generator=self.generator, device=x.device, dtype=x.dtype)

    @torch.no_grad()
    def step(self, noise: bool = True, closure=None):
        """Perform one SGLD update over all parameter groups.

        Args:
            noise: If ``False``, skip the Langevin noise injection (plain gradient step).
            closure: Optional callable that re-evaluates the model and returns the loss.

        Returns:
            The loss returned by ``closure``, or ``None`` when no closure is given.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            prior_prec = group["prior_precision"]
            temperature = group["temperature"]
            use_precond = group["precondition"]
            alpha = group["precond_alpha"]
            eps = group["precond_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                if prior_prec != 0:
                    g = g.add(p, alpha=prior_prec)

                if use_precond:
                    state = self.state[p]
                    if "v" not in state:
                        state["v"] = torch.zeros_like(p)
                    v = state["v"]
                    v.mul_(alpha).addcmul_(g, g, value=1.0 - alpha)
                    precond = 1.0 / (v.sqrt() + eps)
                    p.addcmul_(precond, g, value=-lr)
                    if noise:
                        p.add_(
                            precond.sqrt() * self._randn_like(p),
                            alpha=math.sqrt(2.0 * lr * temperature),
                        )
                else:
                    p.add_(g, alpha=-lr)
                    if noise:
                        p.add_(
                            self._randn_like(p),
                            alpha=math.sqrt(2.0 * lr * temperature),
                        )

        return loss
