import math

import torch


class SGLD(torch.optim.Optimizer):
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
        if self.generator is None:
            return torch.randn_like(x)
        return torch.randn(x.shape, generator=self.generator, device=x.device, dtype=x.dtype)

    @torch.no_grad()
    def step(self, noise: bool = True, closure=None):
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
