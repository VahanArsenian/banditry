import torch
import numpy as np
from torch import Tensor
from torch.distributions import Normal
from abc import ABC, abstractmethod
from pybandits.constants import HALF_LOG_2PI
from pybandits.surrogates.svgp import BaseModel
from pybandits.surrogates.tsmodel import ValueFunction


class Objective(ABC):
    def __init__(self, model, **conf):
        self.model = model

    @property
    @abstractmethod
    def num_obj(self):
        pass

    @property
    @abstractmethod
    def num_constr(self):
        pass

    @abstractmethod
    def eval(self, x : Tensor,  xe : Tensor) -> Tensor:

        pass

    def __call__(self, x : Tensor,  xe : Tensor):
        return self.eval(x, xe)

class SingleObjective(Objective):
    def __init__(self, model : BaseModel, **conf):
        super().__init__(model, **conf)

    @property
    def num_obj(self):
        return 1

    @property
    def num_constr(self):
        return 0
    

class Mean(SingleObjective):
    def __init__(self, model : BaseModel, **conf):
        super().__init__(model, **conf)
        assert(model.num_out == 1)

    def eval(self, x : Tensor, xe : Tensor) -> Tensor:
        py, _ = self.model.predict(x, xe)
        return py


class Sigma(SingleObjective):
    def __init__(self, model : BaseModel, **conf):
        super().__init__(model, **conf)
        assert(model.num_out == 1)

    def eval(self, x : Tensor, xe : Tensor) -> Tensor:
        _, ps2 = self.model.predict(x, xe)
        return ps2.sqrt()


class LCB(SingleObjective):
    def __init__(self, model: BaseModel, best_y=None, **conf):
        super().__init__(model, **conf)
        self.kappa = conf.get('kappa', 2.0)

    def eval(self, x: torch.FloatTensor, xe: torch.LongTensor) -> torch.FloatTensor:
        with torch.no_grad():
            py, ps2 = self.model.predict(x, xe)
            noise = np.sqrt(2.0) * self.model.noise.sqrt()
            ps = ps2.sqrt().clamp(min=torch.finfo(ps2.dtype).eps).reshape(-1)
            py = py.reshape(-1)
            # lcb = (py + noise * torch.randn(py.shape)) - self.kappa * ps
            lcb = py - self.kappa * ps

            return lcb.reshape(-1, 1)


class MACE(Objective):
    def __init__(self, model, best_y, **conf):
        super().__init__(model, **conf)
        self.kappa = conf.get('kappa', 2.0)
        self.eps   = conf.get('eps', 1e-4)
        self.tau   = best_y
    
    @property
    def num_constr(self):
        return 0

    @property
    def num_obj(self):
        return 3

    @staticmethod
    def Mills_ratio_approximations(ps, normalised_improvement) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        logEIapp = ps.log() - 0.5 * normalised_improvement ** 2 - (normalised_improvement ** 2 - 1).log()
        logPIapp = (-0.5 * normalised_improvement ** 2 - torch.log(-1 * normalised_improvement)
                     - HALF_LOG_2PI)
        return logPIapp, logEIapp
    
    @staticmethod
    def exact_value(ps, normalised_improvement) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        dist = Normal(0., 1.)
        log_phi = dist.log_prob(normalised_improvement)
        cdf_at_imp = dist.cdf(normalised_improvement)
        return cdf_at_imp.log(),  (ps * (cdf_at_imp * normalised_improvement + log_phi.exp())).log()

    def eval(self, x : torch.FloatTensor, xe : torch.LongTensor) -> torch.FloatTensor:
        with torch.no_grad():
            py, ps2 = self.model.predict(x, xe)
            noise = np.sqrt(2.0) * self.model.noise.sqrt()
            ps = ps2.sqrt().clamp(min = torch.finfo(ps2.dtype).eps).reshape(-1)
            py = py.reshape(-1)
            # Wenlong Lyu et al. ICML 2018 https://proceedings.mlr.press/v80/lyu18a.html
            # Note that the direction of normalised, jittered improvment is flipped per Lyu et al
            # This is a slight deviation from common EI/PI for maximisiation, 
            # but it is correct, as if the current solution is smaller than the best so far
            # the "normed" is positive, this is why EI and PI are still maximised 
            norm_imp = ((self.tau - self.eps - py - noise * torch.randn(py.shape)) / ps)

            probability_of_imp, expected_imp = self.exact_value(ps, norm_imp)
            approximated_PI, approximated_EI = self.Mills_ratio_approximations(ps, norm_imp)

            use_approx = ~((norm_imp > -6) & torch.isfinite(expected_imp) & torch.isfinite(probability_of_imp))
            out = torch.zeros(x.shape[0], 3)

            out[:, 1] = torch.where(use_approx, approximated_EI, expected_imp)
            out[:, 2] = torch.where(use_approx, approximated_PI, probability_of_imp)

            out *= -1

            # lcb = (py + noise * torch.randn(py.shape)) - self.kappa * ps
            lcb = py - self.kappa * ps

            out[:, 0] = lcb.reshape(-1)

            return out

class ThompsonObjective(Objective):
    """Adapter: exposes a sampled ValueFunction as an Acquisition."""

    def __init__(self, model: ValueFunction):
        self.model = model

    def eval(self, x: Tensor, xe: Tensor) -> Tensor:
        pred = self.model(x, xe)
        if pred.dim() == 1:
            pred = pred.view(-1, 1)
        return pred

    @property
    def num_obj(self):
        return self.model.num_out

    @property
    def num_constr(self):
        return 0

