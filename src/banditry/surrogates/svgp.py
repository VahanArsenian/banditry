import torch
from torch import FloatTensor, LongTensor
from abc import ABC, abstractmethod
import numpy as np
import torch.nn as nn
import gpytorch

from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.models import ApproximateGP

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.priors import GammaPrior
from gpytorch.constraints import GreaterThan
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy, NaturalVariationalDistribution, TrilNaturalVariationalDistribution
from gpytorch.kernels import MaternKernel, ScaleKernel, ProductKernel
from banditry.variable_domains.transforms import TorchMinMaxScaler, TorchStandardScaler, DummyFeatureExtractor


def filter_nan(x: FloatTensor, xe: LongTensor, y: FloatTensor, keep_rule='any') -> tuple[FloatTensor, LongTensor, FloatTensor]:
    assert x is None or torch.isfinite(x).all()
    assert xe is None or torch.isfinite(xe).all()
    assert torch.isfinite(y).any(), "No valid data in the dataset"

    if keep_rule == 'any':
        valid_id = torch.isfinite(y).any(dim = 1)
    else:
        valid_id = torch.isfinite(y).all(dim = 1)
    x_filtered = x[valid_id] if x is not None else None
    xe_filtered = xe[valid_id] if xe is not None else None
    y_filtered = y[valid_id]
    return x_filtered, xe_filtered, y_filtered


def default_kern(x, xe, y, total_dim = None, ard_kernel = True, use_product_kernel=True, max_x = 1000):
    # if use_product_kernel is true, then returns a kernel that is product factorised between categorical (enumeration type variables in general)
    # and numerical ones. It aims to mitigate the curse of dimensionality for maximal information gain by reducing the function class
    if use_product_kernel:
        has_num = x  is not None and x.shape[1]  > 0
        has_enum = xe is not None and xe.shape[1] > 0
        kerns = []
        if has_num:
            ard_num_dims = x.shape[1] if ard_kernel else None
            kernel = MaternKernel(nu=1.5, ard_num_dims=ard_num_dims, active_dims=torch.arange(x.shape[1]))
            if ard_kernel:
                # if automatic relevance must be determined (it is usually critical for kernel methods to function properly)
                # otherwise one must perform adaptive scaling 
                lscales = kernel.lengthscale.detach().clone().view(1, -1)
                for i in range(x.shape[1]):
                    idx = np.random.choice(x.shape[0], min(x.shape[0], max_x), replace=False)
                    lscales[0, i] = torch.pdist(x[idx, i].view(-1, 1)).median().clamp(min=0.02)
                kernel.lengthscale = lscales
            kerns.append(kernel)
        if has_enum:
            kernel = MaternKernel(nu=1.5, active_dims=torch.arange(x.shape[1], total_dim))
            kerns.append(kernel)
        final_kern = ScaleKernel(ProductKernel(*kerns), outputscale_prior=GammaPrior(0.5, 0.5))
        final_kern.outputscale = y[torch.isfinite(y)].var()
        return final_kern
    else:
        if ard_kernel:
            kernel = ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=total_dim))
        else:
            kernel = ScaleKernel(MaternKernel(nu=1.5))
        kernel.outputscale = y[torch.isfinite(y)].var()
        return kernel


# TODO: Does not support sampling value functions from the surrogate
class BaseModel(ABC):
    support_ts = False
    support_grad = False
    support_multi_output = False
    support_warm_start = False

    def __init__(self, 
                 num_cont: int,
                 num_enum: int, 
                 num_out: int, 
                 **conf):
        self.num_cont = num_cont
        self.num_enum = num_enum
        self.num_out = num_out
        self.conf = conf
        assert(self.num_cont >= 0)
        assert(self.num_enum >= 0)
        assert(self.num_out > 0)
        assert(self.num_cont + self.num_enum > 0)
        if self.num_enum > 0:
            assert 'num_uniqs' in self.conf
            assert isinstance(self.conf['num_uniqs'], list)
            assert len(self.conf['num_uniqs']) == self.num_enum
        if not self.support_multi_output:
            assert self.num_out == 1, "Model only support single-output"

    @abstractmethod
    def fit(self,
            Xc: FloatTensor,
            Xe: LongTensor,
            y: FloatTensor):
        pass

    @abstractmethod
    def predict(self,
                Xc: FloatTensor,
                Xe: LongTensor) -> tuple[FloatTensor, FloatTensor]:
        pass


    @property
    def noise(self)->FloatTensor:
        return torch.zeros(self.num_out)

    def sample_y(self, Xc: FloatTensor, Xe: LongTensor, n_samples: int = 1) -> FloatTensor:
        py, ps2 = self.predict(Xc, Xe)
        ps = ps2.sqrt()
        samp = torch.zeros(n_samples, py.shape[0], self.num_out)
        for i in range(n_samples):
            samp[i] = py + ps * torch.randn(py.shape)
        return samp


class SVGPLayer(ApproximateGP):
    def __init__(self, mean, kern, u, learn_u, use_ngd):
        num_inducing = u.shape[0]
        if use_ngd:
            variational_distribution = (NaturalVariationalDistribution(num_inducing) 
            if u.dtype in [torch.float32, torch.float64] else TrilNaturalVariationalDistribution(num_inducing))
        else:
            variational_distribution = CholeskyVariationalDistribution(num_inducing)
        variational_strategy = VariationalStrategy(self, u, variational_distribution, learn_inducing_locations=learn_u)
        super().__init__(variational_strategy)

        self.mean = mean
        self.cov = kern

    def forward(self, x_all):
        m = self.mean(x_all)
        K = self.cov(x_all)
        return MultivariateNormal(m, K)

class SVGPList(gpytorch.Module):
    def __init__(self, gp_list):
        super().__init__()
        self.num_out = len(gp_list)
        self.gp = nn.ModuleList(gp_list)

    def __getitem__(self, i):
        return self.gp[i]
    
    def forward(self, x_all, y):
        if not self.training:
            return [self.gp[i](x_all) for i in range(self.num_out)]
        else:
            dist_list = []
            for i in range(self.num_out):
                valid_idx = torch.isfinite(y[:, i])
                dist_list.append(self.gp[i](x_all[valid_idx]))
            return dist_list


class SVGPModel(gpytorch.Module):
    def __init__(self, x, xe, y, num_inducing = 128, **conf):
        super().__init__()
        self.num_out = y.shape[1]
        self.fe = deepcopy(conf.get('fe', DummyFeatureExtractor(x.shape[1], xe.shape[1], conf.get('num_uniqs'), conf.get('emb_sizes'))))
        self.gp = SVGPList([
            SVGPLayer(
                mean = deepcopy(conf.get('mean', ConstantMean())), 
                kern = deepcopy(conf.get('kern', 
                                         default_kern(x, xe, y[:, i], self.fe.total_dim, 
                                                              conf.get('ard_kernel', True), conf.get('product_kernel', True)
                                                    )
                                        )
                                ),
                u = self.init_u(x, xe, num_inducing),
                learn_u = conf.get('learn_u', True), 
                use_ngd = conf.get('use_ngd', False), 
                )
            for i in range(self.num_out)
        ])

    def forward(self, x, xe, y = None):
        x_all = self.fe(x, xe)
        return self.gp(x_all, y)

    def init_u(self, x, xe, num_inducing):
        num_data = x.shape[0]
        # having inducing points more than data points is meaningless
        num_inducing = min(num_inducing, num_data)
        u_idx = np.random.choice(num_data, num_inducing, replace=False)
        with torch.no_grad():
            u = self.fe(x[u_idx], xe[u_idx]).detach().clone()
            return u


class SVGP(BaseModel):
    support_grad = True
    support_multi_output = True
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)

        self.use_ngd = conf.get('use_ngd', False)
        self.lr = conf.get('lr', 1e-2)
        self.lr_vp = conf.get('lr_vp', 1e-1)
        self.lr_fe = conf.get('lr_fe', 1e-3)
        self.num_inducing = conf.get('num_inducing', 128)
        self.ard_kernel = conf.get('ard_kernel', True)
        self.pred_likeli = conf.get('pred_likeli', True)
        self.beta = conf.get('beta', 1.0)

        self.batch_size = conf.get('batch_size', 64)
        self.num_epochs = conf.get('num_epochs', 300)
        self.verbose = conf.get('verbose', False)
        self.print_every = conf.get('print_every', 10)
        self.noise_lb = conf.get('noise_lb', 1e-5)
        self.xscaler = TorchMinMaxScaler((-1, 1))
        self.yscaler = TorchStandardScaler()

    def fit_scaler(self, Xc: FloatTensor, Xe: LongTensor, y: FloatTensor):
        if Xc is not None and Xc.shape[1] > 0:
            self.xscaler.fit(Xc)
        self.yscaler.fit(y)
    
    def xtrans(self, Xc: FloatTensor, Xe: LongTensor, y: FloatTensor = None):
        if Xc is not None and Xc.shape[1] > 0:
            Xc_t = self.xscaler.transform(Xc)
        else:
            Xc_t = torch.zeros(Xe.shape[0], 0)

        if Xe is None:
            Xe_t = torch.zeros(Xc.shape[0], 0).long()
        else:
            Xe_t = Xe.long()

        if y is not None:
            y_t = self.yscaler.transform(y)
            return Xc_t, Xe_t, y_t
        else:
            return Xc_t, Xe_t

    def fit(self, Xc: FloatTensor, Xe: LongTensor, y: FloatTensor):
        Xc, Xe, y = filter_nan(Xc, Xe, y, 'any')
        self.fit_scaler(Xc, Xe, y)
        Xc, Xe, y = self.xtrans(Xc, Xe, y)

        assert(Xc.shape[1] == self.num_cont)
        assert(Xe.shape[1] == self.num_enum)
        assert(y.shape[1] == self.num_out)

        n_constr = GreaterThan(self.noise_lb)
        self.gp = SVGPModel(Xc, Xe, y, **self.conf)
        self.lik = nn.ModuleList([GaussianLikelihood(noise_constraint=n_constr) for _ in range(self.num_out)])

        self.gp.train()
        self.lik.train()

        ds = TensorDataset(Xc, Xe, y)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=y.shape[0] > self.batch_size)
        if self.use_ngd:
            opt = torch.optim.Adam([
                {'params': self.gp.fe.parameters(), 'lr': self.lr_fe},
                {'params': self.gp.gp.hyperparameters()},
                {'params': self.lik.parameters()},
                ], lr=self.lr)
            opt_ng = gpytorch.optim.NGD(self.gp.variational_parameters(), lr=self.lr_vp, num_data=y.shape[0])
        else:
            opt = torch.optim.Adam([
                {'params': self.gp.fe.parameters(), 'lr': self.lr_fe},
                {'params': self.gp.gp.hyperparameters()},
                {'params': self.gp.gp.variational_parameters(), 'lr': self.lr_vp},
                {'params': self.lik.parameters()},
                ], lr=self.lr)


        mll = [gpytorch.mlls.VariationalELBO(self.lik[i], self.gp.gp[i], num_data=y.shape[0], beta=self.beta) for i in range(self.num_out)]
        for epoch in range(self.num_epochs):
            epoch_loss = 0.
            epoch_cnt = 1e-6
            for bxc, bxe, by in dl:
                dist_list = self.gp(bxc, bxe, by) 
                loss = 0
                valid = torch.isfinite(by)
                for i, dist in enumerate(dist_list):
                    loss += -1 * mll[i](dist, by[valid[:, i], i]) * valid[:, i].sum()
                loss /= by.shape[0]

                if self.use_ngd:
                    opt.zero_grad()
                    opt_ng.zero_grad()
                    loss.backward()
                    opt.step()
                    opt_ng.step()
                else:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                epoch_loss += loss.item()
                epoch_cnt += 1
            epoch_loss /= epoch_cnt
            if self.verbose and ((epoch + 1) % self.print_every == 0 or epoch == 0):
                import banditry.logging_utils as log
                log.debug('After %d epochs, loss = %g' % (epoch + 1, epoch_loss))
        self.gp.eval()
        self.lik.eval()

    def predict(self, Xc, Xe):
        Xc, Xe = self.xtrans(Xc, Xe)
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.debug(False):
            pred = self.gp(Xc, Xe)
            if self.pred_likeli:
                for i in range(self.num_out):
                    pred[i] = self.lik[i](pred[i])
            mu_ = torch.cat([pred[i].mean.reshape(-1, 1) for i in range(self.num_out)], dim=1)
            var_ = torch.cat([pred[i].variance.reshape(-1, 1) for i in range(self.num_out)], dim=1)
        mu = self.yscaler.inverse_transform(mu_)
        var = var_ * self.yscaler.std**2
        return mu, var.clamp(min=torch.finfo(var.dtype).eps)

    def sample_y(self, Xc, Xe, n_samples = 1) -> FloatTensor:

        Xc, Xe = self.xtrans(Xc, Xe)
        with gpytorch.settings.debug(False):
            pred = self.gp(Xc, Xe)
            if self.pred_likeli:
                for i in range(self.num_out):
                    pred[i] = self.lik[i](pred[i])
            samp = [pred[i].rsample(torch.Size((n_samples,))).reshape(n_samples, -1, 1) for i in range(self.num_out)]
            samp = torch.cat(samp, dim=-1)
            return self.yscaler.inverse_transform(samp)

    @property
    def noise(self):
        noise = torch.FloatTensor([lik.noise for lik in self.lik]).view(self.num_out).detach()
        return noise * self.yscaler.std**2
