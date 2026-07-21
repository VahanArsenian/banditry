import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingTransform(nn.Module):
    def __init__(self, num_uniqs, **conf):
        super().__init__()
        self.emb_sizes = conf.get('emb_sizes')
        if self.emb_sizes is None:
            self.emb_sizes = [min(50, 1 + v // 2) for v in num_uniqs]
        
        self.emb = nn.ModuleList([])
        for num_uniq, emb_size in zip(num_uniqs, self.emb_sizes):
            self.emb.append(nn.Embedding(num_uniq, emb_size))

    @property
    def num_out_list(self) -> list[int]:
        return self.emb_sizes
    
    @property
    def num_out(self)->int:
        return sum(self.emb_sizes)

    def forward(self, xe):
        return torch.cat([self.emb[i](xe[:, i]).view(xe.shape[0], -1) for i in range(len(self.emb))], dim = 1)


class OneHotTransform(nn.Module):
    def __init__(self, num_uniqs):
        super().__init__()
        self.num_uniqs = num_uniqs

    @property
    def num_out_list(self) -> list[int]:
        return self.num_uniqs

    @property
    def num_out(self)->int:
        return sum(self.num_uniqs)

    def forward(self, xe):
        return torch.cat([F.one_hot(xe[:, i], self.num_uniqs[i]) for i in range(xe.shape[1])], dim = 1).float()


class TorchIdentityScaler(nn.Module):
    def __init__(self):
        super().__init__()

    def fit(self, _: torch.FloatTensor):
        return self

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x

    def transform(self, x : torch.FloatTensor) -> torch.FloatTensor:
        return self.forward(x)

    def inverse_transform(self, x : torch.FloatTensor) -> torch.FloatTensor:
        return x


class TorchStandardScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(self, x: torch.FloatTensor):
        assert(x.dim() == 2)
        with torch.no_grad():
            finite = torch.isfinite(x)
            count = finite.sum(dim=0)
            denom = count.clamp(min=1).to(dtype=x.dtype)

            safe_x = torch.where(finite, x, torch.zeros_like(x))
            mean = safe_x.sum(dim=0) / denom

            centered = torch.where(finite, x - mean.view(1, -1), torch.zeros_like(x))
            var = centered.square().sum(dim=0) / denom
            std = var.sqrt()

            invalid = (count == 0) | (~torch.isfinite(mean)) | (~torch.isfinite(std)) | (std <= 0)
            self.mean = mean.clone()
            self.std = std.clone()
            self.mean[invalid] = 0.
            self.std[invalid] = 1.
            self.fitted = True
        return self

    def _stats_for(self, x: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        if not self.fitted or self.mean is None or self.std is None:
            raise RuntimeError("TorchStandardScaler must be fitted before calling transform.")
        view_shape = [1] * (x.dim() - 1) + [-1]
        mean = self.mean.to(device=x.device, dtype=x.dtype).view(*view_shape)
        std = self.std.to(device=x.device, dtype=x.dtype).view(*view_shape)
        std = std.clamp(min=torch.finfo(x.dtype).eps)
        return mean, std

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.transform(x)

    def transform(self, x: torch.FloatTensor) -> torch.FloatTensor:
        mean, std = self._stats_for(x)
        return (x - mean) / std

    def inverse_transform(self, x: torch.FloatTensor) -> torch.FloatTensor:
        mean, std = self._stats_for(x)
        return x * std + mean

class TorchMinMaxScaler(nn.Module):
    def __init__(self, range: tuple = (0, 1)):
        super().__init__()
        self.range_lb = float(range[0])
        self.range_ub = float(range[1])
        assert (self.range_ub > self.range_lb )

        self.scale_ = None
        self.min_ = None
        self.fitted = False

    def fit(self, x: torch.FloatTensor):
        assert(x.dim() == 2)
        with torch.no_grad():
            finite = torch.isfinite(x)
            valid_count = finite.sum(dim=0)

            pos_inf = torch.full_like(x, torch.inf)
            neg_inf = torch.full_like(x, -torch.inf)
            data_min = torch.where(finite, x, pos_inf).amin(dim=0)
            data_max = torch.where(finite, x, neg_inf).amax(dim=0)

            invalid = (valid_count == 0) | (~torch.isfinite(data_min)) | (~torch.isfinite(data_max))
            data_range = data_max - data_min
            safe_range = torch.where((data_range > 0) & (~invalid), data_range, torch.ones_like(data_range))

            scale = (self.range_ub - self.range_lb) / safe_range
            min_offset = self.range_lb - data_min * scale

            self.scale_ = scale.clone()
            self.min_ = min_offset.clone()
            self.scale_[invalid] = 1.
            self.min_[invalid] = 0.
            self.fitted = True
        return self

    def _params_for(self, x: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        if not self.fitted or self.scale_ is None or self.min_ is None:
            raise RuntimeError("TorchMinMaxScaler must be fitted before calling transform.")
        view_shape = [1] * (x.dim() - 1) + [-1]
        scale = self.scale_.to(device=x.device, dtype=x.dtype).view(*view_shape)
        min_offset = self.min_.to(device=x.device, dtype=x.dtype).view(*view_shape)
        scale = scale.clamp(min=torch.finfo(x.dtype).eps)
        return scale, min_offset

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.transform(x)

    def transform(self, x: torch.FloatTensor) -> torch.FloatTensor:
        scale, min_offset = self._params_for(x)
        return scale * x + min_offset

    def inverse_transform(self, x: torch.FloatTensor) -> torch.FloatTensor:
        scale, min_offset = self._params_for(x)
        return (x - min_offset) / scale



class DummyFeatureExtractor(nn.Module):
    def __init__(self, num_cont, num_enum, num_uniqs = None, emb_sizes = None):
        super().__init__()
        self.num_cont = num_cont
        self.num_enum = num_enum
        self.total_dim = num_cont
        if num_enum > 0:
            assert num_uniqs is not None
            self.emb_trans  = EmbeddingTransform(num_uniqs, emb_sizes = emb_sizes)
            self.total_dim += self.emb_trans.num_out

    def forward(self, x: torch.FloatTensor, xe: torch.LongTensor):
        x_all = x
        if self.num_enum > 0:
            x_all = torch.cat([x, self.emb_trans(xe)], dim = 1)
        return x_all