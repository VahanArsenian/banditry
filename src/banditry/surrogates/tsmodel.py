from collections.abc import Callable
from copy import deepcopy

import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor

from banditry.variable_domains.transforms import (
    DummyFeatureExtractor,
    TorchMinMaxScaler,
    TorchStandardScaler,
)


def default_network_builder(in_dim: int, out_dim: int) -> nn.Module:
    """Default network factory for :class:`ValueFunction`.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output dimension.

    Returns:
        An ``nn.Sequential`` MLP with two hidden layers of width ``2 * in_dim``,
        each followed by ``LayerNorm`` and ``Tanh``.
    """
    # Non linear Neural net, 2 layers hidden dim 128, activation Swiglu

    layers = []
    layers.append(nn.Linear(in_dim, 2 * in_dim))
    layers.append(nn.LayerNorm(2 * in_dim))
    layers.append(nn.Tanh())

    layers.append(nn.Linear(2 * in_dim, 2 * in_dim))
    layers.append(nn.LayerNorm(2 * in_dim))
    layers.append(nn.Tanh())

    layers.append(nn.Linear(2 * in_dim, out_dim))
    return nn.Sequential(*layers)


class ValueFunction(nn.Module):
    """Neural value model whose weights Thompson sampling draws from the posterior.

    Wraps a feature extractor and a value network into a single module mapping
    ``(Xc, Xe)`` to value predictions of shape ``(n, num_out)``. Thompson sampling
    treats the network weights as random: a posterior draw over the weights yields a
    sampled value function that is then optimised over the design space. Continuous
    inputs are min-max scaled to ``[-1, 1]`` (call :meth:`fit_x_scaler` first) and
    predictions are un-scaled through an attached target scaler when one is set
    (:meth:`set_y_scaler`).

    Config keys (read in ``__init__`` via ``**conf``):
        - ``num_uniqs`` (list[int], required when ``num_enum > 0``): cardinality of each
          categorical column.
        - ``emb_sizes`` (list[int], optional): embedding sizes for the categorical columns.
        - ``fe`` (nn.Module, optional): feature-extractor override; defaults to
          :class:`DummyFeatureExtractor`.

    Args:
        num_cont: Number of continuous input columns.
        num_enum: Number of categorical (index) input columns.
        num_out: Number of outputs.
        yscaler: Optional fitted target scaler used to un-scale predictions.
        network_builder: Callable ``(in_dim, out_dim) -> nn.Module`` building the
            value network; defaults to :func:`default_network_builder`.
        **conf: See "Config keys" above.

    Raises:
        TypeError: If ``network_builder`` does not return an ``nn.Module``.
    """

    def __init__(
        self,
        num_cont: int,
        num_enum: int,
        num_out: int = 1,
        yscaler: TorchStandardScaler | None = None,
        network_builder: Callable[[int, int], nn.Module] = default_network_builder,
        **conf,
    ):
        super().__init__()
        assert num_cont >= 0
        assert num_enum >= 0
        assert num_out > 0
        assert num_cont + num_enum > 0
        if num_enum > 0:
            assert "num_uniqs" in conf
            assert isinstance(conf["num_uniqs"], list)
            assert len(conf["num_uniqs"]) == num_enum

        self.num_cont = num_cont
        self.num_enum = num_enum
        self.num_out = num_out
        self.conf = conf
        self.yscaler = deepcopy(yscaler) if yscaler is not None else TorchStandardScaler()

        self.xscaler = TorchMinMaxScaler((-1, 1))
        self.fe = deepcopy(
            conf.get(
                "fe",
                DummyFeatureExtractor(
                    num_cont,
                    num_enum,
                    conf.get("num_uniqs"),
                    conf.get("emb_sizes"),
                ),
            )
        )

        self.network_builder = network_builder
        self.net = self.network_builder(self.fe.total_dim, self.num_out)
        if not isinstance(self.net, nn.Module):
            raise TypeError("network_builder must return an nn.Module.")

    @property
    def has_y_scaler(self) -> bool:
        return self.yscaler.mean is not None and self.yscaler.std is not None

    def set_y_scaler(self, yscaler: TorchStandardScaler):
        self.yscaler = deepcopy(yscaler)
        return self

    def clear_y_scaler(self):
        self.yscaler = TorchStandardScaler()
        return self

    def fit_x_scaler(self, Xc: FloatTensor):
        if self.num_cont == 0:
            return self
        if Xc is None or Xc.shape[1] == 0:
            raise ValueError("Xc is required to fit xscaler when num_cont > 0.")
        self.xscaler.fit(Xc.float())
        return self

    @staticmethod
    def _batch_size(Xc: FloatTensor | None, Xe: LongTensor | None) -> int:
        if Xc is not None:
            return Xc.shape[0]
        if Xe is not None:
            return Xe.shape[0]
        raise ValueError("At least one of Xc/Xe must be provided.")

    @staticmethod
    def _device(Xc: FloatTensor | None, Xe: LongTensor | None) -> torch.device:
        if Xc is not None:
            return Xc.device
        if Xe is not None:
            return Xe.device
        return torch.device("cpu")

    def xtrans(
        self,
        Xc: FloatTensor | None,
        Xe: LongTensor | None,
    ) -> tuple[FloatTensor, LongTensor]:
        batch_size = self._batch_size(Xc, Xe)
        device = self._device(Xc, Xe)

        if Xc is not None and Xc.shape[1] > 0:
            Xc_t = Xc.float()
            if not self.xscaler.fitted:
                raise RuntimeError("xscaler is not fitted. Call fit_x_scaler first.")
            Xc_t = self.xscaler.transform(Xc_t)
        else:
            Xc_t = torch.zeros(batch_size, 0, device=device)

        if Xe is None:
            Xe_t = torch.zeros(batch_size, 0, device=device).long()
        else:
            Xe_t = Xe.long()
        return Xc_t, Xe_t

    def forward_scaled(self, Xc: FloatTensor | None, Xe: LongTensor | None) -> FloatTensor:
        Xc_t, Xe_t = self.xtrans(Xc, Xe)
        if Xc_t.shape[1] != self.num_cont:
            raise ValueError(f"Expected {self.num_cont} continuous features, got {Xc_t.shape[1]}.")
        if Xe_t.shape[1] != self.num_enum:
            raise ValueError(f"Expected {self.num_enum} categorical features, got {Xe_t.shape[1]}.")
        x_all = self.fe(Xc_t, Xe_t)
        return self.net(x_all)

    def _unscale_output(self, pred_scaled: FloatTensor) -> FloatTensor:
        if not self.has_y_scaler:
            return pred_scaled
        return self.yscaler.inverse_transform(pred_scaled)

    def forward(self, Xc: FloatTensor | None, Xe: LongTensor | None, *, _pre_scaled: bool = False) -> FloatTensor:
        if _pre_scaled:
            x_all = self.fe(Xc, Xe)
            pred_scaled = self.net(x_all)
        else:
            pred_scaled = self.forward_scaled(Xc, Xe)
        return self._unscale_output(pred_scaled)

    def predict(self, Xc: FloatTensor | None, Xe: LongTensor | None) -> FloatTensor:
        return self.forward(Xc, Xe)
