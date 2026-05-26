"""PINN 神经网络模型库.

提供 4 种适用于声场 Helmholtz PDE 的架构, 按需切换:

    * "dnn"     : 基线全连接 + Tanh
    * "fourier" : Random Fourier Feature + Tanh MLP   (默认推荐)
    * "siren"   : sin 激活 + 特殊初始化, 高阶导数天然光滑
    * "modified": Wang 2021 Modified MLP (双编码器门控)

输入:  归一化后的坐标 (x_norm, z_norm) ∈ [-1, 1]^2
输出:  复声压 (p_real, p_imag)
"""
from __future__ import annotations

import math
from typing import List

import torch
from torch import nn


# =========================================================================== #
# 1. 基线 DNN (Tanh MLP)
# =========================================================================== #
class PINN(nn.Module):
    """全连接 PINN (baseline).

    Parameters
    ----------
    num_layers : int
        隐藏层数量 (不含输入层/输出层).
    num_neurons : int
        每个隐藏层的神经元数.
    activation : str
        "tanh" (默认) | "sin" | "gelu" | "relu"
    """

    def __init__(
        self,
        num_layers: int = 7,
        num_neurons: int = 50,
        activation: str = "tanh",
        in_dim: int = 2,
        out_dim: int = 2,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.num_neurons = int(num_neurons)
        self.activation_name = activation

        act = _get_activation(activation)
        sizes: List[int] = [in_dim] + [num_neurons] * num_layers + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)
        self.apply(_xavier_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def summary(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return (f"PINN(dnn, layers={self.num_layers}, neurons={self.num_neurons}, "
                f"act={self.activation_name}, params={n})")


# =========================================================================== #
# 2. Fourier Feature PINN  (Tancik et al. 2020)
# =========================================================================== #
class FourierFeatureMapping(nn.Module):
    """γ(v) = [cos(2π B v), sin(2π B v)]

    B 由 N(0, sigma^2) 采样, 训练过程固定不变.
    - sigma 越大 -> 表达高频能力越强, 但过大会引入噪声; 经验值 1.0 ~ 10.0.
    - mapping_size 即 B 的输出维度 m, 最终特征维度为 2m.
    """

    def __init__(self, in_dim: int = 2, mapping_size: int = 128, sigma: float = 5.0):
        super().__init__()
        B = torch.randn(in_dim, mapping_size) * sigma
        self.register_buffer("B", B)  # 不可训练
        self.mapping_size = mapping_size
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2.0 * math.pi * x @ self.B  # (N, m)
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)  # (N, 2m)


class FourierFeaturePINN(nn.Module):
    """Random Fourier Feature → Tanh MLP → 输出.

    对 Helmholtz 高频振荡 (远场) 精度提升显著 (声场 TL 的 RMSE 通常降 3-10×).
    """

    def __init__(
        self,
        num_layers: int = 7,
        num_neurons: int = 128,
        mapping_size: int = 128,
        sigma: float = 5.0,
        in_dim: int = 2,
        out_dim: int = 2,
        activation: str = "tanh",
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.num_neurons = int(num_neurons)
        self.mapping_size = int(mapping_size)
        self.sigma = float(sigma)
        self.activation_name = activation

        self.ff = FourierFeatureMapping(in_dim, mapping_size, sigma)
        feature_dim = 2 * mapping_size
        act = _get_activation(activation)

        sizes = [feature_dim] + [num_neurons] * num_layers + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)
        self.apply(_xavier_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.ff(x))

    def summary(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return (f"FourierFeaturePINN(mapping={self.mapping_size}, sigma={self.sigma}, "
                f"layers={self.num_layers}, neurons={self.num_neurons}, params={n})")


# =========================================================================== #
# 3. SIREN  (Sitzmann et al. 2020)
# =========================================================================== #
class SIRENLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 is_first: bool = False, w0: float = 30.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.w0 = w0
        self.is_first = is_first
        self._init()

    def _init(self) -> None:
        with torch.no_grad():
            if self.is_first:
                # 首层 U(-1/in, 1/in)
                self.linear.weight.uniform_(-1.0 / self.linear.in_features,
                                            1.0 / self.linear.in_features)
            else:
                # 后续层 U(-sqrt(6/in)/w0, +sqrt(6/in)/w0)
                bound = math.sqrt(6.0 / self.linear.in_features) / self.w0
                self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


class SIREN(nn.Module):
    """SIREN (Sitzmann et al. 2020) - hybrid 频率配置.

    - 首层 w0 = w0_first (大), 让网络快速捕获输入空间的多频结构.
    - 后续层 w0 = w0_hidden (默认 1.0), 让网络自适应学习内部频率,
      避免高 w0 带来的 PDE 残差爆炸 + 常数解陷阱.
    - 输出层用普通 Xavier 初始化 (它是无 sin 的线性层).

    经验:
        声场 (波数 k≈8 归一化后) 推荐 w0_first ∈ [10, 20].
        过大 (w0_first=30) 容易让 PDE loss 数值爆炸 -> 网络塌缩为常数.
    """

    def __init__(
        self,
        num_layers: int = 6,
        num_neurons: int = 128,
        in_dim: int = 2,
        out_dim: int = 2,
        w0_first: float = 15.0,
        w0_hidden: float = 1.0,
        output_scale: float = 1.0,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.num_neurons = int(num_neurons)
        self.w0_first = float(w0_first)
        self.w0_hidden = float(w0_hidden)
        self.output_scale = float(output_scale)

        layers: List[nn.Module] = [
            SIRENLayer(in_dim, num_neurons, is_first=True, w0=w0_first)
        ]
        for _ in range(num_layers - 1):
            layers.append(SIRENLayer(num_neurons, num_neurons,
                                     is_first=False, w0=w0_hidden))
        self.hidden = nn.Sequential(*layers)

        # 输出层: 普通 Linear, 用标准 Xavier 初始化
        self.out_layer = nn.Linear(num_neurons, out_dim)
        nn.init.xavier_uniform_(self.out_layer.weight)
        if self.out_layer.bias is not None:
            nn.init.zeros_(self.out_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_scale * self.out_layer(self.hidden(x))

    def summary(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return (f"SIREN(layers={self.num_layers}, neurons={self.num_neurons}, "
                f"w0_first={self.w0_first}, w0_hidden={self.w0_hidden}, "
                f"out_scale={self.output_scale}, params={n})")


# =========================================================================== #
# 4. Modified MLP  (Wang et al., JCP 2021)
# =========================================================================== #
class ModifiedMLP(nn.Module):
    """双编码器门控 MLP.

        U = σ(W_u x + b_u)
        V = σ(W_v x + b_v)
        h_0 = σ(W_0 x + b_0)
        h_{l+1} = (1 - f_l) ⊙ U + f_l ⊙ V,  f_l = σ(W_l h_l + b_l)
        y = W_out h_L

    经验上 PINN 收敛速度比普通 MLP 快 1.5~2×, 最终精度略好, 非常稳定.
    """

    def __init__(
        self,
        num_layers: int = 7,
        num_neurons: int = 128,
        in_dim: int = 2,
        out_dim: int = 2,
        activation: str = "tanh",
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.num_neurons = int(num_neurons)
        self.activation_name = activation

        act = _get_activation(activation)
        self.act = act()

        self.encoder_u = nn.Linear(in_dim, num_neurons)
        self.encoder_v = nn.Linear(in_dim, num_neurons)
        self.first = nn.Linear(in_dim, num_neurons)
        self.hidden = nn.ModuleList(
            [nn.Linear(num_neurons, num_neurons) for _ in range(num_layers - 1)]
        )
        self.out_layer = nn.Linear(num_neurons, out_dim)
        self.apply(_xavier_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.act(self.encoder_u(x))
        v = self.act(self.encoder_v(x))
        h = self.act(self.first(x))
        for layer in self.hidden:
            f = self.act(layer(h))
            h = (1.0 - f) * u + f * v
        return self.out_layer(h)

    def summary(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return (f"ModifiedMLP(layers={self.num_layers}, neurons={self.num_neurons}, "
                f"act={self.activation_name}, params={n})")


# =========================================================================== #
# 激活与初始化工具
# =========================================================================== #
class Sine(nn.Module):
    """用于 activation='sin' 的普通 sin 激活 (非 SIREN)."""
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


def _get_activation(name: str):
    name = name.lower()
    if name == "tanh":
        return nn.Tanh
    if name == "gelu":
        return nn.GELU
    if name in ("sin", "sine"):
        return Sine
    if name == "relu":
        return nn.ReLU
    if name == "silu" or name == "swish":
        return nn.SiLU
    raise ValueError(f"未知的激活函数: {name}")


def _xavier_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# =========================================================================== #
# 工厂函数
# =========================================================================== #
def build_pinn(
    num_layers: int,
    num_neurons: int,
    activation: str = "tanh",
    device: torch.device | str = "cpu",
    network_type: str = "fourier",
    mapping_size: int = 128,
    fourier_sigma: float = 5.0,
    siren_w0: float = 30.0,
) -> nn.Module:
    """统一工厂.

    Parameters
    ----------
    network_type : str
        "dnn" | "fourier" | "siren" | "modified"
    mapping_size / fourier_sigma :
        Fourier 特征参数 (仅 network_type='fourier' 有效).
    siren_w0 :
        SIREN 频率系数 (仅 network_type='siren' 有效).
    """
    nt = network_type.lower()
    if nt == "dnn":
        model = PINN(num_layers=num_layers, num_neurons=num_neurons, activation=activation)
    elif nt == "fourier":
        model = FourierFeaturePINN(
            num_layers=num_layers, num_neurons=num_neurons,
            mapping_size=mapping_size, sigma=fourier_sigma, activation=activation,
        )
    elif nt == "siren":
        # 论文 hybrid 配置: 首层 w0=siren_w0 大频率, 后续层 w0=1.0 让网络自学习频率.
        model = SIREN(
            num_layers=num_layers, num_neurons=num_neurons,
            w0_first=siren_w0, w0_hidden=1.0, output_scale=1.0,
        )
    elif nt in ("modified", "modified_mlp", "mmlp"):
        model = ModifiedMLP(num_layers=num_layers, num_neurons=num_neurons, activation=activation)
    else:
        raise ValueError(
            f"未知 network_type={network_type!r}; "
            f"可选: dnn / fourier / siren / modified"
        )
    return model.to(device)
