"""Sequential Marching PINN — 沿传播方向 (x) 分段.

将 [0, length] 划分为若干传播段, 每段独立子网络 (可选权重共享).
重叠过渡带内由**左侧 (近源) 段**主导输出; 训练时在界面施加
声压连续 + 单向近→远传递约束.

与 Domain Decomposition 区别:
  * 强调单向传播因果 (marching), 默认左段优先而非“距源最近子域中心”
  * 可与包络分解 / PE 方程联用
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
from torch import nn

from .pinn import build_pinn


class SequentialMarchingPINN(nn.Module):
    """顺序推进式分段 PINN."""

    def __init__(
        self,
        num_segments: int,
        length: float,
        depth: float,
        segment_length: float,
        overlap: float = 50.0,
        shared_network: bool = False,
        network_type: str = "fourier",
        num_layers: int = 5,
        num_neurons: int = 128,
        activation: str = "tanh",
        mapping_size: int = 128,
        fourier_sigma: float = 5.0,
        siren_w0: float = 15.0,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        if num_segments < 2:
            raise ValueError("SequentialMarchingPINN 至少需要 2 个传播段")
        self.num_segments = int(num_segments)
        self.length = float(length)
        self.depth = float(depth)
        self.overlap = max(float(overlap), 0.0)
        self.segment_length = float(segment_length)
        self.shared_network = bool(shared_network)

        bounds_x: List[Tuple[float, float]] = []
        core_bounds: List[Tuple[float, float]] = []
        for i in range(self.num_segments):
            core_lo = i * self.segment_length
            core_hi = min((i + 1) * self.segment_length, self.length)
            if i == self.num_segments - 1:
                core_hi = self.length
            core_bounds.append((core_lo, core_hi))
            lo = max(0.0, core_lo - (self.overlap if i > 0 else 0.0))
            hi = min(
                self.length,
                core_hi + (self.overlap if i < self.num_segments - 1 else 0.0),
            )
            bounds_x.append((lo, hi))

        self.core_bounds = core_bounds
        self.register_buffer(
            "bounds_x_buf",
            torch.tensor(bounds_x, dtype=torch.float32),
            persistent=False,
        )
        self.interface_x: List[float] = [
            core_bounds[i][1] for i in range(self.num_segments - 1)
        ]

        if shared_network:
            net = build_pinn(
                num_layers=num_layers, num_neurons=num_neurons,
                activation=activation, network_type=network_type,
                mapping_size=mapping_size, fourier_sigma=fourier_sigma,
                siren_w0=siren_w0, device="cpu",
            )
            self.nets = nn.ModuleList([net] * self.num_segments)
        else:
            self.nets = nn.ModuleList([
                build_pinn(
                    num_layers=num_layers, num_neurons=num_neurons,
                    activation=activation, network_type=network_type,
                    mapping_size=mapping_size, fourier_sigma=fourier_sigma,
                    siren_w0=siren_w0, device="cpu",
                )
                for _ in range(self.num_segments)
            ])
        self.to(device)

    @property
    def bounds_x(self) -> List[Tuple[float, float]]:
        return [
            (float(self.bounds_x_buf[i, 0].item()),
             float(self.bounds_x_buf[i, 1].item()))
            for i in range(self.num_segments)
        ]

    def overlap_interval(self, i: int) -> Tuple[float, float] | None:
        if i < 0 or i >= self.num_segments - 1:
            return None
        lo_i, hi_i = self.bounds_x[i]
        lo_j, hi_j = self.bounds_x[i + 1]
        x_lo = max(lo_i, lo_j)
        x_hi = min(hi_i, hi_j)
        if x_hi <= x_lo + 1e-6:
            return None
        return (x_lo, x_hi)

    def coverage_mask(self, x_phys: torch.Tensor) -> torch.Tensor:
        x = x_phys.unsqueeze(1)
        lo = self.bounds_x_buf[:, 0].unsqueeze(0)
        hi = self.bounds_x_buf[:, 1].unsqueeze(0)
        return (x >= lo) & (x <= hi)

    def select_segment_index(self, x_phys: torch.Tensor) -> torch.Tensor:
        """重叠区: 选索引最小 (最靠声源/左侧) 的覆盖段 — marching 因果."""
        cov = self.coverage_mask(x_phys)
        n_seg = self.num_segments
        # 大索引惩罚 → 取最小可用段号
        seg_ids = torch.arange(n_seg, device=x_phys.device, dtype=torch.float32)
        large = torch.tensor(1e9, device=x_phys.device, dtype=seg_ids.dtype)
        scored = torch.where(cov, seg_ids.unsqueeze(0).expand_as(cov), large)
        chosen = scored.argmin(dim=1)
        no_cov = ~cov.any(dim=1)
        if torch.any(no_cov):
            fallback = (x_phys / max(self.segment_length, 1e-6)).long().clamp(0, n_seg - 1)
            chosen = torch.where(no_cov, fallback, chosen)
        return chosen

    def forward(self, xz_norm_global: torch.Tensor) -> torch.Tensor:
        x_norm = xz_norm_global[:, 0]
        x_phys = (x_norm + 1.0) * (self.length / 2.0)
        seg_idx = self.select_segment_index(x_phys)
        out = torch.zeros(
            xz_norm_global.shape[0], 2,
            dtype=xz_norm_global.dtype, device=xz_norm_global.device,
        )
        for i in range(self.num_segments):
            mask = (seg_idx == i)
            if not torch.any(mask):
                continue
            sub_out = self.nets[i](xz_norm_global[mask])
            out = out.clone()
            out[mask] = sub_out
        return out

    def forward_both(self, xz_norm_global: torch.Tensor, i: int, j: int):
        return self.nets[i](xz_norm_global), self.nets[j](xz_norm_global)

    def summary(self) -> str:
        params = sum(p.numel() for p in self.parameters())
        shared = "shared" if self.shared_network else "independent"
        return (
            f"SequentialMarchingPINN(segments={self.num_segments}, "
            f"seg_len={self.segment_length:.0f}m, overlap={self.overlap:.0f}m, "
            f"nets={shared}, params={params})"
        )


def build_marching_pinn(
    length: float,
    depth: float,
    num_segments: int = 0,
    segment_length: float = 200.0,
    overlap: float = 50.0,
    shared_network: bool = False,
    num_layers: int = 5,
    num_neurons: int = 128,
    activation: str = "tanh",
    network_type: str = "fourier",
    mapping_size: int = 128,
    fourier_sigma: float = 5.0,
    siren_w0: float = 15.0,
    device: torch.device | str = "cpu",
) -> SequentialMarchingPINN:
    if num_segments <= 0:
        num_segments = max(2, int(math.ceil(length / max(segment_length, 1.0))))
    return SequentialMarchingPINN(
        num_segments=num_segments,
        length=length,
        depth=depth,
        segment_length=segment_length,
        overlap=overlap,
        shared_network=shared_network,
        network_type=network_type,
        num_layers=num_layers,
        num_neurons=num_neurons,
        activation=activation,
        mapping_size=mapping_size,
        fourier_sigma=fourier_sigma,
        siren_w0=siren_w0,
        device=device,
    )
