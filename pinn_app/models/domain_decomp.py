"""Domain Decomposition PINN (XPINN / 重叠域拼接).

沿 x 方向划分子域, 相邻子域在**过渡带**首尾重叠; 推理时在重叠区由
**距声源更近**的子网输出 (波从近场向外传播). 训练时在过渡带施加
函数值/导数连续损失, 并可选用单向耦合: 远场子网匹配近场子网声压.

适用: length 远大于波长时, 单网难以覆盖全域.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

from .pinn import build_pinn


class DomainDecomposedPINN(nn.Module):
    """重叠域分解 PINN."""

    def __init__(
        self,
        num_subdomains: int,
        length: float,
        depth: float,
        overlap: float = 150.0,
        overlap_fraction: float = 0.25,
        source_r: float = 0.0,
        source_z: float = 0.0,
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
        if num_subdomains < 2:
            raise ValueError("DomainDecomposedPINN 至少需要 2 个子域")
        self.num_subdomains = int(num_subdomains)
        self.length = float(length)
        self.depth = float(depth)
        self.source_r = float(source_r)
        self.source_z = float(source_z)

        sw = self.length / self.num_subdomains
        self.sub_width = sw
        # 有效重叠: 显式 overlap 优先, 否则按子域宽度比例
        eff_overlap = float(overlap) if overlap > 0 else sw * float(overlap_fraction)
        self.overlap = max(eff_overlap, sw * 0.05)

        # 扩展物理边界: 子域首尾相连, 过渡带由相邻两网共同覆盖
        bounds_x: List[Tuple[float, float]] = []
        core_bounds: List[Tuple[float, float]] = []
        for i in range(self.num_subdomains):
            core_lo = i * sw
            core_hi = (i + 1) * sw if i < self.num_subdomains - 1 else self.length
            core_bounds.append((core_lo, core_hi))
            lo = max(0.0, core_lo - (self.overlap if i > 0 else 0.0))
            hi = min(
                self.length,
                core_hi + (self.overlap if i < self.num_subdomains - 1 else 0.0),
            )
            bounds_x.append((lo, hi))

        self.core_bounds = core_bounds
        self.register_buffer(
            "bounds_x_buf",
            torch.tensor(bounds_x, dtype=torch.float32),
            persistent=False,
        )
        # 子域核心中心 (用于声源距离优先)
        centers = [
            (0.5 * (lo + hi), 0.5 * self.depth) for lo, hi in core_bounds
        ]
        self.register_buffer(
            "sub_center_buf",
            torch.tensor(centers, dtype=torch.float32),
            persistent=False,
        )
        # 到声源的子域代表距离 (常数, 越小越“近场”)
        dists = [
            float(((cx - self.source_r) ** 2 + (cz - self.source_z) ** 2) ** 0.5)
            for cx, cz in centers
        ]
        self.register_buffer(
            "sub_source_dist_buf",
            torch.tensor(dists, dtype=torch.float32),
            persistent=False,
        )
        # 名义界面 (核心分界, 用于示意图)
        self.interface_x: List[float] = [
            i * sw for i in range(1, self.num_subdomains)
        ]

        nets = [
            build_pinn(
                num_layers=num_layers, num_neurons=num_neurons,
                activation=activation, network_type=network_type,
                mapping_size=mapping_size, fourier_sigma=fourier_sigma,
                siren_w0=siren_w0, device="cpu",
            )
            for _ in range(self.num_subdomains)
        ]
        self.nets = nn.ModuleList(nets)
        self.to(device)

    # ------------------------------------------------------------------ #
    @property
    def bounds_x(self) -> List[Tuple[float, float]]:
        return [
            (float(self.bounds_x_buf[i, 0].item()),
             float(self.bounds_x_buf[i, 1].item()))
            for i in range(self.num_subdomains)
        ]

    def overlap_interval(self, i: int) -> Tuple[float, float] | None:
        """子域 i 与 i+1 的过渡带 [x_lo, x_hi] (物理坐标)."""
        if i < 0 or i >= self.num_subdomains - 1:
            return None
        lo_i, hi_i = self.bounds_x[i]
        lo_j, hi_j = self.bounds_x[i + 1]
        x_lo = max(lo_i, lo_j)
        x_hi = min(hi_i, hi_j)
        if x_hi <= x_lo + 1e-6:
            return None
        return (x_lo, x_hi)

    # ------------------------------------------------------------------ #
    def coverage_mask(self, x_phys: torch.Tensor) -> torch.Tensor:
        """(N, num_subdomains) bool — 点落在各子域扩展边界内."""
        x = x_phys.unsqueeze(1)  # (N, 1)
        lo = self.bounds_x_buf[:, 0].unsqueeze(0)
        hi = self.bounds_x_buf[:, 1].unsqueeze(0)
        return (x >= lo) & (x <= hi)

    def select_subdomain_index(self, x_phys: torch.Tensor) -> torch.Tensor:
        """声源距离优先: 在覆盖该点的子域中选 sub_source_dist 最小者."""
        cov = self.coverage_mask(x_phys)  # (N, S)
        # 未覆盖点 (数值边界): 按 x 落最近核心
        if not torch.any(cov):
            idx = (x_phys / self.sub_width).long().clamp(0, self.num_subdomains - 1)
            return idx
        dists = self.sub_source_dist_buf.unsqueeze(0).expand_as(cov)
        large = torch.tensor(1e9, device=x_phys.device, dtype=dists.dtype)
        scored = torch.where(cov, dists, large)
        chosen = scored.argmin(dim=1)
        # 若某行全 False, 回退硬分区
        no_cov = ~cov.any(dim=1)
        if torch.any(no_cov):
            fallback = (x_phys / self.sub_width).long().clamp(0, self.num_subdomains - 1)
            chosen = torch.where(no_cov, fallback, chosen)
        return chosen

    # ------------------------------------------------------------------ #
    def forward(self, xz_norm_global: torch.Tensor) -> torch.Tensor:
        """重叠域前向: 过渡带由距声源更近的子网输出."""
        x_norm = xz_norm_global[:, 0]
        x_phys = (x_norm + 1.0) * (self.length / 2.0)
        sub_idx = self.select_subdomain_index(x_phys)

        out = torch.zeros(
            xz_norm_global.shape[0], 2,
            dtype=xz_norm_global.dtype, device=xz_norm_global.device,
        )
        for i in range(self.num_subdomains):
            mask = (sub_idx == i)
            if not torch.any(mask):
                continue
            sub_out = self.nets[i](xz_norm_global[mask])
            out = out.clone()
            out[mask] = sub_out
        return out

    # ------------------------------------------------------------------ #
    def forward_both(self, xz_norm_global: torch.Tensor, i: int, j: int):
        """相邻子网在同一批点上的输出 (过渡带损失)."""
        return self.nets[i](xz_norm_global), self.nets[j](xz_norm_global)

    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        params = sum(p.numel() for p in self.parameters())
        sub_summary = self.nets[0].summary() if hasattr(self.nets[0], "summary") else ""
        return (
            f"DomainDecomposedPINN(subdomains={self.num_subdomains}, "
            f"length={self.length}m, overlap={self.overlap:.1f}m, "
            f"source=({self.source_r:.1f},{self.source_z:.1f}), "
            f"params={params})\n    each sub: {sub_summary}"
        )


def build_domain_decomp_pinn(
    num_subdomains: int,
    length: float,
    depth: float,
    overlap: float = 150.0,
    overlap_fraction: float = 0.25,
    source_r: float = 0.0,
    source_z: float = 0.0,
    num_layers: int = 5,
    num_neurons: int = 128,
    activation: str = "tanh",
    network_type: str = "fourier",
    mapping_size: int = 128,
    fourier_sigma: float = 5.0,
    siren_w0: float = 15.0,
    device: torch.device | str = "cpu",
) -> DomainDecomposedPINN:
    return DomainDecomposedPINN(
        num_subdomains=num_subdomains,
        length=length, depth=depth,
        overlap=overlap, overlap_fraction=overlap_fraction,
        source_r=source_r, source_z=source_z,
        num_layers=num_layers, num_neurons=num_neurons,
        activation=activation, network_type=network_type,
        mapping_size=mapping_size, fourier_sigma=fourier_sigma,
        siren_w0=siren_w0, device=device,
    )
