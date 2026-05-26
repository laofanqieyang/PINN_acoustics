"""PINN PDE 残差: Helmholtz / PE (抛物方程).

均在物理坐标下用 autograd 计算, 支持包络参数化 (网络输出 v, 内部还原 u).
"""
from __future__ import annotations

from typing import Callable, Tuple

import torch


def _derivatives_pressure(
    pr: torch.Tensor,
    pi: torch.Tensor,
    coords_phys: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """一阶、二阶导数 (对物理 x,z)."""
    grad_pr = torch.autograd.grad(
        pr, coords_phys, grad_outputs=torch.ones_like(pr),
        create_graph=True, retain_graph=True,
    )[0]
    grad_pi = torch.autograd.grad(
        pi, coords_phys, grad_outputs=torch.ones_like(pi),
        create_graph=True, retain_graph=True,
    )[0]
    dpr_dx, dpr_dz = grad_pr[:, 0:1], grad_pr[:, 1:2]
    dpi_dx, dpi_dz = grad_pi[:, 0:1], grad_pi[:, 1:2]

    dpr_dxx = torch.autograd.grad(
        dpr_dx, coords_phys, grad_outputs=torch.ones_like(dpr_dx),
        create_graph=True, retain_graph=True,
    )[0][:, 0:1]
    dpr_dzz = torch.autograd.grad(
        dpr_dz, coords_phys, grad_outputs=torch.ones_like(dpr_dz),
        create_graph=True, retain_graph=True,
    )[0][:, 1:2]
    dpi_dxx = torch.autograd.grad(
        dpi_dx, coords_phys, grad_outputs=torch.ones_like(dpi_dx),
        create_graph=True, retain_graph=True,
    )[0][:, 0:1]
    dpi_dzz = torch.autograd.grad(
        dpi_dz, coords_phys, grad_outputs=torch.ones_like(dpi_dz),
        create_graph=True, retain_graph=True,
    )[0][:, 1:2]
    return dpr_dx, dpr_dz, dpr_dxx, dpr_dzz, dpi_dx, dpi_dz, dpi_dxx, dpi_dzz


def helmholtz_residual(
    pr: torch.Tensor,
    pi: torch.Tensor,
    coords_phys: torch.Tensor,
    k: float,
    source_real: torch.Tensor,
    source_imag: torch.Tensor,
) -> torch.Tensor:
    """∇²p + k²p = S, 返回 (N,2) 残差."""
    dpr_dx, dpr_dz, dpr_dxx, dpr_dzz, dpi_dx, dpi_dz, dpi_dxx, dpi_dzz = _derivatives_pressure(
        pr, pi, coords_phys,
    )
    k2 = float(k) ** 2
    res_real = dpr_dxx + dpr_dzz + k2 * pr - source_real
    res_imag = dpi_dxx + dpi_dzz + k2 * pi - source_imag
    return torch.cat([res_real, res_imag], dim=1)


def pe_residual(
    pr: torch.Tensor,
    pi: torch.Tensor,
    coords_phys: torch.Tensor,
    k: float,
    source_real: torch.Tensor,
    source_imag: torch.Tensor,
) -> torch.Tensor:
    """标准窄角抛物方程 (沿 +x 传播):

    ∂u/∂x = (i / 2k) ∂²u/∂z²

    实虚分离:
        ∂ur/∂x + (1/2k) ∂²ui/∂z² = S_r
        ∂ui/∂x - (1/2k) ∂²ur/∂z² = S_i
    """
    dpr_dx, dpr_dz, dpr_dxx, dpr_dzz, dpi_dx, dpi_dz, dpi_dxx, dpi_dzz = _derivatives_pressure(
        pr, pi, coords_phys,
    )
    coef = 1.0 / (2.0 * max(float(k), 1e-9))
    res_real = dpr_dx + coef * dpi_dzz - source_real
    res_imag = dpi_dx - coef * dpr_dzz - source_imag
    return torch.cat([res_real, res_imag], dim=1)


def compute_pde_residual(
    raw_out: torch.Tensor,
    coords_phys: torch.Tensor,
    *,
    use_envelope: bool,
    k0: float,
    k: float,
    use_pe: bool,
    source_fn: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """统一入口: raw_out 为网络 (vr,vi) 或 (pr,pi)."""
    x = coords_phys[:, 0:1]
    z = coords_phys[:, 1:2]
    vr, vi = raw_out[:, 0:1], raw_out[:, 1:2]

    if use_envelope:
        from .envelope import envelope_to_pressure
        pr, pi = envelope_to_pressure(vr, vi, x, k0)
    else:
        pr, pi = vr, vi

    s_real, s_imag = source_fn(x, z)
    if use_pe:
        return pe_residual(pr, pi, coords_phys, k, s_real, s_imag)
    return helmholtz_residual(pr, pi, coords_phys, k, s_real, s_imag)
