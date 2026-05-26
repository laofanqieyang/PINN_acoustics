"""长距离声场 PINN 物理公式模块.

- envelope: 包络分解 u = v·exp(i k₀ x)
- pde_residuals: Helmholtz / 包络 Helmholtz / 抛物方程 (PE) 残差
"""
from .envelope import (
    envelope_to_pressure,
    pressure_to_envelope,
    resolve_k0,
    phase_cos_sin,
)
from .pde_residuals import (
    helmholtz_residual,
    pe_residual,
    compute_pde_residual,
)

__all__ = [
    "envelope_to_pressure",
    "pressure_to_envelope",
    "resolve_k0",
    "phase_cos_sin",
    "helmholtz_residual",
    "pe_residual",
    "compute_pde_residual",
]
