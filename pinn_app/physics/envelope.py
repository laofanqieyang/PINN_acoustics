"""包络分解 (Envelope Decomposition).

u(x, z) = v(x, z) · exp(i k₀ x)

网络学习慢变包络 v = v_real + i·v_imag; 物理声压 u 由相位因子还原.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch


def resolve_k0(
    frequency: float,
    sound_speed: float,
    envelope_k0: float = 0.0,
    wave_number_formula: str = "legacy_f_over_c",
) -> float:
    """参考传播波数 k₀. envelope_k0>0 时优先使用用户指定值."""
    if envelope_k0 > 0:
        return float(envelope_k0)
    if (wave_number_formula or "").lower() in ("2pi_f_over_c", "2pi", "standard"):
        return 2.0 * math.pi * float(frequency) / max(float(sound_speed), 1e-9)
    # 与历史 Helmholtz 代码一致 (f/c), 便于对比旧实验
    return float(frequency) / max(float(sound_speed), 1e-9)


def resolve_k(
    frequency: float,
    sound_speed: float,
    wave_number_formula: str = "legacy_f_over_c",
) -> float:
    """PDE 中使用的波数 k (可与 k₀ 公式独立配置)."""
    return resolve_k0(frequency, sound_speed, 0.0, wave_number_formula)


def phase_cos_sin(x: torch.Tensor, k0: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """x: (N,1) 物理坐标 → cos(k₀x), sin(k₀x)."""
    phase = k0 * x
    return torch.cos(phase), torch.sin(phase)


def envelope_to_pressure(
    vr: torch.Tensor,
    vi: torch.Tensor,
    x: torch.Tensor,
    k0: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """v → 物理声压 (p_real, p_imag)."""
    c, s = phase_cos_sin(x, k0)
    pr = vr * c - vi * s
    pi = vr * s + vi * c
    return pr, pi


def pressure_to_envelope(
    pr: torch.Tensor,
    pi: torch.Tensor,
    x: torch.Tensor,
    k0: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """物理声压 → 包络 (用于包络模式下的数据监督)."""
    c, s = phase_cos_sin(x, k0)
    vr = pr * c + pi * s
    vi = pi * c - pr * s
    return vr, vi
