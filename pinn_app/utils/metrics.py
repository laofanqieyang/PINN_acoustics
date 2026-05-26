"""误差指标.

统一支持 numpy 数组输入, 返回 float.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _align(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a).astype(np.float64)
    b = np.asarray(b).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return a, b


def compute_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    pred, true = _align(pred, true)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def compute_mae(pred: np.ndarray, true: np.ndarray) -> float:
    pred, true = _align(pred, true)
    return float(np.mean(np.abs(pred - true)))


def compute_relative_l2(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> float:
    """相对 L2 误差: ||pred - true||_2 / ||true||_2"""
    pred, true = _align(pred, true)
    num = np.sqrt(np.sum((pred - true) ** 2))
    den = np.sqrt(np.sum(true ** 2)) + eps
    return float(num / den)


def compute_max_error(pred: np.ndarray, true: np.ndarray) -> float:
    pred, true = _align(pred, true)
    return float(np.max(np.abs(pred - true)))


def compute_correlation(pred: np.ndarray, true: np.ndarray) -> float:
    pred, true = _align(pred, true)
    a = pred.flatten()
    b = true.flatten()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compute_all_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """一次性返回 RMSE / MAE / Relative L2 / MaxErr / Corr."""
    return {
        f"{prefix}rmse": compute_rmse(pred, true),
        f"{prefix}mae": compute_mae(pred, true),
        f"{prefix}rel_l2": compute_relative_l2(pred, true),
        f"{prefix}max_err": compute_max_error(pred, true),
        f"{prefix}corr": compute_correlation(pred, true),
    }


def compute_regional_rmse(
    pred: np.ndarray,
    true: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    source_r: float,
    source_z: float,
    near: float = 150.0,
    mid: float = 300.0,
) -> Dict[str, float]:
    """按距源距离分 near/mid/far 三档分别计算 RMSE.

    与 stratified_block 采样的分层一致, 便于论文 Table 中报告
    "近场 / 中场 / 远场 RMSE".
    """
    pred = np.asarray(pred); true = np.asarray(true)
    d = np.sqrt((X - source_r) ** 2 + (Z - source_z) ** 2)
    near_mask = d < near
    mid_mask = (d >= near) & (d < mid)
    far_mask = d >= mid

    def _rmse_masked(m: np.ndarray) -> float:
        if not m.any():
            return float("nan")
        return float(np.sqrt(np.mean((pred[m] - true[m]) ** 2)))

    return {
        "near_rmse": _rmse_masked(near_mask),
        "mid_rmse":  _rmse_masked(mid_mask),
        "far_rmse":  _rmse_masked(far_mask),
        "near_count": int(near_mask.sum()),
        "mid_count":  int(mid_mask.sum()),
        "far_count":  int(far_mask.sum()),
    }
