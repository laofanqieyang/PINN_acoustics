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
