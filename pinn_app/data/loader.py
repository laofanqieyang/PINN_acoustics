"""数据加载模块

负责:
    * 读取声压实部/虚部 CSV 文件
    * 生成 (x, z) 网格
    * 构造训练所需的 tensor (含归一化)
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch


# --------------------------------------------------------------------------- #
# 1. 声压文件读取
# --------------------------------------------------------------------------- #
def _sniff_csv_format(path: Path) -> tuple[bool, str]:
    """嗅探 CSV 是否带表头, 以及分隔符.

    Returns
    -------
    has_header : bool
    sep        : str  (逗号/制表符/空白)
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
    # 猜分隔符
    if "\t" in first_line:
        sep = "\t"
    elif "," in first_line:
        sep = ","
    elif ";" in first_line:
        sep = ";"
    else:
        sep = r"\s+"  # 空白分隔
    # 检查首行是否全为数字
    if sep == r"\s+":
        tokens = first_line.split()
    else:
        tokens = [t.strip() for t in first_line.split(sep)]
    has_header = False
    for t in tokens:
        if t == "":
            continue
        try:
            float(t)
        except ValueError:
            has_header = True
            break
    return has_header, sep


def _read_numeric_csv(path: Path) -> np.ndarray:
    """智能读取数值型 CSV, 自动识别表头和分隔符."""
    has_header, sep = _sniff_csv_format(path)
    df = pd.read_csv(
        path,
        header=0 if has_header else None,
        sep=sep,
        engine="python" if sep == r"\s+" else "c",
    )
    # 有些 CSV 会带"索引列", 再过滤一遍非数值列
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")  # 删全 NaN 列
    if df.isna().any().any():
        print(f"[loader] 警告: {path.name} 存在非数值单元, 已用 0 填充.")
        df = df.fillna(0.0)
    arr = df.values.astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{path}: 读取后不是二维数组, 实际 shape={arr.shape}")
    return arr


def load_pressure_data(
    real_path: str | Path,
    imag_path: str | Path,
    expected_shape: Tuple[int, int] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """读取声压实部与虚部文件。

    Parameters
    ----------
    real_path, imag_path : str | Path
        CSV 文件路径，格式为 (nz, nx) 的二维数组（无表头）。
    expected_shape : (nz, nx), 可选
        如给出则校验/警告维度不匹配。

    Returns
    -------
    p_real, p_imag : np.ndarray, shape=(nz, nx)
    """
    real_path = Path(real_path)
    imag_path = Path(imag_path)
    if not real_path.exists():
        raise FileNotFoundError(f"声压实部文件不存在: {real_path}")
    if not imag_path.exists():
        raise FileNotFoundError(f"声压虚部文件不存在: {imag_path}")

    p_real = _read_numeric_csv(real_path)
    p_imag = _read_numeric_csv(imag_path)

    if p_real.shape != p_imag.shape:
        raise ValueError(
            f"实部与虚部数据形状不匹配: {p_real.shape} vs {p_imag.shape}"
        )

    if expected_shape is not None and tuple(expected_shape) != p_real.shape:
        # 不强制报错，仅给出提示——用户可能在 UI 中填入与文件不同的 (nx, nz)
        print(
            f"[loader] 警告: 文件分辨率 {p_real.shape} 与输入的 (nz, nx)={expected_shape} 不一致，"
            f"将以文件分辨率为准。"
        )
    return p_real, p_imag


# --------------------------------------------------------------------------- #
# 2. 网格生成
# --------------------------------------------------------------------------- #
def generate_grid(
    length: float,
    depth: float,
    nx: int,
    nz: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成物理坐标网格。

    Returns
    -------
    X, Z : np.ndarray, shape=(nz, nx)
        meshgrid 后的网格坐标。
    x_flat, z_flat : np.ndarray, shape=(nz*nx,)
        展平后的坐标。
    """
    x = np.linspace(0.0, length, nx, dtype=np.float32)
    z = np.linspace(0.0, depth, nz, dtype=np.float32)
    X, Z = np.meshgrid(x, z)
    return X, Z, X.flatten(), Z.flatten()


# --------------------------------------------------------------------------- #
# 3. 归一化工具
# --------------------------------------------------------------------------- #
class MinMaxNormalizer:
    """把物理坐标 (x, z) 线性映射到 [-1, 1]。

    网络输入使用归一化坐标，物理层面的梯度计算会额外乘以雅可比系数。
    """

    def __init__(self, length: float, depth: float):
        self.length = float(length)
        self.depth = float(depth)

    def encode(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        x_n = 2.0 * x / self.length - 1.0
        z_n = 2.0 * z / self.depth - 1.0
        return np.stack([x_n, z_n], axis=-1).astype(np.float32)

    def encode_torch(self, xz: torch.Tensor) -> torch.Tensor:
        """xz shape = (N, 2), 物理坐标 -> 归一化坐标"""
        scale = torch.tensor(
            [2.0 / self.length, 2.0 / self.depth],
            dtype=xz.dtype,
            device=xz.device,
        )
        shift = torch.tensor([1.0, 1.0], dtype=xz.dtype, device=xz.device)
        return xz * scale - shift

    @property
    def jacobian(self) -> Tuple[float, float]:
        """d(normalized)/d(physical)"""
        return 2.0 / self.length, 2.0 / self.depth


# --------------------------------------------------------------------------- #
# 4. 构建训练张量
# --------------------------------------------------------------------------- #
def build_training_tensors(
    p_real: np.ndarray,
    p_imag: np.ndarray,
    length: float,
    depth: float,
    device: torch.device,
) -> dict:
    """将二维声压场展开为 (N, 2) 坐标 + (N, 1) 真值。

    Returns
    -------
    dict with keys:
        X, Z                : meshgrid (np)
        coords_phys         : (N, 2) 物理坐标 (np)
        coords_tensor       : (N, 2) 物理坐标 (torch, on device)
        p_real_tensor       : (N, 1) torch
        p_imag_tensor       : (N, 1) torch
        normalizer          : MinMaxNormalizer
        shape               : (nz, nx)
    """
    nz, nx = p_real.shape
    X, Z, x_flat, z_flat = generate_grid(length, depth, nx, nz)

    coords_phys = np.stack([x_flat, z_flat], axis=-1).astype(np.float32)
    normalizer = MinMaxNormalizer(length, depth)

    coords_tensor = torch.from_numpy(coords_phys).to(device)
    p_real_tensor = torch.from_numpy(p_real.flatten().astype(np.float32))[:, None].to(device)
    p_imag_tensor = torch.from_numpy(p_imag.flatten().astype(np.float32))[:, None].to(device)

    return {
        "X": X,
        "Z": Z,
        "coords_phys": coords_phys,
        "coords_tensor": coords_tensor,
        "p_real_tensor": p_real_tensor,
        "p_imag_tensor": p_imag_tensor,
        "normalizer": normalizer,
        "shape": (nz, nx),
    }


def sample_boundary_points(
    length: float,
    depth: float,
    n_per_edge: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按 4 条边均匀采样边界点。

    Returns
    -------
    top, bottom, left, right : (n_per_edge, 2) 物理坐标
    """
    xs = np.linspace(0.0, length, n_per_edge, dtype=np.float32)
    zs = np.linspace(0.0, depth, n_per_edge, dtype=np.float32)
    top = np.stack([xs, np.zeros_like(xs)], axis=-1)
    bottom = np.stack([xs, np.full_like(xs, depth)], axis=-1)
    left = np.stack([np.zeros_like(zs), zs], axis=-1)
    right = np.stack([np.full_like(zs, length), zs], axis=-1)
    return top, bottom, left, right


def sample_collocation_points(
    length: float,
    depth: float,
    num_points: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """域内随机 PDE 配点 (物理坐标)。"""
    rng = rng or np.random.default_rng()
    x = rng.uniform(0.0, length, num_points).astype(np.float32)
    z = rng.uniform(0.0, depth, num_points).astype(np.float32)
    return np.stack([x, z], axis=-1)
