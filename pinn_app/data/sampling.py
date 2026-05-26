"""训练观测点采样策略模块.

对比不同的样本点划分方法对训练时间与精度的影响.

所有策略统一返回 :class:`SamplingResult`:
    * indices       : 选中的展平索引 (相对 nz*nx 网格)
    * coords        : (N, 2) 物理坐标
    * info          : 描述性信息 (块数/远近场分布等), 用于日志与可视化标题

每种策略都是对真实数据网格的"子集选择"——网络仍会在这些点上做数据 MSE 监督.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class SamplingResult:
    indices: np.ndarray            # (M,) int64, 展平网格索引
    coords: np.ndarray             # (M, 2) float32, 物理坐标
    info: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.indices.shape[0])


# --------------------------------------------------------------------------- #
# 工具: (i_z, i_x) -> 展平索引 (网格按 row-major flatten: idx = i_z*nx + i_x)
# --------------------------------------------------------------------------- #
def _flatten_index(iz: np.ndarray, ix: np.ndarray, nx: int) -> np.ndarray:
    return iz.astype(np.int64) * int(nx) + ix.astype(np.int64)


def _phys_to_grid_idx(x: np.ndarray, z: np.ndarray, length: float, depth: float,
                      nx: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    """物理坐标 -> 最近网格行列索引."""
    ix = np.clip(np.round(x / length * (nx - 1)).astype(np.int64), 0, nx - 1)
    iz = np.clip(np.round(z / depth * (nz - 1)).astype(np.int64), 0, nz - 1)
    return iz, ix


# --------------------------------------------------------------------------- #
# 1. 均匀随机采样 (uniform random)
# --------------------------------------------------------------------------- #
def sample_uniform(
    nx: int, nz: int, length: float, depth: float,
    num_train_obs: int, seed: int = 42,
) -> SamplingResult:
    rng = np.random.default_rng(seed)
    total = nx * nz
    n = min(num_train_obs, total)
    sel = rng.choice(total, size=n, replace=False)
    iz_all = sel // nx
    ix_all = sel %  nx
    x = ix_all.astype(np.float32) / max(nx - 1, 1) * length
    z = iz_all.astype(np.float32) / max(nz - 1, 1) * depth
    coords = np.stack([x, z], axis=-1).astype(np.float32)
    return SamplingResult(
        indices=sel.astype(np.int64),
        coords=coords,
        info={"method": "uniform", "n": int(n), "total": int(total)},
    )


# --------------------------------------------------------------------------- #
# 2. 距源分层块采样 (stratified_block, 参考 train_acoustic_pinn_wangge.py)
# --------------------------------------------------------------------------- #
def sample_stratified_block(
    nx: int, nz: int, length: float, depth: float,
    source_r: float, source_z: float,
    num_blocks_x: int = 20, num_blocks_z: int = 20,
    near_dist_threshold: float = 150.0,
    mid_dist_threshold: float = 300.0,
    points_per_near_block: int = 30,
    points_per_mid_block: int = 60,
    points_per_far_block: int = 125,
    seed: int = 42,
) -> SamplingResult:
    """将域划分为 nbx × nbz 块, 按块中心到声源的距离分配采样数.

    距源越远的块, 采样越密 (远场信息量更大, 但传统网络容易学不到).
    """
    rng = np.random.default_rng(seed)
    bw = length / max(num_blocks_x, 1)
    bh = depth  / max(num_blocks_z, 1)

    near_blocks = mid_blocks = far_blocks = 0
    all_x: list[np.ndarray] = []
    all_z: list[np.ndarray] = []

    for j in range(num_blocks_z):
        z_min = j * bh
        z_max = (j + 1) * bh
        for i in range(num_blocks_x):
            x_min = i * bw
            x_max = (i + 1) * bw
            cx = 0.5 * (x_min + x_max)
            cz = 0.5 * (z_min + z_max)
            d = float(np.hypot(cx - source_r, cz - source_z))
            if d < near_dist_threshold:
                n = points_per_near_block
                near_blocks += 1
            elif d < mid_dist_threshold:
                n = points_per_mid_block
                mid_blocks += 1
            else:
                n = points_per_far_block
                far_blocks += 1
            xs = rng.uniform(x_min, x_max, n)
            zs = rng.uniform(z_min, z_max, n)
            all_x.append(xs); all_z.append(zs)

    x = np.concatenate(all_x).astype(np.float32)
    z = np.concatenate(all_z).astype(np.float32)
    iz, ix = _phys_to_grid_idx(x, z, length, depth, nx, nz)
    # 去重: 同一格点只保留一份 (避免重复 batch)
    flat = _flatten_index(iz, ix, nx)
    flat_unique, keep = np.unique(flat, return_index=True)
    coords_grid = np.stack([
        (flat_unique % nx).astype(np.float32) / max(nx - 1, 1) * length,
        (flat_unique // nx).astype(np.float32) / max(nz - 1, 1) * depth,
    ], axis=-1)
    return SamplingResult(
        indices=flat_unique.astype(np.int64),
        coords=coords_grid.astype(np.float32),
        info={
            "method": "stratified_block",
            "n": int(flat_unique.size),
            "near_blocks": near_blocks, "mid_blocks": mid_blocks,
            "far_blocks": far_blocks,
            "near_dist_threshold": near_dist_threshold,
            "mid_dist_threshold": mid_dist_threshold,
        },
    )


# --------------------------------------------------------------------------- #
# 3. Latin Hypercube Sampling
# --------------------------------------------------------------------------- #
def sample_lhs(
    nx: int, nz: int, length: float, depth: float,
    num_train_obs: int, seed: int = 42,
) -> SamplingResult:
    """LHS 在 (x, z) 上分层, 再就近映射到网格点."""
    rng = np.random.default_rng(seed)
    n = num_train_obs

    # 简易 LHS: 每个维度分 n 段, 各段内均匀采 1 个 + 随机置乱
    cut = np.linspace(0.0, 1.0, n + 1)
    u_x = rng.uniform(cut[:-1], cut[1:])
    u_z = rng.uniform(cut[:-1], cut[1:])
    rng.shuffle(u_x)
    rng.shuffle(u_z)
    x = (u_x * length).astype(np.float32)
    z = (u_z * depth).astype(np.float32)

    iz, ix = _phys_to_grid_idx(x, z, length, depth, nx, nz)
    flat = _flatten_index(iz, ix, nx)
    flat_unique = np.unique(flat)
    coords_grid = np.stack([
        (flat_unique % nx).astype(np.float32) / max(nx - 1, 1) * length,
        (flat_unique // nx).astype(np.float32) / max(nz - 1, 1) * depth,
    ], axis=-1)
    return SamplingResult(
        indices=flat_unique.astype(np.int64),
        coords=coords_grid.astype(np.float32),
        info={"method": "lhs", "n": int(flat_unique.size)},
    )


# --------------------------------------------------------------------------- #
# 4. 网格等间距 (grid_uniform)
# --------------------------------------------------------------------------- #
def sample_grid_uniform(
    nx: int, nz: int, length: float, depth: float,
    num_train_obs: int, seed: int = 42,
) -> SamplingResult:
    """等间距网格抽样: 自动决定 stride 使总点数 ≈ num_train_obs."""
    total = nx * nz
    if num_train_obs >= total:
        sel = np.arange(total, dtype=np.int64)
        stride_used = 1
    else:
        # 估计每个方向的下采样间隔
        stride_used = max(1, int(round(np.sqrt(total / max(num_train_obs, 1)))))
        iz_g, ix_g = np.meshgrid(
            np.arange(0, nz, stride_used),
            np.arange(0, nx, stride_used),
            indexing="ij",
        )
        sel = _flatten_index(iz_g.flatten(), ix_g.flatten(), nx)
    iz_all = sel // nx
    ix_all = sel %  nx
    coords = np.stack([
        ix_all.astype(np.float32) / max(nx - 1, 1) * length,
        iz_all.astype(np.float32) / max(nz - 1, 1) * depth,
    ], axis=-1).astype(np.float32)
    return SamplingResult(
        indices=sel.astype(np.int64),
        coords=coords,
        info={
            "method": "grid_uniform", "n": int(sel.size),
            "stride": int(stride_used),
        },
    )


# --------------------------------------------------------------------------- #
# 5. 分层 + 问题区域加密 (problem_region_aug)
# --------------------------------------------------------------------------- #
def sample_problem_region_aug(
    nx: int, nz: int, length: float, depth: float,
    source_r: float, source_z: float,
    num_blocks_x: int = 20, num_blocks_z: int = 20,
    near_dist_threshold: float = 150.0,
    mid_dist_threshold: float = 300.0,
    points_per_near_block: int = 30,
    points_per_mid_block: int = 60,
    points_per_far_block: int = 125,
    pr_x_min_norm: float = 0.0,  # 问题区域 (归一化 [0,1])
    pr_x_max_norm: float = 0.2,
    pr_z_min_norm: float = 0.6,
    pr_z_max_norm: float = 1.0,
    pr_extra: int = 3000,
    seed: int = 42,
) -> SamplingResult:
    """先做 stratified_block 采样, 再在 (归一化) 问题区域内补充均匀点."""
    base = sample_stratified_block(
        nx, nz, length, depth, source_r, source_z,
        num_blocks_x, num_blocks_z,
        near_dist_threshold, mid_dist_threshold,
        points_per_near_block, points_per_mid_block, points_per_far_block,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 17)
    x_extra = rng.uniform(pr_x_min_norm * length, pr_x_max_norm * length, pr_extra).astype(np.float32)
    z_extra = rng.uniform(pr_z_min_norm * depth,  pr_z_max_norm * depth,  pr_extra).astype(np.float32)
    iz, ix = _phys_to_grid_idx(x_extra, z_extra, length, depth, nx, nz)
    flat_extra = _flatten_index(iz, ix, nx)

    combined = np.unique(np.concatenate([base.indices, flat_extra]))
    coords = np.stack([
        (combined %  nx).astype(np.float32) / max(nx - 1, 1) * length,
        (combined // nx).astype(np.float32) / max(nz - 1, 1) * depth,
    ], axis=-1).astype(np.float32)
    info = dict(base.info)
    info.update({
        "method": "problem_region_aug",
        "n": int(combined.size),
        "extra_added": int(combined.size - base.size),
        "problem_region_xz_norm": (pr_x_min_norm, pr_x_max_norm,
                                   pr_z_min_norm, pr_z_max_norm),
    })
    return SamplingResult(indices=combined.astype(np.int64), coords=coords, info=info)


# --------------------------------------------------------------------------- #
# 6. 残差自适应采样 (Residual-based Adaptive Sampling, RAS) — 初始种子
# --------------------------------------------------------------------------- #
def sample_residual_adaptive(
    nx: int, nz: int, length: float, depth: float,
    num_train_obs: int, seed: int = 42,
    initial_fraction: float = 0.25,
) -> SamplingResult:
    """RAS 的**初始**观测子集: 均匀随机种子.

    训练过程中由 :meth:`Trainer._ras_refine_observations` 按 PDE/数据残差
    从候选池迭代加点, 直至达到 ``num_train_obs`` 上限.
    """
    n0 = max(100, int(num_train_obs * max(min(initial_fraction, 1.0), 0.05)))
    base = sample_uniform(nx, nz, length, depth, num_train_obs=n0, seed=seed)
    info = dict(base.info)
    info.update({
        "method": "residual_adaptive",
        "n": int(base.size),
        "target_n": int(num_train_obs),
        "initial_fraction": float(initial_fraction),
        "note": "训练中按残差自适应追加观测点",
    })
    return SamplingResult(indices=base.indices, coords=base.coords, info=info)


# --------------------------------------------------------------------------- #
# 统一入口
# --------------------------------------------------------------------------- #
def build_observation_sampling(cfg, nx: int, nz: int) -> SamplingResult:
    """根据 AppConfig 选择并执行采样策略.

    Parameters
    ----------
    cfg : AppConfig
    nx, nz : int
        真实数据网格大小 (来自 CSV)
    """
    method = (cfg.sampling_method or "uniform").lower()
    common = dict(nx=nx, nz=nz, length=cfg.length, depth=cfg.depth,
                  seed=cfg.random_seed)
    if method == "uniform":
        return sample_uniform(num_train_obs=cfg.num_train_obs, **common)
    if method in ("stratified", "stratified_block", "block"):
        return sample_stratified_block(
            source_r=cfg.source_r, source_z=cfg.source_z,
            num_blocks_x=cfg.num_blocks_x, num_blocks_z=cfg.num_blocks_z,
            near_dist_threshold=cfg.near_dist_threshold,
            mid_dist_threshold=cfg.mid_dist_threshold,
            points_per_near_block=cfg.points_per_near_block,
            points_per_mid_block=cfg.points_per_mid_block,
            points_per_far_block=cfg.points_per_far_block,
            **common,
        )
    if method == "lhs":
        return sample_lhs(num_train_obs=cfg.num_train_obs, **common)
    if method in ("grid", "grid_uniform"):
        return sample_grid_uniform(num_train_obs=cfg.num_train_obs, **common)
    if method in ("residual_adaptive", "ras", "residual"):
        return sample_residual_adaptive(
            num_train_obs=cfg.num_train_obs,
            initial_fraction=float(getattr(cfg, "ras_initial_fraction", 0.25)),
            **common,
        )
    if method in ("problem_region", "problem_region_aug", "aug"):
        return sample_problem_region_aug(
            source_r=cfg.source_r, source_z=cfg.source_z,
            num_blocks_x=cfg.num_blocks_x, num_blocks_z=cfg.num_blocks_z,
            near_dist_threshold=cfg.near_dist_threshold,
            mid_dist_threshold=cfg.mid_dist_threshold,
            points_per_near_block=cfg.points_per_near_block,
            points_per_mid_block=cfg.points_per_mid_block,
            points_per_far_block=cfg.points_per_far_block,
            pr_x_min_norm=cfg.problem_region_x_min,
            pr_x_max_norm=cfg.problem_region_x_max,
            pr_z_min_norm=cfg.problem_region_z_min,
            pr_z_max_norm=cfg.problem_region_z_max,
            pr_extra=cfg.problem_region_extra_points,
            **common,
        )
    raise ValueError(
        f"未知 sampling_method={cfg.sampling_method!r}; "
        f"可选: uniform / stratified_block / lhs / grid_uniform / "
        f"problem_region_aug / residual_adaptive"
    )


SAMPLING_METHODS = (
    "uniform", "stratified_block", "lhs", "grid_uniform",
    "problem_region_aug", "residual_adaptive",
)
