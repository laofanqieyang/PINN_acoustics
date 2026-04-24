"""可视化工具.

所有函数都返回 matplotlib.figure.Figure 对象, 既可以在 Streamlit 中 st.pyplot(fig)
也可以直接 fig.savefig(...) 保存到输出目录.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


# --------------------------------------------------------------------------- #
# 1. 中文字体自动检测
# --------------------------------------------------------------------------- #
def setup_chinese_font() -> str:
    """自动选择本机可用的中文字体."""
    candidates = [
        "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
        "Noto Sans CJK SC", "Noto Sans CJK", "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [chosen]
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


# --------------------------------------------------------------------------- #
# 2. 声场绘制
# --------------------------------------------------------------------------- #
def plot_field(
    X: np.ndarray,
    Z: np.ndarray,
    field: np.ndarray,
    title: str = "",
    cmap: str = "jet",
    cbar_label: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    invert_y: bool = True,
    figsize: Tuple[float, float] = (6.0, 4.5),
) -> plt.Figure:
    """绘制二维声场热力图."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(X, Z, field, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel("距离 x (m)")
    ax.set_ylabel("深度 z (m)")
    ax.set_title(title)
    if invert_y:
        ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_field_triptych(
    X: np.ndarray,
    Z: np.ndarray,
    p_real: np.ndarray,
    p_imag: np.ndarray,
    tl: np.ndarray,
    suptitle: str = "",
    tl_vmin: float = 40.0,
    tl_vmax: float = 120.0,
) -> plt.Figure:
    """实部 / 虚部 / TL 三联图."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    im0 = axes[0].pcolormesh(X, Z, p_real, cmap="RdBu_r", shading="auto")
    axes[0].set_title("声压实部")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(X, Z, p_imag, cmap="RdBu_r", shading="auto")
    axes[1].set_title("声压虚部")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(X, Z, tl, cmap="jet", shading="auto", vmin=tl_vmin, vmax=tl_vmax)
    axes[2].set_title("传输损失 TL (dB)")
    fig.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("距离 x (m)")
        ax.set_ylabel("深度 z (m)")
        ax.invert_yaxis()

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_field_comparison(
    X: np.ndarray,
    Z: np.ndarray,
    true_fields: Tuple[np.ndarray, np.ndarray, np.ndarray],
    pred_fields: Tuple[np.ndarray, np.ndarray, np.ndarray],
    tl_vmin: float = 40.0,
    tl_vmax: float = 120.0,
) -> plt.Figure:
    """真实 vs 预测 2x3 对比图."""
    pr_t, pi_t, tl_t = true_fields
    pr_p, pi_p, tl_p = pred_fields

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    titles_row = ["真实", "预测"]
    for row_idx, (pr, pi, tl) in enumerate([(pr_t, pi_t, tl_t), (pr_p, pi_p, tl_p)]):
        im0 = axes[row_idx, 0].pcolormesh(X, Z, pr, cmap="RdBu_r", shading="auto")
        axes[row_idx, 0].set_title(f"{titles_row[row_idx]} 声压实部")
        fig.colorbar(im0, ax=axes[row_idx, 0])

        im1 = axes[row_idx, 1].pcolormesh(X, Z, pi, cmap="RdBu_r", shading="auto")
        axes[row_idx, 1].set_title(f"{titles_row[row_idx]} 声压虚部")
        fig.colorbar(im1, ax=axes[row_idx, 1])

        im2 = axes[row_idx, 2].pcolormesh(
            X, Z, tl, cmap="jet", shading="auto", vmin=tl_vmin, vmax=tl_vmax
        )
        axes[row_idx, 2].set_title(f"{titles_row[row_idx]} 传输损失 (dB)")
        fig.colorbar(im2, ax=axes[row_idx, 2])

    for ax in axes.flat:
        ax.set_xlabel("距离 x (m)")
        ax.set_ylabel("深度 z (m)")
        ax.invert_yaxis()

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 3. Loss 曲线
# --------------------------------------------------------------------------- #
def plot_loss_curve(
    steps: Sequence[int],
    loss_dict: Dict[str, Sequence[float]],
    title: str = "训练损失曲线",
    log_scale: bool = True,
    figsize: Tuple[float, float] = (8.0, 5.0),
) -> plt.Figure:
    """汇总 loss 曲线 (所有分量叠加)."""
    fig, ax = plt.subplots(figsize=figsize)
    for name, values in loss_dict.items():
        if len(values) == 0:
            continue
        ax.plot(steps[: len(values)], values, label=name, linewidth=1.3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_individual_losses(
    steps: Sequence[int],
    loss_dict: Dict[str, Sequence[float]],
    log_scale: bool = True,
) -> Dict[str, plt.Figure]:
    """每个 loss 分量生成一张独立的图, 返回 {name: fig}."""
    figs: Dict[str, plt.Figure] = {}
    for name, values in loss_dict.items():
        if len(values) == 0:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(steps[: len(values)], values, color="#1f77b4", linewidth=1.4)
        ax.set_xlabel("Step")
        ax.set_ylabel(f"{name} loss")
        ax.set_title(f"{name} 损失曲线")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        figs[name] = fig
    return figs


# --------------------------------------------------------------------------- #
# 4. 误差分布 & PDE 残差
# --------------------------------------------------------------------------- #
def plot_error_map(
    X: np.ndarray,
    Z: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    title: str = "预测误差分布",
) -> plt.Figure:
    err = np.abs(pred - true)
    return plot_field(X, Z, err, title=title, cmap="viridis", cbar_label="|pred - true|")


def plot_pde_residual(
    X: np.ndarray,
    Z: np.ndarray,
    residual: np.ndarray,
    title: str = "PDE 残差",
) -> plt.Figure:
    return plot_field(X, Z, np.abs(residual), title=title, cmap="inferno",
                      cbar_label="|Helmholtz residual|")


# --------------------------------------------------------------------------- #
# 5. 剖面对比
# --------------------------------------------------------------------------- #
def plot_profile_comparison(
    coord_axis: np.ndarray,
    true_profile: np.ndarray,
    pred_profile: np.ndarray,
    xlabel: str = "深度 z (m)",
    ylabel: str = "传输损失 (dB)",
    title: str = "深度剖面对比",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(coord_axis, true_profile, label="真实", linewidth=1.6)
    ax.plot(coord_axis, pred_profile, label="预测", linewidth=1.6, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 6. 保存所有 loss 图
# --------------------------------------------------------------------------- #
def save_all_loss_figures(
    steps: Sequence[int],
    loss_dict: Dict[str, Sequence[float]],
    out_dir: str | Path,
    log_scale: bool = True,
) -> Dict[str, str]:
    """保存汇总 loss 图 + 每个分量单独图到 out_dir, 返回 {name: filepath}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, str] = {}

    fig_all = plot_loss_curve(steps, loss_dict, log_scale=log_scale)
    p = out_dir / "loss_all.png"
    fig_all.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig_all)
    saved["all"] = str(p)

    figs = plot_individual_losses(steps, loss_dict, log_scale=log_scale)
    for name, fig in figs.items():
        p = out_dir / f"loss_{name}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved[name] = str(p)

    return saved


# --------------------------------------------------------------------------- #
# 7. 传输损失计算的小工具 (也放在这里方便绘图)
# --------------------------------------------------------------------------- #
def compute_tl(p_real: np.ndarray, p_imag: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    amp = np.sqrt(p_real ** 2 + p_imag ** 2)
    return -20.0 * np.log10(amp + eps)
