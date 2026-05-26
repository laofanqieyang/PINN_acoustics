"""可视化工具.

所有绘图函数返回 :class:`matplotlib.figure.Figure`, 既可 ``st.pyplot(fig)`` 也可 ``fig.savefig(...)``.
本模块在导入时会自动应用 :func:`setup_publication_style`, 让所有图表满足 SCI 论文标准:
    * DPI 300, 紧凑布局
    * 字体大小 / 线宽统一, 标题加粗
    * 刻度向外、双侧对称
    * 颜色条规范、科学计数法
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib.ticker import ScalarFormatter
import numpy as np


# =========================================================================== #
# 1. 字体 + 发表级 rcParams
# =========================================================================== #
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


def setup_publication_style(
    base_font_size: int = 11,
    title_size: int = 13,
    tick_size: int = 10,
    line_width: float = 1.4,
    axes_lw: float = 1.2,
) -> None:
    """统一应用 SCI 论文级别的 matplotlib 参数.

    特点:
        * 较小但清晰的字体, 适合 1-2 栏图
        * 紧凑边距, 适合 PNG / PDF 双格式输出
        * 颜色条小而精, 数字用科学计数法
        * 双侧刻度, 向外, 不破坏热力图
    """
    setup_chinese_font()
    mpl.rcParams.update({
        # 字体 / 大小
        "font.size":            base_font_size,
        "axes.titlesize":       title_size,
        "axes.titleweight":     "bold",
        "axes.labelsize":       base_font_size,
        "axes.labelweight":     "normal",
        "xtick.labelsize":      tick_size,
        "ytick.labelsize":      tick_size,
        "legend.fontsize":      tick_size,
        "figure.titlesize":     title_size + 1,
        "figure.titleweight":   "bold",

        # 关闭 axis tick formatter 的 mathtext (中文字体不带数学字符集,
        # 会导致 ParseException 渲染崩溃)
        "axes.formatter.use_mathtext": False,
        "axes.unicode_minus":   False,   # 用 ASCII '-' 而不是 Unicode minus

        # 线宽 / 坐标轴
        "lines.linewidth":      line_width,
        "axes.linewidth":       axes_lw,
        "patch.linewidth":      0.8,
        "xtick.major.width":    axes_lw,
        "ytick.major.width":    axes_lw,
        "xtick.minor.width":    axes_lw * 0.7,
        "ytick.minor.width":    axes_lw * 0.7,
        "xtick.major.size":     4.0,
        "ytick.major.size":     4.0,
        "xtick.minor.size":     2.0,
        "ytick.minor.size":     2.0,

        # 刻度方向 (向外 = 论文常见)
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.top":            False,
        "ytick.right":          False,

        # 图像 / 保存
        "figure.dpi":           110,        # 屏幕显示
        "savefig.dpi":          300,        # 保存
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.08,
        "savefig.facecolor":    "white",
        "figure.facecolor":     "white",
        "figure.constrained_layout.use": False,

        # 网格 (默认关闭, 由函数显式打开)
        "axes.grid":            False,
        "grid.alpha":           0.25,
        "grid.linewidth":       0.6,
        "grid.linestyle":       "--",

        # 图例
        "legend.frameon":       True,
        "legend.framealpha":    0.9,
        "legend.edgecolor":     "0.7",
        "legend.fancybox":      False,
    })


# 模块加载时立即应用一次
setup_publication_style()


def save_figure_publication(
    fig: plt.Figure,
    path: str | Path,
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
    close: bool = True,
) -> List[str]:
    """以发表质量保存 Figure. 支持同时输出多种格式 (png / pdf / svg / eps).

    Parameters
    ----------
    path : str | Path
        基础路径 (不含扩展名, 或含 .png 等已知扩展名).
    formats : 序列, 默认 ("png",)
        如 ("png", "pdf") 同时输出两种格式, 便于直接放入论文.
    """
    path = Path(path)
    if path.suffix.lower() in {".png", ".pdf", ".svg", ".eps", ".jpg", ".jpeg"}:
        base = path.with_suffix("")
        formats = [path.suffix.lower().lstrip(".")] if not formats else formats
    else:
        base = path
    written: List[str] = []
    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        p = base.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.08,
                    facecolor="white")
        written.append(str(p))
    if close:
        plt.close(fig)
    return written


# =========================================================================== #
# 2. 颜色条 / 刻度小工具
# =========================================================================== #
def _attach_colorbar(fig, ax, mappable, label: str = "",
                     shrink: float = 0.85, pad: float = 0.04,
                     use_sci: bool = True) -> "plt.cm.ScalarMappable":
    cbar = fig.colorbar(mappable, ax=ax, shrink=shrink, pad=pad)
    if label:
        cbar.set_label(label, fontsize=mpl.rcParams["axes.labelsize"])
    if use_sci:
        try:
            # useMathText=False 避免中文字体环境下 mathtext 渲染崩溃
            sf = ScalarFormatter(useMathText=False, useOffset=False)
            sf.set_powerlimits((-3, 4))
            cbar.ax.yaxis.set_major_formatter(sf)
            cbar.ax.yaxis.get_offset_text().set_fontsize(
                mpl.rcParams["xtick.labelsize"] - 1
            )
        except Exception:
            pass
    cbar.outline.set_linewidth(mpl.rcParams["axes.linewidth"])
    cbar.ax.tick_params(width=mpl.rcParams["axes.linewidth"])
    return cbar


def _set_axes_aspect_equal(ax) -> None:
    """对于物理空间图设置等比例, 避免畸形."""
    try:
        ax.set_aspect("equal", adjustable="box")
    except Exception:
        pass


# =========================================================================== #
# 3. 声场绘制
# =========================================================================== #
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
    figsize: Tuple[float, float] = (6.5, 4.8),
    equal_aspect: bool = False,
) -> plt.Figure:
    """绘制二维声场热力图."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(X, Z, field, cmap=cmap, shading="auto",
                       vmin=vmin, vmax=vmax, rasterized=True)
    _attach_colorbar(fig, ax, im, label=cbar_label)
    ax.set_xlabel("距离 x (m)")
    ax.set_ylabel("深度 z (m)")
    ax.set_title(title)
    if invert_y:
        ax.invert_yaxis()
    if equal_aspect:
        _set_axes_aspect_equal(ax)
    fig.tight_layout()
    return fig


def plot_field_triptych(
    X: np.ndarray,
    Z: np.ndarray,
    p_real: np.ndarray,
    p_imag: np.ndarray,
    tl: np.ndarray,
    suptitle: str = "",
    tl_vmin: Optional[float] = 40.0,
    tl_vmax: Optional[float] = 120.0,
    figsize: Tuple[float, float] = (15.5, 4.5),
) -> plt.Figure:
    """实部 / 虚部 / TL 三联图 (论文常用横向布局)."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 实部 / 虚部使用对称色阶 (RdBu_r), 中心 0
    p_real_lim = float(np.nanmax(np.abs(p_real)))
    p_imag_lim = float(np.nanmax(np.abs(p_imag)))

    im0 = axes[0].pcolormesh(X, Z, p_real, cmap="RdBu_r", shading="auto",
                             vmin=-p_real_lim, vmax=p_real_lim, rasterized=True)
    axes[0].set_title("声压实部 Re(p)")
    _attach_colorbar(fig, axes[0], im0)

    im1 = axes[1].pcolormesh(X, Z, p_imag, cmap="RdBu_r", shading="auto",
                             vmin=-p_imag_lim, vmax=p_imag_lim, rasterized=True)
    axes[1].set_title("声压虚部 Im(p)")
    _attach_colorbar(fig, axes[1], im1)

    im2 = axes[2].pcolormesh(X, Z, tl, cmap="jet", shading="auto",
                             vmin=tl_vmin, vmax=tl_vmax, rasterized=True)
    axes[2].set_title("传输损失 TL (dB)")
    _attach_colorbar(fig, axes[2], im2, label="TL (dB)")

    for ax in axes:
        ax.set_xlabel("距离 x (m)")
        ax.set_ylabel("深度 z (m)")
        ax.invert_yaxis()

    if suptitle:
        fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    return fig


def plot_field_comparison(
    X: np.ndarray,
    Z: np.ndarray,
    true_fields: Tuple[np.ndarray, np.ndarray, np.ndarray],
    pred_fields: Tuple[np.ndarray, np.ndarray, np.ndarray],
    tl_vmin: Optional[float] = 40.0,
    tl_vmax: Optional[float] = 120.0,
    figsize: Tuple[float, float] = (16.5, 9.0),
) -> plt.Figure:
    """真实 vs 预测 2x3 对比图. 行: 真实 / 预测; 列: 实部 / 虚部 / TL."""
    pr_t, pi_t, tl_t = true_fields
    pr_p, pi_p, tl_p = pred_fields

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    titles_row = ["Ground Truth", "PINN Prediction"]

    # 行内统一色阶, 避免左右对比时颜色不一致造成误导
    pr_lim = float(max(np.nanmax(np.abs(pr_t)), np.nanmax(np.abs(pr_p))))
    pi_lim = float(max(np.nanmax(np.abs(pi_t)), np.nanmax(np.abs(pi_p))))

    for row_idx, (pr, pi, tl) in enumerate([(pr_t, pi_t, tl_t), (pr_p, pi_p, tl_p)]):
        im0 = axes[row_idx, 0].pcolormesh(X, Z, pr, cmap="RdBu_r", shading="auto",
                                          vmin=-pr_lim, vmax=pr_lim, rasterized=True)
        axes[row_idx, 0].set_title(f"{titles_row[row_idx]}: Re(p)")
        _attach_colorbar(fig, axes[row_idx, 0], im0)

        im1 = axes[row_idx, 1].pcolormesh(X, Z, pi, cmap="RdBu_r", shading="auto",
                                          vmin=-pi_lim, vmax=pi_lim, rasterized=True)
        axes[row_idx, 1].set_title(f"{titles_row[row_idx]}: Im(p)")
        _attach_colorbar(fig, axes[row_idx, 1], im1)

        im2 = axes[row_idx, 2].pcolormesh(
            X, Z, tl, cmap="jet", shading="auto",
            vmin=tl_vmin, vmax=tl_vmax, rasterized=True,
        )
        axes[row_idx, 2].set_title(f"{titles_row[row_idx]}: TL (dB)")
        _attach_colorbar(fig, axes[row_idx, 2], im2, label="TL (dB)")

    for ax in axes.flat:
        ax.set_xlabel("距离 x (m)")
        ax.set_ylabel("深度 z (m)")
        ax.invert_yaxis()

    fig.tight_layout()
    return fig


# =========================================================================== #
# 4. Loss 曲线
# =========================================================================== #
def plot_loss_curve(
    steps: Sequence[int],
    loss_dict: Dict[str, Sequence[float]],
    title: str = "训练损失曲线",
    log_scale: bool = True,
    figsize: Tuple[float, float] = (8.0, 5.0),
) -> plt.Figure:
    """汇总 loss 曲线 (所有分量叠加)."""
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (name, values) in enumerate(loss_dict.items()):
        if len(values) == 0:
            continue
        ax.plot(steps[: len(values)], values, label=name,
                color=colors[i % len(colors)], linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
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
        ax.plot(steps[: len(values)], values, color="#1f77b4", linewidth=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"{name} loss")
        ax.set_title(f"{name} 损失曲线")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        figs[name] = fig
    return figs


# =========================================================================== #
# 5. 误差分布图 (Feature 3: 小 RMSE 也能呈现强对比)
# =========================================================================== #
def plot_error_map(
    X: np.ndarray,
    Z: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    title: str = "预测误差分布",
    scale: str = "percentile",
    percentile_lo: float = 1.0,
    percentile_hi: float = 99.0,
    cmap: str = "viridis",
    log_floor: float = 1e-12,
    figsize: Tuple[float, float] = (6.5, 4.8),
) -> plt.Figure:
    """绘制 |pred - true| 误差热力图.

    Parameters
    ----------
    scale : {"percentile", "log", "linear"}
        默认 "percentile" — 用第 1 / 99 百分位裁剪颜色范围, 即使全场 RMSE 很小
        也能呈现强对比 (避免少数离群值压缩整个色阶).
        "log" — LogNorm, 跨多个数量级的误差能同时看清.
        "linear" — 不做特殊处理 (最差).
    percentile_lo, percentile_hi : float
        scale="percentile" 时使用的百分位.
    """
    err = np.abs(pred - true)
    err_pos = err[np.isfinite(err)]
    title_extra = ""

    if scale == "percentile":
        if err_pos.size > 0:
            vmin = float(np.percentile(err_pos, percentile_lo))
            vmax = float(np.percentile(err_pos, percentile_hi))
            if vmax <= vmin:
                vmax = float(err_pos.max()) if err_pos.max() > vmin else vmin + 1e-12
        else:
            vmin, vmax = 0.0, 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        title_extra = f"  [p{int(percentile_lo)}–p{int(percentile_hi)} 色阶]"
    elif scale == "log":
        # 用 log 色阶, 跨多个数量级
        safe = np.maximum(err, log_floor)
        vmin = float(np.percentile(safe[np.isfinite(safe)], percentile_lo)) \
            if err_pos.size > 0 else log_floor
        vmax = float(safe.max()) if err_pos.size > 0 else 1.0
        vmin = max(vmin, log_floor)
        if vmax <= vmin:
            vmax = vmin * 10
        norm = LogNorm(vmin=vmin, vmax=vmax)
        title_extra = "  [log 色阶]"
    else:
        norm = None

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(X, Z, err, cmap=cmap, shading="auto",
                       norm=norm, rasterized=True)
    _attach_colorbar(fig, ax, im, label="|pred - true|")
    ax.set_xlabel("距离 x (m)")
    ax.set_ylabel("深度 z (m)")

    # 在标题里附上全场统计, 方便论文
    rmse = float(np.sqrt(np.nanmean(err ** 2))) if err_pos.size > 0 else 0.0
    mae = float(np.nanmean(err)) if err_pos.size > 0 else 0.0
    ax.set_title(f"{title}{title_extra}\nRMSE={rmse:.3e}, MAE={mae:.3e}")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_error_map_multi_scale(
    X: np.ndarray, Z: np.ndarray,
    pred: np.ndarray, true: np.ndarray,
    title_prefix: str = "误差",
    figsize: Tuple[float, float] = (16.5, 4.5),
) -> plt.Figure:
    """同一误差三种色阶并排显示: 线性 / 百分位 / log.

    适合在 RMSE 很小时观察细节: 线性图常常 "一片绿", 百分位/log 则有强对比.
    """
    err = np.abs(pred - true)
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # (1) 线性
    im0 = axes[0].pcolormesh(X, Z, err, cmap="viridis", shading="auto", rasterized=True)
    axes[0].set_title(f"{title_prefix} · 线性色阶")
    _attach_colorbar(fig, axes[0], im0)

    # (2) 百分位裁剪
    err_pos = err[np.isfinite(err)]
    if err_pos.size > 0:
        vlo = float(np.percentile(err_pos, 1.0))
        vhi = float(np.percentile(err_pos, 99.0))
    else:
        vlo, vhi = 0.0, 1.0
    if vhi <= vlo:
        vhi = vlo + 1e-12
    im1 = axes[1].pcolormesh(X, Z, err, cmap="viridis", shading="auto",
                             vmin=vlo, vmax=vhi, rasterized=True)
    axes[1].set_title(f"{title_prefix} · 百分位 [p1–p99]")
    _attach_colorbar(fig, axes[1], im1)

    # (3) log
    safe = np.maximum(err, 1e-12)
    vmin_l = max(float(np.percentile(safe[np.isfinite(safe)], 1.0)), 1e-12)
    vmax_l = float(safe.max()) if err_pos.size > 0 else 1.0
    if vmax_l <= vmin_l:
        vmax_l = vmin_l * 10
    im2 = axes[2].pcolormesh(X, Z, err, cmap="viridis", shading="auto",
                             norm=LogNorm(vmin=vmin_l, vmax=vmax_l), rasterized=True)
    axes[2].set_title(f"{title_prefix} · log 色阶")
    _attach_colorbar(fig, axes[2], im2)

    rmse = float(np.sqrt(np.nanmean(err ** 2))) if err_pos.size > 0 else 0.0
    mae = float(np.nanmean(err)) if err_pos.size > 0 else 0.0
    for ax in axes:
        ax.set_xlabel("距离 x (m)")
        ax.set_ylabel("深度 z (m)")
        ax.invert_yaxis()
    fig.suptitle(f"误差三视图  ·  RMSE={rmse:.3e}, MAE={mae:.3e}", y=1.02)
    fig.tight_layout()
    return fig


def plot_pde_residual(
    X: np.ndarray,
    Z: np.ndarray,
    residual: np.ndarray,
    title: str = "PDE 残差",
) -> plt.Figure:
    return plot_field(X, Z, np.abs(residual), title=title, cmap="inferno",
                      cbar_label="|Helmholtz residual|")


# =========================================================================== #
# 6. 剖面对比
# =========================================================================== #
def plot_profile_comparison(
    coord_axis: np.ndarray,
    true_profile: np.ndarray,
    pred_profile: np.ndarray,
    xlabel: str = "深度 z (m)",
    ylabel: str = "传输损失 (dB)",
    title: str = "深度剖面对比",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(coord_axis, true_profile, label="Ground Truth",
            color="#1f77b4", linewidth=1.8)
    ax.plot(coord_axis, pred_profile, label="PINN Prediction",
            color="#d62728", linewidth=1.8, linestyle="--")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    fig.tight_layout()
    return fig


# =========================================================================== #
# 7. 保存全部 loss 图
# =========================================================================== #
def save_all_loss_figures(
    steps: Sequence[int],
    loss_dict: Dict[str, Sequence[float]],
    out_dir: str | Path,
    log_scale: bool = True,
    formats: Sequence[str] = ("png",),
) -> Dict[str, List[str]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, List[str]] = {}

    fig_all = plot_loss_curve(steps, loss_dict, log_scale=log_scale)
    saved["all"] = save_figure_publication(fig_all, out_dir / "loss_all", formats=formats)

    figs = plot_individual_losses(steps, loss_dict, log_scale=log_scale)
    for name, fig in figs.items():
        saved[name] = save_figure_publication(
            fig, out_dir / f"loss_{name}", formats=formats,
        )
    return saved


# =========================================================================== #
# 8. TL 计算
# =========================================================================== #
def compute_tl(p_real: np.ndarray, p_imag: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    amp = np.sqrt(p_real ** 2 + p_imag ** 2)
    return -20.0 * np.log10(amp + eps)


# =========================================================================== #
# 9. 采样分布可视化 (Feature 2)
# =========================================================================== #
def sampling_viz_style(method_name: str) -> str:
    """根据采样方法返回预览风格: region | lhs | grid | plain."""
    m = (method_name or "").lower()
    if m in ("stratified_block", "stratified", "block",
             "problem_region_aug", "problem_region", "aug"):
        return "region"
    if m == "lhs":
        return "lhs"
    if m in ("grid", "grid_uniform"):
        return "grid"
    return "plain"


def _draw_lhs_guides(ax, n: int, length: float, depth: float) -> None:
    """LHS: 在 (x,z) 上画分层网格线."""
    n_bins = max(2, int(round(n ** 0.5)))
    for k in range(1, n_bins):
        ax.axvline(k / n_bins * length, color="#bbbbbb", linewidth=0.6, alpha=0.85)
        ax.axhline(k / n_bins * depth, color="#bbbbbb", linewidth=0.6, alpha=0.85)


def _draw_grid_guides(ax, length: float, depth: float, stride: int = 1,
                      nx: int | None = None, nz: int | None = None) -> None:
    """grid_uniform: 等间距网格示意."""
    if nx and nz and stride > 0:
        xs = np.arange(0, nx, stride) / max(nx - 1, 1) * length
        zs = np.arange(0, nz, stride) / max(nz - 1, 1) * depth
    else:
        n_lines = 12
        xs = np.linspace(0, length, n_lines)
        zs = np.linspace(0, depth, n_lines)
    for xv in xs:
        ax.axvline(float(xv), color="#888888", linewidth=0.5, alpha=0.7)
    for zv in zs:
        ax.axhline(float(zv), color="#888888", linewidth=0.5, alpha=0.7)


def plot_sampling_distribution(
    coords: np.ndarray,
    length: float,
    depth: float,
    source_r: float | None = None,
    source_z: float | None = None,
    method_name: str = "",
    n_total: int | None = None,
    figsize: Tuple[float, float] = (8.0, 5.8),
    show_density_hexbin: bool = False,
    color_mode: str = "auto",
    viz_style: str = "auto",
    near_dist: float | None = None,
    mid_dist: float | None = None,
    problem_region: Optional[Tuple[float, float, float, float]] = None,
    sampling_info: Optional[dict] = None,
    nx: int | None = None,
    nz: int | None = None,
) -> plt.Figure:
    """绘制训练观测点在 (x, z) 域的**彩色**信息图.

    Parameters
    ----------
    color_mode : {"auto", "distance", "region", "uniform", "lhs", "plain"}
    viz_style : {"auto", "region", "lhs", "grid", "plain"}
        控制背景示意: 仅 stratified 类画近/中场弧圈; lhs 画分层线;
        grid 画网格; 其余 plain.
    sampling_info : 可选, grid_uniform 的 stride 等.
    near_dist, mid_dist : float, 可选
        距源近场 / 中场阈值 (m). 提供后会在图中画虚线圆.
    problem_region : (x_min, x_max, z_min, z_max) (m, 物理坐标), 可选
        要在图上高亮的矩形区域 (用于 problem_region_aug).

    标题分两行: 主标含采样方法和点数, 副标记录全部参数 (length / depth /
    nx*nz / 声源位置), 便于复现实验.
    """
    coords = np.asarray(coords)
    info = sampling_info or {}
    fig, ax = plt.subplots(figsize=figsize)

    style = viz_style
    if style == "auto":
        style = sampling_viz_style(method_name)

    # --------------- 背景示意 --------------- #
    if style == "lhs":
        _draw_lhs_guides(ax, len(coords), length, depth)
    elif style == "grid":
        stride = int(info.get("stride", 1))
        _draw_grid_guides(ax, length, depth, stride=stride, nx=nx, nz=nz)

    # --------------- 色彩 --------------- #
    mode = color_mode
    if mode == "auto":
        if style == "region":
            mode = "region"
        elif style == "lhs":
            mode = "lhs"
        else:
            mode = "plain"

    if show_density_hexbin and len(coords) > 8000:
        hb = ax.hexbin(coords[:, 0], coords[:, 1], gridsize=50,
                       cmap="viridis", mincnt=1)
        _attach_colorbar(fig, ax, hb, label="样本点密度")
    elif mode == "region" and source_r is not None and source_z is not None \
            and near_dist is not None and mid_dist is not None:
        dist = np.hypot(coords[:, 0] - source_r, coords[:, 1] - source_z)
        near_mask = dist < near_dist
        mid_mask = (dist >= near_dist) & (dist < mid_dist)
        far_mask = dist >= mid_dist
        ax.scatter(coords[far_mask, 0], coords[far_mask, 1], s=5.0,
                   alpha=0.6, c="#2ca02c", edgecolors="none",
                   label=f"远场 (≥ {mid_dist:.0f} m)  N={int(far_mask.sum())}")
        ax.scatter(coords[mid_mask, 0], coords[mid_mask, 1], s=5.0,
                   alpha=0.65, c="#ff7f0e", edgecolors="none",
                   label=f"中场 ({near_dist:.0f}–{mid_dist:.0f} m)  N={int(mid_mask.sum())}")
        ax.scatter(coords[near_mask, 0], coords[near_mask, 1], s=6.0,
                   alpha=0.8, c="#d62728", edgecolors="none",
                   label=f"近场 (< {near_dist:.0f} m)  N={int(near_mask.sum())}")
    elif mode == "lhs":
        n_bins = max(2, int(round(len(coords) ** 0.5)))
        bx = np.clip((coords[:, 0] / max(length, 1e-9) * n_bins).astype(int), 0, n_bins - 1)
        bz = np.clip((coords[:, 1] / max(depth, 1e-9) * n_bins).astype(int), 0, n_bins - 1)
        layer_id = bx * n_bins + bz
        sc = ax.scatter(coords[:, 0], coords[:, 1], s=5.0, c=layer_id,
                        cmap="tab20", alpha=0.85, edgecolors="none")
        _attach_colorbar(fig, ax, sc, label="LHS 分层编号")
    elif mode == "distance" and source_r is not None and source_z is not None:
        dist = np.hypot(coords[:, 0] - source_r, coords[:, 1] - source_z)
        sc = ax.scatter(coords[:, 0], coords[:, 1], s=5.0, c=dist,
                        cmap="viridis", alpha=0.8, edgecolors="none")
        _attach_colorbar(fig, ax, sc, label="距源距离 (m)")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=4.0, alpha=0.65,
                   color="#1f77b4", edgecolors="none", label=f"样本点 N={len(coords)}")

    # --------------- 距离环: 仅 stratified 类 --------------- #
    if style == "region" and source_r is not None and source_z is not None:
        theta = np.linspace(0, 2 * np.pi, 200)
        for r, color, ls in [
            (near_dist, "#d62728", ":"),
            (mid_dist, "#ff7f0e", ":"),
        ]:
            if r is None or r <= 0:
                continue
            ax.plot(source_r + r * np.cos(theta), source_z + r * np.sin(theta),
                    color=color, linestyle=ls, linewidth=1.2, alpha=0.7)

    # --------------- 问题区域矩形 --------------- #
    if problem_region is not None:
        from matplotlib.patches import Rectangle
        xmin, xmax, zmin, zmax = problem_region
        rect = Rectangle(
            (xmin, zmin), xmax - xmin, zmax - zmin,
            linewidth=1.6, edgecolor="magenta", facecolor="magenta",
            alpha=0.10, linestyle="--",
            label=f"问题区域 [{xmin:.0f}, {xmax:.0f}] × [{zmin:.0f}, {zmax:.0f}] m",
        )
        ax.add_patch(rect)

    # --------------- 声源标记 --------------- #
    if source_r is not None and source_z is not None:
        ax.plot(source_r, source_z, marker="*", color="red",
                markersize=20, markeredgecolor="black", markeredgewidth=1.0,
                label=f"声源 ({source_r:.1f}, {source_z:.1f}) m", zorder=10)

    # --------------- 坐标轴 (按真实物理范围) --------------- #
    ax.set_xlim(0, length); ax.set_ylim(0, depth)
    ax.invert_yaxis()
    ax.set_xlabel(f"水平距离 x (m)   ·   总范围 0 ~ {length:.0f} m")
    ax.set_ylabel(f"深度 z (m)   ·   总深度 0 ~ {depth:.0f} m")

    # --------------- 标题 --------------- #
    method_label = method_name if method_name else "training sample distribution"
    title_lines = [f"采样方法: {method_label}"]
    if n_total is not None:
        title_lines[-1] += f"   N = {n_total:,}"
    # 参数副标
    sub_parts = [f"L={length:.0f} m", f"D={depth:.0f} m"]
    if source_r is not None and source_z is not None:
        sub_parts.append(f"源=({source_r:.1f}, {source_z:.1f})")
    if near_dist is not None:
        sub_parts.append(f"近场<{near_dist:.0f}m")
    if mid_dist is not None:
        sub_parts.append(f"中场<{mid_dist:.0f}m")
    title_lines.append(" · ".join(sub_parts))
    ax.set_title("\n".join(title_lines), fontsize=mpl.rcParams["axes.titlesize"])

    # 图例 (右上角, 半透明)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right",
                  fontsize=max(mpl.rcParams["legend.fontsize"] - 1, 8),
                  framealpha=0.9)

    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    return fig


def plot_sampling_methods_panel(
    method_to_result: Dict[str, "SamplingResult"],
    length: float,
    depth: float,
    source_r: float | None = None,
    source_z: float | None = None,
    near_dist: float | None = None,
    mid_dist: float | None = None,
    ncols: int = 3,
    panel_size: Tuple[float, float] = (5.2, 4.0),
    nx: int | None = None,
    nz: int | None = None,
) -> plt.Figure:
    """多种采样方法横向对比: 各方法使用对应示意 (弧圈/分层/网格)."""
    n = len(method_to_result)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        squeeze=False,
    )

    for idx, (method, sr) in enumerate(method_to_result.items()):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        style = sampling_viz_style(method)
        if style == "lhs":
            _draw_lhs_guides(ax, sr.size, length, depth)
            n_bins = max(2, int(round(sr.size ** 0.5)))
            bx = np.clip((sr.coords[:, 0] / max(length, 1e-9) * n_bins).astype(int), 0, n_bins - 1)
            bz = np.clip((sr.coords[:, 1] / max(depth, 1e-9) * n_bins).astype(int), 0, n_bins - 1)
            lid = bx * n_bins + bz
            ax.scatter(sr.coords[:, 0], sr.coords[:, 1], s=3.5, c=lid,
                       cmap="tab20", alpha=0.8, edgecolors="none")
        elif style == "grid":
            stride = int(sr.info.get("stride", 1))
            _draw_grid_guides(ax, length, depth, stride=stride, nx=nx, nz=nz)
            ax.scatter(sr.coords[:, 0], sr.coords[:, 1], s=4.0,
                       color="#1f77b4", alpha=0.75, edgecolors="none")
        elif style == "region" and source_r is not None and source_z is not None \
                and near_dist is not None and mid_dist is not None:
            dist = np.hypot(sr.coords[:, 0] - source_r, sr.coords[:, 1] - source_z)
            near_mask = dist < near_dist
            mid_mask = (dist >= near_dist) & (dist < mid_dist)
            far_mask = dist >= mid_dist
            ax.scatter(sr.coords[far_mask, 0], sr.coords[far_mask, 1], s=3.5,
                       c="#2ca02c", alpha=0.6, edgecolors="none")
            ax.scatter(sr.coords[mid_mask, 0], sr.coords[mid_mask, 1], s=3.5,
                       c="#ff7f0e", alpha=0.65, edgecolors="none")
            ax.scatter(sr.coords[near_mask, 0], sr.coords[near_mask, 1], s=4.0,
                       c="#d62728", alpha=0.8, edgecolors="none")
            theta = np.linspace(0, 2 * np.pi, 200)
            for rd, col in [(near_dist, "#d62728"), (mid_dist, "#ff7f0e")]:
                if rd is None or rd <= 0:
                    continue
                ax.plot(source_r + rd * np.cos(theta),
                        source_z + rd * np.sin(theta),
                        color=col, linestyle=":", linewidth=1.0, alpha=0.7)
        else:
            ax.scatter(sr.coords[:, 0], sr.coords[:, 1], s=3.5, alpha=0.65,
                       color="#1f77b4", edgecolors="none")

        if source_r is not None and source_z is not None:
            ax.plot(source_r, source_z, marker="*", color="red",
                    markersize=12, markeredgecolor="black", markeredgewidth=0.5,
                    zorder=10)

        ax.set_xlim(0, length); ax.set_ylim(0, depth)
        ax.invert_yaxis()
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
        ax.set_title(f"{method}\nN = {sr.size:,}", fontsize=11)
        ax.grid(True, alpha=0.2)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle(
        f"训练样本点划分方法 · 分布对比  "
        f"(L={length:.0f} m, D={depth:.0f} m)",
        y=1.01,
    )
    fig.tight_layout()
    return fig


# =========================================================================== #
# 10. 训练时间成本图
# =========================================================================== #
def plot_time_vs_step(
    steps: Sequence[int],
    elapsed_seconds: Sequence[float],
    title: str = "训练时间累积",
    figsize: Tuple[float, float] = (7.0, 4.5),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(steps, elapsed_seconds, color="#2ca02c", linewidth=1.6)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Elapsed time (s)")
    ax.set_title(title); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_step_time_per_iter(
    steps: Sequence[int],
    elapsed_seconds: Sequence[float],
    title: str = "单步耗时 (滚动均值)",
    figsize: Tuple[float, float] = (7.0, 4.5),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    if len(steps) >= 2:
        steps_arr = np.asarray(steps, dtype=np.float64)
        times_arr = np.asarray(elapsed_seconds, dtype=np.float64)
        dt = np.diff(times_arr) / np.maximum(np.diff(steps_arr), 1.0)
        ax.plot(steps_arr[1:], dt, color="#d62728", linewidth=1.3)
        ax.set_ylabel("s / iteration")
    else:
        ax.text(0.5, 0.5, "数据不足", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("Iteration"); ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_loss_vs_time(
    elapsed_seconds: Sequence[float],
    loss_total: Sequence[float],
    title: str = "Loss vs 训练耗时",
    log_scale: bool = True,
    figsize: Tuple[float, float] = (7.0, 4.5),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(elapsed_seconds, loss_total, color="#1f77b4", linewidth=1.5)
    ax.set_xlabel("Elapsed time (s)"); ax.set_ylabel("Total loss")
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# =========================================================================== #
# 11. 预测-真实散点图
# =========================================================================== #
def plot_pred_vs_true_scatter(
    pred: np.ndarray,
    true: np.ndarray,
    quantity_name: str = "TL (dB)",
    max_points: int = 50000,
    figsize: Tuple[float, float] = (5.6, 5.6),
) -> plt.Figure:
    """散点图: 横轴真实, 纵轴预测; 理想为对角线."""
    pred = np.asarray(pred).flatten()
    true = np.asarray(true).flatten()
    if pred.size > max_points:
        idx = np.random.choice(pred.size, max_points, replace=False)
        pred = pred[idx]; true = true[idx]
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(true, pred, s=1.2, alpha=0.35, color="#1f77b4", edgecolors="none")
    lo = float(min(true.min(), pred.min()))
    hi = float(max(true.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1.4,
            label="y = x (ideal)")
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    corr = float(np.corrcoef(pred, true)[0, 1]) if pred.std() > 1e-12 else 0.0
    ax.set_xlabel(f"真实 {quantity_name}")
    ax.set_ylabel(f"预测 {quantity_name}")
    ax.set_title(f"Predicted vs Ground Truth\nRMSE = {rmse:.4g}   Corr = {corr:.4f}")
    ax.grid(True, alpha=0.3); ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return fig
