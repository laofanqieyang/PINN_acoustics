"""把一次训练的全部环境参数 + 指标 + 采样信息 + 迁移学习信息导出到单个 Excel 文件.

设计目标:
    * 一个 .xlsx 完整记录一次实验, 方便归档与课题组分享
    * 多 sheet 结构, 每个 sheet 对应一个领域 (参数/指标/采样/迁移/运行时)
    * 列宽自动适配; 关键信息加粗
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd


# 字段中文描述 (列 1: 字段, 列 2: 中文, 列 3: 取值)
_PARAM_LABELS: Dict[str, str] = {
    "pres_real_path":          "声压实部 CSV 路径",
    "pres_imag_path":          "声压虚部 CSV 路径",
    "length":                  "声场水平距离 length (m)",
    "depth":                   "声场深度 depth (m)",
    "nx":                      "水平方向数据点数 nx",
    "nz":                      "深度方向数据点数 nz",
    "learning_rate":           "学习率 lr",
    "batch_size":              "batch_size",
    "pde_weight":              "PDE 权重",
    "num_layers":              "神经网络隐藏层数",
    "num_neurons":             "每层神经元数",
    "network_type":            "网络架构类型",
    "fourier_mapping_size":    "Fourier mapping size",
    "fourier_sigma":           "Fourier 带宽 σ",
    "siren_w0":                "SIREN 首层 w0",
    "activation":              "隐藏层激活",
    "epochs":                  "训练步数 steps",
    "data_weight":             "数据损失权重",
    "boundary_weight":         "边界损失权重",
    "frequency":               "声源频率 (Hz)",
    "sound_speed":             "声速 (m/s)",
    "source_r":                "声源水平位置 (m)",
    "source_z":                "声源深度 (m)",
    "source_sigma":            "声源宽度 sigma",
    "source_amplitude":        "声源幅度",
    "visualize_interval":      "可视化刷新间隔",
    "log_interval":            "Loss 记录间隔",
    "num_collocation":         "PDE 配点数 / step",
    "num_boundary":            "边界点数 / 每边",
    "random_seed":             "随机种子",
    "output_dir":              "输出根目录",
    "experiment_name":         "实验名称",
    "device":                  "计算设备",
    "sampling_method":         "样本划分方法",
    "num_train_obs":           "训练观测点总数 (上限)",
    "num_blocks_x":            "x 方向块数 (stratified)",
    "num_blocks_z":            "z 方向块数 (stratified)",
    "near_dist_threshold":     "近场阈值 (m)",
    "mid_dist_threshold":      "中场阈值 (m)",
    "points_per_near_block":   "近场每块点数",
    "points_per_mid_block":    "中场每块点数",
    "points_per_far_block":    "远场每块点数",
    "problem_region_x_min":    "问题区域 x_min (归一化)",
    "problem_region_x_max":    "问题区域 x_max (归一化)",
    "problem_region_z_min":    "问题区域 z_min (归一化)",
    "problem_region_z_max":    "问题区域 z_max (归一化)",
    "problem_region_extra_points": "问题区域额外加密点数",
    "pretrained_ckpt":         "预训练 .pt 路径 (迁移)",
    "freeze_first_n_layers":   "冻结前 N 层 Linear",
    "transfer_lr_scale":       "迁移学习 lr 缩放",
    "fourier_b_rescale":       "Fourier B 矩阵缩放",
    "pretrained_frequency":    "预训练时的频率",
    "gradient_clip":           "梯度裁剪上限",
    "nan_skip_threshold":      "NaN/Inf 累计阈值",
    "cuda_empty_cache_every":  "CUDA 缓存清理间隔",
    "max_loss_points_in_memory": "Loss 历史内存上限",
}


def _to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {"value": obj}


def _coerce(v: Any) -> Any:
    """把不可被 openpyxl 序列化的类型转成字符串/数字."""
    import numpy as np
    if v is None:
        return ""
    if isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (list, tuple, set)):
        return ", ".join(str(_coerce(x)) for x in v)
    if isinstance(v, dict):
        return ", ".join(f"{k}={_coerce(val)}" for k, val in v.items())
    if isinstance(v, Path):
        return str(v)
    return str(v)


def save_parameters_xlsx(
    out_path: str | Path,
    config: Any,
    final_metrics: Optional[Mapping[str, float]] = None,
    rmse_summary: Optional[Mapping[str, Any]] = None,
    sampling_info: Optional[Mapping[str, Any]] = None,
    transfer_info: Optional[Mapping[str, Any]] = None,
    runtime_info: Optional[Mapping[str, Any]] = None,
) -> str:
    """把一次训练的所有环境写到一个多 sheet 的 .xlsx.

    Sheets:
        ① "训练参数"   — AppConfig 全部字段 (中文标签 + 字段名 + 取值)
        ② "误差指标"   — final_metrics & rmse_summary (全场 + 近/中/远场)
        ③ "样本采样"   — sampling_info (策略 / 点数 / 块统计)
        ④ "迁移学习"   — 仅当启用迁移时
        ⑤ "运行时信息" — 设备、耗时、时间戳、产物路径
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg_dict = _to_dict(config)
    metrics_dict = dict(final_metrics or {})
    rmse_dict = dict(rmse_summary or {})
    sampling_dict = dict(sampling_info or {})
    transfer_dict = dict(transfer_info or {})
    runtime_dict = dict(runtime_info or {})

    # ---------- Sheet 1: 训练参数 ---------- #
    rows = []
    known_keys = list(_PARAM_LABELS.keys())
    for k in known_keys:
        if k in cfg_dict:
            rows.append({
                "字段名 (key)": k,
                "中文描述":      _PARAM_LABELS[k],
                "取值":          _coerce(cfg_dict[k]),
            })
    # 其余未在 _PARAM_LABELS 出现的 key
    for k, v in cfg_dict.items():
        if k not in _PARAM_LABELS:
            rows.append({
                "字段名 (key)": k,
                "中文描述":      "",
                "取值":          _coerce(v),
            })
    df_params = pd.DataFrame(rows)

    # ---------- Sheet 2: 误差指标 ---------- #
    metric_rows = []
    # 优先输出 rmse_summary 的核心字段
    primary_keys = [
        "tl_rmse", "real_rmse", "imag_rmse",
        "tl_mae", "real_mae", "imag_mae",
        "tl_corr", "real_corr", "imag_corr",
        "tl_near_rmse", "tl_mid_rmse", "tl_far_rmse",
        "real_near_rmse", "real_mid_rmse", "real_far_rmse",
        "imag_near_rmse", "imag_mid_rmse", "imag_far_rmse",
    ]
    summary_used = set()
    for k in primary_keys:
        v = rmse_dict.get(k, metrics_dict.get(k))
        if v is None:
            continue
        metric_rows.append({"指标 (key)": k, "数值": _coerce(v)})
        summary_used.add(k)
    # 其余字段
    for k, v in rmse_dict.items():
        if k in summary_used:
            continue
        metric_rows.append({"指标 (key)": k, "数值": _coerce(v)})
    for k, v in metrics_dict.items():
        if k in rmse_dict or k in summary_used:
            continue
        metric_rows.append({"指标 (key)": k, "数值": _coerce(v)})
    df_metrics = pd.DataFrame(metric_rows) if metric_rows else pd.DataFrame()

    # ---------- Sheet 3: 样本采样 ---------- #
    sampling_rows = [
        {"字段 (key)": k, "取值": _coerce(v)} for k, v in sampling_dict.items()
    ]
    df_sampling = pd.DataFrame(sampling_rows) if sampling_rows else pd.DataFrame()

    # ---------- Sheet 4: 迁移学习 ---------- #
    transfer_rows = [
        {"字段 (key)": k, "取值": _coerce(v)} for k, v in transfer_dict.items()
    ]
    df_transfer = pd.DataFrame(transfer_rows) if transfer_rows else pd.DataFrame()

    # ---------- Sheet 5: 运行时 ---------- #
    runtime_rows = [
        {"字段 (key)": k, "取值": _coerce(v)} for k, v in runtime_dict.items()
    ]
    if not runtime_rows:
        runtime_rows = [{"字段 (key)": "saved_at",
                          "取值": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]
    df_runtime = pd.DataFrame(runtime_rows)

    # ---------- 写入 (xlsxwriter 引擎美化) ---------- #
    try:
        engine = "xlsxwriter"
        writer = pd.ExcelWriter(out_path, engine=engine)
    except Exception:
        engine = "openpyxl"
        writer = pd.ExcelWriter(out_path, engine=engine)

    with writer:
        df_params.to_excel(writer, sheet_name="训练参数", index=False)
        if not df_metrics.empty:
            df_metrics.to_excel(writer, sheet_name="误差指标", index=False)
        if not df_sampling.empty:
            df_sampling.to_excel(writer, sheet_name="样本采样", index=False)
        if not df_transfer.empty:
            df_transfer.to_excel(writer, sheet_name="迁移学习", index=False)
        df_runtime.to_excel(writer, sheet_name="运行时信息", index=False)

        # 美化: xlsxwriter
        if engine == "xlsxwriter":
            book = writer.book
            header_fmt = book.add_format({
                "bold": True, "bg_color": "#D9E1F2", "border": 1,
                "align": "left", "valign": "vcenter",
            })
            cell_fmt = book.add_format({"align": "left", "valign": "vcenter"})
            for sheet_name, df in [
                ("训练参数", df_params),
                ("误差指标", df_metrics),
                ("样本采样", df_sampling),
                ("迁移学习", df_transfer),
                ("运行时信息", df_runtime),
            ]:
                if df is None or df.empty:
                    continue
                ws = writer.sheets[sheet_name]
                # 设置列宽
                for col_idx, col in enumerate(df.columns):
                    series = df[col].astype(str)
                    max_len = max(
                        [len(col)] + [len(s) for s in series.tolist()[:200]]
                    )
                    width = min(60, max(12, max_len + 2))
                    ws.set_column(col_idx, col_idx, width, cell_fmt)
                # 写表头格式
                for col_idx, col in enumerate(df.columns):
                    ws.write(0, col_idx, col, header_fmt)
                ws.freeze_panes(1, 0)

    return str(out_path)
