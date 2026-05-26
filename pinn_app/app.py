"""PINN 水下声场预测 - 交互式训练主界面 (Streamlit).

运行:
    cd d:/lzz/Auto_train_PINN
    streamlit run pinn_app/app.py
"""
from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch

# 允许 "streamlit run pinn_app/app.py" 或 "python -m pinn_app.app" 两种方式运行
try:
    from pinn_app.config import AppConfig, default_config
    from pinn_app.training.trainer import Trainer
    from pinn_app.utils.logger import TrainingLogger
    from pinn_app.utils.metrics import compute_all_metrics
    from pinn_app.utils.visualization import (
        compute_tl,
        plot_field,
        plot_field_comparison,
        plot_field_triptych,
        plot_individual_losses,
        plot_loss_curve,
        plot_error_map,
        plot_error_map_multi_scale,
        plot_pde_residual,
        plot_profile_comparison,
        plot_sampling_distribution,
        plot_sampling_methods_panel,
        sampling_viz_style,
        save_all_loss_figures,
        setup_chinese_font,
        setup_publication_style,
    )
    from pinn_app.data.sampling import build_observation_sampling, SAMPLING_METHODS
except ModuleNotFoundError:
    # streamlit run 的情况下 Python 路径可能缺少父目录, 手动加入
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pinn_app.config import AppConfig, default_config
    from pinn_app.training.trainer import Trainer
    from pinn_app.utils.logger import TrainingLogger
    from pinn_app.utils.metrics import compute_all_metrics
    from pinn_app.utils.visualization import (
        compute_tl,
        plot_field,
        plot_field_comparison,
        plot_field_triptych,
        plot_individual_losses,
        plot_loss_curve,
        plot_error_map,
        plot_error_map_multi_scale,
        plot_pde_residual,
        plot_profile_comparison,
        plot_sampling_distribution,
        plot_sampling_methods_panel,
        sampling_viz_style,
        save_all_loss_figures,
        setup_chinese_font,
        setup_publication_style,
    )
    from pinn_app.data.sampling import build_observation_sampling, SAMPLING_METHODS


# =========================================================================== #
# 全局设置
# =========================================================================== #
st.set_page_config(
    page_title="PINN 水下声场交互式训练平台",
    layout="wide",
    initial_sidebar_state="expanded",
)
setup_chinese_font()
setup_publication_style()


# =========================================================================== #
# UI 模块 1: 参数输入
# =========================================================================== #
def get_user_inputs() -> AppConfig:
    """在侧边栏渲染参数输入表单, 返回 AppConfig."""
    st.sidebar.title("⚙️ 训练参数")

    with st.sidebar.expander("📁 1-2. 数据文件路径", expanded=True):
        pres_real_path = st.text_input(
            "声压实部文件路径 (CSV)",
            value=st.session_state.get("pres_real_path", "pres_real.csv"),
            key="pres_real_path",
        )
        pres_imag_path = st.text_input(
            "声压虚部文件路径 (CSV)",
            value=st.session_state.get("pres_imag_path", "pres_imag.csv"),
            key="pres_imag_path",
        )

    with st.sidebar.expander("📐 3-6. 声场空间参数", expanded=True):
        length = st.number_input("声场水平距离 length (m)",
                                 min_value=1.0, max_value=1e6, value=500.0, step=10.0)
        depth = st.number_input("声场深度 depth (m)",
                                min_value=1.0, max_value=1e5, value=500.0, step=10.0)
        nx = st.number_input("水平方向数据点数 nx",
                             min_value=10, max_value=5000, value=500, step=10)
        nz = st.number_input("深度方向数据点数 nz",
                             min_value=10, max_value=5000, value=500, step=10)
        st.caption(f"总数据点数 = nx × nz = **{int(nx) * int(nz):,}**")

    with st.sidebar.expander("🧠 10-11. 神经网络结构", expanded=True):
        network_options = {
            "fourier":  "Fourier Feature + MLP  (推荐, 声场振荡最强)",
            "siren":    "SIREN (sin 激活, 高阶导光滑)",
            "modified": "Modified MLP (Wang 2021, 收敛最快)",
            "dnn":      "普通 Tanh DNN (基线)",
        }
        network_type = st.selectbox(
            "网络架构",
            options=list(network_options.keys()),
            format_func=lambda k: network_options[k],
            index=0,
            help=(
                "Fourier Feature: 把输入投影到 [cos(2πBx), sin(2πBx)] 消除频谱偏差, "
                "对高频声场精度提升最显著.\n"
                "SIREN: 所有激活都是 sin, 对 Helmholtz 二阶导天然光滑.\n"
                "Modified MLP: 双编码器门控, PINN 社区公认的训练加速器."
            ),
        )
        num_layers = st.number_input("神经网络隐藏层数 num_layers",
                                     min_value=1, max_value=32, value=7, step=1)
        num_neurons = st.number_input("每层神经元个数 num_neurons",
                                      min_value=4, max_value=1024, value=128, step=4)

        # 架构特定的超参
        fourier_mapping_size = 128
        fourier_sigma = 5.0
        siren_w0 = 30.0
        activation = "tanh"
        if network_type == "fourier":
            fourier_mapping_size = int(st.number_input(
                "Fourier mapping_size (m)", min_value=16, max_value=1024,
                value=128, step=16,
            ))
            fourier_sigma = float(st.number_input(
                "Fourier 带宽 σ", min_value=0.1, max_value=50.0,
                value=5.0, step=0.5,
                help="σ 越大能表达的频率越高, 经验 1~10; 远场高频场景取大.",
            ))
        elif network_type == "siren":
            siren_w0 = float(st.number_input(
                "SIREN 首层 w0", min_value=1.0, max_value=60.0,
                value=15.0, step=1.0,
                help=(
                    "首层 sin 的频率系数 (后续层固定 w0=1 让网络自学频率). "
                    "声场归一化波数≈8, 推荐 10~20. "
                    "过大 (≥30) 会让 PDE 残差爆炸 → 网络塌缩为常数解."
                ),
            ))
        elif network_type in ("dnn", "modified"):
            activation = st.selectbox(
                "隐藏层激活", options=["tanh", "gelu", "silu", "sin"], index=0,
            )

    with st.sidebar.expander("🏋️ 7-9, 12. 训练超参", expanded=True):
        learning_rate = st.number_input("学习率 lr",
                                        min_value=1e-6, max_value=1.0,
                                        value=1e-3, step=1e-4, format="%.6f")
        batch_size = st.number_input("batch_size",
                                     min_value=8, max_value=65536, value=128, step=8)
        pde_weight = st.number_input("PDE 权重",
                                     min_value=0.0, max_value=1e4, value=2.0, step=0.5)
        epochs = st.number_input("训练步数 steps",
                                 min_value=100, max_value=5_000_000,
                                 value=20000, step=500)

    with st.sidebar.expander("🎯 物理 / PDE 采样 (可选)", expanded=False):
        frequency = st.number_input("声源频率 (Hz)", value=50.0, step=1.0)
        sound_speed = st.number_input("声速 (m/s)", value=1541.0, step=1.0)
        source_r = st.number_input("声源水平位置 (m)", value=1.0, step=0.5)
        source_z = st.number_input("声源深度 (m)", value=20.0, step=1.0)
        source_sigma = st.number_input("声源宽度 sigma", value=10.0, step=0.5)
        source_amplitude = st.number_input("声源幅度", value=1.0, step=0.1)
        data_weight = st.number_input("数据权重", value=100.0, step=10.0)
        boundary_weight = st.number_input("边界权重", value=5.0, step=0.5)
        num_collocation = st.number_input("PDE 配点数/step", value=5000, step=500, min_value=100)
        num_boundary = st.number_input("边界点数/边", value=400, step=50, min_value=20)

    # ------------------------------------------------------------------ #
    # 训练观测点采样方法 (Feature 3)
    # ------------------------------------------------------------------ #
    with st.sidebar.expander("🎲 训练样本点划分方法 (Feature 3)", expanded=False):
        sampling_options = {
            "uniform":             "uniform · 全网格均匀随机",
            "stratified_block":    "stratified_block · 距源分层 (近少远多)",
            "lhs":                 "lhs · 拉丁超立方",
            "grid_uniform":        "grid_uniform · 等间距网格",
            "problem_region_aug":  "problem_region_aug · 分层 + 问题区域加密",
            "residual_adaptive":   "residual_adaptive · 残差自适应 (RAS)",
        }
        sampling_method = st.selectbox(
            "采样策略", options=list(sampling_options.keys()),
            format_func=lambda k: sampling_options[k], index=0,
            help=(
                "uniform: 在全网格上均匀随机选 num_train_obs 个点; \n"
                "stratified_block: 把域划成 20×20 块, 按到声源距离 (近/中/远) 分配不同密度; \n"
                "lhs: 拉丁超立方采样, 在 (x,z) 上分层均匀; \n"
                "grid_uniform: 等间距网格; \n"
                "problem_region_aug: stratified_block + 指定问题区域加密 (参考 wangge.py); \n"
                "residual_adaptive: 初始稀疏种子 + 训练中按 PDE/数据残差自动加密难点."
            ),
        )
        num_train_obs = st.number_input(
            "训练观测点总数 num_train_obs",
            min_value=100, max_value=10_000_000, value=30000, step=1000,
            help="若策略是 stratified_block / problem_region_aug, 总数由块参数决定, "
                 "该值仅作上限提示.",
        )
        if sampling_method in ("stratified_block", "problem_region_aug"):
            col_a, col_b = st.columns(2)
            num_blocks_x = col_a.number_input(
                "x 方向块数 nbx", min_value=2, max_value=200, value=20, step=1)
            num_blocks_z = col_b.number_input(
                "z 方向块数 nbz", min_value=2, max_value=200, value=20, step=1)
            col_a, col_b = st.columns(2)
            near_dist_threshold = col_a.number_input(
                "近场阈值 (m)", value=150.0, step=10.0)
            mid_dist_threshold = col_b.number_input(
                "中场阈值 (m)", value=300.0, step=10.0)
            col_a, col_b, col_c = st.columns(3)
            points_per_near_block = col_a.number_input(
                "近场/块", value=30, min_value=1, step=5)
            points_per_mid_block = col_b.number_input(
                "中场/块", value=60, min_value=1, step=5)
            points_per_far_block = col_c.number_input(
                "远场/块", value=125, min_value=1, step=5)
        else:
            num_blocks_x = 20; num_blocks_z = 20
            near_dist_threshold = 150.0; mid_dist_threshold = 300.0
            points_per_near_block = 30
            points_per_mid_block = 60; points_per_far_block = 125
        if sampling_method == "problem_region_aug":
            st.caption("问题区域 (归一化到 [0,1] 矩形):")
            col_a, col_b = st.columns(2)
            pr_x_min = col_a.number_input(
                "x_min_norm", value=0.0, min_value=0.0, max_value=1.0, step=0.05)
            pr_x_max = col_b.number_input(
                "x_max_norm", value=0.2, min_value=0.0, max_value=1.0, step=0.05)
            col_a, col_b = st.columns(2)
            pr_z_min = col_a.number_input(
                "z_min_norm", value=0.6, min_value=0.0, max_value=1.0, step=0.05)
            pr_z_max = col_b.number_input(
                "z_max_norm", value=1.0, min_value=0.0, max_value=1.0, step=0.05)
            pr_extra = st.number_input(
                "额外加密点数", value=3000, min_value=100, step=500)
        else:
            pr_x_min = 0.0; pr_x_max = 0.2
            pr_z_min = 0.6; pr_z_max = 1.0
            pr_extra = 3000
        if sampling_method == "residual_adaptive":
            st.caption("RAS: 训练时按残差自动追加观测点, 适合 Fourier 等强表达网络.")
            col_a, col_b = st.columns(2)
            ras_initial_fraction = col_a.number_input(
                "初始点数比例", min_value=0.05, max_value=1.0, value=0.25, step=0.05,
                help="初始观测 = num_train_obs × 该比例",
            )
            ras_refine_every = col_b.number_input(
                "残差评估间隔 (step)", min_value=100, max_value=50000, value=2000, step=100,
            )
            col_a, col_b, col_c = st.columns(3)
            ras_points_per_refine = col_a.number_input(
                "每次新增点数", min_value=50, max_value=5000, value=500, step=50)
            ras_candidate_size = col_b.number_input(
                "候选池大小", min_value=1000, max_value=50000, value=8000, step=500)
            ras_residual_metric = col_c.selectbox(
                "残差指标", options=["pde", "data"], index=0,
                help="pde: Helmholtz 方程残差; data: 与真值声压误差",
            )
        else:
            ras_initial_fraction = 0.25
            ras_refine_every = 2000
            ras_points_per_refine = 500
            ras_candidate_size = 8000
            ras_residual_metric = "pde"

    # ------------------------------------------------------------------ #
    # GradNorm 自适应损失权重
    # ------------------------------------------------------------------ #
    with st.sidebar.expander("⚖️ GradNorm 自适应损失权重", expanded=False):
        use_gradnorm = st.checkbox(
            "启用 GradNorm (推荐)", value=True,
            help="自动平衡 data / pde / bc / interface 各任务损失的下降速率",
            key="use_gradnorm",
        )
        gradnorm_alpha = st.number_input(
            "α (训练率指数)", min_value=0.1, max_value=5.0, value=1.5, step=0.1,
            help="越大 → 对慢任务权重提升越激进; 0.5–2.0 之间最常用",
        )
        gradnorm_lr = st.number_input(
            "权重学习率", min_value=1e-5, max_value=1e-1, value=1e-3,
            step=1e-4, format="%.5f",
        )
        gradnorm_update_every = st.number_input(
            "权重更新间隔 (step)", min_value=1, max_value=1000, value=10, step=1,
            help="每多少 step 跑一次 GradNorm; 越小越自适应, 但开销更大",
        )
        gradnorm_warmup_steps = st.number_input(
            "Warmup 步数", min_value=0, max_value=100000, value=100, step=10,
            help="前 N 步用固定权重等待损失稳定, 再启动 GradNorm",
        )
        gradnorm_min_weight = st.number_input(
            "权重下限", min_value=1e-4, max_value=1.0, value=0.01, step=0.001,
            format="%.4f",
        )

    # ------------------------------------------------------------------ #
    # Domain Decomposition
    # ------------------------------------------------------------------ #
    with st.sidebar.expander("🌐 Domain Decomposition (长域分解)", expanded=False):
        dd_force = st.selectbox(
            "启用模式",
            options=["auto", "on", "off"],
            format_func=lambda v: {
                "auto": "auto — length > 阈值自动启用",
                "on":   "on — 强制启用",
                "off":  "off — 强制关闭",
            }[v],
            index=0,
            help="auto: length 超过下方阈值自动启用 (默认 1000m)",
        )
        dd_threshold = st.number_input(
            "触发阈值 length (m)", min_value=100.0, max_value=1e6,
            value=1000.0, step=100.0,
        )
        dd_num_sub = st.number_input(
            "子域数 (0 = 自动)", min_value=0, max_value=32,
            value=0, step=1,
            help="0 = 按 length / 500m 上取整; 推荐 2~6 个子域",
        )
        dd_overlap = st.number_input(
            "相邻子域重叠 (m)", min_value=0.0, max_value=2000.0,
            value=150.0, step=10.0,
            help="过渡带宽度; 0 则按子域宽度×下方比例自动计算",
        )
        dd_overlap_fraction = st.number_input(
            "重叠比例 (overlap=0 时)", min_value=0.05, max_value=0.5,
            value=0.25, step=0.05,
        )
        dd_one_way = st.checkbox(
            "单向近→远耦合 (远场子网匹配近场声压)",
            value=True,
            help="在过渡带约束 u_{i+1} ≈ u_i, 模拟波从声源向外传播",
        )
        dd_coupling_weight = st.number_input(
            "单向耦合权重", min_value=0.0, max_value=100.0, value=5.0, step=0.5,
        )
        dd_interface_points = st.number_input(
            "每界面采样点数", min_value=20, max_value=5000, value=200, step=20,
        )
        dd_interface_weight = st.number_input(
            "界面连续性权重 (初值)", min_value=0.0, max_value=1e4,
            value=10.0, step=1.0,
            help="GradNorm 启用时会自动调整, 这里是初始值",
        )
        dd_deriv_weight = st.number_input(
            "界面导数连续相对权重", min_value=0.0, max_value=10.0,
            value=0.1, step=0.05,
        )
        # 实时预览是否会启用
        _enabled_now = (
            dd_force == "on"
            or (dd_force == "auto" and float(length) > float(dd_threshold))
        )
        if _enabled_now:
            _n_sub = int(dd_num_sub) if dd_num_sub > 0 else max(2, int(round(length / 500)))
            st.success(f"✓ DD 将启用: {_n_sub} 个子域")
        else:
            st.info("当前不启用 DD (length 未超阈值)")

    # ------------------------------------------------------------------ #
    # 长距离物理增强 (技术2: 包络 / PE / Marching)
    # ------------------------------------------------------------------ #
    with st.sidebar.expander("📡 长距离物理增强 (技术2)", expanded=False):
        use_envelope = st.checkbox(
            "包络分解 Envelope", value=False,
            help="u = v·exp(i k₀x), 网络学习慢变包络 v, 降低高频振荡学习难度",
        )
        envelope_k0 = 0.0
        supervise_envelope = False
        if use_envelope:
            envelope_k0 = st.number_input(
                "k₀ (0=自动)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1,
                help="参考相位波数; 0 时按下方波数公式自动计算",
            )
            supervise_envelope = st.checkbox(
                "数据损失在包络 v 上", value=False,
                help="默认在物理声压 u 上监督; 勾选则在包络空间监督",
            )
        use_pe_pde = st.checkbox(
            "抛物方程 PE-PINN", value=False,
            help="用 ∂u/∂x = (i/2k)∂²u/∂z² 替代 Helmholtz 作 PDE 残差 (适合长距离 +x 传播)",
        )
        wave_number_formula = st.selectbox(
            "波数 k 公式",
            options=["legacy_f_over_c", "2pi_f_over_c"],
            format_func=lambda v: {
                "legacy_f_over_c": "f/c (兼容旧实验)",
                "2pi_f_over_c": "2πf/c (标准 Helmholtz)",
            }[v],
            index=0,
        )
        st.caption("包络与 PE 可同时启用; Marching 与 DD 互斥 (Marching 优先).")
        marching_force = st.selectbox(
            "Sequential Marching",
            options=["auto", "on", "off"],
            format_func=lambda v: {
                "auto": "auto — length > 阈值",
                "on": "on — 强制启用",
                "off": "off — 关闭",
            }[v],
            index=0,
        )
        marching_threshold = st.number_input(
            "Marching 触发阈值 (m)", value=1000.0, step=100.0,
        )
        col_a, col_b = st.columns(2)
        marching_num_segments = col_a.number_input(
            "传播段数 (0=自动)", min_value=0, max_value=32, value=0, step=1,
        )
        marching_segment_length = col_b.number_input(
            "段长 (m, 自动时用)", min_value=50.0, max_value=2000.0,
            value=200.0, step=50.0,
        )
        marching_overlap = st.number_input(
            "Marching 重叠 (m)", min_value=0.0, max_value=500.0, value=80.0, step=10.0,
        )
        marching_shared_network = st.checkbox("各段共享网络权重", value=False)
        marching_sequential_train = st.checkbox(
            "按段顺序训练 (冻结已训段)", value=False,
        )
        marching_steps_per_segment = 0
        if marching_sequential_train:
            marching_steps_per_segment = st.number_input(
                "每段训练步数 (0=总步/段数)", min_value=0, max_value=10_000_000,
                value=0, step=1000,
            )
        col_a, col_b = st.columns(2)
        marching_interface_weight = col_a.number_input(
            "界面权重", value=10.0, step=1.0,
        )
        marching_coupling_weight = col_b.number_input(
            "单向耦合权重", value=5.0, step=0.5,
        )
        _march_on = (
            marching_force == "on"
            or (marching_force == "auto" and float(length) > float(marching_threshold))
        )
        if _march_on:
            _ns = int(marching_num_segments) if marching_num_segments > 0 else max(
                2, int(round(float(length) / max(marching_segment_length, 1)))
            )
            st.success(f"✓ Marching 将启用: {_ns} 段 (DD 自动关闭)")
        elif use_envelope or use_pe_pde:
            st.info(f"物理模式: {'Envelope+' if use_envelope else ''}{'PE' if use_pe_pde else 'Helmholtz'}")

    # ------------------------------------------------------------------ #
    # 训练稳定性 (Feature 4 - 长程训练崩溃修复)
    # ------------------------------------------------------------------ #
    with st.sidebar.expander("🛡️ 训练稳定性 (Feature 4)", expanded=False):
        gradient_clip = st.number_input(
            "梯度裁剪上限 (0=关闭)", value=1.0, min_value=0.0, max_value=100.0, step=0.1,
            help="对所有可训练参数的梯度做 clip_grad_norm_, 防止长程训练梯度爆炸.",
        )
        nan_skip_threshold = st.number_input(
            "NaN/Inf 累计阈值", value=50, min_value=1, max_value=10000, step=1,
            help="累计出现 NaN 损失超过此值则自动停止训练.",
        )
        cuda_empty_cache_every = st.number_input(
            "CUDA 缓存清理间隔", value=1000, min_value=0, max_value=100000, step=100,
            help="每多少 step 调用一次 torch.cuda.empty_cache(); 0 = 关闭.",
        )
        max_loss_points_in_memory = st.number_input(
            "Loss 历史内存上限", value=20000, min_value=1000, max_value=500000, step=1000,
            help="超出后自动下采样 (每 2 个保留 1 个), 防止 5M step 时内存爆炸.",
        )

    with st.sidebar.expander("📺 可视化 / 输出", expanded=True):
        visualize_interval = st.number_input(
            "每隔多少 step 在界面刷新",
            min_value=10, max_value=100000, value=500, step=50,
        )
        log_interval = st.number_input(
            "Loss 记录间隔",
            min_value=1, max_value=10000, value=50, step=10,
        )
        experiment_name = st.text_input(
            "实验名称 (输出子目录)",
            value=f"pinn_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        output_dir = st.text_input("输出根目录", value="outputs")
        device_choice = st.selectbox("计算设备", options=["auto", "cuda", "cpu"], index=0)

    # 迁移学习: 默认空, 由专用 tab 控件设置 (此处仅占位)
    pretrained_ckpt = st.session_state.get("transfer_pretrained_ckpt", "")
    freeze_first_n_layers = int(st.session_state.get("transfer_freeze_n", 0))
    transfer_lr_scale = float(st.session_state.get("transfer_lr_scale", 0.1))
    fourier_b_rescale = bool(st.session_state.get("transfer_fourier_b_rescale", False))
    pretrained_frequency = float(st.session_state.get("transfer_pretrained_frequency", 50.0))

    return AppConfig(
        pres_real_path=pres_real_path,
        pres_imag_path=pres_imag_path,
        length=float(length),
        depth=float(depth),
        nx=int(nx),
        nz=int(nz),
        learning_rate=float(learning_rate),
        batch_size=int(batch_size),
        pde_weight=float(pde_weight),
        num_layers=int(num_layers),
        num_neurons=int(num_neurons),
        network_type=str(network_type),
        fourier_mapping_size=int(fourier_mapping_size),
        fourier_sigma=float(fourier_sigma),
        siren_w0=float(siren_w0),
        activation=str(activation),
        epochs=int(epochs),
        frequency=float(frequency),
        sound_speed=float(sound_speed),
        source_r=float(source_r),
        source_z=float(source_z),
        source_sigma=float(source_sigma),
        source_amplitude=float(source_amplitude),
        data_weight=float(data_weight),
        boundary_weight=float(boundary_weight),
        num_collocation=int(num_collocation),
        num_boundary=int(num_boundary),
        visualize_interval=int(visualize_interval),
        log_interval=int(log_interval),
        experiment_name=experiment_name,
        output_dir=output_dir,
        device=device_choice,
        # 采样 (Feature 3)
        sampling_method=str(sampling_method),
        num_train_obs=int(num_train_obs),
        num_blocks_x=int(num_blocks_x),
        num_blocks_z=int(num_blocks_z),
        near_dist_threshold=float(near_dist_threshold),
        mid_dist_threshold=float(mid_dist_threshold),
        points_per_near_block=int(points_per_near_block),
        points_per_mid_block=int(points_per_mid_block),
        points_per_far_block=int(points_per_far_block),
        problem_region_x_min=float(pr_x_min),
        problem_region_x_max=float(pr_x_max),
        problem_region_z_min=float(pr_z_min),
        problem_region_z_max=float(pr_z_max),
        problem_region_extra_points=int(pr_extra),
        ras_initial_fraction=float(ras_initial_fraction),
        ras_refine_every=int(ras_refine_every),
        ras_points_per_refine=int(ras_points_per_refine),
        ras_candidate_size=int(ras_candidate_size),
        ras_residual_metric=str(ras_residual_metric),
        # 迁移学习 (Feature 2) - 由"🔁 迁移学习"页通过 session_state 注入
        pretrained_ckpt=str(pretrained_ckpt),
        freeze_first_n_layers=int(freeze_first_n_layers),
        transfer_lr_scale=float(transfer_lr_scale),
        fourier_b_rescale=bool(fourier_b_rescale),
        pretrained_frequency=float(pretrained_frequency),
        # 稳定性 (Feature 4)
        gradient_clip=float(gradient_clip),
        nan_skip_threshold=int(nan_skip_threshold),
        cuda_empty_cache_every=int(cuda_empty_cache_every),
        max_loss_points_in_memory=int(max_loss_points_in_memory),
        # GradNorm
        use_gradnorm=bool(use_gradnorm),
        gradnorm_alpha=float(gradnorm_alpha),
        gradnorm_lr=float(gradnorm_lr),
        gradnorm_update_every=int(gradnorm_update_every),
        gradnorm_warmup_steps=int(gradnorm_warmup_steps),
        gradnorm_min_weight=float(gradnorm_min_weight),
        # Domain Decomposition
        domain_decomp_force=str(dd_force),
        domain_decomp_threshold=float(dd_threshold),
        domain_decomp_num_subdomains=int(dd_num_sub),
        domain_decomp_overlap=float(dd_overlap),
        domain_decomp_interface_points=int(dd_interface_points),
        domain_decomp_interface_weight=float(dd_interface_weight),
        domain_decomp_deriv_weight=float(dd_deriv_weight),
        domain_decomp_overlap_fraction=float(dd_overlap_fraction),
        domain_decomp_one_way_coupling=bool(dd_one_way),
        domain_decomp_coupling_weight=float(dd_coupling_weight),
        use_envelope_decomposition=bool(use_envelope),
        envelope_k0=float(envelope_k0),
        supervise_envelope=bool(supervise_envelope),
        use_pe_pde=bool(use_pe_pde),
        wave_number_formula=str(wave_number_formula),
        marching_force=str(marching_force),
        marching_threshold=float(marching_threshold),
        marching_num_segments=int(marching_num_segments),
        marching_segment_length=float(marching_segment_length),
        marching_overlap=float(marching_overlap),
        marching_shared_network=bool(marching_shared_network),
        marching_interface_weight=float(marching_interface_weight),
        marching_one_way_coupling=True,
        marching_coupling_weight=float(marching_coupling_weight),
        marching_sequential_train=bool(marching_sequential_train),
        marching_steps_per_segment=int(marching_steps_per_segment),
    )


# =========================================================================== #
# UI 模块 2: 真实数据预览
# =========================================================================== #
def render_data_preview(config: AppConfig, container) -> None:
    """读取并展示真实声压数据."""
    try:
        from pinn_app.data.loader import load_pressure_data, generate_grid
        p_real, p_imag = load_pressure_data(
            config.pres_real_path, config.pres_imag_path,
            expected_shape=(config.nz, config.nx),
        )
        tl = compute_tl(p_real, p_imag)
        nz, nx = p_real.shape
        X, Z, _, _ = generate_grid(config.length, config.depth, nx, nz)
        fig = plot_field_triptych(X, Z, p_real, p_imag, tl, suptitle="真实声场")
        container.pyplot(fig, use_container_width=True)
        container.success(
            f"✅ 数据读取成功, 形状 (nz, nx) = ({nz}, {nx})"
        )
    except Exception as exc:
        container.error(f"❌ 读取数据失败: {exc}")


# =========================================================================== #
# UI 模块 2b: 当前采样方法实时彩色示意图 (随参数自动刷新)
# =========================================================================== #
def render_current_sampling_diagram(config: AppConfig) -> None:
    """每次侧栏改动后自动重绘当前采样方法的彩色示意图.

    - 坐标轴标注 length / depth 真实值
    - 颜色区分: stratified/aug 用三色分区, 其他用距源距离 viridis
    - 自动画近/中场距离圈, problem_region_aug 高亮问题区域矩形
    - 自动落盘到 outputs/_sampling_preview/sampling_<method>_<timestamp>.png
    """
    import matplotlib.pyplot as plt
    try:
        # 优先以 CSV 文件分辨率为准 (避免侧栏 nx,nz 与文件不一致)
        nx, nz = int(config.nx), int(config.nz)
        try:
            from pinn_app.data.loader import load_pressure_data
            p_real, _ = load_pressure_data(
                config.pres_real_path, config.pres_imag_path,
                expected_shape=(config.nz, config.nx),
            )
            nz, nx = p_real.shape
        except Exception:
            # CSV 不可读, 用侧栏值
            pass

        sr = build_observation_sampling(config, nx=nx, nz=nz)

        method = (config.sampling_method or "").lower()
        viz_style = sampling_viz_style(method)
        color_mode = "auto"

        problem_region = None
        if method in ("problem_region_aug", "problem_region", "aug"):
            problem_region = (
                config.problem_region_x_min * config.length,
                config.problem_region_x_max * config.length,
                config.problem_region_z_min * config.depth,
                config.problem_region_z_max * config.depth,
            )

        fig = plot_sampling_distribution(
            sr.coords,
            length=config.length, depth=config.depth,
            source_r=config.source_r, source_z=config.source_z,
            method_name=config.sampling_method,
            n_total=sr.size,
            color_mode=color_mode,
            viz_style=viz_style,
            near_dist=config.near_dist_threshold,
            mid_dist=config.mid_dist_threshold,
            problem_region=problem_region,
            sampling_info=sr.info,
            nx=nx, nz=nz,
        )
        st.pyplot(fig, use_container_width=True)

        # ----- 写入磁盘 (默认每次都保存; 用时间戳避免覆盖) ----- #
        try:
            preview_dir = Path(config.output_dir) / "_sampling_preview"
            preview_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            png_path = preview_dir / f"sampling_{config.sampling_method}_{ts}.png"
            fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
            # 同时保存一个 "latest" 覆盖版, 便于其他地方引用
            latest = preview_dir / f"sampling_{config.sampling_method}_latest.png"
            fig.savefig(latest, dpi=300, bbox_inches="tight", pad_inches=0.1)
            st.caption(f"📁 已保存: `{png_path}`")
        except Exception as exc:
            st.warning(f"⚠ 保存采样图失败: {exc}")
        finally:
            plt.close(fig)

        # 简明统计
        with st.expander("📊 采样统计信息", expanded=False):
            st.json(sr.info)
    except Exception as exc:
        st.error(f"❌ 生成采样示意图失败: {exc}")


# =========================================================================== #
# UI 模块 2c: 多种采样方法横向对比
# =========================================================================== #
def render_sampling_preview(config: AppConfig, container) -> None:
    """按当前参数生成全部采样方法的分布对比, 不进行训练."""
    import copy
    try:
        nx, nz = int(config.nx), int(config.nz)
        # 优先以 CSV 文件分辨率为准
        try:
            from pinn_app.data.loader import load_pressure_data
            p_real, _ = load_pressure_data(
                config.pres_real_path, config.pres_imag_path,
                expected_shape=(config.nz, config.nx),
            )
            nz, nx = p_real.shape
            container.success(
                f"✅ 读取真实数据成功, 以文件分辨率 (nz, nx) = ({nz}, {nx}) 生成采样预览"
            )
        except Exception:
            container.info(
                f"未能读取声压 CSV, 按侧边栏 (nz, nx)=({nz}, {nx}) 生成预览"
            )

        method_to_result = {}
        for method in SAMPLING_METHODS:
            cfg_m = copy.deepcopy(config)
            cfg_m.sampling_method = method
            sr = build_observation_sampling(cfg_m, nx=nx, nz=nz)
            method_to_result[method] = sr

        # 每种方法独立彩图 (彩色按区/距离)
        st.markdown("##### 每种方法独立彩图")
        for method, sr in method_to_result.items():
            viz_style = sampling_viz_style(method)
            problem_region = None
            if method == "problem_region_aug":
                problem_region = (
                    config.problem_region_x_min * config.length,
                    config.problem_region_x_max * config.length,
                    config.problem_region_z_min * config.depth,
                    config.problem_region_z_max * config.depth,
                )
            fig = plot_sampling_distribution(
                sr.coords,
                length=config.length, depth=config.depth,
                source_r=config.source_r, source_z=config.source_z,
                method_name=method, n_total=sr.size,
                color_mode="auto",
                viz_style=viz_style,
                near_dist=config.near_dist_threshold,
                mid_dist=config.mid_dist_threshold,
                problem_region=problem_region,
                sampling_info=sr.info,
                nx=nx, nz=nz,
            )
            container.pyplot(fig, use_container_width=True)
            import matplotlib.pyplot as plt; plt.close(fig)

        # 5 种放在一张大图 (距离 viridis 上色)
        st.markdown("##### 全部方法 · 总览对比")
        fig_all = plot_sampling_methods_panel(
            method_to_result,
            length=config.length, depth=config.depth,
            source_r=config.source_r, source_z=config.source_z,
            near_dist=config.near_dist_threshold,
            mid_dist=config.mid_dist_threshold,
            ncols=3,
            nx=nx, nz=nz,
        )
        container.pyplot(fig_all, use_container_width=True)
        import matplotlib.pyplot as plt; plt.close(fig_all)

        # 可选保存到 outputs/_sampling_preview/
        with st.expander("💾 保存采样预览图到磁盘"):
            target_dir = st.text_input(
                "保存目录",
                value=str(Path(config.output_dir) / "_sampling_preview"),
                key="sampling_preview_dir",
            )
            if st.button("⬇ 保存全部预览图"):
                target = Path(target_dir); target.mkdir(parents=True, exist_ok=True)
                import matplotlib.pyplot as plt
                from pinn_app.utils.visualization import save_figure_publication
                for method, sr in method_to_result.items():
                    viz_style = sampling_viz_style(method)
                    problem_region = None
                    if method == "problem_region_aug":
                        problem_region = (
                            config.problem_region_x_min * config.length,
                            config.problem_region_x_max * config.length,
                            config.problem_region_z_min * config.depth,
                            config.problem_region_z_max * config.depth,
                        )
                    fig = plot_sampling_distribution(
                        sr.coords,
                        length=config.length, depth=config.depth,
                        source_r=config.source_r, source_z=config.source_z,
                        method_name=method, n_total=sr.size,
                        color_mode="auto",
                        viz_style=viz_style,
                        near_dist=config.near_dist_threshold,
                        mid_dist=config.mid_dist_threshold,
                        problem_region=problem_region,
                        sampling_info=sr.info,
                        nx=nx, nz=nz,
                    )
                    save_figure_publication(
                        fig, target / f"sampling_{method}", formats=("png",),
                    )
                fig_all = plot_sampling_methods_panel(
                    method_to_result,
                    length=config.length, depth=config.depth,
                    source_r=config.source_r, source_z=config.source_z,
                    near_dist=config.near_dist_threshold,
                    mid_dist=config.mid_dist_threshold,
                    ncols=3,
                    nx=nx, nz=nz,
                )
                save_figure_publication(
                    fig_all, target / "sampling_methods_compare", formats=("png",),
                )
                st.success(f"✅ 已保存到 {target.resolve()}")
    except Exception as exc:
        container.error(f"❌ 采样预览失败: {exc}")


# =========================================================================== #
# 训练线程
# =========================================================================== #
class TrainingThread(threading.Thread):
    """在独立线程运行 Trainer, 把回调状态推入 queue."""

    def __init__(self, config: AppConfig, msg_queue: "queue.Queue", stop_event: threading.Event):
        super().__init__(daemon=True)
        self.config = config
        self.msg_queue = msg_queue
        self.stop_event = stop_event
        self.result: Optional[dict] = None
        self.error: Optional[str] = None

    def run(self) -> None:
        try:
            def _cb(state: dict) -> None:
                # 丢弃过旧的中间帧, 只保留最新的可视化状态
                try:
                    self.msg_queue.put_nowait({"type": "state", "state": state})
                except queue.Full:
                    pass

            trainer = Trainer(
                config=self.config,
                callback=_cb,
                stop_flag=lambda: self.stop_event.is_set(),
            )
            self.msg_queue.put({
                "type": "event",
                "message": f"模型: {trainer.model.summary()}; 设备: {trainer.device}",
            })
            self.result = trainer.train()
            self.msg_queue.put({"type": "done", "result_keys": list(self.result.keys())})
        except Exception as exc:
            import traceback
            self.error = f"{exc}\n{traceback.format_exc()}"
            self.msg_queue.put({"type": "error", "message": self.error})


# =========================================================================== #
# 训练监控面板 (UI 模块 3)
# =========================================================================== #
def _ensure_session_keys() -> None:
    ss = st.session_state
    ss.setdefault("training_active", False)
    ss.setdefault("training_thread", None)
    ss.setdefault("training_queue", None)
    ss.setdefault("training_stop_event", threading.Event())
    ss.setdefault("training_events", [])
    ss.setdefault("training_latest_state", None)
    ss.setdefault("training_config_snapshot", None)


def render_training_monitor(config: AppConfig) -> None:
    """启动训练并实时刷新界面.

    使用 session_state 持久化训练状态, 即使 Streamlit rerun 也能自动恢复监控视图.
    所有产物保存由 Trainer 自身负责 (磁盘), 与 UI 是否存活无关.
    """
    _ensure_session_keys()
    st.subheader("🚀 训练进度与实时可视化")

    ss = st.session_state
    col_ctrl1, col_ctrl2, _ = st.columns([1, 1, 3])
    start = col_ctrl1.button(
        "▶ 开始训练", type="primary", use_container_width=True,
        disabled=ss.training_active,
    )
    stop = col_ctrl2.button(
        "⏹ 停止训练", use_container_width=True, disabled=not ss.training_active,
    )

    if stop and ss.training_active:
        ss.training_stop_event.set()
        st.warning("已请求停止, 当前 step 结束后会退出.")

    # 触发新训练
    if start and not ss.training_active:
        ss.training_stop_event = threading.Event()
        ss.training_queue = queue.Queue(maxsize=32)
        ss.training_thread = TrainingThread(
            config, ss.training_queue, ss.training_stop_event,
        )
        ss.training_thread.start()
        ss.training_active = True
        ss.training_events = []
        ss.training_latest_state = None
        ss.training_config_snapshot = config

    # 没在训练 -> 显示提示并退出
    if not ss.training_active:
        # 若上次训练有结果, 仍展示已保存的产物
        if ss.training_thread is not None and ss.training_thread.result is not None:
            _render_post_training(ss.training_config_snapshot or config,
                                  ss.training_thread.result,
                                  ss.training_latest_state)
        else:
            st.info("设置好侧边栏参数后, 点击 **开始训练**. 训练中可随意操作其他控件, "
                    "界面会自动恢复; 即便关闭网页, 磁盘也会保存完整结果.")
        return

    # ============ 训练进行中 -> 渲染监控 ============ #
    snapshot_cfg = ss.training_config_snapshot or config

    progress_bar = st.progress(0.0, text="准备启动...")
    status_text = st.empty()
    metrics_area = st.empty()
    loss_area = st.empty()
    field_area = st.empty()
    error_area = st.empty()
    log_area = st.expander("📜 事件日志", expanded=False).empty()

    thread: TrainingThread = ss.training_thread
    msg_queue: "queue.Queue" = ss.training_queue

    # 先渲染"上次已知状态" (rerun 后可立即看到, 不必等下一帧)
    if ss.training_latest_state is not None:
        _render_latest_state(
            ss.training_latest_state, snapshot_cfg,
            progress_bar=progress_bar, status_text=status_text,
            metrics_area=metrics_area, loss_area=loss_area,
            field_area=field_area, error_area=error_area,
        )
    if ss.training_events:
        log_area.write("\n".join(ss.training_events[-200:]))

    # 轮询新消息直到训练结束
    error_msg: Optional[str] = None
    while thread.is_alive() or not msg_queue.empty():
        try:
            msg = msg_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        mtype = msg.get("type")
        if mtype == "state":
            ss.training_latest_state = msg["state"]
            _render_latest_state(
                ss.training_latest_state, snapshot_cfg,
                progress_bar=progress_bar, status_text=status_text,
                metrics_area=metrics_area, loss_area=loss_area,
                field_area=field_area, error_area=error_area,
            )
        elif mtype == "event":
            ss.training_events.append(msg["message"])
            log_area.write("\n".join(ss.training_events[-200:]))
        elif mtype == "error":
            error_msg = msg["message"]
            break
        elif mtype == "done":
            break

    thread.join(timeout=2.0)
    ss.training_active = False

    if error_msg or thread.error:
        st.error(f"❌ 训练出错:\n{error_msg or thread.error}")
        return
    if thread.result is not None:
        st.success(f"🎉 训练完成! 全部产物已保存到: `{snapshot_cfg.experiment_dir}`")
        _render_post_training(snapshot_cfg, thread.result, ss.training_latest_state)


# --------------------------------------------------------------------------- #
def _render_latest_state(
    state: dict,
    config: AppConfig,
    progress_bar,
    status_text,
    metrics_area,
    loss_area,
    field_area,
    error_area,
) -> None:
    step = state["step"]
    total = state["total_steps"]
    frac = step / max(total, 1)
    progress_bar.progress(min(max(frac, 0.0), 1.0),
                         text=f"Step {step}/{total} · {frac*100:.1f}%")

    losses = state["losses"]
    status_text.info(
        f"⏱ 已训练 {state['elapsed']:.1f}s  |  当前 lr={state['lr']:.2e}  "
        f"|  total={losses['total']:.4e}  data={losses['data']:.4e}  "
        f"pde={losses['pde']:.4e}  bc={losses['bc']:.4e}"
    )

    # 关键指标卡片
    m = state["metrics"]
    with metrics_area.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TL RMSE", f"{m.get('tl_rmse', 0):.3f}")
        c2.metric("TL MAE", f"{m.get('tl_mae', 0):.3f}")
        c3.metric("TL Rel-L2", f"{m.get('tl_rel_l2', 0):.3e}")
        c4.metric("TL Corr", f"{m.get('tl_corr', 0):.3f}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("实部 RMSE", f"{m.get('real_rmse', 0):.3e}")
        c2.metric("虚部 RMSE", f"{m.get('imag_rmse', 0):.3e}")
        c3.metric("实部 Corr", f"{m.get('real_corr', 0):.3f}")
        c4.metric("虚部 Corr", f"{m.get('imag_corr', 0):.3f}")

    import matplotlib.pyplot as plt

    def _safe_render(area, fig, caption: str):
        """渲染 figure 时若 matplotlib mathtext 出现 ParseException 等异常,
        不中断整个 UI; 用警告替代图像."""
        try:
            area.pyplot(fig, use_container_width=True)
        except Exception as exc:
            area.warning(f"⚠ {caption} 渲染失败 (matplotlib): {type(exc).__name__}")
        finally:
            try:
                plt.close(fig)
            except Exception:
                pass

    # Loss 曲线
    steps = state["loss_steps"]
    history = state["loss_history"]
    if steps:
        try:
            fig_loss = plot_loss_curve(steps, history, title=f"Loss 曲线 (step {step})")
            _safe_render(loss_area, fig_loss, "Loss 曲线")
        except Exception as exc:
            loss_area.warning(f"⚠ Loss 曲线生成失败: {exc}")

    # 声场预测
    try:
        fig_field = plot_field_triptych(
            state["X"], state["Z"],
            state["pred_real"], state["pred_imag"], state["pred_tl"],
            suptitle=f"预测声场 (step {step})",
        )
        _safe_render(field_area, fig_field, "预测声场")
    except Exception as exc:
        field_area.warning(f"⚠ 预测声场图生成失败: {exc}")

    # 误差分布 (实部误差示例)
    try:
        err_fig = plot_error_map(
            state["X"], state["Z"],
            state["pred_tl"], state["true_tl"],
            title=f"TL 绝对误差分布 (step {step})",
        )
        _safe_render(error_area, err_fig, "TL 误差图")
    except Exception as exc:
        error_area.warning(f"⚠ 误差图生成失败: {exc}")


# --------------------------------------------------------------------------- #
# 训练后的最终科研输出  (只读已保存的产物展示, 保存逻辑全部已在 Trainer 中执行)
# --------------------------------------------------------------------------- #
def _render_post_training(config: AppConfig, result: dict, last_state: Optional[dict]) -> None:
    """读取 trainer 已保存的图片和 CSV, 展示给用户."""
    subdirs = result["subdirs"]
    base = config.experiment_dir
    st.subheader("📁 全部输出文件 (已保存到磁盘)")
    with st.expander("展开查看", expanded=False):
        for root, _, files in os.walk(base):
            rel = os.path.relpath(root, base)
            if files:
                st.write(f"**{rel}/**")
                for f in files:
                    st.write(f" - {f}")

    # 1) 损失曲线
    loss_all = Path(subdirs["loss"]) / "loss_all.png"
    if loss_all.exists():
        st.subheader("📉 损失曲线 (汇总)")
        st.image(str(loss_all))

    # 2) 预测三联图
    pred_fig = Path(subdirs["field"]) / "pred_triptych.png"
    if pred_fig.exists():
        st.subheader("🎯 最终预测声场 (实部 / 虚部 / TL)")
        st.image(str(pred_fig))

    # 3) 真实-预测对比
    cmp_fig = Path(subdirs["field"]) / "compare_true_vs_pred.png"
    if cmp_fig.exists():
        st.subheader("🔍 真实 vs 预测 对比")
        st.image(str(cmp_fig))

    # 4) 误差分布
    err_files = [Path(subdirs["field"]) / f"err_{n}.png" for n in ("real", "imag", "tl")]
    if all(p.exists() for p in err_files):
        st.subheader("📊 误差分布")
        cols = st.columns(3)
        for col, p in zip(cols, err_files):
            col.image(str(p))

    # 5) 剖面
    prof_fig = Path(subdirs["profiles"]) / "profile_tl_mid.png"
    if prof_fig.exists():
        st.subheader("📏 TL 深度剖面对比")
        st.image(str(prof_fig))

    # 6) PDE 残差
    res_fig = Path(subdirs["residual"]) / "pde_residual.png"
    if res_fig.exists():
        st.subheader("🧮 PDE 残差")
        st.image(str(res_fig))

    # 7) 最终指标 CSV
    metric_csv = Path(subdirs["metrics"]) / "final_metrics.csv"
    if metric_csv.exists():
        st.subheader("📈 最终指标")
        st.dataframe(pd.read_csv(metric_csv).T.rename(columns={0: "value"}))

    # 8) 参数 Excel (Feature 1) - 提供下载
    xlsx_path = Path(subdirs["metrics"]) / "parameters.xlsx"
    if xlsx_path.exists():
        st.subheader("📑 训练参数 Excel (一键归档)")
        st.caption(
            "已自动保存本次训练全部参数 / 指标 / 采样信息 / 迁移配置 / 运行时. "
            "建议为每次实验保留这份 Excel."
        )
        try:
            with open(xlsx_path, "rb") as f:
                st.download_button(
                    "⬇ 下载 parameters.xlsx",
                    data=f.read(),
                    file_name=f"{config.experiment_name}_parameters.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        except Exception as exc:
            st.warning(f"读取参数 Excel 失败: {exc}")

    # 9) 误差图: 百分位 vs Log vs 多视图 (Feature 3)
    multi_files = [Path(subdirs["field"]) / f"err_{n}_multi.png"
                   for n in ("tl", "real", "imag")]
    if all(p.exists() for p in multi_files):
        st.subheader("🔬 误差图 (三色阶对比, 适合小 RMSE 显示细节)")
        st.caption(
            "**线性色阶** 在 RMSE 极小时会一片同色; "
            "**百分位 [p1–p99]** 裁掉离群值后可呈现细节; "
            "**log 色阶** 适合跨多个数量级的误差. 三者并排, 论文按需选用."
        )
        for p in multi_files:
            st.image(str(p), use_container_width=True)

    # 10) 采样分布 (Feature 2)
    samp_fig = Path(subdirs["logs"]) / "sampling_distribution.png"
    if samp_fig.exists():
        st.subheader("🎲 训练样本点分布")
        st.image(str(samp_fig))

    # 11) 时间图
    time_files = [
        ("⏱ 训练耗时 vs Step",    Path(subdirs["logs"]) / "time_vs_step.png"),
        ("⏱ 单步耗时",             Path(subdirs["logs"]) / "step_time_per_iter.png"),
        ("⏱ Loss vs 训练耗时",     Path(subdirs["logs"]) / "loss_vs_time.png"),
    ]
    visible = [(t, p) for t, p in time_files if p.exists()]
    if visible:
        st.subheader("⏱ 训练时间成本")
        cols = st.columns(min(3, len(visible)))
        for (title, p), col in zip(visible, cols):
            col.image(str(p), caption=title, use_container_width=True)

    # 12) GradNorm 权重曲线
    gn_fig = Path(subdirs["loss"]) / "gradnorm_weights.png"
    if gn_fig.exists():
        st.subheader("⚖️ GradNorm 自适应权重变化")
        st.caption("GradNorm 自动调整各损失项权重, 让 data / pde / bc / interface 以相近速率收敛.")
        st.image(str(gn_fig))

    # 13) Domain Decomposition 示意图
    dd_fig = Path(subdirs["logs"]) / "domain_decomposition.png"
    if dd_fig.exists():
        st.subheader("🌐 Domain Decomposition 子域示意")
        st.image(str(dd_fig))


# =========================================================================== #
# Feature 2: 跨频率迁移学习 (UI 页)
# =========================================================================== #
def render_transfer_learning_tab() -> None:
    """跨频率迁移学习配置页. 把参数写入 session_state, 训练页读取."""
    st.subheader("🔁 跨频率迁移学习")
    st.markdown(
        """
        从已训练好的 PINN 模型 (`.pt` 文件) 继续训练, 目标是把**低频学到的几何 / 边界先验**
        迁移到**同一海洋环境下的其他频率**.

        **可行性**: 不同频率下的解析声场结构差异较大, 不能直接复用预测,
        但网络底层学到的特征 (边界条件、源附近形状) 对新频率有显著加速作用.
        通常微调比从零训练快 **3-10×**.

        **使用方法**:
        1. 设置预训练 `.pt` 路径
        2. 在左侧侧边栏修改 **频率** 为目标频率
        3. 在左侧侧边栏修改 **声压实/虚部 CSV 路径** 为目标频率对应的数据
        4. 选择冻结层数 / lr 缩放
        5. 切到「🏋️ 训练 & 监控」点击开始训练
        """
    )

    ss = st.session_state
    col1, col2 = st.columns([3, 1])
    def _clear_transfer_pretrained_ckpt() -> None:
        # 必须在回调中修改：同一 rerun 内 text_input 已实例化后不能直接写 ss[key]
        ss["transfer_pretrained_ckpt"] = ""

    pretrained_ckpt = col1.text_input(
        "预训练模型 .pt 路径",
        value=ss.get("transfer_pretrained_ckpt", ""),
        placeholder=r"D:\lzz\Auto_train_PINN\outputs\xxx\model\best.pt",
        key="transfer_pretrained_ckpt",
    )
    col2.button(
        "🧹 关闭迁移",
        use_container_width=True,
        on_click=_clear_transfer_pretrained_ckpt,
    )

    enabled = bool(pretrained_ckpt and Path(pretrained_ckpt).exists())
    if pretrained_ckpt and not enabled:
        st.error(f"路径不存在: {pretrained_ckpt}")
    elif enabled:
        st.success(f"✓ 迁移学习已启用. 训练时会从该 ckpt 加载权重.")

    # 配置
    col_a, col_b, col_c = st.columns(3)
    freeze_n = col_a.number_input(
        "冻结前 N 层 Linear",
        min_value=0, max_value=16, value=int(ss.get("transfer_freeze_n", 0)),
        step=1,
        help="冻结底层特征提取层. 0 = 全量微调; 推荐 2-4 (保留几何先验).",
        key="transfer_freeze_n",
    )
    lr_scale = col_b.number_input(
        "学习率缩放",
        min_value=1e-4, max_value=10.0, value=float(ss.get("transfer_lr_scale", 0.1)),
        step=0.05, format="%.4f",
        help="加载预训练后的学习率乘以此系数 (默认 0.1: 1e-3 → 1e-4).",
        key="transfer_lr_scale",
    )
    pretrained_freq = col_c.number_input(
        "预训练时频率 (Hz)",
        min_value=1.0, max_value=10000.0,
        value=float(ss.get("transfer_pretrained_frequency", 50.0)), step=1.0,
        help="若启用 Fourier B 缩放, 会按 f_new/f_old 重新缩放 Fourier B 矩阵.",
        key="transfer_pretrained_frequency",
    )
    fourier_rescale = st.checkbox(
        "Fourier 网络: 按 f_new / f_old 缩放 B 矩阵",
        value=bool(ss.get("transfer_fourier_b_rescale", False)),
        help="只对 Fourier Feature 网络有效. 高频差异较大时建议开启.",
        key="transfer_fourier_b_rescale",
    )

    # 显示预训练 ckpt 信息
    if enabled:
        try:
            ckpt = torch.load(pretrained_ckpt, map_location="cpu", weights_only=False)
            with st.expander("📋 预训练 checkpoint 信息", expanded=False):
                st.write(f"**Step**: {ckpt.get('step', '?')}")
                st.write(f"**State keys**: {len(ckpt.get('model_state', {}))} 张量")
                if "config" in ckpt:
                    st.write("**Config (节选)**:")
                    cfg = ckpt["config"]
                    show_keys = [
                        "network_type", "num_layers", "num_neurons",
                        "frequency", "learning_rate", "epochs",
                        "sampling_method", "experiment_name",
                    ]
                    st.json({k: cfg.get(k) for k in show_keys if k in cfg})
        except Exception as exc:
            st.warning(f"读取 checkpoint 信息失败: {exc}")


# =========================================================================== #
# Feature 3: 实验对比页
# =========================================================================== #
def render_compare_tab() -> None:
    """扫描 outputs 下所有实验, 横向对比 RMSE / 时间 / 采样."""
    import json as _json
    st.subheader("📊 实验对比")
    st.caption("自动扫描 outputs 下各实验的 rmse_summary.csv, 横向比较.")

    root = Path(st.text_input("outputs 根目录", value="outputs"))
    if not root.exists():
        st.warning(f"目录不存在: {root.resolve()}")
        return

    summaries = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        csv = d / "metrics" / "rmse_summary.csv"
        if not csv.exists():
            continue
        try:
            row = pd.read_csv(csv).iloc[0].to_dict()
            row["experiment"] = d.name
            row["path"] = str(d)
            summaries.append(row)
        except Exception:
            continue

    if not summaries:
        st.info("尚未发现 rmse_summary.csv. 完成至少一次训练后再查看.")
        return

    df = pd.DataFrame(summaries)
    cols_first = [
        "experiment", "network_type", "sampling_method",
        "num_train_obs", "epochs", "actual_steps", "elapsed_seconds",
        "tl_rmse", "real_rmse", "imag_rmse",
        "tl_near_rmse", "tl_mid_rmse", "tl_far_rmse",
        "tl_corr", "real_corr", "imag_corr",
        "frequency", "num_layers", "num_neurons",
    ]
    cols_order = [c for c in cols_first if c in df.columns] + \
                 [c for c in df.columns if c not in cols_first]
    df = df[cols_order]

    st.markdown("#### 全部实验汇总表")
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇ 下载汇总 CSV", data=csv_bytes,
                       file_name="all_experiments_summary.csv", mime="text/csv")

    # 选择若干实验做对比图
    st.markdown("#### 多实验对比图")
    selected = st.multiselect(
        "选择实验 (多选)", options=df["experiment"].tolist(),
        default=df["experiment"].tolist()[: min(5, len(df))],
    )
    if not selected:
        return

    import matplotlib.pyplot as plt
    sub = df[df["experiment"].isin(selected)].copy()

    # 1) 时间 vs TL RMSE 散点 (核心: 时间-精度 trade-off)
    if {"elapsed_seconds", "tl_rmse"}.issubset(sub.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        for _, row in sub.iterrows():
            ax.scatter(row["elapsed_seconds"], row["tl_rmse"], s=80, alpha=0.8)
            ax.annotate(row["experiment"],
                        (row["elapsed_seconds"], row["tl_rmse"]),
                        fontsize=8, alpha=0.8)
        ax.set_xlabel("训练耗时 (s)")
        ax.set_ylabel("TL RMSE (dB)")
        ax.set_title("时间 - 精度 trade-off")
        ax.set_yscale("log"); ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close(fig)

    # 2) 不同采样方法的 RMSE 柱状图
    if "sampling_method" in sub.columns:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, col, ttl in zip(axes,
                                ["tl_rmse", "real_rmse", "imag_rmse"],
                                ["TL RMSE", "实部 RMSE", "虚部 RMSE"]):
            if col in sub.columns:
                xs = sub["experiment"]
                ys = sub[col]
                bars = ax.bar(range(len(xs)), ys, alpha=0.8)
                ax.set_xticks(range(len(xs)))
                ax.set_xticklabels(xs, rotation=45, ha="right", fontsize=8)
                ax.set_ylabel(col); ax.set_title(ttl)
                ax.set_yscale("log"); ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    # 3) 近 / 中 / 远场 RMSE 分区柱状图
    region_cols = ["tl_near_rmse", "tl_mid_rmse", "tl_far_rmse"]
    if all(c in sub.columns for c in region_cols):
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(sub))
        w = 0.25
        for i, c in enumerate(region_cols):
            ax.bar(x + (i - 1) * w, sub[c], width=w, label=c.replace("tl_", "TL "))
        ax.set_xticks(x)
        ax.set_xticklabels(sub["experiment"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("RMSE (dB)")
        ax.set_yscale("log"); ax.legend(); ax.grid(True, alpha=0.3, axis="y")
        ax.set_title("近/中/远场 TL RMSE 分区对比")
        fig.tight_layout()
        st.pyplot(fig); plt.close(fig)

    # 4) 每个选中实验的 loss-vs-time 曲线 (如果有 time_history.csv)
    st.markdown("#### 各实验的 Loss-vs-时间 曲线")
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0
    for exp_name in selected:
        path = Path(sub[sub["experiment"] == exp_name].iloc[0]["path"]) / "logs" / "time_history.csv"
        if not path.exists():
            continue
        try:
            df_t = pd.read_csv(path)
            ax.plot(df_t["elapsed_s"], df_t["total"], label=exp_name, linewidth=1.4)
            plotted += 1
        except Exception:
            continue
    if plotted > 0:
        ax.set_xlabel("累积耗时 (s)"); ax.set_ylabel("Total Loss")
        ax.set_yscale("log"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        ax.set_title("不同实验的 Loss vs 时间")
        st.pyplot(fig)
    else:
        st.info("选中的实验都没有 logs/time_history.csv (旧版本不会生成).")
    plt.close(fig)


# =========================================================================== #
def _finalize_outputs_DEPRECATED(config: AppConfig, result: dict, last_state: Optional[dict]) -> None:
    subdirs = result["subdirs"]
    loss_steps = result["loss_steps"]
    loss_history = result["loss_history"]

    # 1. 保存所有 loss 图 (汇总 + 单独)
    saved_loss = save_all_loss_figures(loss_steps, loss_history, subdirs["loss"])
    st.write("📉 **损失曲线图已保存:**")
    st.json(saved_loss)

    # 2. 预测全场 & 保存结果
    if last_state is None:
        st.warning("未拿到最终预测结果")
        return

    X, Z = last_state["X"], last_state["Z"]
    p_real, p_imag, tl = last_state["pred_real"], last_state["pred_imag"], last_state["pred_tl"]
    tr, ti, tt = last_state["true_real"], last_state["true_imag"], last_state["true_tl"]

    # 2a. 预测声场三联图
    field_fig = plot_field_triptych(X, Z, p_real, p_imag, tl, suptitle="最终预测声场")
    field_path = Path(subdirs["field"]) / "pred_triptych.png"
    field_fig.savefig(field_path, dpi=200, bbox_inches="tight")
    st.subheader("🎯 预测声场 (实部 / 虚部 / TL)")
    st.pyplot(field_fig, use_container_width=True)

    # 2b. 真实 vs 预测 对比图
    compare_fig = plot_field_comparison(X, Z, (tr, ti, tt), (p_real, p_imag, tl))
    compare_path = Path(subdirs["field"]) / "compare_true_vs_pred.png"
    compare_fig.savefig(compare_path, dpi=200, bbox_inches="tight")
    st.subheader("🔍 真实 vs 预测 对比")
    st.pyplot(compare_fig, use_container_width=True)

    # 3. 误差分布
    st.subheader("📊 误差分布图")
    col1, col2, col3 = st.columns(3)
    err_real = plot_error_map(X, Z, p_real, tr, title="实部误差")
    err_imag = plot_error_map(X, Z, p_imag, ti, title="虚部误差")
    err_tl = plot_error_map(X, Z, tl, tt, title="传输损失误差")
    err_real.savefig(Path(subdirs["field"]) / "err_real.png", dpi=200, bbox_inches="tight")
    err_imag.savefig(Path(subdirs["field"]) / "err_imag.png", dpi=200, bbox_inches="tight")
    err_tl.savefig(Path(subdirs["field"]) / "err_tl.png", dpi=200, bbox_inches="tight")
    col1.pyplot(err_real, use_container_width=True)
    col2.pyplot(err_imag, use_container_width=True)
    col3.pyplot(err_tl, use_container_width=True)

    # 4. 剖面对比 (取中间距离处的深度剖面)
    nz, nx = p_real.shape
    mid = nx // 2
    z_axis = Z[:, mid]
    prof_fig = plot_profile_comparison(
        z_axis, tt[:, mid], tl[:, mid],
        xlabel="深度 z (m)", ylabel="传输损失 (dB)",
        title=f"x = {X[0, mid]:.1f} m 处 TL 深度剖面",
    )
    prof_path = Path(subdirs["profiles"]) / "profile_tl_mid.png"
    prof_fig.savefig(prof_path, dpi=200, bbox_inches="tight")
    st.subheader("📏 TL 深度剖面对比 (x=中间)")
    st.pyplot(prof_fig, use_container_width=True)

    # 5. PDE 残差图
    st.subheader("🧮 PDE 残差 (Helmholtz)")
    try:
        trainer = result.get("trainer")
        if trainer is not None:
            x_sub, z_sub, res_mag = trainer.compute_pde_residual_field(max_points=20000)
            import matplotlib.pyplot as plt
            fig_res, ax = plt.subplots(figsize=(7, 4.5))
            sc = ax.scatter(x_sub, z_sub, c=res_mag, cmap="inferno", s=3)
            fig_res.colorbar(sc, ax=ax, label="|Helmholtz residual|")
            ax.set_xlabel("距离 x (m)"); ax.set_ylabel("深度 z (m)")
            ax.set_title("PDE 残差幅值 (下采样)")
            ax.invert_yaxis()
            fig_res.tight_layout()
            res_path = Path(subdirs["residual"]) / "pde_residual.png"
            fig_res.savefig(res_path, dpi=200, bbox_inches="tight")
            st.pyplot(fig_res, use_container_width=True)
            pd.DataFrame({"x": x_sub, "z": z_sub, "residual": res_mag}).to_csv(
                Path(subdirs["residual"]) / "pde_residual.csv", index=False
            )
    except Exception as exc:
        st.warning(f"残差图生成失败: {exc}")

    # 6. 导出指标 CSV
    final_metrics = {}
    final_metrics.update(compute_all_metrics(p_real, tr, prefix="real_"))
    final_metrics.update(compute_all_metrics(p_imag, ti, prefix="imag_"))
    final_metrics.update(compute_all_metrics(tl, tt, prefix="tl_"))
    metric_df = pd.DataFrame([final_metrics])
    metric_path = Path(subdirs["metrics"]) / "final_metrics.csv"
    metric_df.to_csv(metric_path, index=False)
    st.subheader("📈 最终指标 (RMSE / MAE / Relative L2 / Corr)")
    st.dataframe(metric_df.T.rename(columns={0: "value"}))

    # 7. 导出预测声场 (CSV, 供下游分析)
    pd.DataFrame(p_real).to_csv(Path(subdirs["field"]) / "pred_real.csv", index=False, header=False)
    pd.DataFrame(p_imag).to_csv(Path(subdirs["field"]) / "pred_imag.csv", index=False, header=False)
    pd.DataFrame(tl).to_csv(Path(subdirs["field"]) / "pred_tl.csv", index=False, header=False)

    # 8. 输出汇总
    st.success(f"✅ 全部结果已保存到: {config.experiment_dir}")
    with st.expander("📁 输出文件一览"):
        for root, _, files in os.walk(config.experiment_dir):
            rel = os.path.relpath(root, config.experiment_dir)
            st.write(f"**{rel}/**")
            for f in files:
                st.write(f" - {f}")


# =========================================================================== #
# 主入口
# =========================================================================== #
def main() -> None:
    st.title("🌊 PINN 水下声场交互式训练平台")
    st.caption("Helmholtz 方程 · PyTorch + Streamlit · 支持实时训练监控与科研级结果输出")

    config = get_user_inputs()

    tab_data, tab_train, tab_transfer, tab_compare, tab_about = st.tabs([
        "📊 数据预览", "🏋️ 训练 & 监控",
        "🔁 迁移学习", "📈 实验对比", "ℹ️ 说明",
    ])

    with tab_data:
        st.subheader("真实声压数据预览")
        preview_container = st.container()
        do_preview_data = st.button("🔍 加载 & 预览声压数据", use_container_width=True)
        if do_preview_data:
            render_data_preview(config, preview_container)
        else:
            preview_container.info(
                "点击 **加载 & 预览声压数据** 检查侧边栏指定的 CSV 文件."
            )

        st.divider()
        # ----- 自动随参数刷新: 当前采样方法的彩色示意图 ----- #
        st.subheader("🎲 当前采样方法示意图 (实时随侧栏参数刷新)")
        render_current_sampling_diagram(config)

        st.divider()
        with st.expander("🆚 采样方法对比 (耗时一点, 按需展开)", expanded=False):
            if st.button("生成对比图", key="btn_compare_5_methods"):
                render_sampling_preview(config, st.container())

    with tab_train:
        # 显示迁移学习当前状态
        if config.pretrained_ckpt:
            st.info(f"🔁 **迁移学习模式**: 将从 `{config.pretrained_ckpt}` 加载预训练权重 "
                    f"(冻结前 {config.freeze_first_n_layers} 层, lr×{config.transfer_lr_scale})")
        if config.sampling_method != "uniform":
            st.info(f"🎲 **采样方法**: {config.sampling_method} "
                    f"(num_train_obs ≤ {config.num_train_obs})")
        # GradNorm 状态
        if config.use_gradnorm:
            st.info(
                f"⚖️ **GradNorm 启用**: 自动平衡 data/pde/bc"
                + ("/interface" if (config.marching_enabled or config.domain_decomp_enabled) else "")
                + f" 损失权重 (α={config.gradnorm_alpha}, warmup={config.gradnorm_warmup_steps})"
            )
        if config.use_envelope_decomposition or config.use_pe_pde or config.marching_enabled:
            st.info(f"📡 **物理模式**: {config.physics_mode_label}  (k 公式: {config.wave_number_formula})")
        if config.marching_enabled:
            st.info(
                f"➡️ **Sequential Marching**: {config.marching_resolved_num_segments} 段, "
                f"段长≈{config.length / config.marching_resolved_num_segments:.0f}m, "
                f"顺序训练={'是' if config.marching_sequential_train else '否'}"
            )
        elif config.domain_decomp_enabled:
            st.info(
                f"🌐 **Domain Decomposition 启用**: length={config.length}m > "
                f"{config.domain_decomp_threshold}m → 将分成 "
                f"{config.domain_decomp_resolved_num} 个子域 "
                f"(overlap={config.domain_decomp_overlap}m)"
            )
        render_training_monitor(config)

    with tab_transfer:
        render_transfer_learning_tab()

    with tab_compare:
        render_compare_tab()

    with tab_about:
        st.markdown(
            """
### 功能说明

#### 训练产物 (`outputs/<experiment_name>/`)

| 子目录 | 内容 |
|---|---|
| `loss/` | 汇总损失图 + 各分量单独损失图 (PNG) |
| `field/` | 预测三联图、真实-预测 2×3 对比图、误差分布、散点图、预测场 CSV |
| `profiles/` | 固定距离的深度剖面对比 |
| `residual/` | PDE 残差散点图 + CSV |
| `metrics/` | `final_metrics.csv`、**`rmse_summary.csv` / `.txt`** (Feature 1) |
| `logs/` | 事件日志、`time_history.csv`、**`time_vs_step.png`**、**`step_time_per_iter.png`**、**`loss_vs_time.png`**、**`sampling_distribution.png`**、`sampling_info.json` (Feature 3) |
| `model/` | `best.pt`、各步 checkpoint |

#### 四大新增功能

**1. RMSE 单独汇总** (`metrics/rmse_summary.csv` + `.txt`)
   显式输出三类 RMSE: TL / 实部 / 虚部 (全场 + 近/中/远场分区), 便于直接放入论文表格.

**2. 跨频率迁移学习** (🔁 迁移学习 tab)
   - 加载已训练 `.pt`, 修改频率 / CSV 即可微调
   - 可选冻结前 N 层 Linear (保留几何先验)
   - lr 自动按 transfer_lr_scale 缩放
   - Fourier 网络可选按 f_new/f_old 缩放 B 矩阵
   - **典型加速 3-10×** vs 从零训练

**3. 多种样本划分方法** + 训练时间成本图 (📈 实验对比 tab)
   - uniform / stratified_block / lhs / grid_uniform / problem_region_aug
   - 自动保存采样分布图、训练耗时图、loss-vs-时间曲线
   - 实验对比页一键扫描所有 `rmse_summary.csv` 并展示
     时间-精度散点、各采样方法柱状对比、近/中/远场分区对比

**4. 长程训练稳定性** (5M step 不再崩溃)
   - NaN / Inf 自动检测 + 跳过, 累计超阈值自动停止
   - 梯度裁剪 `clip_grad_norm_`
   - 定期 `torch.cuda.empty_cache()` + `gc.collect()`
   - Loss 历史超过内存上限自动下采样
   - 重 IO (field PNG 等) 自适应节流: 5M step 时全程最多 200 次写图
"""
        )


if __name__ == "__main__":
    main()
