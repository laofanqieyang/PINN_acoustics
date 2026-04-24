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
        plot_pde_residual,
        plot_profile_comparison,
        save_all_loss_figures,
        setup_chinese_font,
    )
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
        plot_pde_residual,
        plot_profile_comparison,
        save_all_loss_figures,
        setup_chinese_font,
    )


# =========================================================================== #
# 全局设置
# =========================================================================== #
st.set_page_config(
    page_title="PINN 水下声场交互式训练平台",
    layout="wide",
    initial_sidebar_state="expanded",
)
setup_chinese_font()


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
                "SIREN w0", min_value=1.0, max_value=200.0, value=30.0, step=5.0,
                help="sin 的频率系数, 声场经验 15~30.",
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

    with st.sidebar.expander("🎯 物理 / 采样 (可选)", expanded=False):
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
def render_training_monitor(config: AppConfig) -> None:
    """启动训练并实时刷新界面."""
    st.subheader("🚀 训练进度与实时可视化")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 3])
    start = col_ctrl1.button("▶ 开始训练", type="primary", use_container_width=True)
    stop = col_ctrl2.button("⏹ 停止训练", use_container_width=True)

    if "stop_event" not in st.session_state:
        st.session_state.stop_event = threading.Event()

    if stop:
        st.session_state.stop_event.set()
        st.warning("已请求停止, 当前 step 结束后会退出.")
        return

    if not start:
        st.info("设置好侧边栏参数后, 点击 **开始训练**.")
        return

    # 重置状态
    st.session_state.stop_event = threading.Event()
    msg_queue: "queue.Queue" = queue.Queue(maxsize=32)
    thread = TrainingThread(config, msg_queue, st.session_state.stop_event)

    # UI 占位符
    progress_bar = st.progress(0.0, text="准备启动...")
    status_text = st.empty()
    metrics_area = st.empty()
    loss_area = st.empty()
    field_area = st.empty()
    error_area = st.empty()
    log_area = st.expander("📜 事件日志", expanded=False).empty()
    events_buffer: list[str] = []

    thread.start()
    latest_state: Optional[dict] = None

    while True:
        try:
            msg = msg_queue.get(timeout=0.5)
        except queue.Empty:
            if not thread.is_alive():
                break
            continue

        mtype = msg.get("type")
        if mtype == "state":
            latest_state = msg["state"]
            _render_latest_state(
                latest_state, config,
                progress_bar=progress_bar,
                status_text=status_text,
                metrics_area=metrics_area,
                loss_area=loss_area,
                field_area=field_area,
                error_area=error_area,
            )
        elif mtype == "event":
            events_buffer.append(msg["message"])
            log_area.write("\n".join(events_buffer[-200:]))
        elif mtype == "error":
            status_text.error(f"❌ 训练出错:\n{msg['message']}")
            break
        elif mtype == "done":
            break

    thread.join()

    if thread.error:
        st.error(thread.error)
        return

    # 最终输出
    if thread.result is not None:
        st.success("🎉 训练完成! 正在生成科研级输出...")
        _finalize_outputs(config, thread.result, latest_state)


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

    # Loss 曲线
    steps = state["loss_steps"]
    history = state["loss_history"]
    if steps:
        fig_loss = plot_loss_curve(steps, history, title=f"Loss 曲线 (step {step})")
        loss_area.pyplot(fig_loss, use_container_width=True)
        import matplotlib.pyplot as plt; plt.close(fig_loss)

    # 声场预测
    fig_field = plot_field_triptych(
        state["X"], state["Z"],
        state["pred_real"], state["pred_imag"], state["pred_tl"],
        suptitle=f"预测声场 (step {step})",
    )
    field_area.pyplot(fig_field, use_container_width=True)
    import matplotlib.pyplot as plt; plt.close(fig_field)

    # 误差分布 (实部误差示例)
    err_fig = plot_error_map(
        state["X"], state["Z"],
        state["pred_tl"], state["true_tl"],
        title=f"TL 绝对误差分布 (step {step})",
    )
    error_area.pyplot(err_fig, use_container_width=True)
    plt.close(err_fig)


# --------------------------------------------------------------------------- #
# 训练后的最终科研输出
# --------------------------------------------------------------------------- #
def _finalize_outputs(config: AppConfig, result: dict, last_state: Optional[dict]) -> None:
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

    tab_data, tab_train, tab_about = st.tabs(["📊 数据预览", "🏋️ 训练 & 监控", "ℹ️ 说明"])

    with tab_data:
        st.subheader("真实声压数据预览")
        preview_container = st.container()
        if st.button("🔍 加载 & 预览数据"):
            render_data_preview(config, preview_container)
        else:
            preview_container.info(
                "点击按钮加载侧边栏指定的声压实/虚部 CSV 文件并预览声场."
            )

    with tab_train:
        render_training_monitor(config)

    with tab_about:
        st.markdown(
            """
### 功能说明

本平台按照 `PINN UI.md` 架构实现:

1. **参数输入 (侧边栏)** - 覆盖你要求的 12 项输入参数.
2. **数据层** (`pinn_app/data/loader.py`) - 加载声压 CSV, 生成网格, 归一化.
3. **模型层** (`pinn_app/models/pinn.py`) - 可配置层数/神经元的全连接 PINN.
4. **训练层** (`pinn_app/training/trainer.py`) - Helmholtz PDE loss + 数据 loss + 边界 loss.
5. **可视化层** (`pinn_app/utils/visualization.py`) - 声场/损失/误差/剖面.
6. **评估层** (`pinn_app/utils/metrics.py`) - RMSE / MAE / Relative-L2 / Corr.

训练期间每 `visualize_interval` step 在界面上刷新:
- Loss 曲线 (total / data / pde / bc)
- 预测声场三联图 (实部 / 虚部 / 传输损失)
- TL 绝对误差分布
- 关键指标仪表盘

训练结束后会在 `outputs/<experiment_name>/` 自动生成:
- `loss/` 汇总 + 每个分量单独的损失曲线
- `field/` 预测声场 & 真实-预测对比 & 误差分布 (PNG + CSV)
- `profiles/` 固定距离的深度剖面对比
- `metrics/final_metrics.csv` 最终误差指标
- `logs/` 训练事件与 loss 历史 CSV
- `model/best.pt` 训练好的 PINN 权重
"""
        )


if __name__ == "__main__":
    main()
