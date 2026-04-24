"""参数配置模块

定义训练配置的数据结构与默认值，所有来自 UI 的用户输入都会被封装进 :class:`AppConfig`。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# 物理参数（与参考的 train_acoustic_pinn.py 对齐，可按需修改）
# --------------------------------------------------------------------------- #
DEFAULT_FREQUENCY = 50.0       # 声源频率 (Hz)
DEFAULT_SOUND_SPEED = 1541.0   # 声速 (m/s)
DEFAULT_SOURCE_R = 1.0         # 源的横向位置 (m)
DEFAULT_SOURCE_Z = 20.0        # 源的深度位置 (m)
DEFAULT_SOURCE_SIGMA = 10.0    # 高斯源宽度参数
DEFAULT_SOURCE_AMP = 1.0       # 高斯源幅度


@dataclass
class AppConfig:
    """交互式训练的全部配置。

    覆盖用户要求的 12 项输入参数:
      1.  声压实部文件路径  -> pres_real_path
      2.  声压虚部文件路径  -> pres_imag_path
      3.  声场水平距离      -> length
      4.  声场深度          -> depth
      5.  水平距离方向数据点数 -> nx
      6.  深度方向数据点数  -> nz
      7.  学习率            -> learning_rate
      8.  batch_size        -> batch_size
      9.  PDE 权重          -> pde_weight
      10. 神经网络层数      -> num_layers
      11. 神经元个数        -> num_neurons
      12. 训练步数 step     -> epochs
    """

    # 1-2. 数据文件
    pres_real_path: str = "pres_real.csv"
    pres_imag_path: str = "pres_imag.csv"

    # 3-6. 空间域
    length: float = 500.0
    depth: float = 500.0
    nx: int = 500
    nz: int = 500

    # 7-9. 训练超参
    learning_rate: float = 1e-3
    batch_size: int = 128
    pde_weight: float = 2.0

    # 10-11. 网络结构
    num_layers: int = 7          # 隐藏层层数
    num_neurons: int = 50        # 每层神经元数
    # 网络架构 (可选: "dnn" | "fourier" | "siren" | "modified")
    network_type: str = "fourier"
    # Fourier Feature 参数 (network_type="fourier")
    fourier_mapping_size: int = 128
    fourier_sigma: float = 5.0
    # SIREN 频率参数 (network_type="siren")
    siren_w0: float = 30.0
    activation: str = "tanh"

    # 12. 训练轮数
    epochs: int = 20000

    # 额外的权重 & 物理参数（默认值与参考实现一致）
    data_weight: float = 100.0
    boundary_weight: float = 5.0
    frequency: float = DEFAULT_FREQUENCY
    sound_speed: float = DEFAULT_SOUND_SPEED
    source_r: float = DEFAULT_SOURCE_R
    source_z: float = DEFAULT_SOURCE_Z
    source_sigma: float = DEFAULT_SOURCE_SIGMA
    source_amplitude: float = DEFAULT_SOURCE_AMP

    # UI/训练监控相关
    visualize_interval: int = 500      # 每多少 step 在 UI 上刷新一次
    log_interval: int = 50             # loss 记录间隔
    num_collocation: int = 5000        # PDE 配点数
    num_boundary: int = 400            # 边界点数
    random_seed: int = 42

    # 输出目录
    output_dir: str = "outputs"
    experiment_name: str = "pinn_run"

    # 设备
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    # ----------------------------- 派生属性 ---------------------------- #
    @property
    def wave_number(self) -> float:
        """波数 k = 2*pi*f / c  (此处保持与参考实现一致: f / c)"""
        return self.frequency / self.sound_speed

    @property
    def total_points(self) -> int:
        return int(self.nx) * int(self.nz)

    @property
    def experiment_dir(self) -> Path:
        p = Path(self.output_dir) / self.experiment_name
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ----------------------------- 工具方法 ---------------------------- #
    def to_dict(self) -> dict:
        return asdict(self)

    def ensure_subdirs(self) -> dict:
        """创建所有标准输出子目录。"""
        base = self.experiment_dir
        sub = {
            "field": base / "field",
            "loss": base / "loss",
            "metrics": base / "metrics",
            "residual": base / "residual",
            "profiles": base / "profiles",
            "figs": base / "figs",
            "logs": base / "logs",
            "model": base / "model",
        }
        for p in sub.values():
            p.mkdir(parents=True, exist_ok=True)
        return {k: str(v) for k, v in sub.items()}


def default_config() -> AppConfig:
    return AppConfig()
