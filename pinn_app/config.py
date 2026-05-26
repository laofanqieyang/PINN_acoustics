"""参数配置模块

定义训练配置的数据结构与默认值，所有来自 UI 的用户输入都会被封装进 :class:`AppConfig`。
"""
from __future__ import annotations

import math
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
    # SIREN 首层频率参数 (network_type="siren"); 后续层固定 w0=1.0
    siren_w0: float = 15.0
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

    # ------------------------------------------------------------------ #
    # 样本点划分方法 (Feature 3)
    # ------------------------------------------------------------------ #
    # "uniform"             : 在全网格上均匀随机抽 num_train_obs 个点 (默认)
    # "stratified_block"    : 把域分成 nbx × nbz 块, 按距源距离分层 (近场少 / 远场多)
    # "lhs"                 : Latin Hypercube Sampling, 在 (x,z) 域均匀分布
    # "grid_uniform"        : 等间距网格抽样
    # "problem_region_aug"  : stratified_block + 指定问题区域加密
    # "residual_adaptive"   : 初始均匀种子 + 训练中按 PDE 残差自适应加点 (RAS)
    sampling_method: str = "uniform"
    num_train_obs: int = 30000          # 训练观测点总数 (上限受真实网格 nx*nz 限制)
    # residual_adaptive (RAS) 参数 — 训练过程中在 trainer 内动态加点
    ras_initial_fraction: float = 0.25   # 初始观测点 = num_train_obs * 该比例
    ras_refine_every: int = 2000         # 每隔多少 step 评估残差并加点
    ras_points_per_refine: int = 500     # 每次新增点数
    ras_candidate_size: int = 8000       # 每次评估残差的候选池大小
    ras_residual_metric: str = "pde"     # "pde" | "data" — 按哪种残差排序
    # stratified / problem_region 参数
    num_blocks_x: int = 20
    num_blocks_z: int = 20
    near_dist_threshold: float = 150.0
    mid_dist_threshold: float = 300.0
    points_per_near_block: int = 30
    points_per_mid_block: int = 60
    points_per_far_block: int = 125
    # problem_region 加密 (按归一化 [0,1] 矩形指定)
    problem_region_x_min: float = 0.0
    problem_region_x_max: float = 0.2
    problem_region_z_min: float = 0.6
    problem_region_z_max: float = 1.0
    problem_region_extra_points: int = 3000

    # ------------------------------------------------------------------ #
    # 迁移学习 (Feature 2)
    # ------------------------------------------------------------------ #
    pretrained_ckpt: str = ""           # 预训练 .pt 文件路径; 空字符串 = 从零训练
    freeze_first_n_layers: int = 0      # 冻结前 N 层 Linear (0 = 不冻结, 全量微调)
    transfer_lr_scale: float = 0.1      # 加载预训练后 lr 缩放系数
    fourier_b_rescale: bool = False     # Fourier 网络: 把 B 矩阵按 f_new/f_old 缩放
    pretrained_frequency: float = 50.0  # 预训练时的频率 (用于 Fourier B 缩放参考)

    # ------------------------------------------------------------------ #
    # 训练稳定性 (Feature 4 - 长程训练崩溃修复)
    # ------------------------------------------------------------------ #
    gradient_clip: float = 1.0          # 梯度裁剪上限 (0 = 关闭)
    nan_skip_threshold: int = 50        # 累计 NaN/Inf 次数超过此值则停止训练
    cuda_empty_cache_every: int = 1000  # 每多少 step 清理一次 CUDA 缓存
    max_loss_points_in_memory: int = 20000  # loss 历史在内存中保留的最大点数 (超过下采样)

    # ------------------------------------------------------------------ #
    # GradNorm 自适应损失权重 (Chen et al. 2018)
    # ------------------------------------------------------------------ #
    use_gradnorm: bool = True           # 默认对所有训练启用
    gradnorm_alpha: float = 1.5         # α: 训练率指数, 越大对慢任务越倾斜
    gradnorm_lr: float = 1e-3           # GradNorm 权重的学习率 (与主优化器独立)
    gradnorm_update_every: int = 10     # 每多少 step 更新一次权重 (省成本)
    gradnorm_warmup_steps: int = 100    # 前 N 步用固定权重, 等损失稳定后再启用
    gradnorm_min_weight: float = 0.01   # 权重下限, 防塌缩到 0

    # ------------------------------------------------------------------ #
    # Domain Decomposition (域分解, XPINN 风格)
    # ------------------------------------------------------------------ #
    # 触发条件: length > domain_decomp_threshold 时自动启用 (用户要求 1000m)
    domain_decomp_threshold: float = 1000.0
    domain_decomp_force: str = "auto"   # "auto" | "on" | "off" - 强制开关
    # 子域数: 0 = 自动 (按 length / 500m 上取整, 至少 2)
    domain_decomp_num_subdomains: int = 0
    domain_decomp_overlap: float = 150.0         # 相邻子域重叠宽度 (m); 建议 ≥ 0.2*子域宽度
    domain_decomp_interface_points: int = 300    # 每个过渡带采样点数
    domain_decomp_interface_weight: float = 10.0  # 界面连续性损失权重
    domain_decomp_deriv_weight: float = 0.1      # 界面导数连续权重 (相对值损失)
    domain_decomp_overlap_fraction: float = 0.25 # overlap=0 时按子域宽度比例自动扩展
    domain_decomp_one_way_coupling: bool = True  # 远场子网在过渡带匹配近场子网声压 (单向)
    domain_decomp_coupling_weight: float = 5.0   # 单向耦合损失权重 (相对 interface)

    # ------------------------------------------------------------------ #
    # 包络分解 Envelope Decomposition (技术2 §1)
    # ------------------------------------------------------------------ #
    use_envelope_decomposition: bool = False   # u = v·exp(i k₀x), 网络学 v
    envelope_k0: float = 0.0                   # 0 = 按 wave_number_formula 自动
    supervise_envelope: bool = False           # True: 数据损失在包络 v 上; False: 在 u 上

    # ------------------------------------------------------------------ #
    # 抛物方程 PE-PINN (技术2 §2)
    # ------------------------------------------------------------------ #
    use_pe_pde: bool = False                   # PDE 残差用 PE 替代 Helmholtz

    # ------------------------------------------------------------------ #
    # Sequential Marching PINN (技术2 §3) — 与 DD 互斥, Marching 优先
    # ------------------------------------------------------------------ #
    marching_force: str = "auto"               # "auto" | "on" | "off"
    marching_threshold: float = 1000.0         # auto: length > 阈值启用
    marching_num_segments: int = 0             # 0 = 按 segment_length 推算
    marching_segment_length: float = 200.0     # 每段长度 (m)
    marching_overlap: float = 80.0
    marching_shared_network: bool = False      # 各段共享同一 MLP 权重
    marching_interface_points: int = 300
    marching_interface_weight: float = 10.0
    marching_deriv_weight: float = 0.1
    marching_one_way_coupling: bool = True
    marching_coupling_weight: float = 5.0
    marching_sequential_train: bool = False    # 按段顺序训练 (冻结已训段)
    marching_steps_per_segment: int = 0        # 0 = epochs / num_segments

    # ------------------------------------------------------------------ #
    # 波数公式 (Helmholtz / PE / 包络 k₀)
    # ------------------------------------------------------------------ #
    # "legacy_f_over_c" 保持旧实验 f/c; "2pi_f_over_c" 为标准 2πf/c
    wave_number_formula: str = "legacy_f_over_c"

    # ----------------------------- 派生属性 ---------------------------- #
    @property
    def wave_number(self) -> float:
        """Helmholtz/PE 波数 k."""
        f = (self.wave_number_formula or "legacy_f_over_c").lower()
        if f in ("2pi_f_over_c", "2pi", "standard"):
            return 2.0 * math.pi * float(self.frequency) / max(float(self.sound_speed), 1e-9)
        return float(self.frequency) / max(float(self.sound_speed), 1e-9)

    @property
    def envelope_k0_resolved(self) -> float:
        """包络参考波数 k₀."""
        if self.envelope_k0 > 0:
            return float(self.envelope_k0)
        return self.wave_number

    @property
    def total_points(self) -> int:
        return int(self.nx) * int(self.nz)

    # ----------------------------- Domain Decomposition ----------------------------- #
    @property
    def domain_decomp_enabled(self) -> bool:
        """按 force 设置和 length 阈值决定是否启用 DD."""
        f = (self.domain_decomp_force or "auto").lower()
        if f == "on":
            return True
        if f == "off":
            return False
        return float(self.length) > float(self.domain_decomp_threshold)

    @property
    def domain_decomp_resolved_num(self) -> int:
        """实际子域数: 0 时按 length / 500m 自动决定, 至少 2."""
        if self.domain_decomp_num_subdomains > 0:
            return int(self.domain_decomp_num_subdomains)
        return max(2, int(math.ceil(self.length / 500.0)))

    @property
    def marching_enabled(self) -> bool:
        """是否启用 Sequential Marching (与 DD 互斥)."""
        f = (self.marching_force or "auto").lower()
        if f == "on":
            return True
        if f == "off":
            return False
        return float(self.length) > float(self.marching_threshold)

    @property
    def marching_resolved_num_segments(self) -> int:
        if self.marching_num_segments > 0:
            return max(2, int(self.marching_num_segments))
        seg_len = max(float(self.marching_segment_length), 1.0)
        return max(2, int(math.ceil(float(self.length) / seg_len)))

    @property
    def physics_mode_label(self) -> str:
        """人类可读的物理模式描述."""
        parts = []
        if self.use_envelope_decomposition:
            parts.append("Envelope")
        if self.use_pe_pde:
            parts.append("PE")
        else:
            parts.append("Helmholtz")
        if self.marching_enabled:
            parts.append("Marching")
        elif self.domain_decomp_enabled:
            parts.append("DD")
        return "+".join(parts) if parts else "Standard"

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
