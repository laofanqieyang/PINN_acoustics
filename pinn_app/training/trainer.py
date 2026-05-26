"""PINN 训练器.

实现 Helmholtz 方程的 PINN 训练:

    ∇²p + k² p = S(x, z)

其中 p = p_real + i p_imag, 网络输出 (p_real, p_imag).
边界条件 (参考 train_acoustic_pinn.py):
    * 海面 (z=0) : p = 0 (Dirichlet)
    * 海底 (z=depth) : ∂p/∂n = 0 (Neumann)
    * 左/右边界 : 弱 Neumann
数据约束:
    * 使用整张真实声压场作为观测数据 (可 mini-batch)
"""
from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from ..config import AppConfig
from ..data.loader import (
    MinMaxNormalizer,
    build_training_tensors,
    load_pressure_data,
    sample_boundary_points,
    sample_collocation_points,
)
from ..data.sampling import build_observation_sampling, SamplingResult
from ..models.pinn import build_pinn
from ..models.domain_decomp import DomainDecomposedPINN, build_domain_decomp_pinn
from ..models.marching_pinn import SequentialMarchingPINN, build_marching_pinn
from ..physics.envelope import envelope_to_pressure, pressure_to_envelope
from ..physics.pde_residuals import compute_pde_residual
from .gradnorm import GradNormWeighter
from ..utils.logger import TrainingLogger
from ..utils.metrics import compute_all_metrics, compute_regional_rmse
from ..utils.excel_logger import save_parameters_xlsx
from ..utils.visualization import (
    compute_tl,
    plot_error_map,
    plot_error_map_multi_scale,
    plot_field_comparison,
    plot_field_triptych,
    plot_loss_vs_time,
    plot_pred_vs_true_scatter,
    plot_profile_comparison,
    plot_sampling_distribution,
    sampling_viz_style,
    plot_step_time_per_iter,
    plot_time_vs_step,
    save_all_loss_figures,
    save_figure_publication,
    setup_publication_style,
)


# ============================================================================ #
# 训练配置 (只包含训练相关字段; 来自 AppConfig)
# ============================================================================ #
@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    pde_weight: float
    data_weight: float
    boundary_weight: float
    num_layers: int
    num_neurons: int
    visualize_interval: int
    log_interval: int
    num_collocation: int
    num_boundary: int
    frequency: float
    sound_speed: float
    source_r: float
    source_z: float
    source_sigma: float
    source_amplitude: float
    random_seed: int = 42

    @property
    def wave_number(self) -> float:
        return self.frequency / self.sound_speed

    @classmethod
    def from_app_config(cls, cfg: AppConfig) -> "TrainConfig":
        return cls(
            epochs=int(cfg.epochs),
            batch_size=int(cfg.batch_size),
            learning_rate=float(cfg.learning_rate),
            pde_weight=float(cfg.pde_weight),
            data_weight=float(cfg.data_weight),
            boundary_weight=float(cfg.boundary_weight),
            num_layers=int(cfg.num_layers),
            num_neurons=int(cfg.num_neurons),
            visualize_interval=int(cfg.visualize_interval),
            log_interval=int(cfg.log_interval),
            num_collocation=int(cfg.num_collocation),
            num_boundary=int(cfg.num_boundary),
            frequency=float(cfg.frequency),
            sound_speed=float(cfg.sound_speed),
            source_r=float(cfg.source_r),
            source_z=float(cfg.source_z),
            source_sigma=float(cfg.source_sigma),
            source_amplitude=float(cfg.source_amplitude),
            random_seed=int(cfg.random_seed),
        )


# ============================================================================ #
# 训练器
# ============================================================================ #
class Trainer:
    """PINN 训练器.

    通过 ``callback(state)`` 把每隔 ``visualize_interval`` 步的信息回传给 UI 层:

        state = {
            "step": int,
            "total_steps": int,
            "losses": {"total", "data", "pde", "bc"},
            "metrics": {"rmse_real", ...},
            "pred_real": np.ndarray (nz, nx),
            "pred_imag": np.ndarray (nz, nx),
            "pred_tl":   np.ndarray (nz, nx),
            "X", "Z":    np.ndarray (nz, nx),
            "elapsed":   float (s),
            "lr":        float,
        }
    """

    def __init__(
        self,
        config: AppConfig,
        callback: Optional[Callable[[dict], None]] = None,
        stop_flag: Optional[Callable[[], bool]] = None,
        logger: Optional[TrainingLogger] = None,
    ):
        self.app_config = config
        self.tc = TrainConfig.from_app_config(config)
        self.callback = callback
        self.stop_flag = stop_flag or (lambda: False)

        # 设备
        self.device = self._pick_device(config.device)

        # 随机种子
        np.random.seed(self.tc.random_seed)
        torch.manual_seed(self.tc.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.tc.random_seed)

        # 子目录
        self.subdirs = config.ensure_subdirs()

        # 日志
        self.logger = logger or TrainingLogger(self.subdirs["logs"])
        self.logger.save_config(config.to_dict())
        self.logger.log_event(f"训练设备: {self.device}")

        # 数据
        self._prepare_data()

        # === 物理模式 (技术2: 包络 / PE / Marching) ===
        self.use_envelope = bool(config.use_envelope_decomposition)
        self.use_pe = bool(config.use_pe_pde)
        self.supervise_envelope = bool(config.supervise_envelope)
        self._k0 = float(config.envelope_k0_resolved)
        self._k = float(config.wave_number)
        wnf = (config.wave_number_formula or "legacy_f_over_c").lower()

        # Marching 与 DD 互斥, Marching 优先
        self.use_marching = bool(config.marching_enabled)
        self.use_domain_decomp = bool(config.domain_decomp_enabled) and not self.use_marching
        self.use_interface_loss = self.use_marching or self.use_domain_decomp

        self.logger.log_event(
            f"物理模式: {config.physics_mode_label} | k={self._k:.6g} "
            f"(formula={wnf})"
            + (f", k0={self._k0:.6g} (envelope)" if self.use_envelope else "")
            + (", PE-PDE" if self.use_pe else ", Helmholtz-PDE")
        )

        # === 网络 ===
        if self.use_marching:
            n_seg = config.marching_resolved_num_segments
            seg_len = float(config.length) / n_seg
            self.model = build_marching_pinn(
                length=config.length,
                depth=config.depth,
                num_segments=n_seg,
                segment_length=seg_len,
                overlap=config.marching_overlap,
                shared_network=config.marching_shared_network,
                num_layers=self.tc.num_layers,
                num_neurons=self.tc.num_neurons,
                activation=config.activation,
                network_type=config.network_type,
                mapping_size=config.fourier_mapping_size,
                fourier_sigma=config.fourier_sigma,
                siren_w0=config.siren_w0,
                device=self.device,
            )
            self.logger.log_event(
                f"➡️ Sequential Marching 启用: {n_seg} 段, "
                f"段长≈{seg_len:.0f}m, overlap={config.marching_overlap}m, "
                f"sequential_train={config.marching_sequential_train}"
            )
        elif self.use_domain_decomp:
            n_sub = config.domain_decomp_resolved_num
            self.model = build_domain_decomp_pinn(
                num_subdomains=n_sub,
                length=config.length, depth=config.depth,
                overlap=config.domain_decomp_overlap,
                overlap_fraction=float(config.domain_decomp_overlap_fraction),
                source_r=float(config.source_r),
                source_z=float(config.source_z),
                num_layers=self.tc.num_layers,
                num_neurons=self.tc.num_neurons,
                activation=config.activation,
                network_type=config.network_type,
                mapping_size=config.fourier_mapping_size,
                fourier_sigma=config.fourier_sigma,
                siren_w0=config.siren_w0,
                device=self.device,
            )
            self.logger.log_event(
                f"🌐 Domain Decomposition 启用: length={config.length}m > "
                f"{config.domain_decomp_threshold}m -> {n_sub} 个子域, "
                f"overlap={config.domain_decomp_overlap}m"
            )
        else:
            self.model = build_pinn(
                num_layers=self.tc.num_layers,
                num_neurons=self.tc.num_neurons,
                activation=config.activation,
                device=self.device,
                network_type=config.network_type,
                mapping_size=config.fourier_mapping_size,
                fourier_sigma=config.fourier_sigma,
                siren_w0=config.siren_w0,
            )
        self.logger.log_event(self.model.summary())

        # === 迁移学习: 加载预训练权重 === (Feature 2)
        effective_lr = self.tc.learning_rate
        self.transfer_info: Dict[str, object] = {"enabled": False}
        if getattr(config, "pretrained_ckpt", "") and Path(config.pretrained_ckpt).exists():
            try:
                self._load_pretrained(config)
                effective_lr = self.tc.learning_rate * float(config.transfer_lr_scale)
                self.transfer_info = {
                    "enabled": True,
                    "ckpt_path": str(config.pretrained_ckpt),
                    "freeze_first_n_layers": int(config.freeze_first_n_layers),
                    "transfer_lr_scale": float(config.transfer_lr_scale),
                    "effective_lr": effective_lr,
                    "pretrained_frequency": float(config.pretrained_frequency),
                    "new_frequency": float(config.frequency),
                }
                self.logger.log_event(
                    f"✓ 迁移学习启用: 加载 {config.pretrained_ckpt} | "
                    f"冻结前 {config.freeze_first_n_layers} 层 | "
                    f"lr {self.tc.learning_rate:.2e} → {effective_lr:.2e}"
                )
            except Exception as exc:
                self.logger.log_event(
                    f"⚠ 迁移学习加载失败 (将从零训练): {exc}", level="WARN"
                )

        # 只把可训练参数交给优化器 (冻结层 requires_grad=False)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("所有参数都被冻结, 无法训练. 请减小 freeze_first_n_layers")
        self.optimizer = torch.optim.Adam(trainable, lr=effective_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(self.tc.epochs, 1), eta_min=1e-6
        )

        # === GradNorm (自适应损失权重, Chen et al. 2018) ===
        self.use_gradnorm = bool(getattr(config, "use_gradnorm", True))
        if self.use_gradnorm:
            task_names = ["data", "pde", "bc"]
            init_w = [
                float(self.tc.data_weight),
                float(self.tc.pde_weight),
                float(self.tc.boundary_weight),
            ]
            if self.use_interface_loss:
                task_names.append("interface")
                iw = (
                    float(config.marching_interface_weight)
                    if self.use_marching
                    else float(config.domain_decomp_interface_weight)
                )
                init_w.append(iw)
            self.gradnorm = GradNormWeighter(
                task_names=task_names,
                alpha=float(config.gradnorm_alpha),
                lr=float(config.gradnorm_lr),
                init_weights=init_w,
                update_every=int(config.gradnorm_update_every),
                warmup_steps=int(config.gradnorm_warmup_steps),
                min_weight=float(config.gradnorm_min_weight),
                device=self.device,
            )
            self.logger.log_event(
                f"⚖️ GradNorm 启用: tasks={task_names}, α={config.gradnorm_alpha}, "
                f"warmup={config.gradnorm_warmup_steps}, update_every={config.gradnorm_update_every}"
            )
        else:
            self.gradnorm = None

        # loss 缓存 (供 UI 实时画曲线)
        self.loss_steps: List[int] = []
        self.loss_history: Dict[str, List[float]] = {
            "total": [], "data": [], "pde": [], "bc": []
        }
        if self.use_interface_loss:
            self.loss_history["interface"] = []
        # 权重历史 (GradNorm)
        self.weight_history: Dict[str, List[float]] = {}
        if self.use_gradnorm:
            for n in self.gradnorm.task_names:
                self.weight_history[f"w_{n}"] = []
        # 时间历史 (Feature 3): 每个 log 点记录累计耗时
        self.time_history: List[float] = []
        # 长程训练稳定性 (Feature 4)
        self._nan_count = 0
        self._gradient_clip = float(getattr(config, "gradient_clip", 1.0))
        self._nan_skip_threshold = int(getattr(config, "nan_skip_threshold", 50))
        self._cuda_clean_every = int(getattr(config, "cuda_empty_cache_every", 1000))
        self._max_loss_points = int(getattr(config, "max_loss_points_in_memory", 20000))

        # k^2 常量 (Helmholtz; PE 残差在 pde_residuals 内用 k)
        self._k2 = torch.tensor(self._k ** 2, dtype=torch.float32, device=self.device)

        # 雅可比 (归一化 -> 物理), d(norm)/d(phys)
        jx, jz = self.normalizer.jacobian
        self._jx2 = float(jx) ** 2
        self._jz2 = float(jz) ** 2

    # ------------------------------------------------------------------ #
    # 迁移学习辅助 (Feature 2)
    # ------------------------------------------------------------------ #
    def _load_pretrained(self, config: AppConfig) -> None:
        """加载预训练 .pt 文件并按需冻结前 N 层 Linear."""
        ckpt = torch.load(config.pretrained_ckpt, map_location=self.device,
                          weights_only=False)
        state = ckpt.get("model_state", ckpt)
        # 容错: 若 Fourier B 矩阵尺寸或网络宽度不一致, 仅加载能匹配的键
        own = self.model.state_dict()
        loaded, skipped = 0, []
        for k, v in state.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded += 1
            else:
                skipped.append(k)
        self.model.load_state_dict(own)
        self.logger.log_event(
            f"预训练权重加载: 匹配 {loaded} / 跳过 {len(skipped)} 个键"
        )

        # 可选: Fourier B 缩放
        if (config.network_type or "").lower() == "fourier" and config.fourier_b_rescale:
            try:
                f_old = max(float(config.pretrained_frequency), 1e-9)
                f_new = float(config.frequency)
                scale = f_new / f_old
                if hasattr(self.model, "ff") and hasattr(self.model.ff, "B"):
                    self.model.ff.B.mul_(scale)
                    self.logger.log_event(
                        f"Fourier B 矩阵已按 f_new/f_old={scale:.3f} 缩放"
                    )
            except Exception as exc:
                self.logger.log_event(
                    f"Fourier B 缩放失败: {exc}", level="WARN"
                )

        # 冻结前 N 层 Linear
        n = int(config.freeze_first_n_layers)
        if n > 0:
            frozen = 0
            seen_linear = 0
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    if seen_linear < n:
                        for p in m.parameters():
                            p.requires_grad_(False)
                            frozen += 1
                        seen_linear += 1
            self.logger.log_event(
                f"已冻结前 {seen_linear} 层 Linear ({frozen} 个张量)"
            )

    # ------------------------------------------------------------------ #
    # 初始化辅助
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pick_device(preference: str) -> torch.device:
        if preference == "cpu":
            return torch.device("cpu")
        if preference == "cuda":
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_data(self) -> None:
        """读取真实数据 → 应用采样策略 → 构造训练张量和边界点."""
        p_real, p_imag = load_pressure_data(
            self.app_config.pres_real_path,
            self.app_config.pres_imag_path,
            expected_shape=(self.app_config.nz, self.app_config.nx),
        )
        self.real_shape = p_real.shape  # (nz, nx) 来自文件
        nz, nx = self.real_shape

        # 1) 全网格用于全场预测 / 评估指标 (始终保留)
        pack = build_training_tensors(
            p_real, p_imag,
            length=self.app_config.length,
            depth=self.app_config.depth,
            device=self.device,
        )
        self.X = pack["X"]
        self.Z = pack["Z"]
        self.coords_phys_full = pack["coords_phys"]
        self.coords_tensor_full = pack["coords_tensor"]
        self.p_real_target_full = pack["p_real_tensor"]
        self.p_imag_target_full = pack["p_imag_tensor"]
        self.p_real_np = p_real
        self.p_imag_np = p_imag
        self.tl_true_np = compute_tl(p_real, p_imag)
        self.normalizer: MinMaxNormalizer = pack["normalizer"]

        # 2) 应用采样策略, 得到训练观测子集 (Feature 3)
        sr: SamplingResult = build_observation_sampling(self.app_config, nx=nx, nz=nz)
        self.sampling_result = sr
        self._obs_indices = sr.indices.copy()
        self._ras_enabled = (sr.info.get("method") or "").lower() in (
            "residual_adaptive", "ras", "residual",
        )
        self._ras_target_n = int(
            getattr(self.app_config, "num_train_obs", self._obs_indices.size)
        )
        self._update_observation_tensors()

        # 边界点 (一次性采样)
        top, bottom, left, right = sample_boundary_points(
            self.app_config.length, self.app_config.depth,
            n_per_edge=max(50, self.tc.num_boundary // 4),
        )
        self.top_bc = torch.from_numpy(top).to(self.device)
        self.bottom_bc = torch.from_numpy(bottom).to(self.device)
        self.left_bc = torch.from_numpy(left).to(self.device)
        self.right_bc = torch.from_numpy(right).to(self.device)

        self.logger.log_event(
            f"真实数据形状={self.real_shape}, 全网格点数={nz*nx}, "
            f"采样方法={sr.info.get('method')}, 训练观测点数={self.num_obs}"
            + (f" (RAS 目标 {self._ras_target_n})" if self._ras_enabled else "")
        )

    def _update_observation_tensors(self) -> None:
        """根据当前观测索引刷新 data loss 用的张量."""
        idx_t = torch.from_numpy(self._obs_indices).long().to(self.device)
        self.coords_tensor = self.coords_tensor_full[idx_t]
        self.p_real_target = self.p_real_target_full[idx_t]
        self.p_imag_target = self.p_imag_target_full[idx_t]
        self.num_obs = int(self._obs_indices.size)
        self.sampling_result.indices = self._obs_indices.copy()
        nx, nz = self.real_shape[1], self.real_shape[0]
        self.sampling_result.coords = np.stack([
            (self._obs_indices % nx).astype(np.float32) / max(nx - 1, 1) * self.app_config.length,
            (self._obs_indices // nx).astype(np.float32) / max(nz - 1, 1) * self.app_config.depth,
        ], axis=-1).astype(np.float32)
        self.sampling_result.info["n"] = self.num_obs

    # ------------------------------------------------------------------ #
    # 前向 (归一化输入)
    # ------------------------------------------------------------------ #
    def _raw_forward(self, coords_phys: torch.Tensor) -> torch.Tensor:
        xz_norm = self.normalizer.encode_torch(coords_phys)
        return self.model(xz_norm)

    def _forward(self, coords_phys: torch.Tensor) -> torch.Tensor:
        """物理声压 (p_real, p_imag). 包络模式下由 v·exp(ik₀x) 还原."""
        raw = self._raw_forward(coords_phys)
        if not self.use_envelope:
            return raw
        x = coords_phys[:, 0:1]
        pr, pi = envelope_to_pressure(raw[:, 0:1], raw[:, 1:2], x, self._k0)
        return torch.cat([pr, pi], dim=1)

    def _pde_source_terms(
        self, x: torch.Tensor, z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self._gaussian_source(x, z)
        return g * 10.0, 0.5 * g * 5.0

    # ------------------------------------------------------------------ #
    # 高斯源
    # ------------------------------------------------------------------ #
    def _gaussian_source(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        r0 = self.tc.source_r
        z0 = self.tc.source_z
        sigma = self.tc.source_sigma
        amp = self.tc.source_amplitude
        dist2 = (x - r0) ** 2 + (z - z0) ** 2
        return amp * torch.exp(-dist2 / (2.0 * sigma ** 2))

    # ------------------------------------------------------------------ #
    # PDE 残差 (Helmholtz)
    # ------------------------------------------------------------------ #
    def _pde_residual(self, coords_phys: torch.Tensor) -> torch.Tensor:
        """Helmholtz 或 PE 残差; 支持包络参数化."""
        coords_phys = coords_phys.detach().clone().requires_grad_(True)
        raw = self._raw_forward(coords_phys)
        return compute_pde_residual(
            raw,
            coords_phys,
            use_envelope=self.use_envelope,
            k0=self._k0,
            k=self._k,
            use_pe=self.use_pe,
            source_fn=self._pde_source_terms,
        )

    # ------------------------------------------------------------------ #
    # 界面连续性损失 (Domain Decomposition)
    # ------------------------------------------------------------------ #
    def _interface_loss(self) -> torch.Tensor:
        """DD / Marching 过渡带连续性 + 可选单向近→远耦合 (在物理声压 u 上)."""
        if not self.use_interface_loss:
            return torch.tensor(0.0, device=self.device)

        cfg = self.app_config
        if self.use_marching:
            model = self.model  # SequentialMarchingPINN
            n_sub = model.num_segments
            n_pts = max(int(cfg.marching_interface_points), 16)
            deriv_w = float(cfg.marching_deriv_weight)
            one_way = bool(cfg.marching_one_way_coupling)
            couple_w = float(cfg.marching_coupling_weight)
        else:
            model = self.model  # DomainDecomposedPINN
            n_sub = model.num_subdomains
            n_pts = max(int(cfg.domain_decomp_interface_points), 16)
            deriv_w = float(cfg.domain_decomp_deriv_weight)
            one_way = bool(cfg.domain_decomp_one_way_coupling)
            couple_w = float(cfg.domain_decomp_coupling_weight)

        depth = cfg.depth
        device = self.device
        loss_val = torch.tensor(0.0, device=device)
        loss_dx = torch.tensor(0.0, device=device)
        loss_couple = torch.tensor(0.0, device=device)

        for k in range(n_sub - 1):
            interval = model.overlap_interval(k)
            if interval is None:
                continue
            x_lo, x_hi = interval
            # 过渡带内 (x, z) 均匀随机
            n_here = n_pts
            x_s = torch.rand(n_here, device=device) * (x_hi - x_lo) + x_lo
            z_s = torch.rand(n_here, device=device) * depth
            pts_phys = torch.stack([x_s, z_s], dim=-1).requires_grad_(True)
            u_near = self._forward(pts_phys)
            # 两段各自前向 (同一点、不同子网参数)
            pts_norm = self.normalizer.encode_torch(pts_phys)
            raw_near = model.nets[k](pts_norm)
            raw_far = model.nets[k + 1](pts_norm)
            if self.use_envelope:
                x_p = pts_phys[:, 0:1]
                pr_n, pi_n = envelope_to_pressure(
                    raw_near[:, 0:1], raw_near[:, 1:2], x_p, self._k0,
                )
                pr_f, pi_f = envelope_to_pressure(
                    raw_far[:, 0:1], raw_far[:, 1:2], x_p, self._k0,
                )
                y_near = torch.cat([pr_n, pi_n], dim=1)
                y_far = torch.cat([pr_f, pi_f], dim=1)
            else:
                y_near, y_far = raw_near, raw_far

            loss_val = loss_val + torch.mean((y_near - y_far) ** 2)
            if one_way:
                loss_couple = loss_couple + torch.mean((y_far - y_near.detach()) ** 2)

            try:
                gL = torch.autograd.grad(
                    y_near.sum(), pts_phys, create_graph=True, retain_graph=True,
                )[0][:, 0]
                gR = torch.autograd.grad(
                    y_far.sum(), pts_phys, create_graph=True, retain_graph=True,
                )[0][:, 0]
                loss_dx = loss_dx + torch.mean((gL - gR) ** 2)
            except RuntimeError:
                pass

        total = loss_val + deriv_w * loss_dx
        if one_way and couple_w > 0:
            total = total + couple_w * loss_couple
        return total

    def _ras_refine_observations(self, step: int, rng: np.random.Generator) -> None:
        """Residual-based Adaptive Sampling: 按残差从候选池追加观测点."""
        if not self._ras_enabled:
            return
        if self.num_obs >= self._ras_target_n:
            return
        cfg = self.app_config
        every = max(int(getattr(cfg, "ras_refine_every", 2000)), 1)
        if step % every != 0:
            return

        nx, nz = self.real_shape[1], self.real_shape[0]
        total = nx * nz
        pool_n = min(int(getattr(cfg, "ras_candidate_size", 8000)), total)
        add_n = min(
            int(getattr(cfg, "ras_points_per_refine", 500)),
            self._ras_target_n - self.num_obs,
        )
        if add_n <= 0:
            return

        # 候选: 全网格随机子集, 排除已选
        cand = rng.choice(total, size=pool_n, replace=False).astype(np.int64)
        existing = set(self._obs_indices.tolist())
        cand = cand[np.array([c not in existing for c in cand], dtype=bool)]
        if cand.size == 0:
            return

        ix = (cand % nx).astype(np.float32)
        iz = (cand // nx).astype(np.float32)
        x = ix / max(nx - 1, 1) * cfg.length
        z = iz / max(nz - 1, 1) * cfg.depth
        pts_np = np.stack([x, z], axis=-1).astype(np.float32)
        pts = torch.from_numpy(pts_np).to(self.device)

        metric = (getattr(cfg, "ras_residual_metric", "pde") or "pde").lower()
        self.model.eval()
        with torch.enable_grad():
            if metric == "data":
                pred = self._forward(pts)
                idx_t = torch.from_numpy(cand).long().to(self.device)
                tgt_r = self.p_real_target_full[idx_t]
                tgt_i = self.p_imag_target_full[idx_t]
                res = torch.sqrt(
                    (pred[:, 0:1] - tgt_r) ** 2 + (pred[:, 1:2] - tgt_i) ** 2
                )
                scores = res.detach().squeeze(1).cpu().numpy()
            else:
                residual = self._pde_residual(pts)
                scores = torch.linalg.norm(residual, dim=1).detach().cpu().numpy()
        self.model.train()

        top_k = np.argsort(scores)[-add_n:]
        new_idx = cand[top_k]
        self._obs_indices = np.unique(np.concatenate([self._obs_indices, new_idx]))
        self._update_observation_tensors()
        self.logger.log_event(
            f"RAS step={step}: +{new_idx.size} 观测点 → 总计 {self.num_obs} "
            f"(目标 {self._ras_target_n}, metric={metric})"
        )

    # ------------------------------------------------------------------ #
    # 获取 GradNorm 用的共享参数 (取最后一层 Linear 的 weight)
    # ------------------------------------------------------------------ #
    def _gradnorm_shared_params(self) -> List[torch.nn.Parameter]:
        params: List[torch.nn.Parameter] = []
        if self.use_marching or self.use_domain_decomp:
            model = self.model
            for sub in model.nets:
                last = None
                for m in sub.modules():
                    if isinstance(m, torch.nn.Linear):
                        last = m
                if last is not None and last.weight.requires_grad:
                    params.append(last.weight)
        else:
            last = None
            for m in self.model.modules():
                if isinstance(m, torch.nn.Linear):
                    last = m
            if last is not None and last.weight.requires_grad:
                params.append(last.weight)
        return params

    # ------------------------------------------------------------------ #
    # 边界条件损失
    # ------------------------------------------------------------------ #
    def _boundary_loss(self) -> torch.Tensor:
        # 海面: p = 0
        top_pred = self._forward(self.top_bc)
        loss_top = torch.mean(top_pred ** 2)

        # 海底 & 左右: ∂p/∂n = 0 (弱 Neumann)
        def _neumann(points: torch.Tensor, axis: int) -> torch.Tensor:
            points = points.detach().clone().requires_grad_(True)
            y = self._forward(points)
            grads = []
            for c in range(y.shape[1]):
                yc = y[:, c:c + 1]
                g = torch.autograd.grad(
                    yc, points, grad_outputs=torch.ones_like(yc),
                    create_graph=True, retain_graph=True,
                )[0]
                grads.append(g[:, axis:axis + 1])
            grad_cat = torch.cat(grads, dim=1)
            return torch.mean(grad_cat ** 2)

        loss_bottom = _neumann(self.bottom_bc, axis=1)
        loss_left = _neumann(self.left_bc, axis=0)
        loss_right = _neumann(self.right_bc, axis=0)

        return loss_top + loss_bottom + 0.2 * (loss_left + loss_right)

    # ------------------------------------------------------------------ #
    # 训练主循环
    # ------------------------------------------------------------------ #
    def train(self) -> Dict[str, object]:
        if self.use_marching and bool(self.app_config.marching_sequential_train):
            return self._train_marching_sequential()
        return self._train_standard()

    def _train_standard(self) -> Dict[str, object]:
        cfg = self.tc
        total_steps = cfg.epochs
        bsz = max(cfg.batch_size, 8)
        rng = np.random.default_rng(cfg.random_seed)
        t_start = time.time()
        best_loss = math.inf

        # 长程训练自适应节流: 避免 5M step 时 IO/UI 爆炸
        eff_log_interval = max(cfg.log_interval, total_steps // 20000 if total_steps > 20000 else 1)
        # 重 IO (field PNG) 的实际间隔: 至少 visualize_interval, 但全程最多 200 次
        heavy_io_interval = max(cfg.visualize_interval, total_steps // 200 if total_steps > 200 else 1)

        self.logger.log_event(
            f"开始训练: steps={total_steps}, batch_size={bsz}, "
            f"effective_lr={self.optimizer.param_groups[0]['lr']:.2e}, "
            f"pde_weight={cfg.pde_weight}"
        )
        self.logger.log_event(
            f"长程训练节流: log_interval={eff_log_interval}, "
            f"heavy_io_interval={heavy_io_interval}, "
            f"grad_clip={self._gradient_clip}"
        )

        for step in range(1, total_steps + 1):
            if self.stop_flag():
                self.logger.log_event("收到停止信号, 提前结束训练.", level="WARN")
                break

            self.model.train()
            self.optimizer.zero_grad()

            # --- Data loss ---
            idx = torch.from_numpy(
                rng.integers(0, self.num_obs, size=bsz)
            ).long().to(self.device)
            coords_batch = self.coords_tensor[idx]
            tgt_r = self.p_real_target[idx]
            tgt_i = self.p_imag_target[idx]
            if self.use_envelope and self.supervise_envelope:
                raw = self._raw_forward(coords_batch)
                x_b = coords_batch[:, 0:1]
                tgt_vr, tgt_vi = pressure_to_envelope(tgt_r, tgt_i, x_b, self._k0)
                loss_data = (
                    torch.mean((raw[:, 0:1] - tgt_vr) ** 2)
                    + torch.mean((raw[:, 1:2] - tgt_vi) ** 2)
                )
            else:
                pred = self._forward(coords_batch)
                pred_r, pred_i = pred[:, 0:1], pred[:, 1:2]
                loss_data = (
                    torch.mean((pred_r - tgt_r) ** 2)
                    + torch.mean((pred_i - tgt_i) ** 2)
                )

            # --- PDE loss ---
            colloc_np = sample_collocation_points(
                self.app_config.length, self.app_config.depth,
                cfg.num_collocation, rng=rng,
            )
            colloc = torch.from_numpy(colloc_np).to(self.device)
            residual = self._pde_residual(colloc)
            loss_pde = torch.mean(residual ** 2)

            # --- Boundary loss ---
            loss_bc = self._boundary_loss()

            # --- Interface loss (DD / Marching) ---
            loss_interface = self._interface_loss() if self.use_interface_loss \
                else torch.tensor(0.0, device=self.device)

            # === GradNorm 动态权重 (在主反向之前更新) ===
            if self.use_gradnorm and self.gradnorm is not None:
                tasks = [loss_data, loss_pde, loss_bc]
                if self.use_interface_loss:
                    tasks.append(loss_interface)
                try:
                    gn_info = self.gradnorm.step(
                        tasks, self._gradnorm_shared_params(),
                    )
                except Exception as exc:
                    if step <= 5:
                        self.logger.log_event(f"GradNorm step error: {exc}", level="WARN")
                    gn_info = {}
                w_data = self.gradnorm.weights[0]
                w_pde = self.gradnorm.weights[1]
                w_bc = self.gradnorm.weights[2]
                w_inf = self.gradnorm.weights[3] if self.use_interface_loss \
                    else torch.tensor(0.0, device=self.device)
            else:
                w_data = torch.tensor(cfg.data_weight, device=self.device)
                w_pde = torch.tensor(cfg.pde_weight, device=self.device)
                w_bc = torch.tensor(cfg.boundary_weight, device=self.device)
                w_inf = torch.tensor(
                    float(
                        self.app_config.marching_interface_weight
                        if self.use_marching
                        else self.app_config.domain_decomp_interface_weight
                    ),
                    device=self.device,
                ) if self.use_interface_loss else torch.tensor(0.0, device=self.device)
                gn_info = {}

            # 主 total loss (权重用 .detach() 防止反向通过 weight 影响主模型)
            loss_total = (
                w_data.detach() * loss_data
                + w_pde.detach() * loss_pde
                + w_bc.detach() * loss_bc
            )
            if self.use_interface_loss:
                loss_total = loss_total + w_inf.detach() * loss_interface

            # === NaN/Inf 防护 (Feature 4) ===
            if not torch.isfinite(loss_total):
                self._nan_count += 1
                if self._nan_count <= 5 or self._nan_count % 50 == 0:
                    self.logger.log_event(
                        f"step={step}: 损失为 NaN/Inf, 跳过更新 "
                        f"(累计 {self._nan_count} 次)",
                        level="WARN",
                    )
                self.optimizer.zero_grad()
                if self._nan_count >= self._nan_skip_threshold:
                    self.logger.log_event(
                        f"NaN/Inf 累计 {self._nan_count} 次超过阈值, 提前终止训练. "
                        f"建议: 降低学习率、增大 batch_size 或 gradient_clip",
                        level="ERROR",
                    )
                    break
                continue

            loss_total.backward()
            # 梯度裁剪 (Feature 4): 防止长程训练梯度爆炸
            if self._gradient_clip > 0:
                trainable = [p for p in self.model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, self._gradient_clip)
            self.optimizer.step()
            self.scheduler.step()

            # RAS: 按残差自适应追加观测点
            if self._ras_enabled:
                try:
                    self._ras_refine_observations(step, rng)
                except Exception as exc:
                    if step <= 5:
                        self.logger.log_event(f"RAS refine error: {exc}", level="WARN")

            # 定期 CUDA 缓存清理 (Feature 4): 防止长程内存碎片化
            if self._cuda_clean_every > 0 and step % self._cuda_clean_every == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # 记录 loss
            if step % max(eff_log_interval, 1) == 0 or step == 1:
                entry = {
                    "total": float(loss_total.item()),
                    "data": float(loss_data.item()),
                    "pde": float(loss_pde.item()),
                    "bc": float(loss_bc.item()),
                }
                if self.use_interface_loss:
                    entry["interface"] = float(loss_interface.item())
                self.loss_steps.append(step)
                for k, v in entry.items():
                    if k in self.loss_history:
                        self.loss_history[k].append(v)
                self.time_history.append(time.time() - t_start)
                self.logger.log_loss(step, entry)

                # 记录 GradNorm 动态权重历史
                if self.use_gradnorm and self.gradnorm is not None:
                    cur_w = self.gradnorm.current_weights()
                    for k, v in cur_w.items():
                        key = f"w_{k}"
                        if key in self.weight_history:
                            self.weight_history[key].append(float(v))
                        else:
                            self.weight_history[key] = [float(v)]

                # 历史下采样 (Feature 4): 内存中超出上限时, 每两个保留一个
                if len(self.loss_steps) > self._max_loss_points:
                    self.loss_steps = self.loss_steps[::2]
                    self.time_history = self.time_history[::2]
                    for k in self.loss_history:
                        self.loss_history[k] = self.loss_history[k][::2]
                    self.logger.log_event(
                        f"loss 历史已下采样至 {len(self.loss_steps)} 点 "
                        f"(超过 {self._max_loss_points})"
                    )

            # 周期性: 构造 state -> 磁盘中间产物 -> UI 回调 -> 记录 metric -> checkpoint
            do_visualize = (
                step % max(cfg.visualize_interval, 1) == 0 or step == total_steps
            )
            if do_visualize:
                # 全场预测 + 指标
                state = self._build_callback_state(step, total_steps, t_start)
                # 是否做重 IO (field PNG): 仅 heavy_io_interval 间隔
                do_heavy_io = (step % heavy_io_interval == 0) or (step == total_steps)

                # ① 磁盘保存中间产物 (即使 UI 中断, 用户也能拿到最新结果)
                try:
                    self._save_intermediate_artifacts(state, heavy=do_heavy_io)
                except Exception as exc:
                    self.logger.log_event(
                        f"中间产物保存失败 step={step}: {exc}", level="ERROR"
                    )

                # ② 通知 UI (可能失败, 但不影响磁盘)
                if self.callback is not None:
                    try:
                        self.callback(state)
                    except Exception as exc:
                        self.logger.log_event(f"callback error: {exc}", level="ERROR")

                # ③ 记录 metric
                self.logger.log_metric(step, state["metrics"])
                # 每个 visualize 间隔都把日志 flush 到磁盘, 防止丢失
                try:
                    self.logger.save()
                except Exception:
                    pass

                # ④ checkpoint (保留 best)
                cur = state["losses"]["total"]
                if cur < best_loss:
                    best_loss = cur
                    self._save_checkpoint(step, best=True)

        # 收尾: 保存最终模型 + 最终产物 (loss 图 / 预测场 / 误差图 / 剖面 / 残差 / metrics)
        last_step = self.loss_steps[-1] if self.loss_steps else 0
        self._save_checkpoint(step=last_step, best=False)
        self.logger.save()
        try:
            final_state = self._build_callback_state(last_step, total_steps, t_start)
            self._save_final_artifacts(final_state)
            self.logger.log_event(f"✅ 最终产物已保存到 {self.app_config.experiment_dir}")
        except Exception as exc:
            self.logger.log_event(f"最终产物保存失败: {exc}", level="ERROR")
            final_state = None
        finally:
            self.logger.save()

        return {
            "model": self.model,
            "trainer": self,
            "loss_steps": list(self.loss_steps),
            "loss_history": {k: list(v) for k, v in self.loss_history.items()},
            "logger": self.logger,
            "subdirs": self.subdirs,
            "final_state": final_state,
        }

    def _set_marching_segment_trainable(self, active_seg: int) -> None:
        """顺序训练: 仅当前传播段可更新."""
        model: SequentialMarchingPINN = self.model  # type: ignore
        if model.shared_network:
            for p in model.nets[0].parameters():
                p.requires_grad_(True)
            return
        for i, net in enumerate(model.nets):
            for p in net.parameters():
                p.requires_grad_(i == active_seg)

    def _rebuild_optimizer(self) -> None:
        lr = self.optimizer.param_groups[0]["lr"]
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("Marching 顺序训练: 当前段无可训练参数")
        self.optimizer = torch.optim.Adam(trainable, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(self.tc.epochs, 1), eta_min=1e-6,
        )

    def _sample_collocation_in_segment(
        self,
        x_lo: float,
        x_hi: float,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """在指定 x 段内采样 PDE 配点."""
        x = rng.uniform(x_lo, x_hi, n).astype(np.float32)
        z = rng.uniform(0.0, self.app_config.depth, n).astype(np.float32)
        return np.stack([x, z], axis=-1)

    def _train_marching_sequential(self) -> Dict[str, object]:
        """Sequential Marching: 按 x 段依次训练, 已训段冻结."""
        model: SequentialMarchingPINN = self.model  # type: ignore
        cfg = self.tc
        acfg = self.app_config
        n_seg = model.num_segments
        steps_per = int(acfg.marching_steps_per_segment) or max(cfg.epochs // n_seg, 100)
        total_steps = steps_per * n_seg
        bsz = max(cfg.batch_size, 8)
        rng = np.random.default_rng(cfg.random_seed)
        t_start = time.time()
        best_loss = math.inf
        eff_log_interval = max(cfg.log_interval, total_steps // 20000 if total_steps > 20000 else 1)
        heavy_io_interval = max(cfg.visualize_interval, total_steps // 200 if total_steps > 200 else 1)

        self.logger.log_event(
            f"➡️ Marching 顺序训练: {n_seg} 段 × {steps_per} step/段 = {total_steps} 总步"
        )

        global_step = 0
        for seg in range(n_seg):
            self._set_marching_segment_trainable(seg)
            self._rebuild_optimizer()
            core_lo, core_hi = model.core_bounds[seg]
            self.logger.log_event(
                f"  段 {seg + 1}/{n_seg}: x∈[{core_lo:.0f}, {core_hi:.0f}] m"
            )

            for _ in range(steps_per):
                global_step += 1
                if self.stop_flag():
                    break

                self.model.train()
                self.optimizer.zero_grad()

                idx = torch.from_numpy(
                    rng.integers(0, self.num_obs, size=bsz)
                ).long().to(self.device)
                coords_batch = self.coords_tensor[idx]
                tgt_r = self.p_real_target[idx]
                tgt_i = self.p_imag_target[idx]
                if self.use_envelope and self.supervise_envelope:
                    raw = self._raw_forward(coords_batch)
                    x_b = coords_batch[:, 0:1]
                    tgt_vr, tgt_vi = pressure_to_envelope(tgt_r, tgt_i, x_b, self._k0)
                    loss_data = (
                        torch.mean((raw[:, 0:1] - tgt_vr) ** 2)
                        + torch.mean((raw[:, 1:2] - tgt_vi) ** 2)
                    )
                else:
                    pred = self._forward(coords_batch)
                    loss_data = (
                        torch.mean((pred[:, 0:1] - tgt_r) ** 2)
                        + torch.mean((pred[:, 1:2] - tgt_i) ** 2)
                    )

                colloc_np = self._sample_collocation_in_segment(
                    core_lo, core_hi, cfg.num_collocation, rng,
                )
                colloc = torch.from_numpy(colloc_np).to(self.device)
                loss_pde = torch.mean(self._pde_residual(colloc) ** 2)
                loss_bc = self._boundary_loss()
                loss_interface = self._interface_loss() if self.use_interface_loss \
                    else torch.tensor(0.0, device=self.device)

                if self.use_gradnorm and self.gradnorm is not None:
                    tasks = [loss_data, loss_pde, loss_bc]
                    if self.use_interface_loss:
                        tasks.append(loss_interface)
                    try:
                        self.gradnorm.step(tasks, self._gradnorm_shared_params())
                    except Exception:
                        pass
                    w_data, w_pde, w_bc = self.gradnorm.weights[:3]
                    w_inf = self.gradnorm.weights[3] if self.use_interface_loss \
                        else torch.tensor(0.0, device=self.device)
                else:
                    w_data = torch.tensor(cfg.data_weight, device=self.device)
                    w_pde = torch.tensor(cfg.pde_weight, device=self.device)
                    w_bc = torch.tensor(cfg.boundary_weight, device=self.device)
                    w_inf = torch.tensor(float(acfg.marching_interface_weight), device=self.device)

                loss_total = (
                    w_data.detach() * loss_data
                    + w_pde.detach() * loss_pde
                    + w_bc.detach() * loss_bc
                )
                if self.use_interface_loss:
                    loss_total = loss_total + w_inf.detach() * loss_interface

                if not torch.isfinite(loss_total):
                    self.optimizer.zero_grad()
                    continue

                loss_total.backward()
                if self._gradient_clip > 0:
                    tr = [p for p in self.model.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(tr, self._gradient_clip)
                self.optimizer.step()
                self.scheduler.step()

                if global_step % max(eff_log_interval, 1) == 0:
                    entry = {
                        "total": float(loss_total.item()),
                        "data": float(loss_data.item()),
                        "pde": float(loss_pde.item()),
                        "bc": float(loss_bc.item()),
                    }
                    if self.use_interface_loss:
                        entry["interface"] = float(loss_interface.item())
                    self.loss_steps.append(global_step)
                    for k, v in entry.items():
                        if k in self.loss_history:
                            self.loss_history[k].append(v)
                    self.time_history.append(time.time() - t_start)
                    self.logger.log_loss(global_step, entry)

                if global_step % max(cfg.visualize_interval, 1) == 0:
                    state = self._build_callback_state(global_step, total_steps, t_start)
                    if self.callback is not None:
                        try:
                            self.callback(state)
                        except Exception as exc:
                            self.logger.log_event(f"callback error: {exc}", level="ERROR")

            if self.stop_flag():
                break

        # 解冻全部段供最终预测
        for net in model.nets:
            for p in net.parameters():
                p.requires_grad_(True)
        self._rebuild_optimizer()

        last_step = self.loss_steps[-1] if self.loss_steps else 0
        self._save_checkpoint(step=last_step, best=False)
        self.logger.save()
        try:
            final_state = self._build_callback_state(last_step, total_steps, t_start)
            self._save_final_artifacts(final_state)
        except Exception as exc:
            self.logger.log_event(f"最终产物保存失败: {exc}", level="ERROR")
            final_state = None
        finally:
            self.logger.save()

        return {
            "model": self.model,
            "trainer": self,
            "loss_steps": list(self.loss_steps),
            "loss_history": {k: list(v) for k, v in self.loss_history.items()},
            "logger": self.logger,
            "subdirs": self.subdirs,
            "final_state": final_state,
        }

    # ------------------------------------------------------------------ #
    # 预测 (全场, 分批推理)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_full_field(
        self, batch_size: int = 4096
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """在 *全网格* 上推理, 返回 (p_real, p_imag, TL)."""
        self.model.eval()
        preds = []
        n = self.coords_tensor_full.shape[0]
        for i in range(0, n, batch_size):
            coords_batch = self.coords_tensor_full[i:i + batch_size]
            y = self._forward(coords_batch).cpu().numpy()
            preds.append(y)
        full = np.concatenate(preds, axis=0)
        nz, nx = self.real_shape
        p_real = full[:, 0].reshape(nz, nx)
        p_imag = full[:, 1].reshape(nz, nx)
        tl = compute_tl(p_real, p_imag)
        return p_real.astype(np.float32), p_imag.astype(np.float32), tl.astype(np.float32)

    def compute_pde_residual_field(self, max_points: int = 20000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算全网格上的 PDE 残差 (下采样以节省显存)."""
        nz, nx = self.real_shape
        total = nz * nx
        step = max(1, total // max_points)
        idx = np.arange(0, total, step)
        coords = self.coords_tensor_full[idx]
        was_training = self.model.training
        self.model.train()
        try:
            res = self._pde_residual(coords)  # (M, 2)
            mag = torch.sqrt(torch.sum(res ** 2, dim=1)).detach().cpu().numpy()
        finally:
            if not was_training:
                self.model.eval()
        X_flat = self.X.flatten()[idx]
        Z_flat = self.Z.flatten()[idx]
        return X_flat, Z_flat, mag

    # ------------------------------------------------------------------ #
    # Callback payload
    # ------------------------------------------------------------------ #
    def _build_callback_state(self, step: int, total_steps: int, t_start: float) -> dict:
        p_real, p_imag, tl = self.predict_full_field()
        metrics: Dict[str, float] = {}
        metrics.update(compute_all_metrics(p_real, self.p_real_np, prefix="real_"))
        metrics.update(compute_all_metrics(p_imag, self.p_imag_np, prefix="imag_"))
        metrics.update(compute_all_metrics(tl, self.tl_true_np, prefix="tl_"))

        last_losses = {
            k: v[-1] if v else float("nan") for k, v in self.loss_history.items()
        }

        return {
            "step": step,
            "total_steps": total_steps,
            "losses": last_losses,
            "loss_steps": list(self.loss_steps),
            "loss_history": {k: list(v) for k, v in self.loss_history.items()},
            "time_history": list(self.time_history),
            "metrics": metrics,
            "pred_real": p_real,
            "pred_imag": p_imag,
            "pred_tl": tl,
            "true_real": self.p_real_np,
            "true_imag": self.p_imag_np,
            "true_tl": self.tl_true_np,
            "X": self.X,
            "Z": self.Z,
            "elapsed": time.time() - t_start,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "sampling_info": dict(self.sampling_result.info),
            "transfer_info": dict(self.transfer_info),
        }

    # ------------------------------------------------------------------ #
    # 内部: 构造 RMSE summary 字典 (共享给 CSV / TXT / XLSX 三处)
    # ------------------------------------------------------------------ #
    def _compose_rmse_summary(self, state: dict) -> Dict[str, object]:
        pr, pi, tl = state["pred_real"], state["pred_imag"], state["pred_tl"]
        tr, ti, tt = state["true_real"], state["true_imag"], state["true_tl"]
        X, Z = state["X"], state["Z"]
        final_metrics = state["metrics"]

        summary: Dict[str, object] = {
            "tl_rmse":   float(final_metrics.get("tl_rmse",   float("nan"))),
            "real_rmse": float(final_metrics.get("real_rmse", float("nan"))),
            "imag_rmse": float(final_metrics.get("imag_rmse", float("nan"))),
            "tl_mae":    float(final_metrics.get("tl_mae",    float("nan"))),
            "real_mae":  float(final_metrics.get("real_mae",  float("nan"))),
            "imag_mae":  float(final_metrics.get("imag_mae",  float("nan"))),
            "tl_corr":   float(final_metrics.get("tl_corr",   float("nan"))),
            "real_corr": float(final_metrics.get("real_corr", float("nan"))),
            "imag_corr": float(final_metrics.get("imag_corr", float("nan"))),
            "sampling_method":      self.sampling_result.info.get("method", ""),
            "num_train_obs":        int(self.num_obs),
            "epochs":               int(self.tc.epochs),
            "actual_steps":         int(state["step"]),
            "elapsed_seconds":      float(state["elapsed"]),
            "network_type":         self.app_config.network_type,
            "num_layers":           int(self.tc.num_layers),
            "num_neurons":          int(self.tc.num_neurons),
            "frequency":            float(self.app_config.frequency),
            "pretrained_ckpt":      str(getattr(self.app_config, "pretrained_ckpt", "")),
        }
        # 加 regional RMSE (近场 / 中场 / 远场)
        for name, p, t in [("tl", tl, tt), ("real", pr, tr), ("imag", pi, ti)]:
            reg = compute_regional_rmse(
                p, t, X, Z,
                source_r=self.app_config.source_r, source_z=self.app_config.source_z,
                near=self.app_config.near_dist_threshold,
                mid=self.app_config.mid_dist_threshold,
            )
            for k, v in reg.items():
                summary[f"{name}_{k}"] = v
        return summary

    # ------------------------------------------------------------------ #
    # 产物保存 (磁盘) - 中间 & 最终
    # ------------------------------------------------------------------ #
    def _save_intermediate_artifacts(self, state: dict, heavy: bool = True) -> None:
        """每隔 visualize_interval 保存一次. heavy=False 时跳过重 IO (field PNG).

        长程训练 (5M step) 时, heavy=False 占多数, 仅每 heavy_io_interval 一次完整 IO.
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        subdirs = self.subdirs

        # 1) loss 曲线 (cheap, 每次都保存)
        try:
            save_all_loss_figures(
                state["loss_steps"], state["loss_history"], subdirs["loss"],
            )
        except Exception:
            pass

        # 2) metrics 追加 (cheap)
        try:
            row = {"step": state["step"], "elapsed_s": state["elapsed"], **state["metrics"]}
            csv_path = Path(subdirs["metrics"]) / "metrics_history.csv"
            df = pd.DataFrame([row])
            df.to_csv(csv_path, mode="a",
                      header=not csv_path.exists(), index=False)
        except Exception:
            pass

        # 3) 时间历史 (cheap, 覆盖)
        try:
            if state.get("loss_steps") and state.get("time_history"):
                pd.DataFrame({
                    "step": state["loss_steps"],
                    "elapsed_s": state["time_history"],
                    "total": state["loss_history"]["total"],
                }).to_csv(Path(subdirs["logs"]) / "time_history.csv", index=False)
        except Exception:
            pass

        if not heavy:
            return

        # === 重 IO: field PNG / 误差图 (每 heavy_io_interval 一次) === #
        try:
            fig = plot_field_triptych(
                state["X"], state["Z"],
                state["pred_real"], state["pred_imag"], state["pred_tl"],
                suptitle=f"预测声场 (step {state['step']})",
            )
            fig.savefig(Path(subdirs["field"]) / "pred_latest.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            pass

        try:
            fig = plot_error_map(
                state["X"], state["Z"],
                state["pred_tl"], state["true_tl"],
                title=f"TL 绝对误差 (step {state['step']})",
            )
            fig.savefig(Path(subdirs["field"]) / "err_tl_latest.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            pass

    def _save_final_artifacts(self, state: dict) -> None:
        """训练结束后保存全部科研图、RMSE summary、参数 Excel、时间图、采样分布图等."""
        import matplotlib.pyplot as plt
        import pandas as pd

        # 应用发表级 matplotlib 样式
        setup_publication_style()

        subdirs = self.subdirs
        X, Z = state["X"], state["Z"]
        pr, pi, tl = state["pred_real"], state["pred_imag"], state["pred_tl"]
        tr, ti, tt = state["true_real"], state["true_imag"], state["true_tl"]

        # 1) 损失图 (汇总 + 各分量)
        save_all_loss_figures(
            state["loss_steps"], state["loss_history"], subdirs["loss"],
        )

        # 2) 最终预测三联图
        fig = plot_field_triptych(X, Z, pr, pi, tl, suptitle="最终预测声场")
        save_figure_publication(fig, Path(subdirs["field"]) / "pred_triptych",
                                formats=("png",))

        # 3) 真实-预测对比图
        fig = plot_field_comparison(X, Z, (tr, ti, tt), (pr, pi, tl))
        save_figure_publication(fig, Path(subdirs["field"]) / "compare_true_vs_pred",
                                formats=("png",))

        # 4) 误差分布 (Feature 3: 三种色阶都保存, 用于论文挑选)
        for name, p, t, label in [
            ("real", pr, tr, "实部"),
            ("imag", pi, ti, "虚部"),
            ("tl",   tl, tt, "传输损失"),
        ]:
            # 4a) 默认: 百分位色阶 (对小 RMSE 最友好)
            fig = plot_error_map(X, Z, p, t,
                                 title=f"{label}预测误差", scale="percentile")
            save_figure_publication(
                fig, Path(subdirs["field"]) / f"err_{name}",
                formats=("png",),
            )
            # 4b) Log 色阶
            fig = plot_error_map(X, Z, p, t,
                                 title=f"{label}预测误差", scale="log")
            save_figure_publication(
                fig, Path(subdirs["field"]) / f"err_{name}_log",
                formats=("png",),
            )
            # 4c) 多色阶并排
            fig = plot_error_map_multi_scale(X, Z, p, t, title_prefix=f"{label}误差")
            save_figure_publication(
                fig, Path(subdirs["field"]) / f"err_{name}_multi",
                formats=("png",),
            )

        # 5) TL 深度剖面 (中间距离)
        nz, nx = pr.shape
        mid = nx // 2
        fig = plot_profile_comparison(
            Z[:, mid], tt[:, mid], tl[:, mid],
            xlabel="深度 z (m)", ylabel="传输损失 (dB)",
            title=f"x = {X[0, mid]:.1f} m 处 TL 深度剖面",
        )
        save_figure_publication(fig, Path(subdirs["profiles"]) / "profile_tl_mid",
                                formats=("png",))

        # 6) PDE 残差散点图 (使用百分位色阶, 与误差图风格一致)
        try:
            from matplotlib.colors import LogNorm as _LogNorm
            x_sub, z_sub, res_mag = self.compute_pde_residual_field(max_points=20000)
            fig, ax = plt.subplots(figsize=(7.5, 5.0))
            # 用百分位裁剪让色阶更有对比
            if res_mag.size > 0:
                vmin = float(np.percentile(res_mag, 1.0))
                vmax = float(np.percentile(res_mag, 99.0))
                if vmax <= vmin:
                    vmax = vmin + 1e-12
            else:
                vmin, vmax = 0.0, 1.0
            sc = ax.scatter(x_sub, z_sub, c=res_mag, cmap="inferno", s=4,
                            vmin=vmin, vmax=vmax)
            fig.colorbar(sc, ax=ax, label="|Helmholtz residual|",
                         shrink=0.85, pad=0.04)
            ax.set_xlabel("距离 x (m)"); ax.set_ylabel("深度 z (m)")
            ax.set_title("PDE 残差幅值  [p1–p99 色阶, 下采样]")
            ax.invert_yaxis()
            fig.tight_layout()
            save_figure_publication(fig, Path(subdirs["residual"]) / "pde_residual",
                                    formats=("png",))
            pd.DataFrame({"x": x_sub, "z": z_sub, "residual": res_mag}).to_csv(
                Path(subdirs["residual"]) / "pde_residual.csv", index=False
            )
        except Exception as exc:
            self.logger.log_event(f"残差图保存失败: {exc}", level="WARN")

        # 7) 最终指标 CSV
        final_metrics = state["metrics"]
        pd.DataFrame([final_metrics]).to_csv(
            Path(subdirs["metrics"]) / "final_metrics.csv", index=False
        )

        # 8) 预测场 CSV (供下游分析)
        pd.DataFrame(pr).to_csv(Path(subdirs["field"]) / "pred_real.csv",
                                index=False, header=False)
        pd.DataFrame(pi).to_csv(Path(subdirs["field"]) / "pred_imag.csv",
                                index=False, header=False)
        pd.DataFrame(tl).to_csv(Path(subdirs["field"]) / "pred_tl.csv",
                                index=False, header=False)

        # ============================================================ #
        # 9) RMSE Summary (Feature 1) - 显式只输出三类 RMSE
        # ============================================================ #
        rmse_summary = self._compose_rmse_summary(state)
        pd.DataFrame([rmse_summary]).to_csv(
            Path(subdirs["metrics"]) / "rmse_summary.csv", index=False
        )
        # 人类可读 txt
        lines = ["===== PINN 预测 RMSE 汇总 =====\n"]
        lines.append(f"实验:       {self.app_config.experiment_name}")
        lines.append(f"采样方法:   {rmse_summary['sampling_method']} "
                     f"(训练观测点 {rmse_summary['num_train_obs']})")
        lines.append(f"网络:       {rmse_summary['network_type']} "
                     f"({rmse_summary['num_layers']} 层 × {rmse_summary['num_neurons']})")
        lines.append(f"训练 step:  {rmse_summary['actual_steps']} / {rmse_summary['epochs']}")
        lines.append(f"训练时长:   {rmse_summary['elapsed_seconds']:.1f} s")
        lines.append("\n---- 全场 RMSE ----")
        lines.append(f"  TL  RMSE = {rmse_summary['tl_rmse']:.6f} dB")
        lines.append(f"  实部 RMSE = {rmse_summary['real_rmse']:.6e}")
        lines.append(f"  虚部 RMSE = {rmse_summary['imag_rmse']:.6e}")
        lines.append("\n---- 分区 (距源距离) RMSE ----")
        for q in ("tl", "real", "imag"):
            lines.append(f"  {q}:  近 {rmse_summary[f'{q}_near_rmse']:.6g} "
                         f"| 中 {rmse_summary[f'{q}_mid_rmse']:.6g} "
                         f"| 远 {rmse_summary[f'{q}_far_rmse']:.6g}")
        lines.append("\n---- 相关系数 ----")
        for q in ("tl", "real", "imag"):
            lines.append(f"  {q}_corr = {rmse_summary[f'{q}_corr']:.6f}")
        Path(subdirs["metrics"]) / "rmse_summary.txt"  # touch
        (Path(subdirs["metrics"]) / "rmse_summary.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )

        # 10) 散点图: 预测 vs 真实 (TL / real / imag)
        try:
            for name, p, t, qname in [
                ("tl", tl, tt, "TL (dB)"),
                ("real", pr, tr, "Pressure (real)"),
                ("imag", pi, ti, "Pressure (imag)"),
            ]:
                fig = plot_pred_vs_true_scatter(p, t, quantity_name=qname)
                save_figure_publication(
                    fig, Path(subdirs["field"]) / f"scatter_pred_vs_true_{name}",
                    formats=("png",),
                )
        except Exception as exc:
            self.logger.log_event(f"散点图保存失败: {exc}", level="WARN")

        # ============================================================ #
        # 11) 训练时间成本图 (Feature 3)
        # ============================================================ #
        try:
            steps = state.get("loss_steps") or []
            times = state.get("time_history") or []
            n = min(len(steps), len(times))
            if n >= 2:
                steps_s, times_s = list(steps[:n]), list(times[:n])
                fig = plot_time_vs_step(steps_s, times_s)
                save_figure_publication(fig, Path(subdirs["logs"]) / "time_vs_step",
                                        formats=("png",))
                fig = plot_step_time_per_iter(steps_s, times_s)
                save_figure_publication(fig, Path(subdirs["logs"]) / "step_time_per_iter",
                                        formats=("png",))
                fig = plot_loss_vs_time(times_s, state["loss_history"]["total"][:n])
                save_figure_publication(fig, Path(subdirs["logs"]) / "loss_vs_time",
                                        formats=("png",))
                pd.DataFrame({
                    "step": steps_s, "elapsed_s": times_s,
                    "total": state["loss_history"]["total"][:n],
                    "data":  state["loss_history"]["data"][:n],
                    "pde":   state["loss_history"]["pde"][:n],
                    "bc":    state["loss_history"]["bc"][:n],
                }).to_csv(Path(subdirs["logs"]) / "time_history.csv", index=False)
        except Exception as exc:
            self.logger.log_event(f"时间图保存失败: {exc}", level="WARN")

        # ============================================================ #
        # 12) 采样分布彩图 (按区/距离上色, 标注 length/depth/源/近中场)
        # ============================================================ #
        try:
            method = (self.sampling_result.info.get("method") or "").lower()
            nz, nx = self.real_shape
            problem_region = None
            if method == "problem_region_aug":
                problem_region = (
                    self.app_config.problem_region_x_min * self.app_config.length,
                    self.app_config.problem_region_x_max * self.app_config.length,
                    self.app_config.problem_region_z_min * self.app_config.depth,
                    self.app_config.problem_region_z_max * self.app_config.depth,
                )
            fig = plot_sampling_distribution(
                self.sampling_result.coords,
                length=self.app_config.length, depth=self.app_config.depth,
                source_r=self.app_config.source_r, source_z=self.app_config.source_z,
                method_name=self.sampling_result.info.get("method", ""),
                n_total=self.sampling_result.size,
                color_mode="auto",
                viz_style=sampling_viz_style(method),
                near_dist=self.app_config.near_dist_threshold,
                mid_dist=self.app_config.mid_dist_threshold,
                problem_region=problem_region,
                sampling_info=self.sampling_result.info,
                nx=nx, nz=nz,
            )
            save_figure_publication(
                fig, Path(subdirs["logs"]) / "sampling_distribution",
                formats=("png",),
            )
            # 同时保存采样索引/坐标 CSV
            pd.DataFrame({
                "index": self.sampling_result.indices,
                "x":     self.sampling_result.coords[:, 0],
                "z":     self.sampling_result.coords[:, 1],
            }).to_csv(Path(subdirs["logs"]) / "sampling_indices.csv", index=False)
            import json as _json
            (Path(subdirs["logs"]) / "sampling_info.json").write_text(
                _json.dumps(self.sampling_result.info, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            self.logger.log_event(f"采样分布图保存失败: {exc}", level="WARN")

        # ============================================================ #
        # 12.5) GradNorm 权重曲线 (如果启用)
        # ============================================================ #
        if self.use_gradnorm and self.weight_history:
            try:
                fig, ax = plt.subplots(figsize=(7.5, 4.5))
                steps_arr = state.get("loss_steps") or []
                n = min(len(steps_arr), min(len(v) for v in self.weight_history.values()))
                steps_use = steps_arr[:n]
                colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
                for ci, (name, vals) in enumerate(self.weight_history.items()):
                    ax.plot(steps_use, vals[:n],
                            label=name, color=colors[ci % len(colors)], linewidth=1.5)
                ax.set_xlabel("Iteration"); ax.set_ylabel("Loss weight (GradNorm)")
                ax.set_title("GradNorm 自适应损失权重变化")
                ax.grid(True, alpha=0.3); ax.legend(loc="best")
                fig.tight_layout()
                fig.savefig(Path(subdirs["loss"]) / "gradnorm_weights.png",
                            dpi=200, bbox_inches="tight")
                plt.close(fig)
                pd.DataFrame({"step": steps_use, **{k: v[:n] for k, v in self.weight_history.items()}}) \
                    .to_csv(Path(subdirs["loss"]) / "gradnorm_weights.csv", index=False)
            except Exception as exc:
                self.logger.log_event(f"GradNorm 权重曲线保存失败: {exc}", level="WARN")

        # ============================================================ #
        # 12.6) Domain Decomposition 子域示意图
        # ============================================================ #
        if self.use_domain_decomp and isinstance(self.model, DomainDecomposedPINN):
            try:
                fig, ax = plt.subplots(figsize=(8.0, 5.5))
                ax.set_xlim(0, self.app_config.length)
                ax.set_ylim(0, self.app_config.depth)
                ax.invert_yaxis()
                colors_dd = ["#FFB3BA", "#BAFFC9", "#BAE1FF", "#FFFFBA",
                             "#FFDFBA", "#E0BBE4", "#B5EAD7", "#FFC9DE"]
                from matplotlib.patches import Rectangle
                for i, (lo, hi) in enumerate(self.model.bounds_x):
                    rect = Rectangle(
                        (lo, 0), hi - lo, self.app_config.depth,
                        linewidth=1.2, edgecolor="black",
                        facecolor=colors_dd[i % len(colors_dd)],
                        alpha=0.28,
                        label=f"sub {i+1} ext: [{lo:.0f}, {hi:.0f}] m",
                    )
                    ax.add_patch(rect)
                for i, (clo, chi) in enumerate(self.model.core_bounds):
                    ax.axvline(clo, color=colors_dd[i % len(colors_dd)],
                               linestyle="-", linewidth=0.8, alpha=0.9)
                    if i == self.model.num_subdomains - 1:
                        ax.axvline(chi, color=colors_dd[i % len(colors_dd)],
                                   linestyle="-", linewidth=0.8, alpha=0.9)
                for k in range(self.model.num_subdomains - 1):
                    interval = self.model.overlap_interval(k)
                    if interval is None:
                        continue
                    x_lo, x_hi = interval
                    ov = Rectangle(
                        (x_lo, 0), x_hi - x_lo, self.app_config.depth,
                        linewidth=1.0, edgecolor="purple", facecolor="none",
                        linestyle="--", alpha=0.9,
                        label="过渡带 (双网+声源优先)" if k == 0 else None,
                    )
                    ax.add_patch(ov)
                # 声源
                ax.plot(self.app_config.source_r, self.app_config.source_z,
                        marker="*", color="red", markersize=18,
                        markeredgecolor="black", markeredgewidth=0.8, zorder=10)
                ax.set_xlabel(f"水平距离 x (m)   ·   总范围 0 ~ {self.app_config.length:.0f}")
                ax.set_ylabel(f"深度 z (m)   ·   总深度 0 ~ {self.app_config.depth:.0f}")
                ax.set_title(
                    f"Domain Decomposition: {self.model.num_subdomains} 子域 "
                    f"(overlap={self.model.overlap:.0f}m, 过渡带声源优先拼接)"
                )
                ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
                fig.savefig(Path(subdirs["logs"]) / "domain_decomposition.png",
                            dpi=200, bbox_inches="tight")
                plt.close(fig)
                # 子域信息 JSON
                import json as _json
                dd_info = {
                    "num_subdomains": self.model.num_subdomains,
                    "length": self.model.length,
                    "depth": self.model.depth,
                    "overlap": self.model.overlap,
                    "source_r": self.model.source_r,
                    "source_z": self.model.source_z,
                    "interface_x": list(self.model.interface_x),
                    "bounds_x": list(self.model.bounds_x),
                    "core_bounds": list(self.model.core_bounds),
                    "blend": "source_priority_in_overlap",
                }
                (Path(subdirs["logs"]) / "domain_decomposition.json").write_text(
                    _json.dumps(dd_info, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:
                self.logger.log_event(f"DD 示意图保存失败: {exc}", level="WARN")

        # ============================================================ #
        # 12.7) Sequential Marching 示意图
        # ============================================================ #
        if self.use_marching and isinstance(self.model, SequentialMarchingPINN):
            try:
                import json as _json
                from matplotlib.patches import Rectangle
                fig, ax = plt.subplots(figsize=(8.0, 5.5))
                ax.set_xlim(0, self.app_config.length)
                ax.set_ylim(0, self.app_config.depth)
                ax.invert_yaxis()
                colors_m = ["#FFE5B4", "#B4D7FF", "#C9F0C9", "#F0C9F0", "#FFD4B4"]
                model_m: SequentialMarchingPINN = self.model  # type: ignore
                for i, (lo, hi) in enumerate(model_m.bounds_x):
                    ax.add_patch(Rectangle(
                        (lo, 0), hi - lo, self.app_config.depth,
                        linewidth=1.2, edgecolor="black",
                        facecolor=colors_m[i % len(colors_m)],
                        alpha=0.3,
                        label=f"seg {i+1}: [{lo:.0f},{hi:.0f}]m",
                    ))
                for k in range(model_m.num_segments - 1):
                    iv = model_m.overlap_interval(k)
                    if iv:
                        x_lo, x_hi = iv
                        ax.add_patch(Rectangle(
                            (x_lo, 0), x_hi - x_lo, self.app_config.depth,
                            linewidth=1.0, edgecolor="purple", facecolor="none",
                            linestyle="--",
                        ))
                ax.arrow(
                    self.app_config.source_r, self.app_config.source_z,
                    min(200, self.app_config.length * 0.15), 0,
                    head_width=15, color="red", length_includes_head=True,
                )
                ax.set_title(
                    f"Sequential Marching: {model_m.num_segments} 段 "
                    f"(overlap={model_m.overlap:.0f}m, 左→右因果)"
                )
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
                fig.savefig(Path(subdirs["logs"]) / "marching_decomposition.png",
                            dpi=200, bbox_inches="tight")
                plt.close(fig)
                march_info = {
                    "num_segments": model_m.num_segments,
                    "segment_length": model_m.segment_length,
                    "overlap": model_m.overlap,
                    "shared_network": model_m.shared_network,
                    "bounds_x": list(model_m.bounds_x),
                    "core_bounds": list(model_m.core_bounds),
                    "interface_x": list(model_m.interface_x),
                }
                (Path(subdirs["logs"]) / "marching_decomposition.json").write_text(
                    _json.dumps(march_info, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:
                self.logger.log_event(f"Marching 示意图保存失败: {exc}", level="WARN")

        # ============================================================ #
        # 12.8) 物理模式归档
        # ============================================================ #
        try:
            import json as _json
            physics_info = {
                "physics_mode": self.app_config.physics_mode_label,
                "use_envelope_decomposition": self.use_envelope,
                "use_pe_pde": self.use_pe,
                "supervise_envelope": self.supervise_envelope,
                "k": self._k,
                "k0": self._k0,
                "wave_number_formula": self.app_config.wave_number_formula,
                "use_marching": self.use_marching,
                "use_domain_decomp": self.use_domain_decomp,
            }
            (Path(subdirs["logs"]) / "physics_mode.json").write_text(
                _json.dumps(physics_info, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

        # ============================================================ #
        # 13) 迁移学习信息 (Feature 2)
        # ============================================================ #
        if self.transfer_info.get("enabled"):
            try:
                import json as _json
                (Path(subdirs["logs"]) / "transfer_info.json").write_text(
                    _json.dumps(self.transfer_info, indent=2, ensure_ascii=False,
                                default=str),
                    encoding="utf-8",
                )
            except Exception:
                pass

        # ============================================================ #
        # 14) 参数 Excel (Feature 1: 一次训练全部环境归档为单个 .xlsx)
        # ============================================================ #
        try:
            runtime_info = {
                "saved_at":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "device":            str(self.device),
                "actual_steps":      state["step"],
                "total_steps":       state["total_steps"],
                "elapsed_seconds":   state["elapsed"],
                "elapsed_minutes":   round(state["elapsed"] / 60.0, 2),
                "experiment_dir":    str(self.app_config.experiment_dir),
                "best_ckpt":         str(Path(self.subdirs["model"]) / "best.pt"),
                "num_train_obs_actual": int(self.num_obs),
                "model_summary":     self.model.summary(),
                "nan_count":         int(self._nan_count),
            }
            xlsx_path = save_parameters_xlsx(
                out_path=Path(subdirs["metrics"]) / "parameters.xlsx",
                config=self.app_config,
                final_metrics=state["metrics"],
                rmse_summary=self._compose_rmse_summary(state),
                sampling_info=self.sampling_result.info,
                transfer_info=self.transfer_info,
                runtime_info=runtime_info,
            )
            self.logger.log_event(f"参数 Excel 已保存: {xlsx_path}")
        except Exception as exc:
            self.logger.log_event(f"参数 Excel 保存失败: {exc}", level="WARN")

    # ------------------------------------------------------------------ #
    def _save_checkpoint(self, step: int, best: bool = False) -> str:
        fname = "best.pt" if best else f"step_{step}.pt"
        path = Path(self.subdirs["model"]) / fname
        torch.save(
            {
                "step": step,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.app_config.to_dict(),
            },
            path,
        )
        return str(path)
