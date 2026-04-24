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

import math
import time
from dataclasses import dataclass, field
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
from ..models.pinn import build_pinn
from ..utils.logger import TrainingLogger
from ..utils.metrics import compute_all_metrics
from ..utils.visualization import compute_tl


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

        # 网络 & 优化器
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
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.tc.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(self.tc.epochs, 1), eta_min=1e-6
        )

        # loss 缓存 (供 UI 实时画曲线)
        self.loss_steps: List[int] = []
        self.loss_history: Dict[str, List[float]] = {
            "total": [], "data": [], "pde": [], "bc": []
        }

        # k^2 常量
        self._k2 = torch.tensor(
            self.tc.wave_number ** 2, dtype=torch.float32, device=self.device
        )

        # 雅可比 (归一化 -> 物理), d(norm)/d(phys)
        jx, jz = self.normalizer.jacobian
        self._jx2 = float(jx) ** 2
        self._jz2 = float(jz) ** 2

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
        """读取真实数据, 构造训练张量和边界点."""
        p_real, p_imag = load_pressure_data(
            self.app_config.pres_real_path,
            self.app_config.pres_imag_path,
            expected_shape=(self.app_config.nz, self.app_config.nx),
        )
        self.real_shape = p_real.shape  # (nz, nx) 来自文件
        pack = build_training_tensors(
            p_real, p_imag,
            length=self.app_config.length,
            depth=self.app_config.depth,
            device=self.device,
        )
        self.X = pack["X"]                # np (nz, nx)
        self.Z = pack["Z"]
        self.coords_phys = pack["coords_phys"]    # np (N, 2)
        self.coords_tensor = pack["coords_tensor"]  # torch (N, 2)
        self.p_real_target = pack["p_real_tensor"]
        self.p_imag_target = pack["p_imag_tensor"]
        self.p_real_np = p_real
        self.p_imag_np = p_imag
        self.tl_true_np = compute_tl(p_real, p_imag)
        self.normalizer: MinMaxNormalizer = pack["normalizer"]
        self.num_obs = self.coords_tensor.shape[0]

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
            f"真实数据形状={self.real_shape}, 观测点数={self.num_obs}"
        )

    # ------------------------------------------------------------------ #
    # 前向 (归一化输入)
    # ------------------------------------------------------------------ #
    def _forward(self, coords_phys: torch.Tensor) -> torch.Tensor:
        xz_norm = self.normalizer.encode_torch(coords_phys)
        return self.model(xz_norm)  # (N, 2)

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
        """返回 shape=(N, 2) 的残差 (实部, 虚部)"""
        coords_phys = coords_phys.detach().clone().requires_grad_(True)
        xz_norm = self.normalizer.encode_torch(coords_phys)
        y = self.model(xz_norm)
        pr, pi = y[:, 0:1], y[:, 1:2]

        # 一阶导 d/d(phys)
        grad_pr = torch.autograd.grad(
            pr, coords_phys, grad_outputs=torch.ones_like(pr),
            create_graph=True, retain_graph=True,
        )[0]
        grad_pi = torch.autograd.grad(
            pi, coords_phys, grad_outputs=torch.ones_like(pi),
            create_graph=True, retain_graph=True,
        )[0]
        dpr_dx, dpr_dz = grad_pr[:, 0:1], grad_pr[:, 1:2]
        dpi_dx, dpi_dz = grad_pi[:, 0:1], grad_pi[:, 1:2]

        # 二阶导
        dpr_dxx = torch.autograd.grad(
            dpr_dx, coords_phys, grad_outputs=torch.ones_like(dpr_dx),
            create_graph=True, retain_graph=True,
        )[0][:, 0:1]
        dpr_dzz = torch.autograd.grad(
            dpr_dz, coords_phys, grad_outputs=torch.ones_like(dpr_dz),
            create_graph=True, retain_graph=True,
        )[0][:, 1:2]
        dpi_dxx = torch.autograd.grad(
            dpi_dx, coords_phys, grad_outputs=torch.ones_like(dpi_dx),
            create_graph=True, retain_graph=True,
        )[0][:, 0:1]
        dpi_dzz = torch.autograd.grad(
            dpi_dz, coords_phys, grad_outputs=torch.ones_like(dpi_dz),
            create_graph=True, retain_graph=True,
        )[0][:, 1:2]

        # 源项
        x_coord = coords_phys[:, 0:1]
        z_coord = coords_phys[:, 1:2]
        s_real = self._gaussian_source(x_coord, z_coord) * 10.0
        s_imag = 0.5 * self._gaussian_source(x_coord, z_coord) * 5.0

        res_real = dpr_dxx + dpr_dzz + self._k2 * pr - s_real
        res_imag = dpi_dxx + dpi_dzz + self._k2 * pi - s_imag
        return torch.cat([res_real, res_imag], dim=1)

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
        cfg = self.tc
        total_steps = cfg.epochs
        bsz = max(cfg.batch_size, 8)
        rng = np.random.default_rng(cfg.random_seed)
        t_start = time.time()
        best_loss = math.inf

        self.logger.log_event(
            f"开始训练: steps={total_steps}, batch_size={bsz}, "
            f"lr={cfg.learning_rate}, pde_weight={cfg.pde_weight}"
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
            pred = self._forward(coords_batch)
            pred_r, pred_i = pred[:, 0:1], pred[:, 1:2]
            tgt_r = self.p_real_target[idx]
            tgt_i = self.p_imag_target[idx]
            loss_data = torch.mean((pred_r - tgt_r) ** 2) + torch.mean((pred_i - tgt_i) ** 2)

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

            loss_total = (
                cfg.data_weight * loss_data
                + cfg.pde_weight * loss_pde
                + cfg.boundary_weight * loss_bc
            )

            loss_total.backward()
            self.optimizer.step()
            self.scheduler.step()

            # 记录 loss
            if step % max(cfg.log_interval, 1) == 0 or step == 1:
                entry = {
                    "total": float(loss_total.item()),
                    "data": float(loss_data.item()),
                    "pde": float(loss_pde.item()),
                    "bc": float(loss_bc.item()),
                }
                self.loss_steps.append(step)
                for k, v in entry.items():
                    self.loss_history[k].append(v)
                self.logger.log_loss(step, entry)

            # 周期性 UI 回调 + metric + checkpoint
            if (
                self.callback is not None
                and (step % max(cfg.visualize_interval, 1) == 0 or step == total_steps)
            ):
                state = self._build_callback_state(step, total_steps, t_start)
                try:
                    self.callback(state)
                except Exception as exc:  # UI 侧异常不应中断训练
                    self.logger.log_event(f"callback error: {exc}", level="ERROR")

                # 记录 metric
                self.logger.log_metric(step, state["metrics"])

                # checkpoint (保留 best)
                cur = state["losses"]["total"]
                if cur < best_loss:
                    best_loss = cur
                    self._save_checkpoint(step, best=True)

        # 收尾保存
        self._save_checkpoint(step=self.loss_steps[-1] if self.loss_steps else 0, best=False)
        self.logger.save()

        return {
            "model": self.model,
            "trainer": self,
            "loss_steps": list(self.loss_steps),
            "loss_history": {k: list(v) for k, v in self.loss_history.items()},
            "logger": self.logger,
            "subdirs": self.subdirs,
        }

    # ------------------------------------------------------------------ #
    # 预测 (全场, 分批推理)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_full_field(
        self, batch_size: int = 4096
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (p_real, p_imag, TL), shape 与真实数据一致."""
        self.model.eval()
        preds = []
        n = self.coords_tensor.shape[0]
        for i in range(0, n, batch_size):
            coords_batch = self.coords_tensor[i:i + batch_size]
            y = self._forward(coords_batch).cpu().numpy()
            preds.append(y)
        full = np.concatenate(preds, axis=0)
        nz, nx = self.real_shape
        p_real = full[:, 0].reshape(nz, nx)
        p_imag = full[:, 1].reshape(nz, nx)
        tl = compute_tl(p_real, p_imag)
        return p_real.astype(np.float32), p_imag.astype(np.float32), tl.astype(np.float32)

    def compute_pde_residual_field(self, max_points: int = 20000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算 PDE 残差 (下采样以节省显存). 返回 (X_sub, Z_sub, residual_magnitude)."""
        nz, nx = self.real_shape
        total = nz * nx
        step = max(1, total // max_points)
        idx = np.arange(0, total, step)
        coords = self.coords_tensor[idx]
        was_training = self.model.training
        self.model.train()  # 允许建图求导
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
        }

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
