"""GradNorm: 多任务自适应损失权重 (Chen et al., ICML 2018).

核心想法: 让各任务损失项以相近速率下降. 把权重 w_i 设为可训练参数,
通过最小化 ``Σ |G_i - G_avg · r_i^α|`` 自动平衡, 其中:
    * G_i  = ||∇_W (w_i · L_i)||  在共享层 W 上的梯度范数
    * r_i  = (L_i(t) / L_i(0))  /  mean(L_j(t) / L_j(0))   反向训练率
    * α    为超参 (典型 0.5 ~ 2.0)

实现说明:
    * 用单独的 Adam 优化器更新 w_i, 不参与主模型反向
    * 每 `update_every` 步才更新一次, 降低额外开销
    * 主训练循环正常用 `weights[i] * losses[i]` 求和; GradNorm 模块只读取损失和共享参数,
      不影响主反向图 (除了在 update 时做一次额外的局部 grad)
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn


class GradNormWeighter:
    """自适应任务权重.

    Parameters
    ----------
    task_names : 序列
        如 ("data", "pde", "bc")  或加上 "interface".
    alpha : float
        训练率指数. 越大 → 对慢任务越倾斜. 默认 1.5.
    lr : float
        权重的学习率 (与主网络优化器独立).
    init_weights : 序列, 可选
        初始权重. 若不给则全 1.
    update_every : int
        每多少 step 调一次 ``update()`` (减少梯度计算开销).
    warmup_steps : int
        前 N 步不动权重, 让损失先稳定下来, 否则 L_i(0) 可能不可靠.
    min_weight : float
        权重下限, 防止某任务被压成 0.
    device : str / torch.device
    """

    def __init__(
        self,
        task_names: Sequence[str],
        alpha: float = 1.5,
        lr: float = 1e-3,
        init_weights: Optional[Sequence[float]] = None,
        update_every: int = 10,
        warmup_steps: int = 100,
        min_weight: float = 0.01,
        device: torch.device | str = "cpu",
    ):
        self.task_names = list(task_names)
        self.n_tasks = len(self.task_names)
        self.alpha = float(alpha)
        self.update_every = max(int(update_every), 1)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.min_weight = float(min_weight)
        self.device = torch.device(device)

        if init_weights is None:
            init = [1.0] * self.n_tasks
        else:
            init = [float(w) for w in init_weights]
            if len(init) != self.n_tasks:
                raise ValueError("init_weights 长度必须等于 task 数")

        self.weights = nn.Parameter(
            torch.tensor(init, dtype=torch.float32, device=self.device),
            requires_grad=True,
        )
        self.opt = torch.optim.Adam([self.weights], lr=lr)
        self.initial_losses: Optional[List[float]] = None
        self.step_count = 0
        self.last_grad_norms: Optional[List[float]] = None
        self.last_loss_ratios: Optional[List[float]] = None

    # ----------------------------------------------------------------- #
    @torch.no_grad()
    def current_weights(self) -> Dict[str, float]:
        return {n: float(self.weights[i].item()) for i, n in enumerate(self.task_names)}

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.weights[idx]

    # ----------------------------------------------------------------- #
    def step(
        self,
        losses: Sequence[torch.Tensor],
        shared_params: Iterable[nn.Parameter],
    ) -> Dict[str, float]:
        """根据当前损失值和共享参数更新权重.

        必须在主优化器 ``optimizer.step()`` **之前** 调用 (需要保留 graph).

        Parameters
        ----------
        losses : list of 0-d Tensors
            各任务的损失值 (未加权).
        shared_params : iterable of Parameters
            用于计算 G_i 的共享参数 (一般取最后一个共享 Linear 的 weight).

        Returns
        -------
        info : dict
            含每个任务的当前权重 / 梯度范数 / 训练率, 便于日志.
        """
        if len(losses) != self.n_tasks:
            raise ValueError(
                f"losses 数量 {len(losses)} 与 task 数 {self.n_tasks} 不匹配"
            )
        self.step_count += 1
        info: Dict[str, float] = {}

        # warmup 期: 只记录初始损失, 不动权重
        if self.initial_losses is None:
            self.initial_losses = [float(l.detach().item()) + 1e-12 for l in losses]
        if self.step_count <= self.warmup_steps:
            for i, n in enumerate(self.task_names):
                info[f"w_{n}"] = float(self.weights[i].item())
            return info

        # 节流: 仅每 N 步更新一次
        if self.step_count % self.update_every != 0:
            for i, n in enumerate(self.task_names):
                info[f"w_{n}"] = float(self.weights[i].item())
            return info

        shared_list = [p for p in shared_params if p.requires_grad]
        if not shared_list:
            return {f"w_{n}": float(self.weights[i].item())
                    for i, n in enumerate(self.task_names)}

        # 1) 各任务加权损失的梯度范数
        grad_norms = []
        for i, l in enumerate(losses):
            weighted = self.weights[i] * l
            grads = torch.autograd.grad(
                weighted, shared_list,
                retain_graph=True, create_graph=True, allow_unused=True,
            )
            flat = []
            for g in grads:
                if g is not None:
                    flat.append(g.reshape(-1))
            if not flat:
                grad_norms.append(torch.tensor(0.0, device=self.device,
                                                requires_grad=True))
            else:
                gnorm = torch.norm(torch.cat(flat))
                grad_norms.append(gnorm)
        grad_norms_t = torch.stack(grad_norms)

        # 2) 训练率 r_i = (L_i(t)/L_i(0)) / mean(...)
        with torch.no_grad():
            loss_vals = torch.tensor(
                [float(l.detach().item()) for l in losses],
                dtype=torch.float32, device=self.device,
            )
            init_vals = torch.tensor(
                self.initial_losses, dtype=torch.float32, device=self.device,
            )
            inv_rate = loss_vals / (init_vals + 1e-12)
            mean_rate = inv_rate.mean().clamp_min(1e-12)
            r = (inv_rate / mean_rate).clamp_min(1e-3)
            target = grad_norms_t.detach().mean() * (r ** self.alpha)

        # 3) GradNorm loss: |G_i - target_i| 之和, **仅对 weights 求导**
        #    用 autograd.grad 显式拿到对 weight 的梯度, 避免 backward() 消费主图
        gn_loss = torch.abs(grad_norms_t - target).sum()
        try:
            w_grad = torch.autograd.grad(
                gn_loss, [self.weights], retain_graph=False, create_graph=False,
            )[0]
        except RuntimeError:
            # 极少数情况下 graph 已被释放, 跳过本次更新
            for i, n in enumerate(self.task_names):
                info[f"w_{n}"] = float(self.weights[i].item())
            return info
        self.opt.zero_grad()
        self.weights.grad = w_grad.detach()
        self.opt.step()

        # 4) 重归一化: Σ w_i = n_tasks; 下限钳制
        with torch.no_grad():
            self.weights.clamp_(min=self.min_weight)
            coef = self.n_tasks / self.weights.sum().clamp_min(1e-12)
            self.weights.mul_(coef)

        self.last_grad_norms = [float(g.detach().item()) for g in grad_norms]
        self.last_loss_ratios = [float(x.item()) for x in r]

        for i, n in enumerate(self.task_names):
            info[f"w_{n}"] = float(self.weights[i].item())
            info[f"grad_norm_{n}"] = self.last_grad_norms[i]
            info[f"ratio_{n}"] = self.last_loss_ratios[i]
        return info

    # ----------------------------------------------------------------- #
    def state_dict(self) -> dict:
        return {
            "weights": self.weights.detach().cpu().tolist(),
            "task_names": list(self.task_names),
            "alpha": self.alpha,
            "step_count": self.step_count,
            "initial_losses": list(self.initial_losses or []),
        }

    def load_state_dict(self, sd: dict) -> None:
        if "weights" in sd:
            w = torch.tensor(sd["weights"], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                self.weights.copy_(w)
        self.step_count = int(sd.get("step_count", 0))
        self.initial_losses = list(sd.get("initial_losses") or [])
