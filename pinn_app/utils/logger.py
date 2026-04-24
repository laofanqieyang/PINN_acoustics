"""训练日志记录.

将每一次 step 的 loss / metrics 缓存到内存 (供 UI 实时展示)
并持久化到 CSV / JSON 文件.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class TrainingLogger:
    """训练日志器.

    用法::

        logger = TrainingLogger(log_dir="outputs/exp1/logs")
        logger.log_loss(step=100, losses={"total": 0.12, "data": 0.05, "pde": 0.07})
        logger.log_metric(step=100, metrics={"rmse": 0.1, "mae": 0.08})
        logger.save()
    """

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.loss_history: List[Dict[str, Any]] = []
        self.metric_history: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self._t0 = time.time()

    # --------------------- 记录 --------------------- #
    def log_loss(self, step: int, losses: Dict[str, float]) -> None:
        entry = {"step": int(step), "time": time.time() - self._t0, **losses}
        self.loss_history.append(entry)

    def log_metric(self, step: int, metrics: Dict[str, float]) -> None:
        entry = {"step": int(step), "time": time.time() - self._t0, **metrics}
        self.metric_history.append(entry)

    def log_event(self, message: str, level: str = "INFO") -> None:
        self.events.append(
            {"time": time.time() - self._t0, "level": level, "message": message}
        )

    # --------------------- 查询 --------------------- #
    def loss_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.loss_history)

    def metric_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.metric_history)

    # --------------------- 持久化 --------------------- #
    def save(self) -> Dict[str, str]:
        paths = {}
        if self.loss_history:
            p = self.log_dir / "loss_history.csv"
            self.loss_df().to_csv(p, index=False)
            paths["loss"] = str(p)
        if self.metric_history:
            p = self.log_dir / "metric_history.csv"
            self.metric_df().to_csv(p, index=False)
            paths["metric"] = str(p)
        if self.events:
            p = self.log_dir / "events.json"
            p.write_text(json.dumps(self.events, ensure_ascii=False, indent=2))
            paths["events"] = str(p)
        return paths

    def save_config(self, config_dict: Dict[str, Any]) -> str:
        p = self.log_dir / "config.json"
        p.write_text(json.dumps(config_dict, ensure_ascii=False, indent=2, default=str))
        return str(p)
