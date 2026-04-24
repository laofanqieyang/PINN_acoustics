# PINN 水下声场交互式训练平台

基于 **PyTorch + Streamlit** 的 Physics-Informed Neural Network 训练 / 监控 / 可视化一体化工具，按照 `PINN UI.md` 中的架构实现。

## ✨ 功能特性

- 侧边栏自由输入 12 项参数（文件路径 / 空间 / 网络 / 训练）
- 实时训练监控：Loss 曲线、预测声场（实部 / 虚部 / TL）、误差分布、指标仪表盘
- 每隔 `visualize_interval` 步自动刷新界面，训练进度一目了然
- 训练结束后自动导出科研级产出：
  - `loss/` 汇总损失图 + 每个分量单独的图
  - `field/` 最终预测声场、真实-预测 2×3 对比图、逐点误差分布，同时输出 CSV
  - `profiles/` 固定距离处的深度剖面对比
  - `residual/` Helmholtz PDE 残差场
  - `metrics/final_metrics.csv` RMSE / MAE / Rel-L2 / Corr
  - `logs/` 完整训练日志 + 配置快照
  - `model/best.pt` 训练好的权重

## 🗂️ 目录结构

```
pinn_app/
├── app.py                # Streamlit 主入口 (UI)
├── config.py             # AppConfig: 所有输入参数 (dataclass)
├── data/
│   └── loader.py         # 数据读取 / 网格 / 归一化 / 采样
├── models/
│   └── pinn.py           # 可配置的全连接 PINN
├── training/
│   └── trainer.py        # Helmholtz PINN 训练主循环
├── utils/
│   ├── metrics.py        # RMSE / MAE / Rel-L2 / Corr
│   ├── visualization.py  # 所有绘图函数 (Figure 返回)
│   └── logger.py         # 训练日志记录
└── outputs/              # 训练产出 (每次训练一个子目录)
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> GPU 训练请先确认已安装与本机 CUDA 版本匹配的 PyTorch：<https://pytorch.org/get-started/locally/>

### 2. 启动界面

```bash
cd d:/lzz/Auto_train_PINN
streamlit run pinn_app/app.py
```

打开浏览器访问 Streamlit 显示的地址即可。

> **若遇到 `RuntimeError: Tried to instantiate class '__path__._path'` 这类 PyTorch ↔ Streamlit 兼容问题**，请改用:
>
> ```bash
> streamlit run pinn_app/app.py --server.fileWatcherType none
> ```

### 3. 使用流程

1. 在左侧边栏填写 12 项必需参数：
   1. 声压实部 CSV 路径
   2. 声压虚部 CSV 路径
   3. 声场水平距离 `length`
   4. 声场深度 `depth`
   5. 水平方向数据点数 `nx`
   6. 深度方向数据点数 `nz`
   7. 学习率 `lr`
   8. `batch_size`
   9. PDE 权重 `pde_weight`
   10. 神经网络层数 `num_layers`
   11. 神经元个数 `num_neurons`
   12. 训练步数 `epochs`
2. 在"数据预览"标签页检查真实声场
3. 在"训练 & 监控"标签页点击 **开始训练**
4. 界面每 `visualize_interval` step 自动刷新
5. 训练完成后查看所有科研图与 `outputs/<experiment_name>/`

## 📝 输入数据格式

- `pres_real.csv` / `pres_imag.csv`：无表头二维 CSV，形状 `(nz, nx)`
- 行对应深度方向，列对应水平距离方向

## 🔬 物理模型

Helmholtz 方程：∇² p + k² p = S(x, z)，其中 p 为复声压。

- 海面 (z=0)：Dirichlet p = 0
- 海底 (z=depth)：Neumann ∂p/∂n = 0
- 左右边界：弱 Neumann (权重更低)
- 高斯源项 `S(x, z)`，参数可在侧边栏修改

## 🧩 扩展建议

- 在 `models/pinn.py` 加入 SIREN / Fourier Feature 变体
- 在 `training/trainer.py` 接入 LBFGS 二阶优化作为微调
- 多频 / 多介质：扩展 `config.py` 并在 `trainer._pde_residual` 加入空间变化的 k
- 记录到 TensorBoard / WandB：在 `utils/logger.py` 里注入即可
