# PINN 水下声场交互式训练平台

基于 **PyTorch + Streamlit** 的 Physics-Informed Neural Network (PINN) 训练、监控与可视化一体化工具，求解二维水下声场传播问题（Helmholtz / 抛物方程 PE / 包络分解等可切换）。

平台已迭代为完整科研工具链，包含：**4 种网络架构**、**6 种样本划分策略（含 RAS）**、**跨频率迁移学习**、**GradNorm 自适应损失**、**域分解 DD**、**长距离物理增强（包络 / PE / Sequential Marching）**、**长程训练稳定性** 与 **多实验对比**。

---

## ✨ 功能特性

### 基础能力
- 侧边栏填写核心参数（数据 / 空间 / 网络 / 训练）+ 多组高级物理与稳定性选项
- 实时训练监控：Loss 曲线、预测声场（实部 / 虚部 / TL）、误差分布、指标仪表盘
- 每隔 `visualize_interval` step 自动刷新界面与磁盘产物
- 完整科研级输出：图表、CSV、JSON、Excel 归档至 `outputs/<experiment_name>/`

### 4 种网络架构（可一键切换对比）
| 架构 | 适用场景 | 推荐参数 |
|---|---|---|
| **Fourier Feature + MLP**（默认推荐） | 高频声场、远场振荡 | `mapping_size=128, σ=5~30, 5×128` |
| **SIREN** | 极致光滑性、高阶导友好 | `w0_first=15, 5×128` |
| **Modified MLP**（Wang 2021） | 训练加速 1.5-2× | `5×128, tanh` |
| **DNN**（基线 Tanh） | 对照实验 | `7×50` |

### 6 种训练样本点划分策略
| 策略 | 说明 |
|---|---|
| **uniform** | 全网格均匀随机选 `num_train_obs` 个点（默认） |
| **stratified_block** | 分 nbx × nbz 块，按距源距离分层（近场少 / 远场多） |
| **lhs** | 拉丁超立方分层抽样 |
| **grid_uniform** | 等间距网格抽样 |
| **problem_region_aug** | stratified_block + 指定问题区域加密 |
| **residual_adaptive (RAS)** | 初始稀疏种子 + 训练中按 PDE/数据残差自适应追加点 |

采样预览图按方法区分风格：`stratified` 类显示近/中/远弧形分区；`lhs` 显示 (x,z) 分层线；`grid_uniform` 显示网格示意。

### 长距离物理增强（技术2，可独立开关，默认全关）

详见项目根目录 `技术2.md` 设计说明。侧边栏 **「📡 长距离物理增强 (技术2)」**：

| 模块 | 作用 | 典型场景 |
|---|---|---|
| **包络分解 Envelope** | 网络学慢变包络 v，声压 u = v·exp(i k₀x) | 500 Hz 高频、相位累积难学 |
| **抛物方程 PE-PINN** | PDE 残差改为 ∂u/∂x = (i/2k)∂²u/∂z² | 2 km 级 +x 单向传播 |
| **Sequential Marching** | 沿 x 分段、重叠过渡、左→右因果拼接 | 长域全局 PINN 失效时 |

- **波数公式**：`f/c`（兼容旧实验）或 `2πf/c`（标准 Helmholtz，高频推荐后者）
- **Marching 与 DD 互斥**（启用 Marching 时自动关闭 DD）
- 可选 **按段顺序训练**（冻结已训段，逐段推进）
- 训练结束写入 `logs/physics_mode.json`

### Domain Decomposition（XPINN 风格）
- `length > 阈值` 时 auto 启用，或 force on/off
- 子域首尾重叠、过渡带界面损失 + 可选单向近→远耦合
- 重叠区内按距声源更近子网输出

### GradNorm 自适应损失权重
- 自动平衡 data / pde / bc / interface 各任务下降速率
- 可配置 α、warmup、更新间隔；与 DD / Marching 的 interface 损失联动

### 跨频率迁移学习
- 从已训练 `.pt` 加载权重，微调适配同环境下其他频率
- 可冻结前 N 层 Linear，lr 按缩放系数下调
- Fourier 网络可选按 `f_new / f_old` 缩放 B 矩阵
- 典型加速 **3-10×** vs 从零训练

### 长程训练稳定性（5M step 不再崩溃）
- NaN / Inf 自动检测与跳过，累计超阈值自动停止
- 梯度裁剪、CUDA 缓存清理、Loss 历史下采样
- 重 IO 按总 step 自适应节流（全程最多约 200 次 field PNG）

### 多实验对比页
- 扫描 `outputs/` 下 `rmse_summary.csv`
- 时间-精度散点、采样方法柱状、近/中/远场分区、Loss-vs-时间叠加

---

## 🗂️ 目录结构

```
Auto_train_PINN/
├── README.md
├── Streamlit启动说明.txt          # Streamlit 启动步骤（与 README 同目录）
├── 技术2.md                       # 包络 / PE / Marching 技术设计文档
├── requirements.txt
└── pinn_app/
    ├── app.py                     # Streamlit 主入口 (5 个 tab)
    ├── config.py                  # AppConfig: 全部参数
    ├── data/
    │   ├── loader.py
    │   └── sampling.py            # 6 种采样 + RAS
    ├── physics/
    │   ├── envelope.py            # 包络分解 u = v·exp(ik₀x)
    │   └── pde_residuals.py       # Helmholtz / PE 残差
    ├── models/
    │   ├── pinn.py                # DNN / Fourier / SIREN / Modified
    │   ├── domain_decomp.py       # 域分解 DD
    │   └── marching_pinn.py       # Sequential Marching
    ├── training/
    │   ├── trainer.py             # 训练主循环
    │   └── gradnorm.py            # GradNorm 权重
    └── utils/
        ├── metrics.py
        ├── visualization.py
        ├── excel_logger.py
        └── logger.py
```

训练产出目录 `outputs/<experiment_name>/`：

```
├── field/          预测场、对比图、误差图、CSV
├── loss/           loss 曲线 + gradnorm_weights
├── metrics/        RMSE 汇总、final_metrics、history
├── profiles/       深度剖面
├── residual/       PDE 残差
├── logs/           事件、采样图、时间图、physics_mode.json
│                   domain_decomposition.* / marching_decomposition.*
│                   transfer_info.json
└── model/          best.pt
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

GPU 训练请先安装与本机 CUDA 匹配的 PyTorch：<https://pytorch.org/get-started/locally/>

```powershell
& C:/Users/<you>/.conda/envs/gpu_pytorch/python.exe -m pip install -r requirements.txt
```

### 2. 启动界面

```powershell
cd D:/lzz/Auto_train_PINN
& C:/Users/<you>/.conda/envs/gpu_pytorch/python.exe -m streamlit run pinn_app/app.py --server.port 8501 --server.fileWatcherType none
```

浏览器打开 **http://localhost:8501**。更详细的启动、排错说明见 **`Streamlit启动说明.txt`**。

> `--server.fileWatcherType none` 用于规避 PyTorch ≥ 2.5 与 Streamlit 文件监视的兼容警告。

### 3. 使用流程

1. **侧边栏** — 数据路径、空间参数、网络、训练超参、采样、GradNorm、DD、**长距离物理增强**、稳定性、输出目录
2. **数据预览** — 确认 CSV 与真实声场
3. **训练 & 监控** — 开始训练，实时查看曲线与声场
4. **迁移学习**（可选）— 配置预训练 `.pt`
5. **实验对比** — 横向对比历史实验 RMSE 与耗时

---

## 🔬 物理模型

### 默认：Helmholtz PINN

∇²p + k²p = S(x, z)，p = p_real + i·p_imag。

**波数 k**（侧边栏可选）：
- `legacy_f_over_c`：k = f/c（与早期实验一致）
- `2pi_f_over_c`：k = 2πf/c（标准形式，**高频/长距离推荐**）

**边界条件**：
- 海面 z=0：Dirichlet p = 0
- 海底：Neumann ∂p/∂n = 0
- 左右：弱 Neumann

**包络模式**（可选）：网络输出 v，u = v·exp(i k₀x)，PDE/边界/评估在 u 上计算。

**PE 模式**（可选）：∂u/∂x = (i/2k)∂²u/∂z²，适合沿 +x 长距离传播。

---

## 📡 长距离 / 高频推荐配置（示例：2 km, 500 Hz）

| 项目 | 建议 |
|---|---|
| 物理增强 | ✅ 包络分解 + ✅ PE（或先只开包络做对比） |
| 波数 | `2pi_f_over_c` |
| Marching | `on`，段长 200 m，重叠 80 m |
| 采样 | `residual_adaptive`，metric=`pde` |
| 迁移 | 50 Hz 的 `best.pt` + Fourier B 缩放 |
| DD | 启用 Marching 时自动关闭 |

---

## 🔁 跨频率迁移学习

1. 打开 **🔁 迁移学习** tab，填入 `best.pt` 路径
2. 设置冻结层数、lr 缩放、预训练频率、Fourier B 缩放
3. 侧边栏修改 **频率** 与 **CSV 路径** 为目标频率数据
4. 在 **训练 & 监控** 开始训练 → `logs/transfer_info.json`

---

## 📊 实验产物说明

| 文件 | 内容 |
|---|---|
| `metrics/rmse_summary.csv` / `.txt` | TL / 实虚部 RMSE、分区 RMSE、相关系数 |
| `loss/loss_*.png` | 各分量 loss；`gradnorm_weights.png` 若启用 GradNorm |
| `field/pred_*.png` / `pred_*.csv` | 预测三联图、对比、误差、散点、原始场 |
| `logs/sampling_distribution.png` | 训练点空间分布 |
| `logs/physics_mode.json` | 包络 / PE / Marching / 波数公式归档 |
| `logs/domain_decomposition.*` | DD 子域示意（若启用） |
| `logs/marching_decomposition.*` | Marching 分段示意（若启用） |
| `model/best.pt` | 最优权重 |

---

## 📝 输入数据格式

- `pres_real.csv` / `pres_imag.csv`：二维 CSV，形状 `(nz, nx)`
- 行 = 深度 z，列 = 水平距离 x
- 自动识别表头与分隔符

**Nyquist 提示**：500 Hz 时波长 λ≈3 m，若 dx≈2 m 则网格偏粗，除调网络外也需注意数据分辨率。

---

## 🛠️ 长程训练稳定性

| 参数 | 默认 | 说明 |
|---|---|---|
| `gradient_clip` | 1.0 | 梯度裁剪，0=关闭 |
| `nan_skip_threshold` | 50 | NaN 累计超限则停止 |
| `cuda_empty_cache_every` | 1000 | CUDA 缓存清理间隔 |
| `max_loss_points_in_memory` | 20000 | loss 历史内存上限 |

---

## 🧩 扩展建议

- **新网络**：`models/pinn.py` → `build_pinn`
- **新采样**：`data/sampling.py` → `SAMPLING_METHODS`
- **新 PDE**：`physics/pde_residuals.py` → `compute_pde_residual`
- **新分段策略**：`models/marching_pinn.py` 或 `domain_decomp.py`

---

## 📜 引用与参考

- 物理与采样参考 `PINN_500m/train_acoustic_pinn.py`、`train_acoustic_pinn_wangge.py`
- 长距离方案设计见本项目 **`技术2.md`**
- Fourier Feature：Tancik et al., NeurIPS 2020
- SIREN：Sitzmann et al., NeurIPS 2020
- Modified MLP：Wang et al., JCP 2021
- GradNorm：Chen et al., ICML 2018
