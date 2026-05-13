# Stage3：已发表口径的“等效剪枝率 ↔ 精度”数据（Top‑down，COCO val2017，256×192）

你需要的是“公开论文/已发表口径”的数据，并希望以 Params/MACs/GFLOPs 统一折算出一个“剪枝率”，从而得到 **剪枝率 ↔ 精度（AP）** 的可用表格。

在姿态估计领域，多数已发表论文并不会以“剪枝率(%)”作为统一横轴，而是报告 **#Params / FLOPs(GFLOPs)** 与 AP。这里给出一套**可复现、可解释**的折算方式，把每个方法的 Params/GFLOPs 映射成一个“等效剪枝率（equivalent prune rate）”。

---

## 1. 等效剪枝率定义

以 **HRNet‑W32 256×192** 作为基线（CVPR 2019，COCO val2017，detector person AP=56.4）：

- baseline_params = 28.5M
- baseline_gflops = 7.1

对任意方法 i：

- 等效剪枝率（按参数量）：
  - `eq_prune_rate_params = 1 - params_i / baseline_params`
- 等效剪枝率（按计算量）：
  - `eq_prune_rate_gflops = 1 - gflops_i / baseline_gflops`

解释：
- 这个“等效剪枝率”不是指该方法一定做了 pruning，而是把“更小的 Params/GFLOPs”统一映射成一个 0~1 的压缩比例，方便把不同论文的效率‑精度点放到同一张图里。
- 如果某方法比基线更大，会出现负数（例如 ResNet‑101/152），表示“比 HRNet‑W32 更重”，并非剪枝回退。

---

## 2. 数据 CSV（已发表来源）

数据文件：
- [published_equivalent_prune_curve_topdown_256x192.csv](file:///root/rivermind-data/PoseBH/experiments/docs/published_equivalent_prune_curve_topdown_256x192.csv)

来源说明（均为对应论文的官方/权威仓库的结果表）：
- HRNet‑W32 / ResNet baselines：HRNet 官方仓库结果表（CVPR 2019 / ECCV 2018）  
  https://github.com/HRNet/HRNet-Human-Pose-Estimation
- Lite‑HRNet：Lite‑HRNet 官方仓库结果表（CVPR 2021）  
  https://github.com/HRNet/Lite-HRNet

约束（保证可比性）：
- 仅收录 **Top‑down heatmap、256×192、COCO val2017** 且表格里同时给出 Params 与 FLOPs/GFLOPs 的条目。
- 检测器设置在上述官方表格中均标注为 person AP=56.4（同一口径）。

---

## 2.1 更密集的“曲线式”CSV（含开源补充）

仅靠“已发表论文/官方仓库”能拿到的 **(Params, GFLOPs, AP)** 点通常较稀疏（很多论文不同时报告复杂度与 val2017 256×192 的全指标）。如果允许使用“公开开源结果（非论文）”补充密度，可以使用以下合并版：

- [equivalent_prune_curve_topdown_256x192_dense.csv](file:///root/rivermind-data/PoseBH/experiments/docs/equivalent_prune_curve_topdown_256x192_dense.csv)

该文件包含两类 series：
- `paper_official_repo`：已发表论文对应的官方/权威仓库表格结果（严格口径）
- `open_source_pruning_curve`：开源 repo 公布的“剪枝曲线锚点”（非论文，用于把曲线变得更密集）

注意：
- `open_source_pruning_curve` 的 AP 基线（例如 0.765）与 HRNet 论文表格的 AP（例如 0.744）可能不同，属于训练/实现差异；横轴仍可用 `eq_prune_rate_*` 做效率对齐。

---

## 3. 如何用这份表回答“剪枝率+精度”

你可以任选一种横轴（更推荐 GFLOPs）：
- 横轴：`eq_prune_rate_gflops`（等效按计算量的剪枝率）
- 纵轴：`AP`

或：
- 横轴：`eq_prune_rate_params`（等效按参数量的剪枝率）
- 纵轴：`AP`

这样就得到一条“已发表口径”的 **(剪枝率, AP)** 曲线/散点图，可直接用于对标与写文档。
