# Stage3：半传统方案——边剪枝边蒸馏（APRCP‑HRNet 风格）

本文档描述 PoseBH 项目中“边剪枝边蒸馏”的一套可落地流程，并将其归类为**半传统方案**：剪枝侧采用传统的 HRNet 结构化通道剪枝（APRCP‑HRNet 风格的 pruning area + 重要性通道选择 + 一致性裁剪），蒸馏侧使用 ViT‑MoE 教师对 HRNet 学生进行弱蒸馏以在剪枝后恢复精度。

参考方法：APRCP‑HRNet（Adaptive Pruning Rate Channel Pruning for HRNet）[https://github.com/vvhj/APRCP-HRNet](https://github.com/vvhj/APRCP-HRNet)

数据来源：本仓库实验统计表 [status.csv](file:///root/rivermind-data/PoseBH/experiments/DIST/status.csv)

---

## 1. 方案定位（为什么叫“半传统”）

- 传统剪枝侧：以 HRNet 的通道剪枝为主（结构变化是显式的 `num_channels`），重要性来自 BN γ / conv L1 等常见指标，裁剪需要显式维护多分支的一致性（transition / fuse_layers）。
- 现代蒸馏侧：以强教师（PoseBH‑B / ViT‑MoE‑Proto）提供的软监督为辅，用于缓解结构突变带来的 AP 掉点与训练不稳定。
- 训练流程：不是“一次性剪枝→长时间微调”，而是“周期性剪枝→验证→必要时蒸馏恢复→下一刀”。

在 APRCP‑HRNet 中，常见流程是：先根据通道重要性与 pruning area 选择结构（或生成剪枝结构文件），再 retrain 恢复精度。本仓库将其改造成 mmpose/mmcv runner 体系内的 hook 化流程，形成可持续迭代的“边剪枝边蒸馏”闭环。

---

## 2. 核心思想（对齐 APRCP‑HRNet 的关键点）

APRCP‑HRNet 的两个核心要点（与本仓库实现相对应）：

1) **Pruning area 固定**：只在 HRNet 的特定区域剪枝，而非全网剪，以减少多分支结构对齐风险与精度敏感性。APRCP‑HRNet README 给出了 pruning area 示意图（Prunarea.PNG）。本仓库对应参数为：

- `prune_branches_stage3=(2,)`
- `prune_branches_stage4=(2, 3)`

即仅剪高层低分辨率分支（语义更强、冗余更高），保留高分辨率分支稳定性。

2) **自适应剪枝率（以 r 为输入）**：APRCP‑HRNet 用 `r` 控制目标剪枝率并在候选结构中选择实际可达到的参数量/计算量剪枝幅度。本仓库把“主目标”明确为 **param_prune_rate**，并在每次剪枝点做候选结构搜索，选择最接近目标的结构。

---

## 3. 本仓库实现结构（代码与配置对应）

### 3.1 蒸馏模型封装（Teacher/Student）

蒸馏与训练主体模型：
- [TopDownDistillPrune](file:///root/rivermind-data/PoseBH/experiments/DIST/distill_prune.py)：同时包含
  - 教师网络冻结（ViT‑MoE‑Proto）
  - 学生网络前向与监督损失
  - KD‑Heatmap（空间 KL）与 KD‑Proto（特征/原型对齐 MSE）
  - 剪枝时的 student 重建与权重 remap（重要性通道映射）

蒸馏‑only（不剪枝、不做 recovery 上限约束）：
- [TopDownDistillOnly](file:///root/rivermind-data/PoseBH/experiments/DIST/distill_only.py)
- 配套配置：[hrnet_w32_distill_only_from_pruned_coco_256x192.py](file:///root/rivermind-data/PoseBH/experiments/DIST/hrnet_w32_distill_only_from_pruned_coco_256x192.py)
- 启动脚本：[run_distill_only_hrnet_w32_from_pruned.sh](file:///root/rivermind-data/PoseBH/experiments/DIST/run_distill_only_hrnet_w32_from_pruned.sh)

### 3.2 剪枝调度与 recovery（Hook）

剪枝‑恢复闭环：
- [HRNetPruneRecoverHook](file:///root/rivermind-data/PoseBH/experiments/DIST/custom_hooks.py)

关键能力：
- 剪枝点执行：pre‑eval → prune → post‑eval
- 掉点进入 recovery：恢复到阈值后继续下一刀
- 以 `param_prune_rate` 为主目标进行结构搜索
- 仅在 pruning area 内做结构变化
- **单调剪枝约束**：结构只允许越来越“瘦”，不允许回退（避免 stepN 比 stepN‑1 更宽）

早停：
- [EarlyStopByMetricHook](file:///root/rivermind-data/PoseBH/experiments/DIST/custom_hooks.py)

---

## 4. 算法流程（边剪枝边蒸馏闭环）

以每 10 个 epoch 为一个剪枝周期：

1) **正常训练（蒸馏+监督）**
- 监督损失为主（heatmap MSE），KD 为弱辅助（KD‑hm、KD‑proto）。
- 训练前若干 iter 强制监督占比（sup_ratio）以稳定收敛（可选项）。

2) **剪枝点：剪前验证（Pre‑prune Eval）**
- 在 epoch = k 且满足 interval 时执行一次 COCO val2017 评估，得到 `AP_before`。

3) **结构化剪枝（Prune）**
- 目标：按计划的 `target_ratio`（或由 r 映射）寻找一组 `new_extra`（HRNet 各 stage 各 branch 的 `num_channels`）。
- 重要性：BN γ / conv L1 估计通道重要性，取 top‑k 通道并生成通道索引映射。
- 一致性裁剪：对 transition/fuse_layers/branches 做一致性 channel remap，保证 forward 不崩溃。
- 单调约束：若 `new_extra` 在可剪分支上比当前更宽，则 clamp 回当前值，保证剪枝率不回退。

4) **剪后验证（Post‑prune Eval）**
- 得到 `AP_after`，计算 `drop = AP_before - AP_after`。

5) **进入 recovery（可选）**
- 如果 `drop` 超过阈值，则进入 recovery：下一次剪枝点到来时若 AP 仍未恢复到目标阈值，则跳过剪枝，继续蒸馏训练；直到达到阈值后再继续下一刀。

6) **Checkpoint 策略**
- 每个 epoch 常规 ckpt：`epoch_XXX.pth`
- 剪枝后立即额外保存：`prune_step_K_epoch_XXX.pth`
- 续训推荐从 `prune_step_*` ckpt resume，避免回到剪枝前状态。

---

## 5. 关键工程细节（易踩坑点）

### 5.1 “剪枝率”口径

在本仓库日志中常见两个口径：

- `prune_rate`：按 HRNet stage3+stage4 的通道数“和”估算的比例（受 pruning area 强烈影响）。
- `param_prune_rate`：按学生模型参数量计算的真实比例（更符合压缩目标）。

在固定 pruning area 的前提下，`prune_rate` 可能增长较慢甚至存在离散跳变；判断是否达到目标剪枝幅度，建议以 `param_prune_rate` 为主。

### 5.2 为什么必须“只从 prune_step_* resume”

训练日志中常见顺序是：先保存 `epoch_XXX.pth`，然后才触发剪枝并保存 `prune_step_*_epoch_XXX.pth`。因此从 `epoch_XXX.pth` resume 很可能回到剪枝前结构，导致剪枝进度倒退/重复。

建议：
- 断点续训：优先使用 `prune_step_*_epoch_*.pth`
- 只有纯训练不中途剪枝时，才优先 `epoch_*.pth`

---

## 6. 半传统方案的目标曲线（status.csv）

以下表格直接来自 [status.csv](file:///root/rivermind-data/PoseBH/experiments/DIST/status.csv)，用于描述“输入 r → 实际剪枝率/参数量/GFLOPs/AP”的目标锚点。

| 输入 r | 实际剪枝率 PR(%) | 参数量(M) | GFLOPs | AP(%) | AP@0.5(%) | AP@0.75(%) | AR(%) | 精度状态 | 备注 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.00 | 0.00 | 28.5 | 7.1 | 76.5 | 93.5 | 83.7 | 84.1 | 原生未剪枝基线 | HRNet‑W32 官方开源原生基准 |
| 0.12 | 11.20 | 25.3 | 6.3 | 76.5 | 93.5 | 83.7 | 84.1 | 完全无损 | 基于仓库官方算法规则推导 |
| 0.22 | 20.70 | 22.6 | 5.7 | 76.5 | 93.5 | 83.7 | 84.1 | 完全无损 | 基于仓库官方算法规则推导 |
| 0.36 | 34.00 | 18.8 | 4.8 | 76.5 | 93.6 | 83.7 | 84.1 | 完全无损 | 仓库官方公开锚点 |
| 0.46 | 43.50 | 16.1 | 4.2 | 76.5 | 93.6 | 83.7 | 84.1 | 完全无损 | 基于仓库官方算法规则推导 |
| 0.53 | 50.15 | 14.2 | 3.7 | 76.5 | 93.6 | 83.7 | 84.1 | 完全无损 | 基于仓库官方算法规则推导 |
| 0.61 | 58.20 | 11.9 | 3.2 | 76.5 | 93.6 | 83.7 | 84.1 | 官方无损上限 | 仓库官方公开最优结果（Golden 模式） |
| 0.65 | 61.55 | 10.9 | 3.0 | 76.4 | 93.5 | 83.6 | 84.0 | 几乎无损（掉点 0.1%） | 基于仓库官方算法规则推导 |
| 0.78 | 73.90 | 7.4 | 2.1 | 75.7 | 93.3 | 83.1 | 83.5 | 微损可接受（掉点 0.8%） | 仓库官方公开极限结果（Manual 模式） |
| 0.85 | 80.55 | 5.5 | 1.7 | 74.2 | 92.9 | 82.0 | 82.2 | 显著掉点（掉点 2.3%） | 基于仓库官方算法规则推导 |

这份表格可作为“半传统方案”的目标档位：当训练‑剪枝‑蒸馏闭环稳定后，优先向 `r≈0.61`（58.2% PR 无损）对齐，再逐步探索 `r≈0.78`（极限档）。

---

## 7. 推荐跑法（可复现与可续训）

### 7.1 边剪枝边蒸馏（完整闭环）

配置入口：
- [hrnet_w32_distill_prune_coco_256x192.py](file:///root/rivermind-data/PoseBH/experiments/DIST/hrnet_w32_distill_prune_coco_256x192.py)

启动脚本：
- [run_distill_prune_hrnet_w32_coco.sh](file:///root/rivermind-data/PoseBH/experiments/DIST/run_distill_prune_hrnet_w32_coco.sh)

续训建议：
- 从 `prune_step_*_epoch_*.pth` resume（剪枝后结构一致）

### 7.2 纯蒸馏（对某个剪枝结构做“只蒸馏不再剪枝”的恢复）

用于固定某个剪枝结构后，纯粹靠蒸馏把 AP 往上拉（不引入新的结构变化）：

```bash
bash /root/rivermind-data/PoseBH/experiments/DIST/run_distill_only_hrnet_w32_from_pruned.sh \
  /root/rivermind-data/PoseBH/experiments/DIST/work_dirs/hrnet_w32_distill_prune_coco_256x192/prune_step_3_epoch_100.pth
```

这条路径更接近 APRCP‑HRNet 的“先确定结构→retrain”的传统套路，只是把 retrain 换成了“蒸馏式 retrain”。

---

## 8. 诊断与建议（面向实际训练）

- 若出现“剪枝 step 变了但剪枝率回退/来回跳”：说明候选搜索输出了比当前更宽的结构。应启用/确认单调约束生效，并优先从 `prune_step_*` resume。
- 若剪枝后 AP 掉点很大（例如从 0.74 掉到 0.5）：通常是结构跳变过猛或 remap 出现语义破坏。建议降低单次幅度、增大 recovery 训练窗口，或先用 distill‑only 固定结构恢复。
- 若 `prune_rate` 增长缓慢：在固定 pruning area 下属正常现象。以 `param_prune_rate` 为主指标判断压缩进度。

