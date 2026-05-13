DIST_INNOV：ViT→HRNet 渐进剪枝蒸馏（动态权重 + 姿态感知通道保护 + 跨架构自适应对齐）

本目录面向“ViT 教师 + HRNet 学生”的人体姿态估计蒸馏与结构化剪枝联合训练，提供三项可独立开关的创新点实现，以及完整消融实验配置。

---

## 摘要

我们提出一套面向姿态估计“跨架构蒸馏 + 渐进式结构化剪枝”的统一训练框架，以 ViT 教师为知识源、HRNet 学生为高分辨率定位骨干。在训练过程中，学生网络按固定周期进行通道级结构化剪枝，累计剪枝率逐步提升并伴随恢复阶段。针对固定蒸馏权重、任务无关的通道重要性与跨架构特征分布错位等问题，我们引入三项互补改进：（1）剪枝率与 epoch 双维度绑定的动态蒸馏权重调度；（2）教师姿态原型引导的关节-通道贡献标定与强通道保护（含长尾关节配额提升）；（3）分布归一化与条件缩放偏移的跨架构自适应特征对齐。该实现提供完整消融配置与可复现的训练脚本。

## 符号表

- `T`：教师模型（ViT + Proto Head）
- `S`：学生模型（HRNet）
- `e∈[0,1]`：epoch 进度（由 `epoch/max_epochs` 归一化）
- `r∈[0,1]`：累计剪枝率（优先使用真实参数压缩率 `param_prune_rate`）
- `H_T, H_S`：教师/学生 heatmap
- `P_T`：教师姿态原型特征（proto head 输出）
- `F_S`：学生用于原型蒸馏的特征图（HRNet stage4 分支特征）
- `A`：教师引导的关节注意力（K×H×W）
- `K`：关节数（默认 17）

**快速开始（启动脚本）**

- 默认训练（三项全开）：

```bash
chmod +x experiments/DIST_INNOV/run_distill_innov.sh
bash experiments/DIST_INNOV/run_distill_innov.sh
```

- 指定某个消融配置（示例：强通道保护 long\_tail\_boost）：

```bash
bash experiments/DIST_INNOV/run_distill_innov.sh experiments/DIST_INNOV/ablation_7_pose_prune_strong_longtail.py
```

- 传入额外训练参数（会原样透传给 `tools/train.py`，示例仅供参考）：

```bash
bash experiments/DIST_INNOV/run_distill_innov.sh experiments/DIST_INNOV/full_innov_3methods.py --resume-from /path/to/latest.pth
```

**目录入口**

- 模型封装（蒸馏 + 剪枝 + 对齐 + 通道保护）：[distill_prune_innov.py](file:///root/rivermind-data/PoseBH/experiments/DIST_INNOV/distill_prune_innov.py)
- 训练状态注入（epoch/max_epochs/param_prune_rate）：[custom_hooks_innov.py](file:///root/rivermind-data/PoseBH/experiments/DIST_INNOV/custom_hooks_innov.py)
- 剪枝调度与恢复（step_schedule/recover/force_prune）：[custom_hooks.py](file:///root/rivermind-data/PoseBH/experiments/DIST/custom_hooks.py)
- 默认全开配置（教师/学生权重路径写死在 config 中）：[full_innov_3methods.py](file:///root/rivermind-data/PoseBH/experiments/DIST_INNOV/full_innov_3methods.py)

## 方法

### 问题定义

给定教师模型 `T`（ViT + Proto Head）与学生模型 `S`（HRNet），在渐进式结构化剪枝过程中（累计剪枝率 `r` 单调上升）进行蒸馏训练，目标是在满足压缩目标的同时最大化姿态估计精度（尤其是高剪枝率阶段的恢复能力与长尾关节鲁棒性）。

训练过程中每隔固定 epoch 触发一次结构化通道剪枝（在 stage3/stage4 指定分支上缩减 `num_channels` 并重映射权重），并可能进入恢复阶段（当 AP 掉点超过阈值时暂停剪枝）。

### 总体目标函数

学生优化目标写为：

```text
L = λ_sup * L_sup
  + λ_hm(e, r) * L_KD_hm
  + λ_proto(e, r) * L_KD_proto
```

- `e` 为 epoch 进度（归一化到 `[0,1]`），`r` 为累计剪枝率（默认使用真实参数压缩率 `param_prune_rate`，缺失时回退到 `prune_rate/target_ratio/...`）。
- 实现层面，`e` 与 `r` 由 hook 写入 `model.set_prune_state(...)`，从而保证调度可复现且不依赖日志解析。

### 算法 1：渐进剪枝蒸馏训练流程（概述）

**输入**：教师 `T`、学生 `S`、训练数据 `D`、剪枝周期 `ΔE`、最大剪枝率 `r_max`  
**输出**：剪枝后的学生 `S'`

1. 初始化：冻结教师参数；初始化学生；初始化累计剪枝率 `r <- 0`；初始化 EMA 缓存（关节/全局通道分数、关节可见性）。
2. 对每个 epoch：
   - 训练迭代：对每个 batch，前向得到 `H_T, P_T, H_S, F_S`；计算 `L_sup, L_KD_hm, L_KD_proto`；更新学生参数。
   - 在线打分：用 `H_T, P_T` 构造注意力 `A`；对被剪枝分支计算关节-通道贡献 `S_joint` 与全局分数 `S_global`；对分数做 EMA 更新。
   - 若 epoch 满足剪枝触发条件（每 `ΔE` 一次，且未处于恢复状态）：
     1. 根据当前 `r` 与 step_schedule 得到下一目标剪枝率 `r' <- min(r + Δr, r_max)`；
     2. 将目标通道数写入 HRNet `extra.num_channels` 并重建学生；
     3. 使用“保护集合 + TopK 填充”生成通道索引映射并裁剪/重映射权重；
     4. 评估 AP，若掉点超过阈值则进入恢复模式，否则提交 `r <- r'`。

### 创新点 1：剪枝感知动态蒸馏权重调度（epoch + 剪枝率二维绑定）

**动机**：固定蒸馏权重在低剪枝率阶段容易拖慢收敛，高剪枝率阶段又难以恢复；此外，纯 epoch 绑定的动态权重无法反映“剪枝进度”这一关键状态。

**实现**：将 `λ_hm(e,r)` 与 `λ_proto(e,r)` 分解为“剪枝率分段插值 × epoch 缩放”：

1. 剪枝率分段插值：设阈值 `(b1,b2)` 把剪枝率分三段，端点权重为 `(w0,w1,w2)`，则

   ```text
   λ(r) = InterpPiecewise(r; b1, b2, w0, w1, w2)
   ```

2. epoch 缩放（HM 与 Proto 可分别设置）：

   ```text
   s(e) = s_min + (1 - s_min) * e^p
   ```

最终：

```text
λ_hm(e, r)    = λ_hm(r)    * s_hm(e)
λ_proto(e, r) = λ_proto(r) * s_proto(e)
```

代码对应：

- 权重生成：`_dynamic_kd_weights(...)`
- 剪枝率来源：`dynamic_kd.prune_ratio_source='param_prune_rate'`（推荐）
- 日志：`kd_hm_w/kd_proto_w/prune_ratio`

### 创新点 2：姿态感知通道重要性剪枝 + 强姿态通道保护（Pose-Aware Protected Pruning）

**动机**：HRNet 常见通道剪枝以 BN-gamma / Conv-L1 等统计量为主，与“17 个关节定位目标”脱钩；均匀裁剪会优先破坏高分辨率定位所需的关键通道，导致严重掉点，且长尾关节（可见性低）更易退化。

本实现构建“教师姿态原型引导的关节注意力”，再用该注意力对学生通道贡献进行**按关节标定**，并在剪枝时引入显式的**保护集合**以实现“差异化通道剪枝保护”。

**(a) 教师姿态注意力（Proto-modulated Heatmap Attention）**
教师 heatmap 先做空间 softmax（温度 `τ`）：

```text
A_hm = softmax_HW(H_T / τ)
```

再用教师 proto 能量图 `E_P = mean_C(|P_T|)` 调制并归一化：

```text
A = norm(A_hm * norm(E_P))
```

代码：`_hm_attn_from_teacher(...)`。

**(b) 按关节通道贡献打分（Joint-Channel Contribution）**
给定学生候选分支特征 `x in R^(N×C×H×W)`，计算按关节的通道贡献：

```text
S_joint(k, c) = Σ_{n,h,w} |x_{n,c,h,w}| * A_{n,k,h,w}
```

实现采用 `einsum('nchw,nkhw->kc')` 得到 `K×C` 的 `joint_scores`。

全局通道重要性通过关节权重 `w_k` 汇总：

```text
S_global(c) = Σ_k w_k * S_joint(k, c)
```

其中 `w_k` 由训练 batch 的 `target_weight` 自动统计并归一化，同时维护关节可见性 EMA 作为长尾识别依据。

**(c) 强姿态通道保护（Protected Set + Quota Allocation）**
剪枝触发时设目标保留通道数为 `K`，从 `joint_scores` 构建保护集合 `P`：

- `protect_mode='joint_union'`：为每个关节分配最少 `per_joint_min` 的保护配额，从每个关节 top 通道取并集形成 `P`。
- `protect_mode='long_tail_boost'`：基于关节可见性 EMA 选出长尾关节（低分位数），其配额乘 `tail_boost`，以减少长尾关节掉点。

最终保留集合为：

```text
K_keep = P ∪ TopK(S_global)
```

并在必要时用 `S_global` 对 `P` 做二次裁剪，使 `|K_keep| = K`。

**(e) 保护模式与差异化配额（实现细节）**

- `protect_ratio`：保护集合预算，占最终保留通道数的比例（例如 0.6 表示至少 60% 保留通道来自保护集合候选）。
- `per_joint_min`：每个关节的最低保护配额（避免长尾关节被完全忽略）。
- `long_tail_boost`：
  - 用关节可见性 EMA（由 `target_weight` 统计）选择低分位数关节集合 `Omega`；
  - 对 `k in Omega` 的配额乘 `tail_boost`，再全局归一化到 `protect_ratio·K`。
- 保护实现入口：`_build_protected_set(...)` 与 `_select_with_protection(...)`；剪枝映射入口：`_build_hrnet_channel_index_map_pose_protect(...)`。

**(d) 稳健融合与回退**
为提升早期训练稳定性与避免异常分数：

- 当 pose 分数缺失/维度不匹配时回退 BN-gamma；
- 使用轻量融合做稳健打分（`bn_mix`）：

  ```text
  S = norm(S_global) + α * norm(S_bn)
  ```

代码对应：

- joint/global 打分与 EMA：`TopDownDistillPruneInnov._maybe_update_pose_scores`
- 保护集合与选择：`_build_protected_set(...)`、`_select_with_protection(...)`
- 剪枝映射入口：`_build_hrnet_channel_index_map_pose_protect(...)`

### 创新点 3：跨架构自适应特征对齐（Distribution-aware Conditional Alignment）

**动机**：ViT 全局表征与 HRNet 局部/多分辨率表征存在分布偏差，静态 1×1 映射难以消除该偏差，导致跨架构蒸馏噪声。

**实现**：对学生特征先投影并做空间归一化，再由教师 proto 生成条件向量预测 per-channel 的缩放与偏移：

```text
Y      = Norm(Conv1x1(F_S))
[γ, β] = MLP(GAP(P_T))
Y_hat  = Y * (1 + γ) + β
```

其中 MLP 最后一层初始化为零，保证初始行为接近 identity，训练更稳定。实现类：`CrossArchAdaptiveAlign`。

## 实现与复现细节

### 关键代码路径（建议论文复现实验引用）

- 动态蒸馏权重：`_dynamic_kd_weights`（读取 `prune_state` 中的 epoch 与剪枝率）
- 注意力构造：`_hm_attn_from_teacher`（heatmap softmax + proto 能量调制）
- 热图辅助对齐诊断（方案一）：`metric_hm_align_cos_*` / `metric_hm_align_l1_*`（仅记录日志，不参与反传）
- 关节-通道打分：`TopDownDistillPruneInnov._maybe_update_pose_scores`（`einsum` 计算 `joint_scores`）
- 强保护映射：`_build_hrnet_channel_index_map_pose_protect`（保护集合 + TopK 填充）

### 稳定性与健壮性策略

- EMA 缓存全部存 CPU，剪枝触发时不额外 forward，避免显存峰值与同步开销。
- 所有分数张量均进行 NaN/Inf 清理；维度不匹配时回退到 BN-gamma。
- 对齐模块避免 forward 动态创建可学习参数；cond_map 通道不匹配时进行安全裁剪/补零对齐。

## 配置字段（核心）

- `model.type='TopDownDistillPruneInnov'`
- `model.student_init_ckpt`：学生初始化权重（支持整网或仅 backbone）。
  - 若 ckpt 仅包含 `conv1/layer1/...` 等键（不含 `keypoint_head`），则只加载到 `student.backbone`。
  - 若 ckpt 含 `keypoint_head`（整网），则加载到 `student`（自动兼容 `state_dict`/`module.` 前缀与 `student.` 前缀）。
- `distill_cfg.dynamic_kd`：`enable/prune_ratio_source/stage_boundaries/hm_stage_weights/proto_stage_weights/epoch_*`
- `distill_cfg.pose_prune`：`enable/protect_mode/protect_ratio/per_joint_min/tail_percentile/tail_boost/bn_mix/score_power`
- `prune_hook`（`HRNetPruneRecoverHook`）：
  - 若学生初始精度较高（如 AP≈76.5），建议 `start_epoch=1` + `immediate_prune_at_start=True`，实现“首轮立即剪枝，后续按 interval 周期剪枝”

## 消融配置

- Baseline：固定 KD + bn_gamma 剪枝：[ablation_0_baseline_static.py](file:///root/rivermind-data/PoseBH/experiments/DIST_INNOV/ablation_0_baseline_static.py)
- 动态权重：[ablation_1_dynamic_weight.py](file:///root/rivermind-data/PoseBH/experiments/DIST_INNOV/ablation_1_dynamic_weight.py)
- 姿态剪枝（基础）：[ablation_2_pose_prune.py](file:///root/rivermind-data/PoseBH/experiments/DIST_INNOV/ablation_2_pose_prune.py)
- 自适应对齐：[ablation_3_adaptive_align.py](file:///root/rivermind-data/PoseBH/experiments/DIST_INNOV/ablation_3_adaptive_align.py)
- 动态权重 + 姿态剪枝：[ablation_4_dynamic_plus_pose_prune.py](file:///root/rivermind-data/PoseBH/experiments/DIST_INNOV/ablation_4_dynamic_plus_pose_prune.py)
- 动态权重 + 自适应对齐：[ablation_5_dynamic_plus_adaptive_align.py](file:///root/rivermind-data/PoseBH/experiments/DIST_INNOV/ablation_5_dynamic_plus_adaptive_align.py)
- 姿态剪枝 + 自适应对齐：[ablation_6_pose_prune_plus_adaptive_align.py](file:///root/rivermind-data/PoseBH/experiments/DIST_INNOV/ablation_6_pose_prune_plus_adaptive_align.py)
- 姿态剪枝（强保护 long_tail_boost）：[ablation_7_pose_prune_strong_longtail.py](file:///root/rivermind-data/PoseBH/experiments/DIST_INNOV/ablation_7_pose_prune_strong_longtail.py)
- 三者全开（默认启用强保护）：[full_innov_3methods.py](file:///root/rivermind-data/PoseBH/experiments/DIST_INNOV/full_innov_3methods.py)

## 结构/权重映射自检脚本
当需要确认教师/学生网络结构与 checkpoint 键名映射关系时，运行：

```bash
python experiments/DIST_INNOV/inspect_teacher_student_mapping.py \
  --cfg experiments/DIST_INNOV/full_innov_3methods.py \
  --teacher-ckpt /root/rivermind-data/PoseBH/weights/posebh/base.pth \
  --student-ckpt /root/rivermind-data/PoseBH/weights/hrnet/pose_hrnet_w32_256x192.pth \
  --out experiments/DIST_INNOV/mapping_report.json
```

脚本会输出并保存：
- teacher/student `state_dict` 的键前缀统计
- ckpt 与模型 `state_dict` 的匹配数/缺失数/shape mismatch 样例
- student ckpt 的原始匹配与“映射后匹配”（针对 HRNet 无前缀权重的自动映射）
