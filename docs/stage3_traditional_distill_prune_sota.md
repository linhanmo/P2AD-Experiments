# Stage3：传统方案——“先定结构再蒸馏恢复”（ViT→HRNet‑W32，COCO val2017 256×192）

本文档给出一套**真正的传统**蒸馏+剪枝流程：先离线确定剪枝结构（或剪枝后权重），再对固定结构进行蒸馏式 finetune 恢复精度；整个过程不引入“在线恢复闭环（recovery mode）”的上限/下限控制，也不在训练过程中持续改变网络结构。

该文档用于对照本仓库的“半传统方案”（在线周期性剪枝+恢复闭环），并补充一个可对标顶会公开数据的**传统基线参照表**（见 [traditional_sota_topdown_256x192.csv](file:///root/rivermind-data/PoseBH/experiments/docs/traditional_sota_topdown_256x192.csv)）。

---

## 0. 重要说明：你要的“顶会 SOTA：ViT→HRNet‑W32 + 剪枝 + 蒸馏”公开数据并不常见

公开论文/官方仓库里，**很少有人把三件事同时写清并公布可复现数值**：

- teacher 是 ViT 系（例如 ViTPose）
- student 是 HRNet‑W32（而不是 LitePose/Lite‑HRNet/自研轻量骨干）
- 同时做**结构化剪枝**与**知识蒸馏**，并且固定在 COCO val2017、256×192、top‑down heatmap 的统一检测器设置下报告 AP

因此本文做法是：

1) 用顶会/权威公开仓库给出“同一设置下可对标”的**teacher/学生基线数据**（HRNet / Lite‑HRNet / ViTPose）；  
2) 给出“传统剪枝+蒸馏”的**标准可落地流程**，用于你在本仓库内生成“ViT→剪枝后 HRNet‑W32”的传统实验曲线。

---

## 1. 传统方案定义（与本仓库“半传统”区别）

### 传统方案（本文）
- 剪枝发生在**离线阶段**（或在独立脚本里完成），得到一个固定结构（`HRNet.extra.stage{3,4}.num_channels`）以及对应权重。
- 后续训练阶段只做“固定结构的 finetune / distill”，结构不再变化。
- 早停可以保留（传统方案里常见），但不会设定类似“必须恢复到某个上限才能继续剪下一刀”的在线逻辑。

### 半传统方案（本仓库 Stage3 现状）
- 剪枝由 hook 在训练中周期性触发：pre‑eval → prune → post‑eval → 必要时进入 recovery → 下一刀。
- 优点：剪枝过程更自动化；缺点：结构变化与训练耦合，DDP/日志/回退问题更容易踩坑。

---

## 2. 顶会/权威公开的参照数据（256×192，COCO val2017 / val）

本节只列出**公开来源明确**、且与 top‑down 256×192 设置相近的数据点，用于你做“传统蒸馏+剪枝”时的对标。

### 2.1 学生基线：HRNet‑W32（CVPR 2019）
HRNet 官方仓库给出的 COCO val2017（256×192）结果：
- AP=0.744、#Params=28.5M、GFLOPs=7.1（detector person AP=56.4）  
来源：HRNet‑Human‑Pose‑Estimation README 的 COCO val2017 表格。  
https://github.com/HRNet/HRNet-Human-Pose-Estimation

### 2.2 轻量学生参照：Lite‑HRNet（CVPR 2021）
Lite‑HRNet 官方仓库给出的 COCO val2017（256×192）结果（同样基于 detector person AP=56.4）：
- Lite‑HRNet‑30：AP=0.672、#Params=1.8M、FLOPs=319.2M  
来源：Lite‑HRNet README 的 COCO val2017 表格。  
https://github.com/HRNet/Lite-HRNet

### 2.3 ViT 教师参照：ViTPose（NeurIPS 2022）
ViTPose 官方仓库给出的 COCO val（256×192）结果（classic decoder；使用 person detector 56 mAP）：
- ViTPose‑B：AP=75.8、AR=81.1
- ViTPose‑L：AP=78.3、AR=83.5
- ViTPose‑H：AP=79.1、AR=84.1  
来源：ViTPose README 的 “Results on MS COCO val set (single‑task training)” 表格。  
https://github.com/ViTAE-Transformer/ViTPose

将这些数据整理成 CSV：  
[traditional_sota_topdown_256x192.csv](file:///root/rivermind-data/PoseBH/experiments/docs/traditional_sota_topdown_256x192.csv)

---

## 3. 传统剪枝+蒸馏的标准流程（ViT→HRNet‑W32）

该流程按“传统论文写法”拆成 3 段：基线 → 剪枝定结构 → 蒸馏恢复。

### 3.1 基线阶段（不剪枝）
目的：拿到学生在本仓库设置下的 baseline（AP/AR/Params/GFLOPs），并固化检测器、数据增强、输入分辨率等。

在本仓库里，基线与训练框架/数据管线已固定（mmpose/mmcv runner）。

### 3.2 剪枝阶段（离线定结构）
目的：得到一个“剪枝后固定结构”的学生。

传统做法一般是迭代式结构化通道剪枝：
- 重要性指标：BN γ 或 conv L1（APRCP‑HRNet 风格常用）
- pruning area：固定剪枝区域（例如只剪 stage3 branch2、stage4 branch2/3），减少多分支对齐风险
- 一致性裁剪：transition / fuse_layers / branches 必须同步 remap

关键点是：这个阶段完成后，学生结构固定下来，后续不再变。

### 3.3 蒸馏恢复阶段（固定结构 finetune）
目的：在固定结构上做知识蒸馏，让剪枝后的学生恢复精度。

典型损失（与本仓库实现一致）：
- 监督：GT heatmap（JointsMSELoss）
- KD‑Heatmap：teacher heatmap 的空间 KL（弱蒸馏）
- KD‑Proto/Feature：teacher 的中间表示对齐（可选）

训练策略建议（传统思路）：
- 固定结构后，优先保证监督主导（避免 teacher 过强导致 student “学偏”）
- 以验证集 AP 作为 early stop 的 monitor（可选）

---

## 4. 本仓库里怎么跑“传统方案”（最小可复现）

### 4.1 固定某个剪枝后 ckpt，只做蒸馏恢复（推荐）
本仓库已提供 distill‑only 入口：从指定剪枝后 ckpt 中读取 `student_backbone_extra` 并加载 `student.*` 权重，然后只训练蒸馏+监督。

- 模型代码：[distill_only.py](file:///root/rivermind-data/PoseBH/experiments/DIST/distill_only.py)
- 配置：[hrnet_w32_distill_only_from_pruned_coco_256x192.py](file:///root/rivermind-data/PoseBH/experiments/DIST/hrnet_w32_distill_only_from_pruned_coco_256x192.py)
- 启动脚本：[run_distill_only_hrnet_w32_from_pruned.sh](file:///root/rivermind-data/PoseBH/experiments/DIST/run_distill_only_hrnet_w32_from_pruned.sh)

示例（把剪枝后权重当成学生初始化）：

```bash
bash /root/rivermind-data/PoseBH/experiments/DIST/run_distill_only_hrnet_w32_from_pruned.sh \
  /root/rivermind-data/PoseBH/experiments/DIST/work_dirs/hrnet_w32_distill_prune_coco_256x192/prune_step_3_epoch_100.pth
```

这条路径就是“传统方案”的蒸馏恢复段。

### 4.2 传统方案的数据记录方式（建议）
建议在每个固定结构上记录：
- student 结构（stage3/stage4 num_channels）
- Params / GFLOPs（以 param_prune_rate 为主）
- COCO val2017：AP / AP50 / AP75 / AR
- 训练设置（teacher 版本、loss 权重、训练轮数、是否使用 EMA/AMP 等）

并以 CSV 形式维护一个“传统方案实验表”，与 [status.csv](file:///root/rivermind-data/PoseBH/experiments/DIST/status.csv) 并列，避免把在线闭环数据与离线传统数据混在一起。

---

## 5. 传统方案与半传统方案的对照建议

- 当你需要“论文式可复现对比”时：用传统方案（固定结构→蒸馏恢复）。
- 当你需要“自动逼近某个目标剪枝率并稳定跑完”时：用半传统方案（在线闭环），但需要额外工程约束（只从 prune_step ckpt resume、单调剪枝、日志补全）。

