# HRNet-W48 剪枝曲线（论文/论文公开页面整理）

本目录新增两份“可直接用于画曲线/对比”的结构化表格：

- [published_hrnet_w48_pruning_from_papers.csv](file:///root/rivermind-data/PoseBH/experiments/docs/published_hrnet_w48_pruning_from_papers.csv)：仅包含 **HRNet-W48** 的官方基线与剪枝点（APRCP、HACR 等）
- [published_hrnet_w48_pruning_non256_from_papers.csv](file:///root/rivermind-data/PoseBH/experiments/docs/published_hrnet_w48_pruning_non256_from_papers.csv)：**非 256×192** 的 W48 剪枝点（APRCP/HACR 当前公开表格主要在 384×288 报告）
- [published_hrnet_family_lightweight_reference_from_papers.csv](file:///root/rivermind-data/PoseBH/experiments/docs/published_hrnet_family_lightweight_reference_from_papers.csv)：**HRNet-family 轻量模型参考**（Lite-HRNet / Slim-HRNet），注意不一定是 W48

## 字段解释（W48 主表）

`published_hrnet_w48_pruning_from_papers.csv` 的字段与现有 `published_equivalent_prune_curve_topdown_256x192.csv` 保持一致：

- `AP/AP50/AP75/AR`：COCO 指标（通常为 val2017 / test-dev，具体以 source 表述为准）
- `params_M/gflops`：论文表格中给出的模型规模
- `eq_prune_rate_params/eq_prune_rate_gflops`：等价剪枝率（对应该行 `baseline_params_M/baseline_gflops`）
  - 对 APRCP 行：直接取其表格中括号内的 PR（pruned ratio）
  - 对 HRNet 官方基线：为 0

## 重要注意事项（对齐风险）

- **本项目当前口径若限定为 256×192**：公开可直接复核的 W48 “剪枝曲线点”非常少（很多方法只在 384×288 报告），因此主表会只保留 256×192 的条目；其余条目移动到 `published_hrnet_w48_pruning_non256_from_papers.csv`。
- **同名 “HRNet-W48 baseline” 在不同论文里数值可能不同**：例如 HRNet 官方 repo（CVPR2019）与 APRCP 表格给出的 W48 baseline AP 不一致，这是因为训练 recipe / 输入分辨率 / detector setting 可能不同。对比时应优先使用“同一论文表格内”的 baseline 做相对比较。
- **HACR（PRL 2023）**：公开 DOI 页面明确给出 “HRNet-W48 Params↓58.2%、FLOPs↓50.1%、无精度损失”，但该页面未直接给出 AP 数字；因此主表中该行 AP 留空，仅保留压缩率与推导出的 Params/GFLOPs。
- **Lite-HRNet / Slim-HRNet**：两者多为“HRNet-family 轻量化设计/结构剪枝”而非“对 HRNet-W48 做通道剪枝曲线”，因此单独放在 reference 表中。

## 主要来源（可复核链接）

- HRNet 官方结果表（含 pose_hrnet_w48 256x192 / 384x288）：https://github.com/HRNet/HRNet-Human-Pose-Estimation/blob/master/README.md
- APRCP-HRNet 表格（包含 HRNet-W48 多个 r 的 Params/GFLOPs/AP）：https://github.com/vvhj/APRCP-HRNet/blob/master/README.md
- HACR（PRL 2023）DOI 页面（包含 58.2% Params / 50.1% FLOPs / no loss）：https://doi.org/10.1016/j.patrec.2023.03.007
- Slim-HRNet（JRTIP 2026）摘要页（包含 COCO AP=72.6%、GFLOPs=3.35）：https://link.springer.com/article/10.1007/s11554-026-01897-x
- Lite-HRNet（CVPR 2021 openaccess）：https://openaccess.thecvf.com/content/CVPR2021/html/Yu_Lite-HRNet_A_Lightweight_High-Resolution_Network_CVPR_2021_paper.html
