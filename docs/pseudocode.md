Algorithm 1: P2AD Training Pipeline

Require: Pretrained HRNet-W48 model M₀
Require: PoseBH teacher model T
Require: Training set D = {(xᵢ, yᵢ)} with ground-truth heatmaps
Require: Target pruning ratio R_target
Require: Number of pruning steps N, epochs per step E
Ensure: Compact pruned and distilled model M*

// ── Stage 1: Pose-Aware Channel Importance Pruning ──
1:  M ← M₀
2:  for each convolutional channel c in M do
3:      // Forward pass with channel c zeroed
4:      H_c ← M(x | channel c ← 0) on a calibration subset
5:      // Compute error gain from ground-truth heatmaps
6:      ε_c ← ∥ H_c − y ∥²
7:      ε_0 ← ∥ H_0 − y ∥²   // baseline error without pruning
8:      s(c) ← ε_c − ε_0     // importance score: error gain
9:  end for
10: // Apply branch-differentiated budgets
11: for each branch b ∈ {high-res, low-res} do
12:     S_b ← {s(c) for channels c in branch b}
13:     if b is high-resolution branch then
14:         keep_ratio_b ← 1 − ρ_high   // conservative budget
15:     else
16:         keep_ratio_b ← 1 − ρ_low    // aggressive budget
17:     end if
18:     k_b ← ⌈ |S_b| × keep_ratio_b ⌉
19:     Retain top k_b channels in branch b according to S_b
20: end for
21: // Optional: golden-section search for optimal pruning rate
22: Initialize interval [α, β] ← [0.3, 0.8]
23: while β − α > ε do
24:     r₁ ← α + 0.382(β − α), r₂ ← α + 0.618(β − α)
25:     Evaluate objective F(r) = Acc(r)/Acc_max + AP(r)/AP_max + r
26:     if F(r₁) > F(r₂) then β ← r₂ else α ← r₁ end if
27: end while
28: r* ← (α + β)/2
29: M_pruned ← Remove channels with lowest scores to reach r*
30: if r* does not reach R_target then increase ρ_high, ρ_low accordingly

// ── Stage 2: Pruning-Aware Dynamic Distillation ──
31: Initialize λ_schedule ← [λ_high, λ_mid, λ_low]
32: Initialize α_schedule ← [1.0, 0.5, 0.0]   // 1: global, 0: local
33: for step = 1 to N do
34:     Adjust pruning ratio progressively toward R_target
35:     if step ≤ N/3 then
36:         stage ← structural stabilization
37:     else if step ≤ 2N/3 then
38:         stage ← accuracy recovery
39:     else
40:         stage ← fine-tuning
41:     end if
42:     λ ← λ_schedule[stage], α ← α_schedule[stage]
43:     for epoch = 1 to E do
44:         for each mini-batch (x, y) ∈ D do
45:             // Forward passes
46:             H_s, F_s ← M_pruned(x)   // student heatmaps, features
47:             H_t, F_t ← T(x)          // teacher heatmaps, features
48:             // Task loss
49:             L_task ← MSE(H_s, y)
50:             // Distillation losses
51:             L_global ← ∥ F_s − F_t ∥²   // align high-level features
52:             L_local  ← KL(H_s || H_t)    // align output distributions
53:             // Combined loss with dynamic weighting
54:             L ← L_task + λ [α · L_global + (1 − α) · L_local]
55:             Update M_pruned via AdamW optimizer
56:         end for
57:     end for
58: end for
59: return M* ← M_pruned