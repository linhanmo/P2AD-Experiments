import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.models.builder import POSENETS, build_posenet

from experiments.DIST.distill_prune import (
    TopDownDistillPrune,
    _infer_student_feat,
    _infer_teacher_proto,
    _remap_hrnet_pruned_weights,
    _spatial_kl,
)


def _norm_spatial(x: torch.Tensor, eps: float = 1.0e-5) -> torch.Tensor:
    mean = x.mean(dim=(2, 3), keepdim=True)
    var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)


def _get_prune_ratio_from_state(state: Dict) -> float:
    for k in ['param_prune_rate', 'prune_rate', 'target_ratio', 'mid_ratio', 'high_ratio']:
        v = state.get(k, None)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            continue
    return 0.0


def _get_prune_ratio(state: Dict, source: str) -> float:
    src = str(source).strip()
    if src:
        v = state.get(src, None)
        try:
            if v is not None:
                return max(0.0, min(1.0, float(v)))
        except Exception:
            pass
    return max(0.0, min(1.0, float(_get_prune_ratio_from_state(state))))


def _piecewise_stage_weights(
    prune_ratio: float,
    stage_bounds: Tuple[float, float],
    hm_stage: Tuple[float, float, float],
    proto_stage: Tuple[float, float, float],
):
    pr = max(0.0, min(1.0, float(prune_ratio)))
    b1, b2 = float(stage_bounds[0]), float(stage_bounds[1])
    b1 = max(0.0, min(1.0, b1))
    b2 = max(b1 + 1.0e-6, min(1.0, b2))

    def _lerp(a, b, t):
        return float(a) + (float(b) - float(a)) * float(t)

    if pr <= b1:
        t = pr / max(b1, 1.0e-6)
        hm = _lerp(hm_stage[0], hm_stage[1], t)
        proto = _lerp(proto_stage[0], proto_stage[1], t)
        return hm, proto
    if pr <= b2:
        t = (pr - b1) / max(b2 - b1, 1.0e-6)
        hm = _lerp(hm_stage[1], hm_stage[2], t)
        proto = _lerp(proto_stage[1], proto_stage[2], t)
        return hm, proto
    return float(hm_stage[2]), float(proto_stage[2])


def _get_epoch_progress(state: Dict) -> float:
    try:
        epoch = float(state.get('epoch', 0.0))
        max_epochs = float(state.get('max_epochs', 1.0))
        if max_epochs <= 1:
            return 0.0
        return max(0.0, min(1.0, epoch / (max_epochs - 1.0)))
    except Exception:
        return 0.0


def _dynamic_kd_weights(distill_cfg: Dict, prune_state: Dict) -> Tuple[float, float]:
    dyn = distill_cfg.get('dynamic_kd', None)
    if not isinstance(dyn, dict) or not bool(dyn.get('enable', False)):
        hm_w = float(distill_cfg.get('kd_hm_weight', distill_cfg.get('hm_weight', 0.0)))
        proto_w = float(distill_cfg.get('kd_proto_weight', distill_cfg.get('proto_weight', 0.0)))
        return hm_w, proto_w

    prune_ratio_source = str(dyn.get('prune_ratio_source', '')).strip()
    prune_ratio = _get_prune_ratio(prune_state, prune_ratio_source)
    stage_bounds = tuple(dyn.get('stage_boundaries', (0.2, 0.4)))
    hm_stage = tuple(dyn.get('hm_stage_weights', (0.05, 0.10, 0.20)))
    proto_stage = tuple(dyn.get('proto_stage_weights', (0.20, 0.15, 0.05)))
    hm_pr, proto_pr = _piecewise_stage_weights(prune_ratio, stage_bounds, hm_stage, proto_stage)

    e = _get_epoch_progress(prune_state)
    min_scale = float(dyn.get('epoch_min_scale', 0.5))
    min_scale_hm = float(dyn.get('epoch_min_scale_hm', min_scale))
    min_scale_proto = float(dyn.get('epoch_min_scale_proto', min_scale))
    min_scale_hm = max(0.0, min(1.0, min_scale_hm))
    min_scale_proto = max(0.0, min(1.0, min_scale_proto))
    power = float(dyn.get('epoch_power', 1.0))
    power_hm = float(dyn.get('epoch_power_hm', power))
    power_proto = float(dyn.get('epoch_power_proto', power))
    e_clamped = max(0.0, min(1.0, e))
    e_scale_hm = min_scale_hm + (1.0 - min_scale_hm) * (e_clamped ** max(power_hm, 1.0e-6))
    e_scale_proto = min_scale_proto + (1.0 - min_scale_proto) * (e_clamped ** max(power_proto, 1.0e-6))

    hm = float(hm_pr) * float(e_scale_hm)
    proto = float(proto_pr) * float(e_scale_proto)

    hm_max = dyn.get('hm_max', None)
    proto_max = dyn.get('proto_max', None)
    if hm_max is not None:
        hm = min(float(hm_max), hm)
    if proto_max is not None:
        proto = min(float(proto_max), proto)
    hm = max(0.0, float(hm))
    proto = max(0.0, float(proto))
    return hm, proto


def _hm_attn_from_teacher(
    t_hm: torch.Tensor,
    proto_energy: Optional[torch.Tensor],
    temperature: float = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    n, k, h, w = t_hm.shape
    q = t_hm / max(float(temperature), 1.0e-6)
    q = q.reshape(n, k, h * w)
    attn = F.softmax(q, dim=-1).reshape(n, k, h, w)
    if proto_energy is None:
        return attn
    if proto_energy.shape[-2:] != (h, w):
        proto_energy = F.interpolate(proto_energy, size=(h, w), mode='bilinear', align_corners=False)
    pe = proto_energy.clamp_min(0.0)
    pe = pe / (pe.sum(dim=(2, 3), keepdim=True) + float(eps))
    out = attn * pe
    out = out / (out.sum(dim=(2, 3), keepdim=True) + float(eps))
    return out


def _energy_map(x: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    if not isinstance(x, torch.Tensor) or x.dim() != 4:
        return None
    y = x.detach().abs().mean(dim=1, keepdim=True)
    if not torch.isfinite(y).all():
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = y.clamp_min(0.0)
    s = y.sum(dim=(2, 3), keepdim=True)
    y = y / (s + float(eps))
    return y


def _mask_from_teacher_hm(
    t_hm: torch.Tensor,
    t_proto: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    use_proto_energy: bool = True,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    if not isinstance(t_hm, torch.Tensor) or t_hm.dim() != 4:
        return None
    attn = _hm_attn_from_teacher(t_hm.detach(), None, temperature=float(temperature), eps=float(eps))
    m = attn.mean(dim=1, keepdim=True)
    if use_proto_energy and isinstance(t_proto, torch.Tensor) and t_proto.dim() == 4:
        pe = _energy_map(t_proto.detach(), eps=float(eps))
        if pe is not None:
            if pe.shape[-2:] != m.shape[-2:]:
                pe = F.interpolate(pe, size=m.shape[-2:], mode='bilinear', align_corners=False)
            m = m * pe
    m = m.clamp_min(0.0)
    m = m / (m.sum(dim=(2, 3), keepdim=True) + float(eps))
    return m


def _masked_cosine(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    if a is None or b is None or m is None:
        return None
    if a.shape != b.shape:
        return None
    if m.shape[-2:] != a.shape[-2:]:
        m = F.interpolate(m, size=a.shape[-2:], mode='bilinear', align_corners=False)
        m = m / (m.sum(dim=(2, 3), keepdim=True) + float(eps))
    va = (a * m).reshape(a.size(0), -1)
    vb = (b * m).reshape(b.size(0), -1)
    na = torch.sqrt((va * va).sum(dim=1) + float(eps))
    nb = torch.sqrt((vb * vb).sum(dim=1) + float(eps))
    cos = (va * vb).sum(dim=1) / (na * nb + float(eps))
    return cos.mean()


def _masked_l1(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    if a is None or b is None or m is None:
        return None
    if a.shape != b.shape:
        return None
    if m.shape[-2:] != a.shape[-2:]:
        m = F.interpolate(m, size=a.shape[-2:], mode='bilinear', align_corners=False)
    m = m.clamp_min(0.0)
    m = m / (m.sum(dim=(2, 3), keepdim=True) + float(eps))
    return ((a - b).abs() * m).sum(dim=(1, 2, 3)).mean()


def _copy_tensor_slices(dst: torch.Tensor, src: torch.Tensor) -> Optional[torch.Tensor]:
    if not isinstance(dst, torch.Tensor) or not isinstance(src, torch.Tensor):
        return None
    if dst.dtype != src.dtype:
        src = src.to(dtype=dst.dtype)
    if dst.device != src.device:
        src = src.to(device=dst.device)
    if dst.dim() != src.dim():
        return None
    slices = tuple(slice(0, min(dst.size(i), src.size(i))) for i in range(dst.dim()))
    out = dst.clone()
    out[slices] = src[slices].clone()
    return out


def _update_ema(dst: Optional[torch.Tensor], src: torch.Tensor, momentum: float) -> torch.Tensor:
    m = max(0.0, min(1.0, float(momentum)))
    if dst is None:
        return src.detach().float().cpu()
    d = dst.detach().float().cpu()
    s = src.detach().float().cpu()
    if d.numel() != s.numel():
        return s
    return d * m + s * (1.0 - m)


def _topk_idx_from_scores(scores: torch.Tensor, keep: int) -> torch.Tensor:
    keep = max(1, min(int(keep), int(scores.numel())))
    top = torch.topk(scores, k=keep, largest=True, sorted=True).indices
    return top.sort().values


def _bn_gamma_scores_for_branch(student_state_dict: Dict[str, torch.Tensor], stage: str, branch: int, ch: int) -> torch.Tensor:
    prefix = f'backbone.{stage}.branches.{int(branch)}.'
    s = torch.zeros((int(ch),), dtype=torch.float32)
    for k, v in student_state_dict.items():
        if not (k.startswith(prefix) and '.bn' in k and k.endswith('.weight')):
            continue
        if not isinstance(v, torch.Tensor) or v.dim() != 1:
            continue
        if int(v.numel()) != int(ch):
            continue
        s += v.detach().abs().float().cpu()
    if not torch.isfinite(s).all():
        s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    return s


def _normalize_scores(x: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        return x
    v = x.float()
    m = v.mean()
    s = v.std(unbiased=False)
    return (v - m) / (s + float(eps))


def _safe_topk_from_vector(scores: torch.Tensor, k: int) -> torch.Tensor:
    if not isinstance(scores, torch.Tensor) or scores.dim() != 1:
        return None
    kk = max(1, min(int(k), int(scores.numel())))
    top = torch.topk(scores, k=kk, largest=True, sorted=True).indices
    return top.sort().values


def _build_protected_set(
    joint_scores: Optional[torch.Tensor],
    keep: int,
    base_scores: torch.Tensor,
    joint_vis: Optional[torch.Tensor],
    protect_cfg: Dict,
) -> torch.Tensor:
    if not isinstance(joint_scores, torch.Tensor) or joint_scores.dim() != 2:
        return torch.empty((0,), dtype=torch.long)
    k_j, c = int(joint_scores.size(0)), int(joint_scores.size(1))
    keep = max(1, min(int(keep), int(c)))

    mode = str(protect_cfg.get('protect_mode', 'none')).strip().lower()
    if mode in ['', 'none', 'off', 'false', '0']:
        return torch.empty((0,), dtype=torch.long)

    protect_ratio = float(protect_cfg.get('protect_ratio', 0.5))
    protect_ratio = max(0.0, min(1.0, protect_ratio))
    protect_keep = int(round(float(keep) * float(protect_ratio)))
    protect_keep = max(0, min(int(keep), int(protect_keep)))
    if protect_keep <= 0:
        return torch.empty((0,), dtype=torch.long)

    per_joint_min = int(protect_cfg.get('per_joint_min', 1))
    per_joint_min = max(0, per_joint_min)

    base_quota = int(max(0, round(float(protect_keep) / max(1.0, float(k_j)))))
    base_quota = max(per_joint_min, base_quota)
    quotas = torch.full((k_j,), int(base_quota), dtype=torch.long)

    if mode == 'long_tail_boost':
        if isinstance(joint_vis, torch.Tensor) and joint_vis.numel() == k_j:
            v = joint_vis.detach().float().cpu()
            if not torch.isfinite(v).all():
                v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            tail_p = float(protect_cfg.get('tail_percentile', 0.3))
            tail_p = max(0.0, min(1.0, tail_p))
            tail_k = int(max(1, round(float(k_j) * float(tail_p))))
            tail_k = max(1, min(k_j, tail_k))
            tail_idx = torch.argsort(v)[:tail_k]
            boost = float(protect_cfg.get('tail_boost', 2.0))
            boost = max(1.0, float(boost))
            quotas[tail_idx] = torch.clamp((quotas[tail_idx].float() * boost).round().long(), min=per_joint_min)

    total = int(quotas.sum().item())
    if total > 0 and total != protect_keep:
        q = quotas.float() * (float(protect_keep) / float(total))
        q = torch.clamp(q.round().long(), min=per_joint_min)
        quotas = q
    if int(quotas.sum().item()) > protect_keep:
        overflow = int(quotas.sum().item()) - int(protect_keep)
        for _ in range(overflow):
            j = int(torch.argmax(quotas).item())
            if quotas[j] > per_joint_min:
                quotas[j] -= 1

    protected = []
    for j in range(k_j):
        qj = int(quotas[j].item())
        if qj <= 0:
            continue
        idx = _safe_topk_from_vector(joint_scores[j].float().cpu(), qj)
        if idx is not None:
            protected.append(idx)
    if not protected:
        return torch.empty((0,), dtype=torch.long)
    prot = torch.unique(torch.cat(protected, dim=0))
    if int(prot.numel()) > protect_keep:
        bs = base_scores.detach().float().cpu()
        prot_bs = bs[prot]
        sel = torch.topk(prot_bs, k=int(protect_keep), largest=True, sorted=True).indices
        prot = prot[sel]
    return prot.sort().values


def _select_with_protection(
    base_scores: torch.Tensor,
    keep: int,
    protected_idx: Optional[torch.Tensor],
) -> torch.Tensor:
    c = int(base_scores.numel())
    keep = max(1, min(int(keep), int(c)))
    if not isinstance(protected_idx, torch.Tensor) or protected_idx.numel() <= 0:
        return _safe_topk_from_vector(base_scores.float().cpu(), keep)
    prot = torch.unique(protected_idx.detach().long().cpu())
    prot = prot[(prot >= 0) & (prot < c)]
    prot = prot.sort().values
    if int(prot.numel()) >= keep:
        bs = base_scores.detach().float().cpu()
        sel = torch.topk(bs[prot], k=int(keep), largest=True, sorted=True).indices
        return prot[sel].sort().values

    bs = base_scores.detach().float().cpu()
    mask = torch.ones((c,), dtype=torch.bool)
    mask[prot] = False
    rest_scores = bs.clone()
    rest_scores[~mask] = -1.0e18
    rest_k = int(keep - int(prot.numel()))
    rest = _safe_topk_from_vector(rest_scores, rest_k)
    out = torch.unique(torch.cat([prot, rest], dim=0))
    if int(out.numel()) > keep:
        sel = torch.topk(bs[out], k=int(keep), largest=True, sorted=True).indices
        out = out[sel]
    return out.sort().values


def _build_hrnet_channel_index_map_pose_proto(
    student_state_dict: Dict[str, torch.Tensor],
    new_extra: Dict,
    cached_scores: Dict,
) -> Dict:
    stage3_new = list(new_extra.get('stage3', {}).get('num_channels', []))
    stage4_new = list(new_extra.get('stage4', {}).get('num_channels', []))

    stage3_old = {}
    stage4_old = {}
    for k, v in student_state_dict.items():
        if k.startswith('backbone.stage3') and k.endswith('.bn1.weight'):
            parts = k.split('.')
            if 'branches' in parts:
                try:
                    b = int(parts[parts.index('branches') + 1])
                    stage3_old[b] = int(v.numel())
                except Exception:
                    pass
        if k.startswith('backbone.stage4') and k.endswith('.bn1.weight'):
            parts = k.split('.')
            if 'branches' in parts:
                try:
                    b = int(parts[parts.index('branches') + 1])
                    stage4_old[b] = int(v.numel())
                except Exception:
                    pass

    idx_map = {'stage3': {}, 'stage4': {}}
    for b, old_ch in stage3_old.items():
        keep = int(stage3_new[b]) if b < len(stage3_new) else int(old_ch)
        sc = None
        try:
            sc = cached_scores.get('stage3', {}).get(int(b), None)
        except Exception:
            sc = None
        bn = _bn_gamma_scores_for_branch(student_state_dict, 'stage3', int(b), int(old_ch))
        if not isinstance(sc, torch.Tensor) or sc.numel() != int(old_ch):
            sc = bn
        else:
            sc = _normalize_scores(sc) + _normalize_scores(bn) * 0.15
        idx_map['stage3'][b] = _safe_topk_from_vector(sc.float().cpu(), keep)

    for b, old_ch in stage4_old.items():
        keep = int(stage4_new[b]) if b < len(stage4_new) else int(old_ch)
        sc = None
        try:
            sc = cached_scores.get('stage4', {}).get(int(b), None)
        except Exception:
            sc = None
        bn = _bn_gamma_scores_for_branch(student_state_dict, 'stage4', int(b), int(old_ch))
        if not isinstance(sc, torch.Tensor) or sc.numel() != int(old_ch):
            sc = bn
        else:
            sc = _normalize_scores(sc) + _normalize_scores(bn) * 0.15
        idx_map['stage4'][b] = _safe_topk_from_vector(sc.float().cpu(), keep)

    return idx_map


def _build_hrnet_channel_index_map_pose_protect(
    student_state_dict: Dict[str, torch.Tensor],
    new_extra: Dict,
    cached_global: Dict,
    cached_joint: Dict,
    joint_vis: Optional[torch.Tensor],
    pose_cfg: Dict,
) -> Dict:
    stage3_new = list(new_extra.get('stage3', {}).get('num_channels', []))
    stage4_new = list(new_extra.get('stage4', {}).get('num_channels', []))

    stage3_old = {}
    stage4_old = {}
    for k, v in student_state_dict.items():
        if k.startswith('backbone.stage3') and k.endswith('.bn1.weight'):
            parts = k.split('.')
            if 'branches' in parts:
                try:
                    b = int(parts[parts.index('branches') + 1])
                    stage3_old[b] = int(v.numel())
                except Exception:
                    pass
        if k.startswith('backbone.stage4') and k.endswith('.bn1.weight'):
            parts = k.split('.')
            if 'branches' in parts:
                try:
                    b = int(parts[parts.index('branches') + 1])
                    stage4_old[b] = int(v.numel())
                except Exception:
                    pass

    bn_mix = float(pose_cfg.get('bn_mix', 0.15))
    bn_mix = max(0.0, float(bn_mix))
    protect_cfg = dict(pose_cfg)

    idx_map = {'stage3': {}, 'stage4': {}}
    for b, old_ch in stage3_old.items():
        keep = int(stage3_new[b]) if b < len(stage3_new) else int(old_ch)
        g = None
        j = None
        try:
            g = cached_global.get('stage3', {}).get(int(b), None)
        except Exception:
            g = None
        try:
            j = cached_joint.get('stage3', {}).get(int(b), None)
        except Exception:
            j = None
        bn = _bn_gamma_scores_for_branch(student_state_dict, 'stage3', int(b), int(old_ch))
        if not isinstance(g, torch.Tensor) or g.numel() != int(old_ch):
            base = bn
        else:
            base = _normalize_scores(g) + _normalize_scores(bn) * float(bn_mix)
        base = base.float().cpu()
        prot = _build_protected_set(j, keep, base, joint_vis, protect_cfg)
        idx_map['stage3'][b] = _select_with_protection(base, keep, prot)

    for b, old_ch in stage4_old.items():
        keep = int(stage4_new[b]) if b < len(stage4_new) else int(old_ch)
        g = None
        j = None
        try:
            g = cached_global.get('stage4', {}).get(int(b), None)
        except Exception:
            g = None
        try:
            j = cached_joint.get('stage4', {}).get(int(b), None)
        except Exception:
            j = None
        bn = _bn_gamma_scores_for_branch(student_state_dict, 'stage4', int(b), int(old_ch))
        if not isinstance(g, torch.Tensor) or g.numel() != int(old_ch):
            base = bn
        else:
            base = _normalize_scores(g) + _normalize_scores(bn) * float(bn_mix)
        base = base.float().cpu()
        prot = _build_protected_set(j, keep, base, joint_vis, protect_cfg)
        idx_map['stage4'][b] = _select_with_protection(base, keep, prot)

    return idx_map


@POSENETS.register_module()
class TopDownDistillPruneInnov(TopDownDistillPrune):
    def __init__(self, *args, student_init_ckpt=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._pose_channel_scores = {'stage3': {}, 'stage4': {}}
        self._pose_joint_scores = {'stage3': {}, 'stage4': {}}
        self._joint_vis_ema = None
        self._stage3_cache = None
        self._stage3_hook_handle = None
        self._hm_sup_heads = None
        self._hm_sup_cfg = None
        self._maybe_load_student_ckpt(student_init_ckpt)
        self._install_feature_hooks()
        self._rebuild_heatmap_supervision_heads()

    def _maybe_load_student_ckpt(self, ckpt_path):
        if ckpt_path is None:
            return
        p = str(ckpt_path)
        if not p:
            return
        try:
            ckpt = torch.load(p, map_location='cpu')
        except Exception:
            return
        sd = None
        if isinstance(ckpt, dict):
            for k in ['state_dict', 'model', 'student', 'student_state_dict']:
                v = ckpt.get(k, None)
                if isinstance(v, dict):
                    sd = v
                    break
        if sd is None and isinstance(ckpt, dict):
            sd = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
        if isinstance(ckpt, dict) is False and hasattr(ckpt, 'keys') and hasattr(ckpt, 'items'):
            try:
                sd = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
            except Exception:
                sd = None
        if not isinstance(sd, dict) or not sd:
            return

        keys = [str(k) for k in sd.keys()]
        is_whole_model = any(('keypoint_head' in k) or ('proto_head' in k) for k in keys)
        has_backbone_prefix = any(k.startswith('backbone.') or k.startswith('student.backbone.') for k in keys)
        backbone_only_like = (not is_whole_model) and (not has_backbone_prefix) and any(k.startswith('conv1.') or k.startswith('layer1.') for k in keys)
        has_final_layer = any(k.startswith('final_layer.') for k in keys)
        has_stage_prefix = any(k.startswith('stage2.') or k.startswith('stage3.') or k.startswith('stage4.') for k in keys)

        if backbone_only_like and has_stage_prefix:
            mapped = {}
            for k, v in sd.items():
                if not isinstance(v, torch.Tensor):
                    continue
                kk = str(k)
                if kk.startswith('module.'):
                    kk = kk[len('module.') :]
                if kk.startswith('student.'):
                    kk = kk[len('student.') :]
                if kk.startswith('final_layer.'):
                    if kk in ['final_layer.weight', 'final_layer.bias']:
                        mapped[f'keypoint_head.final_layer.{kk.split(".", 1)[1]}'] = v
                    continue
                mapped[f'backbone.{kk}'] = v
            try:
                self.student.load_state_dict(mapped, strict=False)
            except Exception:
                return
            return

        new_sd = {}
        for k, v in sd.items():
            if not isinstance(v, torch.Tensor):
                continue
            kk = str(k)
            if kk.startswith('module.'):
                kk = kk[len('module.') :]
            if kk.startswith('student.'):
                kk = kk[len('student.') :]
            new_sd[kk] = v
        try:
            self.student.load_state_dict(new_sd, strict=False)
        except Exception:
            return

    def _install_feature_hooks(self):
        if self._stage3_hook_handle is not None:
            try:
                self._stage3_hook_handle.remove()
            except Exception:
                pass
            self._stage3_hook_handle = None
        self._stage3_cache = None
        try:
            backbone = getattr(self.student, 'backbone', None)
            stage3 = getattr(backbone, 'stage3', None)
            if stage3 is None:
                return

            def _hook(_m, _inp, out):
                self._stage3_cache = out

            self._stage3_hook_handle = stage3.register_forward_hook(_hook)
        except Exception:
            self._stage3_hook_handle = None

    def _rebuild_heatmap_supervision_heads(self):
        cfg = self.distill_cfg.get('heatmap_supervision', None)
        if not isinstance(cfg, dict) or not bool(cfg.get('enable', False)):
            self._hm_sup_heads = None
            self._hm_sup_cfg = None
            return
        self._hm_sup_cfg = copy.deepcopy(cfg)

        branches = cfg.get('branches', None)
        if not isinstance(branches, (list, tuple)) or not branches:
            branches = [('stage3', 2, 0.1), ('stage4', 2, 0.1), ('stage4', 3, 0.1)]

        extra = self.student_cfg.get('backbone', {}).get('extra', {})
        s3_nc = list(extra.get('stage3', {}).get('num_channels', ()))
        s4_nc = list(extra.get('stage4', {}).get('num_channels', ()))

        out_k = cfg.get('out_channels', None)
        if out_k is None:
            out_k = getattr(getattr(self.student, 'keypoint_head', None), 'out_channels', None)
        if out_k is None:
            out_k = cfg.get('num_joints', 17)
        out_k = int(out_k)
        new_heads = nn.ModuleDict()
        for it in branches:
            try:
                stage = str(it[0])
                bid = int(it[1])
            except Exception:
                continue
            if stage == 'stage3':
                if bid < 0 or bid >= len(s3_nc):
                    continue
                in_ch = int(s3_nc[bid])
            elif stage == 'stage4':
                if bid < 0 or bid >= len(s4_nc):
                    continue
                in_ch = int(s4_nc[bid])
            else:
                continue
            key = f'{stage}_b{bid}'
            head = nn.Conv2d(in_ch, out_k, kernel_size=1, stride=1, padding=0, bias=True)
            nn.init.kaiming_normal_(head.weight, mode='fan_out', nonlinearity='relu')
            if head.bias is not None:
                nn.init.zeros_(head.bias)
            new_heads[key] = head

        try:
            device = next(self.student.parameters()).device
            new_heads = new_heads.to(device)
        except Exception:
            pass

        old_heads = self._hm_sup_heads
        self._hm_sup_heads = new_heads
        if isinstance(old_heads, nn.ModuleDict):
            for k, m in self._hm_sup_heads.items():
                if k not in old_heads:
                    continue
                try:
                    ow = getattr(old_heads[k], 'weight', None)
                    ob = getattr(old_heads[k], 'bias', None)
                    if isinstance(ow, torch.Tensor) and isinstance(m.weight, torch.Tensor):
                        copied = _copy_tensor_slices(m.weight, ow)
                        if copied is not None:
                            m.weight.data.copy_(copied)
                    if hasattr(m, 'bias') and isinstance(m.bias, torch.Tensor) and isinstance(ob, torch.Tensor):
                        copied = _copy_tensor_slices(m.bias, ob)
                        if copied is not None:
                            m.bias.data.copy_(copied)
                except Exception:
                    continue

    def prune_student_backbone_extra(self, new_extra, channel_map=None, importance_criterion='bn_gamma'):
        criterion = str(importance_criterion).lower()
        pp = self.distill_cfg.get('pose_prune', None)
        pose_enabled = bool(isinstance(pp, dict) and bool(pp.get('enable', False)))
        use_pose = pose_enabled or (criterion in ['pose_proto', 'pose_aware', 'pose_proto_guided'])
        old_sd = self.student.state_dict()
        new_cfg = copy.deepcopy(self.student_cfg)
        new_cfg['pretrained'] = None
        new_cfg.setdefault('backbone', {})
        new_cfg['backbone']['extra'] = copy.deepcopy(new_extra)
        new_student = build_posenet(new_cfg)
        device = next(self.student.parameters()).device
        new_student = new_student.to(device)

        if use_pose:
            pp = {} if not isinstance(pp, dict) else pp
            channel_map = _build_hrnet_channel_index_map_pose_protect(
                old_sd,
                new_extra,
                cached_global=self._pose_channel_scores,
                cached_joint=self._pose_joint_scores,
                joint_vis=self._joint_vis_ema,
                pose_cfg=pp,
            )

        new_sd = new_student.state_dict()
        new_sd = _remap_hrnet_pruned_weights(old_sd, new_sd, channel_map)
        new_student.load_state_dict(new_sd, strict=False)
        self.student = new_student
        self.student_cfg = new_cfg
        self._install_feature_hooks()
        self._rebuild_heatmap_supervision_heads()

    def _maybe_update_pose_scores(self, t_hm, t_proto, s_stage3, s_stage4, joint_weights=None):
        pp = self.distill_cfg.get('pose_prune', None)
        if not isinstance(pp, dict) or not bool(pp.get('enable', False)):
            return
        momentum = float(pp.get('ema_momentum', 0.95))
        attn_temp = float(pp.get('attn_temperature', 1.0))
        branches_s3 = tuple(pp.get('prune_branches_stage3', (2,)))
        branches_s4 = tuple(pp.get('prune_branches_stage4', (2, 3)))
        score_power = float(pp.get('score_power', 1.0))
        score_power = max(0.5, min(4.0, score_power))

        with torch.no_grad():
            proto_energy = None
            if isinstance(t_proto, torch.Tensor):
                proto_energy = t_proto.detach().abs().mean(dim=1, keepdim=True)

            if not isinstance(t_hm, torch.Tensor):
                return
            attn = _hm_attn_from_teacher(t_hm.detach(), proto_energy, temperature=attn_temp)

        def _joint_and_global_scores(x: torch.Tensor):
            if x.dim() != 4:
                return None, None
            if attn.shape[-2:] != x.shape[-2:]:
                a = F.interpolate(attn, size=x.shape[-2:], mode='bilinear', align_corners=False)
            else:
                a = attn
            x_abs = x.detach().abs()
            js = torch.einsum('nchw,nkhw->kc', x_abs, a.detach()).float()
            js = js / max(1.0, float(x_abs.size(0)))
            if score_power != 1.0:
                js = torch.pow(js.clamp_min(0.0), float(score_power))
            if not torch.isfinite(js).all():
                js = torch.nan_to_num(js, nan=0.0, posinf=0.0, neginf=0.0)
            if isinstance(joint_weights, torch.Tensor) and joint_weights.dim() == 1 and joint_weights.numel() == js.size(0):
                jw = joint_weights.detach().float().cpu().view(-1, 1)
                g = (js.detach().cpu() * jw).sum(dim=0)
            else:
                g = js.detach().cpu().mean(dim=0)
            if not torch.isfinite(g).all():
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            return js.detach().cpu(), g

        if isinstance(s_stage3, (list, tuple)):
            for b in branches_s3:
                if not isinstance(b, int) or b < 0 or b >= len(s_stage3):
                    continue
                x = s_stage3[b]
                js, g = _joint_and_global_scores(x)
                if js is None or g is None:
                    continue
                self._pose_joint_scores['stage3'][int(b)] = _update_ema(
                    self._pose_joint_scores['stage3'].get(int(b), None),
                    js,
                    momentum=momentum,
                )
                self._pose_channel_scores['stage3'][int(b)] = _update_ema(
                    self._pose_channel_scores['stage3'].get(int(b), None),
                    g,
                    momentum=momentum,
                )

        if isinstance(s_stage4, (list, tuple)):
            for b in branches_s4:
                if not isinstance(b, int) or b < 0 or b >= len(s_stage4):
                    continue
                x = s_stage4[b]
                js, g = _joint_and_global_scores(x)
                if js is None or g is None:
                    continue
                self._pose_joint_scores['stage4'][int(b)] = _update_ema(
                    self._pose_joint_scores['stage4'].get(int(b), None),
                    js,
                    momentum=momentum,
                )
                self._pose_channel_scores['stage4'][int(b)] = _update_ema(
                    self._pose_channel_scores['stage4'].get(int(b), None),
                    g,
                    momentum=momentum,
                )

    def forward_train(self, img, target, target_weight, img_metas, **kwargs):
        self._train_iter += 1
        self.teacher.eval()
        img_sources = None
        if img_metas is not None:
            try:
                img_sources = torch.tensor(
                    [int(m.get('dataset_idx', 0)) for m in img_metas],
                    device=img.device,
                    dtype=torch.long,
                )
            except Exception:
                img_sources = None
        with torch.no_grad():
            try:
                if img_sources is not None:
                    t_feat = self.teacher.backbone(img, img_sources)
                else:
                    t_feat = self.teacher.backbone(img)
            except TypeError:
                t_feat = self.teacher.backbone(img)
            if hasattr(self.teacher, 'neck'):
                t_feat = self.teacher.neck(t_feat)
            t_hm = self.teacher.keypoint_head(t_feat)
            t_proto = None
            if hasattr(self.teacher, 'proto_head'):
                t_proto = self.teacher.proto_head(t_feat)
                t_proto = _infer_teacher_proto(t_proto)

        self._stage3_cache = None
        s_feat = self.student.backbone(img)
        s_stage3 = self._stage3_cache
        s_stage4 = s_feat if isinstance(s_feat, (list, tuple)) else None
        if hasattr(self.student, 'neck'):
            s_feat = self.student.neck(s_feat)
        s_hm = self.student.keypoint_head(s_feat)

        jw = None
        try:
            if isinstance(target_weight, torch.Tensor) and target_weight.dim() >= 2:
                jw = target_weight.detach().float().mean(dim=0)
                if jw.dim() > 1:
                    jw = jw.reshape(-1)
                if not torch.isfinite(jw).all():
                    jw = torch.nan_to_num(jw, nan=0.0, posinf=0.0, neginf=0.0)
                s = float(jw.sum().item())
                if s > 0:
                    jw = jw / s
        except Exception:
            jw = None
        if jw is not None:
            self._joint_vis_ema = _update_ema(self._joint_vis_ema, jw.detach().float().cpu(), momentum=0.98)
        self._maybe_update_pose_scores(t_hm, t_proto, s_stage3, s_stage4, joint_weights=jw)

        losses = {}
        sup_losses = self.student.keypoint_head.get_loss(s_hm, target, target_weight)
        heatmap_loss_weight = float(self.distill_cfg.get('heatmap_loss_weight', 1.0))
        if 'heatmap_loss' in sup_losses:
            losses['loss_sup'] = sup_losses['heatmap_loss'] * heatmap_loss_weight
        else:
            for k, v in sup_losses.items():
                losses[f'loss_sup_{k}'] = v * heatmap_loss_weight

        hm_sup_cfg = self._hm_sup_cfg
        hm_sup_heads = self._hm_sup_heads
        if isinstance(hm_sup_cfg, dict) and isinstance(hm_sup_heads, nn.ModuleDict) and hm_sup_heads:
            tgt_mode = str(hm_sup_cfg.get('target', 'gt')).strip().lower()
            loss_type = str(hm_sup_cfg.get('loss', 'mse')).strip().lower()
            mix_alpha = float(hm_sup_cfg.get('mix_alpha', 0.5))
            mix_alpha = max(0.0, min(1.0, mix_alpha))
            temperature = float(hm_sup_cfg.get('temperature', 1.0))

            ref = None
            if tgt_mode == 'teacher':
                ref = t_hm.detach()
            elif tgt_mode == 'mix':
                if isinstance(t_hm, torch.Tensor) and isinstance(target, torch.Tensor):
                    tt = t_hm.detach()
                    if tt.shape[-2:] != target.shape[-2:]:
                        tt = F.interpolate(tt, size=target.shape[-2:], mode='bilinear', align_corners=False)
                    ref = target.detach() * mix_alpha + tt * (1.0 - mix_alpha)
            else:
                ref = target.detach() if isinstance(target, torch.Tensor) else None

            if isinstance(ref, torch.Tensor):
                if ref.shape[-2:] != target.shape[-2:]:
                    ref = F.interpolate(ref, size=target.shape[-2:], mode='bilinear', align_corners=False)

                branches = hm_sup_cfg.get('branches', None)
                if not isinstance(branches, (list, tuple)) or not branches:
                    branches = [('stage3', 2, 0.1), ('stage4', 2, 0.1), ('stage4', 3, 0.1)]

                for it in branches:
                    try:
                        stage = str(it[0])
                        bid = int(it[1])
                        w = float(it[2]) if len(it) >= 3 else float(hm_sup_cfg.get('default_weight', 0.1))
                    except Exception:
                        continue
                    if w <= 0:
                        continue
                    xs = None
                    if stage == 'stage3' and isinstance(s_stage3, (list, tuple)) and 0 <= bid < len(s_stage3):
                        xs = s_stage3[bid]
                    if stage == 'stage4' and isinstance(s_stage4, (list, tuple)) and 0 <= bid < len(s_stage4):
                        xs = s_stage4[bid]
                    if not isinstance(xs, torch.Tensor):
                        continue
                    key = f'{stage}_b{bid}'
                    if key not in hm_sup_heads:
                        continue
                    head = hm_sup_heads[key]
                    pred = head(xs)
                    if pred.shape[-2:] != ref.shape[-2:]:
                        pred = F.interpolate(pred, size=ref.shape[-2:], mode='bilinear', align_corners=False)
                    if pred.size(1) != ref.size(1):
                        continue
                    if loss_type in ['kl', 'spatial_kl']:
                        losses[f'loss_hm_sup_{stage}_b{bid}'] = _spatial_kl(pred, ref, temperature=temperature) * float(w)
                    else:
                        losses[f'loss_hm_sup_{stage}_b{bid}'] = F.mse_loss(pred, ref) * float(w)

        hga = self.distill_cfg.get('heatmap_guided_align', None)
        if isinstance(hga, dict) and bool(hga.get('enable', False)):
            with torch.no_grad():
                m = _mask_from_teacher_hm(
                    t_hm,
                    t_proto=t_proto,
                    temperature=float(hga.get('temperature', 1.0)),
                    use_proto_energy=bool(hga.get('use_proto_energy', True)),
                )
                t_src = t_feat if isinstance(t_feat, torch.Tensor) else _infer_teacher_proto(t_proto)
                t_e = _energy_map(t_src, eps=1.0e-8) if isinstance(t_src, torch.Tensor) else None
                layers = hga.get('student_layers', None)
                if not isinstance(layers, (list, tuple)):
                    layers = [('stage3', 2), ('stage4', 2), ('stage4', 3)]
                for it in layers:
                    try:
                        stage = str(it[0])
                        bid = int(it[1])
                    except Exception:
                        continue
                    xs = None
                    if stage == 'stage3' and isinstance(s_stage3, (list, tuple)) and 0 <= bid < len(s_stage3):
                        xs = s_stage3[bid]
                    if stage == 'stage4' and isinstance(s_stage4, (list, tuple)) and 0 <= bid < len(s_stage4):
                        xs = s_stage4[bid]
                    es = _energy_map(xs, eps=1.0e-8)
                    if es is None or t_e is None or m is None:
                        continue
                    te = t_e
                    if te.shape[-2:] != es.shape[-2:]:
                        te = F.interpolate(te, size=es.shape[-2:], mode='bilinear', align_corners=False)
                        te = te / (te.sum(dim=(2, 3), keepdim=True) + 1.0e-8)
                    cos = _masked_cosine(es, te, m)
                    l1 = _masked_l1(es, te, m)
                    if isinstance(cos, torch.Tensor):
                        losses[f'metric_hm_align_cos_{stage}_b{bid}'] = cos.detach()
                    if isinstance(l1, torch.Tensor):
                        losses[f'metric_hm_align_l1_{stage}_b{bid}'] = l1.detach()

        hm_w, proto_w = _dynamic_kd_weights(self.distill_cfg, self.get_prune_state())
        if hm_w > 0:
            t = float(self.distill_cfg.get('temperature', 1.0))
            losses['loss_kd_hm'] = _spatial_kl(s_hm, t_hm, temperature=t) * float(hm_w)
            losses['kd_hm_w'] = s_hm.new_tensor(float(hm_w))
            losses['prune_ratio'] = s_hm.new_tensor(float(_get_prune_ratio_from_state(self.get_prune_state())))

        if proto_w > 0 and isinstance(t_proto, torch.Tensor):
            s_map = _infer_student_feat(s_feat)
            normalize_teacher = bool(self.distill_cfg.get('normalize_teacher_proto', True))
            aligned = self._proto_adaptor(s_map) if self._proto_adaptor is not None else None
            if aligned is not None:
                if aligned.shape[-2:] != t_proto.shape[-2:]:
                    aligned = F.interpolate(aligned, size=t_proto.shape[-2:], mode='bilinear', align_corners=False)
                tgt = t_proto
                if normalize_teacher:
                    tgt = _norm_spatial(tgt, eps=1.0e-5)
                losses['loss_kd_proto'] = F.mse_loss(aligned, tgt) * float(proto_w)
                losses['kd_proto_w'] = s_hm.new_tensor(float(proto_w))

        warmup_iters = int(self.distill_cfg.get('sup_ratio_enforce_iters', 0))
        min_sup_ratio = float(self.distill_cfg.get('min_sup_ratio', 0.0))
        if warmup_iters > 0 and min_sup_ratio > 0 and self._train_iter <= warmup_iters:
            sup = losses.get('loss_sup', None)
            if isinstance(sup, torch.Tensor):
                kd_sum = 0.0
                kd_keys = [k for k in losses.keys() if k.startswith('loss_kd_')]
                for k in kd_keys:
                    v = losses.get(k, None)
                    if isinstance(v, torch.Tensor):
                        kd_sum = kd_sum + v
                if isinstance(kd_sum, torch.Tensor):
                    max_kd = sup * (1.0 - min_sup_ratio) / max(min_sup_ratio, 1.0e-12)
                    scale = (max_kd / (kd_sum + 1.0e-12)).clamp(max=1.0)
                    if kd_keys and float(scale.item()) < 1.0:
                        for k in kd_keys:
                            v = losses.get(k, None)
                            if isinstance(v, torch.Tensor):
                                losses[k] = v * scale
                        losses['sup_ratio'] = sup / (sup + kd_sum * scale + 1.0e-12)

        return losses
