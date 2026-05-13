import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmpose.models.builder import POSENETS, build_posenet
from mmpose.models.detectors.base import BasePose


def _infer_student_feat(student_backbone_out):
    if isinstance(student_backbone_out, (list, tuple)):
        return student_backbone_out[0]
    return student_backbone_out


def _infer_teacher_proto(teacher_proto_out):
    if isinstance(teacher_proto_out, (list, tuple)):
        return teacher_proto_out[0]
    return teacher_proto_out


def _spatial_kl(student_logits, teacher_logits, temperature=1.0):
    s = student_logits
    q = teacher_logits
    t = float(temperature)
    if t <= 0:
        t = 1.0
    s = s / t
    q = q / t
    n, k, h, w = s.shape
    s = s.reshape(n, k, h * w)
    q = q.reshape(n, k, h * w)
    log_p = F.log_softmax(s, dim=-1)
    p_t = F.softmax(q, dim=-1)
    kl = F.kl_div(log_p, p_t, reduction='batchmean') * (t * t)
    return kl


def _strip_prefix(k, prefixes):
    for p in prefixes:
        if k.startswith(p):
            return k[len(p) :]
    return None


def _extract_student_state_dict(ckpt_state_dict):
    out = {}
    for k, v in ckpt_state_dict.items():
        nk = _strip_prefix(k, prefixes=['student.', 'module.student.'])
        if nk is None:
            continue
        out[nk] = v
    return out


def _get_student_extra_from_ckpt_meta(ckpt):
    meta = ckpt.get('meta', None)
    if not isinstance(meta, dict):
        return None
    state = meta.get('hrnet_prune_state', None)
    if not isinstance(state, dict):
        return None
    ex = state.get('student_backbone_extra', None)
    if isinstance(ex, dict):
        return ex
    return None


@POSENETS.register_module()
class TopDownDistillOnly(BasePose):
    def __init__(
        self,
        teacher,
        student,
        distill_cfg=None,
        student_init_ckpt=None,
        train_cfg=None,
        test_cfg=None,
    ):
        super().__init__()
        self.fp16_enabled = False

        self.teacher_cfg = copy.deepcopy(teacher)
        self.student_cfg = copy.deepcopy(student)
        self.distill_cfg = {} if distill_cfg is None else copy.deepcopy(distill_cfg)
        self.student_init_ckpt = None if student_init_ckpt is None else str(student_init_ckpt)

        self.teacher = build_posenet(self.teacher_cfg)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        ckpt = None
        if self.student_init_ckpt:
            ckpt = torch.load(self.student_init_ckpt, map_location='cpu')
            if isinstance(ckpt, dict):
                ex = _get_student_extra_from_ckpt_meta(ckpt)
                if isinstance(ex, dict):
                    self.student_cfg.setdefault('backbone', {})
                    self.student_cfg['backbone'].setdefault('extra', {})
                    self.student_cfg['backbone']['extra'] = copy.deepcopy(ex)
            self.student_cfg['pretrained'] = None

        self.student = build_posenet(self.student_cfg)
        if ckpt is not None and isinstance(ckpt, dict):
            sd = ckpt.get('state_dict', None)
            if isinstance(sd, dict):
                student_sd = _extract_student_state_dict(sd)
                if student_sd:
                    self.student.load_state_dict(student_sd, strict=False)

        self._proto_adaptor = None
        adaptor_out = self.distill_cfg.get('proto_adaptor_out_channels', None)
        if adaptor_out is not None:
            in_ch = self._infer_student_feat_channels()
            self._proto_adaptor = nn.Conv2d(
                in_ch,
                int(adaptor_out),
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            nn.init.kaiming_normal_(self._proto_adaptor.weight, mode='fan_out', nonlinearity='relu')

        self._train_iter = 0
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _infer_student_feat_channels(self):
        extra = (
            self.student_cfg.get('backbone', {})
            .get('extra', {})
            .get('stage4', {})
            .get('num_channels', None)
        )
        if extra is None:
            return 32
        if isinstance(extra, (list, tuple)) and len(extra) > 0:
            return int(extra[0])
        return 32

    def forward(
        self,
        img,
        target=None,
        target_weight=None,
        img_metas=None,
        return_loss=True,
        return_heatmap=False,
        **kwargs,
    ):
        if return_loss:
            return self.forward_train(img, target, target_weight, img_metas, **kwargs)
        return self.forward_test(img, img_metas, return_heatmap=return_heatmap, **kwargs)

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

        s_feat = self.student.backbone(img)
        if hasattr(self.student, 'neck'):
            s_feat = self.student.neck(s_feat)
        s_hm = self.student.keypoint_head(s_feat)

        losses = {}
        sup_losses = self.student.keypoint_head.get_loss(s_hm, target, target_weight)
        heatmap_loss_weight = float(self.distill_cfg.get('heatmap_loss_weight', 1.0))
        if 'heatmap_loss' in sup_losses:
            losses['loss_sup'] = sup_losses['heatmap_loss'] * heatmap_loss_weight
        else:
            for k, v in sup_losses.items():
                losses[f'loss_sup_{k}'] = v * heatmap_loss_weight

        hm_w = float(self.distill_cfg.get('kd_hm_weight', self.distill_cfg.get('hm_weight', 0.0)))
        if hm_w > 0:
            t = float(self.distill_cfg.get('temperature', 1.0))
            losses['loss_kd_hm'] = _spatial_kl(s_hm, t_hm, temperature=t) * hm_w

        proto_w = float(self.distill_cfg.get('kd_proto_weight', self.distill_cfg.get('proto_weight', 0.0)))
        if proto_w > 0 and t_proto is not None and self._proto_adaptor is not None:
            s_map = _infer_student_feat(s_feat)
            s_map = self._proto_adaptor(s_map)
            if s_map.shape[-2:] != t_proto.shape[-2:]:
                s_map = F.interpolate(s_map, size=t_proto.shape[-2:], mode='bilinear', align_corners=False)
            losses['loss_kd_proto'] = F.mse_loss(s_map, t_proto) * proto_w

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

    def forward_test(self, img, img_metas, return_heatmap=False, **kwargs):
        return self.student.forward_test(img, img_metas, return_heatmap=return_heatmap, **kwargs)

    def show_result(self, **kwargs):
        return self.student.show_result(**kwargs)

