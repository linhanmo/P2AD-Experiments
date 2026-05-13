import argparse
import json
from collections import Counter, defaultdict

import torch


def _as_dict(x):
    return x if isinstance(x, dict) else {}


def _load_cfg(cfg_path):
    from mmcv import Config

    return Config.fromfile(cfg_path)


def _build_posenet(cfg_dict):
    from mmpose.models.builder import build_posenet

    return build_posenet(cfg_dict)


def _count_params(m):
    return int(sum(int(p.numel()) for p in m.parameters()))


def _prefix_counts(keys, depth=2, topk=30):
    c = Counter()
    for k in keys:
        parts = str(k).split(".")
        p = ".".join(parts[: max(1, int(depth))])
        c[p] += 1
    return c.most_common(int(topk))


def _ckpt_to_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "student", "teacher"]:
            v = obj.get(k, None)
            if isinstance(v, dict):
                return v
        t = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
        if t:
            return t
    if hasattr(obj, "items"):
        try:
            t = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}
            if t:
                return t
        except Exception:
            return None
    return None


def _map_hrnet_bn_to_norm(k: str) -> str:
    return k


def _map_student_ckpt_for_topdown(student, sd):
    keys = [str(k) for k in sd.keys()]
    has_stage_prefix = any(k.startswith("stage2.") or k.startswith("stage3.") or k.startswith("stage4.") for k in keys)
    has_final_layer = any(k.startswith("final_layer.") for k in keys)
    mapped = {}
    if has_stage_prefix:
        for k, v in sd.items():
            if not isinstance(v, torch.Tensor):
                continue
            kk = str(k)
            if kk.startswith("module."):
                kk = kk[len("module.") :]
            if kk.startswith("student."):
                kk = kk[len("student.") :]
            if has_final_layer and kk in ["final_layer.weight", "final_layer.bias"]:
                mapped[f"keypoint_head.final_layer.{kk.split('.', 1)[1]}"] = v
                continue
            kk = _map_hrnet_bn_to_norm(kk)
            mapped[f"backbone.{kk}"] = v
        return mapped
    return None


def _analyze_load(target_sd, source_sd):
    target_keys = set(target_sd.keys())
    source_keys = set(source_sd.keys())
    matched = sorted(list(target_keys & source_keys))
    missing = sorted(list(target_keys - source_keys))
    unexpected = sorted(list(source_keys - target_keys))
    shape_mismatch = []
    for k in matched:
        a = target_sd[k]
        b = source_sd[k]
        try:
            if tuple(a.shape) != tuple(b.shape):
                shape_mismatch.append((k, tuple(a.shape), tuple(b.shape)))
        except Exception:
            pass
    return dict(
        target_total=len(target_keys),
        source_total=len(source_keys),
        matched=len(matched),
        missing=len(missing),
        unexpected=len(unexpected),
        shape_mismatch=len(shape_mismatch),
        shape_mismatch_samples=shape_mismatch[:50],
        missing_samples=missing[:50],
        unexpected_samples=unexpected[:50],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--teacher-ckpt", default=None)
    ap.add_argument("--student-ckpt", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = _load_cfg(args.cfg)
    model_cfg = _as_dict(getattr(cfg, "model", None))
    teacher_cfg = _as_dict(model_cfg.get("teacher", None))
    student_cfg = _as_dict(model_cfg.get("student", None))

    teacher = _build_posenet(teacher_cfg)
    student = _build_posenet(student_cfg)

    report = dict()
    report["cfg"] = str(args.cfg)
    report["teacher"] = dict(type=teacher.__class__.__name__, params=_count_params(teacher))
    report["student"] = dict(type=student.__class__.__name__, params=_count_params(student))

    t_sd = teacher.state_dict()
    s_sd = student.state_dict()
    report["teacher_state"] = dict(
        total=len(t_sd),
        prefix2=_prefix_counts(list(t_sd.keys()), depth=2),
        prefix3=_prefix_counts(list(t_sd.keys()), depth=3),
    )
    report["student_state"] = dict(
        total=len(s_sd),
        prefix2=_prefix_counts(list(s_sd.keys()), depth=2),
        prefix3=_prefix_counts(list(s_sd.keys()), depth=3),
    )

    if args.teacher_ckpt:
        obj = torch.load(args.teacher_ckpt, map_location="cpu")
        sd = _ckpt_to_state_dict(obj) or {}
        report["teacher_ckpt"] = dict(
            path=str(args.teacher_ckpt),
            total=len(sd),
            prefix2=_prefix_counts(list(sd.keys()), depth=2),
            analysis=_analyze_load(t_sd, sd),
        )

    if args.student_ckpt:
        obj = torch.load(args.student_ckpt, map_location="cpu")
        sd = _ckpt_to_state_dict(obj) or {}
        rep = dict(
            path=str(args.student_ckpt),
            total=len(sd),
            prefix2=_prefix_counts(list(sd.keys()), depth=2),
        )
        rep["analysis_raw"] = _analyze_load(s_sd, sd)
        mapped = _map_student_ckpt_for_topdown(student, sd)
        if isinstance(mapped, dict):
            rep["analysis_mapped"] = _analyze_load(s_sd, mapped)
        report["student_ckpt"] = rep

    out_path = args.out
    if not out_path:
        out_path = "experiments/DIST_INNOV/mapping_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(out_path)
    print(json.dumps(report.get("teacher_state", {}), ensure_ascii=False, indent=2))
    print(json.dumps(report.get("student_state", {}), ensure_ascii=False, indent=2))
    if "student_ckpt" in report:
        print(json.dumps(report["student_ckpt"].get("analysis_raw", {}), ensure_ascii=False, indent=2))
        if "analysis_mapped" in report["student_ckpt"]:
            print(json.dumps(report["student_ckpt"]["analysis_mapped"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
