import copy

import torch
from mmcv.runner import HOOKS, Hook, get_dist_info

from mmpose.core import build_optimizers
from mmpose.models.builder import build_posenet


def _logbuffer_get_last(log_buffer, key):
    try:
        return log_buffer.get_last(key)
    except Exception:
        try:
            out = getattr(log_buffer, 'output', None)
            if isinstance(out, dict):
                return out.get(key, None)
        except Exception:
            return None
    return None


def _round_nearest_multiple(x, m, min_value, max_value=None, min_keep=None):
    m = int(m)
    min_value = int(min_value)
    if x <= min_value:
        return int(min_value)
    v = int(round(float(x) / float(m))) * m
    v = max(v, min_value)
    if min_keep is not None:
        try:
            mk = int(min_keep)
        except Exception:
            mk = int(float(min_keep))
        if v < mk:
            v = int(((mk + m - 1) // m) * m)
    if max_value is not None:
        v = min(int(max_value), v)
    return int(v)


def _pruned_hrnet_extra(
    base_extra,
    mid_ratio,
    high_ratio,
    round_to=64,
    prune_stages=(3, 4),
    mid_limit=0.2,
    high_limit=0.4,
):
    extra = copy.deepcopy(base_extra)

    stage3 = extra.get('stage3', {})
    stage4 = extra.get('stage4', {})

    s3 = list(stage3.get('num_channels', []))
    s4 = list(stage4.get('num_channels', []))

    new_s3 = []
    for ch in s3:
        ch = int(ch)
        if 3 not in prune_stages:
            new_s3.append(ch)
            continue
        if ch < round_to:
            new_s3.append(ch)
            continue
        kept = ch * (1.0 - float(mid_ratio))
        min_keep = ch * (1.0 - float(mid_limit))
        new_s3.append(_round_nearest_multiple(kept, round_to, round_to, max_value=ch, min_keep=min_keep))

    new_s4 = []
    for ch in s4:
        ch = int(ch)
        if 4 not in prune_stages:
            new_s4.append(ch)
            continue
        if ch < round_to:
            new_s4.append(ch)
            continue
        kept = ch * (1.0 - float(high_ratio))
        min_keep = ch * (1.0 - float(high_limit))
        new_s4.append(_round_nearest_multiple(kept, round_to, round_to, max_value=ch, min_keep=min_keep))

    stage3['num_channels'] = tuple(new_s3)
    stage4['num_channels'] = tuple(new_s4)
    extra['stage3'] = stage3
    extra['stage4'] = stage4
    return extra


def _estimate_prune_rate(base_extra, new_extra):
    def _sum_channels(ex):
        s3 = list(ex.get('stage3', {}).get('num_channels', []))
        s4 = list(ex.get('stage4', {}).get('num_channels', []))
        return float(sum(int(x) for x in s3) + sum(int(x) for x in s4))

    base_sum = _sum_channels(base_extra)
    new_sum = _sum_channels(new_extra)
    if base_sum <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (new_sum / base_sum)))


def _candidate_extra_for_target_ratio(
    base_extra,
    target_ratio,
    round_to=16,
    prune_stages=(3, 4),
    prune_branches_stage3=(2,),
    prune_branches_stage4=(2, 3),
    branch_weight_stage3=None,
    branch_weight_stage4=None,
    alpha=1.0,
):
    extra = copy.deepcopy(base_extra)
    stage3 = extra.get('stage3', {})
    stage4 = extra.get('stage4', {})

    s3 = list(stage3.get('num_channels', []))
    s4 = list(stage4.get('num_channels', []))

    w3 = [0.25, 0.6, 1.0]
    w4 = [0.2, 0.5, 0.8, 1.0]
    try:
        if isinstance(branch_weight_stage3, (list, tuple)) and branch_weight_stage3:
            w3 = [float(x) for x in branch_weight_stage3]
    except Exception:
        pass
    try:
        if isinstance(branch_weight_stage4, (list, tuple)) and branch_weight_stage4:
            w4 = [float(x) for x in branch_weight_stage4]
    except Exception:
        pass

    new_s3 = []
    for i, ch in enumerate(s3):
        ch = int(ch)
        if 3 not in prune_stages or i not in prune_branches_stage3 or ch < round_to:
            new_s3.append(ch)
            continue
        r = float(alpha) * float(target_ratio) * float(w3[i] if i < len(w3) else w3[-1])
        r = max(0.0, min(0.95, r))
        kept = ch * (1.0 - r)
        new_s3.append(_round_nearest_multiple(kept, round_to, round_to, max_value=ch))

    new_s4 = []
    for i, ch in enumerate(s4):
        ch = int(ch)
        if 4 not in prune_stages or i not in prune_branches_stage4 or ch < round_to:
            new_s4.append(ch)
            continue
        r = float(alpha) * float(target_ratio) * float(w4[i] if i < len(w4) else w4[-1])
        r = max(0.0, min(0.95, r))
        kept = ch * (1.0 - r)
        new_s4.append(_round_nearest_multiple(kept, round_to, round_to, max_value=ch))

    stage3['num_channels'] = tuple(int(x) for x in new_s3)
    stage4['num_channels'] = tuple(int(x) for x in new_s4)
    extra['stage3'] = stage3
    extra['stage4'] = stage4
    return extra


def _count_model_params_from_cfg(model_cfg):
    m = build_posenet(model_cfg)
    try:
        return int(sum(int(p.numel()) for p in m.parameters()))
    finally:
        del m


def _search_extra_for_param_prune_target(
    student_cfg,
    base_extra,
    base_params,
    target_ratio,
    round_to=16,
    prune_stages=(3, 4),
    prune_branches_stage3=(2,),
    prune_branches_stage4=(2, 3),
    branch_weight_stage3=None,
    branch_weight_stage4=None,
):
    target_ratio = float(target_ratio)
    if base_params is None or base_params <= 0:
        return (
            _candidate_extra_for_target_ratio(
                base_extra,
                target_ratio,
                round_to=round_to,
                prune_stages=prune_stages,
                branch_weight_stage3=branch_weight_stage3,
                branch_weight_stage4=branch_weight_stage4,
                alpha=1.0,
            ),
            None,
        )

    cfg_base = copy.deepcopy(student_cfg)
    cfg_base['pretrained'] = None
    cfg_base.setdefault('backbone', {})

    alphas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    best = None
    best_over = None
    best_over_gap = None
    best_gap = None

    for a in alphas:
        ex = _candidate_extra_for_target_ratio(
            base_extra,
            target_ratio,
            round_to=round_to,
            prune_stages=prune_stages,
            prune_branches_stage3=prune_branches_stage3,
            prune_branches_stage4=prune_branches_stage4,
            branch_weight_stage3=branch_weight_stage3,
            branch_weight_stage4=branch_weight_stage4,
            alpha=a,
        )
        cfg_base['backbone']['extra'] = copy.deepcopy(ex)
        p = _count_model_params_from_cfg(cfg_base)
        r = max(0.0, min(1.0, 1.0 - (float(p) / float(base_params))))
        gap = abs(r - target_ratio)
        if best is None or gap < best_gap:
            best = (ex, r)
            best_gap = gap
        if r + 1.0e-12 >= target_ratio:
            over_gap = r - target_ratio
            if best_over is None or over_gap < best_over_gap:
                best_over = (ex, r)
                best_over_gap = over_gap

    if best_over is not None:
        return best_over[0], float(best_over[1])
    return best[0], float(best[1])


@HOOKS.register_module()
class EarlyStopByMetricHook(Hook):
    def __init__(self, monitor='AP', min_delta=0.001, patience=2, interval=10):
        self.monitor = monitor
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.interval = int(interval)
        self.best = None
        self.bad_count = 0

    def after_train_epoch(self, runner):
        current_epoch = runner.epoch + 1
        if current_epoch % self.interval != 0:
            return
        try:
            meta = runner.meta if runner.meta is not None else {}
            state = meta.get('hrnet_prune_state', None)
            if isinstance(state, dict) and state.get('recovering', False):
                return
        except Exception:
            pass
        val = _logbuffer_get_last(runner.log_buffer, self.monitor)
        if val is None:
            return
        if self.best is None or val > self.best + self.min_delta:
            self.best = float(val)
            self.bad_count = 0
            return
        self.bad_count += 1
        if self.bad_count >= self.patience:
            runner._stop = True


@HOOKS.register_module()
class HRNetStructuredPruneHook(Hook):
    def __init__(
        self,
        start_epoch=30,
        interval=10,
        mid_final=0.2,
        high_final=0.4,
        round_to=64,
        min_ap_drop=0.003,
        importance_criterion='bn_gamma',
        save_prune_ckpt=True,
        prune_stages=(3, 4),
        prune_protected_layers=(),
        prune_ramp_steps=None,
    ):
        self.start_epoch = int(start_epoch)
        self.interval = int(interval)
        self.mid_final = float(mid_final)
        self.high_final = float(high_final)
        self.round_to = int(round_to)
        self.min_ap_drop = float(min_ap_drop)
        self.importance_criterion = str(importance_criterion)
        self.save_prune_ckpt = bool(save_prune_ckpt)
        self.prune_stages = tuple(int(x) for x in prune_stages)
        self.prune_protected_layers = tuple(str(x) for x in prune_protected_layers)
        self.prune_ramp_steps = None if prune_ramp_steps is None else int(prune_ramp_steps)

        self.base_extra = None
        self.prune_step = 0
        self.last_eval_ap = None
        self.last_prune_eval_ap = None

    def before_run(self, runner):
        model = getattr(runner.model, 'module', runner.model)
        student_cfg = getattr(model, 'student_cfg', None)
        if student_cfg is None:
            return
        self.base_extra = copy.deepcopy(student_cfg.get('backbone', {}).get('extra', {}))
        try:
            if hasattr(model, 'student') and hasattr(model.student, 'parameters'):
                self.base_student_params = int(sum(int(p.numel()) for p in model.student.parameters()))
            else:
                self.base_student_params = None
        except Exception:
            self.base_student_params = None
        meta = runner.meta if runner.meta is not None else {}
        state = meta.get('hrnet_prune_state', None)
        if isinstance(state, dict):
            self.prune_step = int(state.get('prune_step', 0))
            self.last_eval_ap = state.get('last_eval_ap', None)
            self.last_prune_eval_ap = state.get('last_prune_eval_ap', None)
            base_extra = state.get('base_extra', None)
            if isinstance(base_extra, dict):
                self.base_extra = copy.deepcopy(base_extra)
            base_params = state.get('base_student_params', None)
            if base_params is not None:
                try:
                    self.base_student_params = int(base_params)
                except Exception:
                    pass

    def after_train_epoch(self, runner):
        current_epoch = runner.epoch + 1
        if self.base_extra is None:
            return
        if current_epoch < self.start_epoch:
            return
        if current_epoch % self.interval != 0:
            return

        ap_before = _logbuffer_get_last(runner.log_buffer, 'AP')
        try:
            if ap_before is not None:
                runner.logger.info(f'Pre-prune eval epoch {current_epoch}: AP={float(ap_before):.4f}')
        except Exception:
            pass

        total_epochs = int(getattr(runner, 'max_epochs', current_epoch))
        total_steps = max(1, (max(0, total_epochs - self.start_epoch) // self.interval))
        next_step = min(total_steps, self.prune_step + 1)
        ramp_steps = total_steps if self.prune_ramp_steps is None else max(1, min(total_steps, self.prune_ramp_steps))
        progress = min(1.0, float(next_step) / float(ramp_steps))

        mid_ratio = self.mid_final * progress
        high_ratio = self.high_final * progress
        new_extra = _pruned_hrnet_extra(
            self.base_extra,
            mid_ratio,
            high_ratio,
            round_to=self.round_to,
            prune_stages=self.prune_stages,
            mid_limit=self.mid_final,
            high_limit=self.high_final,
        )

        model = getattr(runner.model, 'module', runner.model)
        try:
            params_before = (
                int(sum(int(p.numel()) for p in model.student.parameters()))
                if hasattr(model, 'student') and hasattr(model.student, 'parameters')
                else None
            )
        except Exception:
            params_before = None
        if hasattr(model, 'prune_student_backbone_extra'):
            channel_map = None
            try:
                crit = str(self.importance_criterion).lower()
                if not (crit.startswith('pose') or crit.startswith('kpt')):
                    from experiments.DIST.distill_prune import _build_hrnet_channel_index_map

                    if hasattr(model, 'student') and hasattr(model.student, 'state_dict'):
                        channel_map = _build_hrnet_channel_index_map(
                            model.student.state_dict(),
                            new_extra,
                            criterion=self.importance_criterion,
                        )
            except Exception:
                channel_map = None
            model.prune_student_backbone_extra(
                new_extra,
                channel_map=channel_map,
                importance_criterion=self.importance_criterion,
            )
            model.set_prune_state(step=next_step, mid_ratio=mid_ratio, high_ratio=high_ratio)

        runner.optimizer = build_optimizers(runner.model, runner.cfg.optimizer)
        self.prune_step = next_step

        prune_rate = _estimate_prune_rate(self.base_extra, new_extra)
        try:
            params_after = (
                int(sum(int(p.numel()) for p in model.student.parameters()))
                if hasattr(model, 'student') and hasattr(model.student, 'parameters')
                else None
            )
        except Exception:
            params_after = None
        param_prune_rate = None
        if self.base_student_params is not None and params_after is not None and self.base_student_params > 0:
            param_prune_rate = max(0.0, min(1.0, 1.0 - (float(params_after) / float(self.base_student_params))))
        try:
            b3 = tuple(self.base_extra.get('stage3', {}).get('num_channels', ()))
            b4 = tuple(self.base_extra.get('stage4', {}).get('num_channels', ()))
            n3 = tuple(new_extra.get('stage3', {}).get('num_channels', ()))
            n4 = tuple(new_extra.get('stage4', {}).get('num_channels', ()))
            params_msg = ''
            if param_prune_rate is not None:
                params_msg = f', param_prune_rate={param_prune_rate:.4f}'
            runner.logger.info(
                f'Prune step {next_step}/{total_steps} (ramp={ramp_steps}) epoch {current_epoch}: '
                f'mid_ratio={mid_ratio:.4f}, high_ratio={high_ratio:.4f}, prune_rate={prune_rate:.4f}, '
                f'stage3 {b3}->{n3}, stage4 {b4}->{n4}{params_msg}'
            )
        except Exception:
            pass
        lb = {'prune_rate': prune_rate}
        if param_prune_rate is not None:
            lb['param_prune_rate'] = float(param_prune_rate)
        runner.log_buffer.update(lb)

        if runner.meta is None:
            runner.meta = {}
        runner.meta['hrnet_prune_state'] = dict(
            prune_step=self.prune_step,
            last_eval_ap=self.last_eval_ap,
            last_prune_eval_ap=self.last_prune_eval_ap,
            base_extra=copy.deepcopy(self.base_extra),
            student_backbone_extra=copy.deepcopy(new_extra),
            mid_ratio=float(mid_ratio),
            high_ratio=float(high_ratio),
            base_student_params=self.base_student_params,
            student_params=params_after,
        )
        runner.meta['hrnet_pruned_epoch'] = dict(
            epoch=current_epoch,
            ap_before=ap_before,
        )

        eval_hook = None
        for h in getattr(runner, 'hooks', []):
            if hasattr(h, '_do_evaluate'):
                eval_hook = h
                break
        if eval_hook is not None:
            try:
                eval_hook._do_evaluate(runner)
            except Exception:
                pass
            ap_after = _logbuffer_get_last(runner.log_buffer, 'AP')
            try:
                if ap_after is not None:
                    runner.logger.info(
                        f'Post-prune eval epoch {current_epoch}: '
                        f'AP={float(ap_after):.4f}, prune_step={self.prune_step}, prune_rate={prune_rate:.4f}'
                    )
            except Exception:
                pass

        if self.save_prune_ckpt:
            rank, _ = get_dist_info()
            if rank == 0:
                try:
                    filename_tmpl = f'prune_step_{self.prune_step}_epoch_{current_epoch}.pth'
                    runner.save_checkpoint(
                        runner.work_dir,
                        filename_tmpl=filename_tmpl,
                        create_symlink=False,
                    )
                except Exception:
                    pass


@HOOKS.register_module()
class HRNetPruneRecoverHook(Hook):
    def __init__(
        self,
        start_epoch=70,
        interval=10,
        force_prune_start_epoch=None,
        immediate_prune_at_start=False,
        prune_step_ratio=0.1,
        max_prune_ratio=0.5,
        max_prune_ratio_stage3=None,
        max_prune_ratio_stage4=None,
        post_switch_ratio=0.4,
        post_step_ratio=0.05,
        step_schedule=None,
        round_to=16,
        branch_weight_stage3=None,
        branch_weight_stage4=None,
        min_ap_drop=0.003,
        recover_margin=0.0,
        recover_drop_tolerance=0.003,
        recovery_schedule_base_ap=0.76,
        recovery_schedule_drop_per_10p=0.003,
        recovery_schedule_ratio_unit=0.1,
        use_ratio_based_recovery=True,
        force_prune_gap=0.01,
        force_prune_max_steps=6,
        target_ap=0.7573,
        force_prune=True,
        force_eval_if_missing_ap=True,
        importance_criterion='bn_gamma',
        save_prune_ckpt=True,
        prune_stages=(3, 4),
        prune_branches_stage3=(2,),
        prune_branches_stage4=(2, 3),
        prune_protected_layers=(),
    ):
        self.start_epoch = int(start_epoch)
        self.interval = int(interval)
        self.force_prune_start_epoch = None if force_prune_start_epoch is None else int(force_prune_start_epoch)
        self.immediate_prune_at_start = bool(immediate_prune_at_start)
        self.prune_step_ratio = float(prune_step_ratio)
        self.max_prune_ratio = float(max_prune_ratio)
        self.max_prune_ratio_stage3 = None
        self.max_prune_ratio_stage4 = None
        try:
            if isinstance(max_prune_ratio_stage3, dict) and max_prune_ratio_stage3:
                self.max_prune_ratio_stage3 = {int(k): float(v) for k, v in max_prune_ratio_stage3.items()}
        except Exception:
            self.max_prune_ratio_stage3 = None
        try:
            if isinstance(max_prune_ratio_stage4, dict) and max_prune_ratio_stage4:
                self.max_prune_ratio_stage4 = {int(k): float(v) for k, v in max_prune_ratio_stage4.items()}
        except Exception:
            self.max_prune_ratio_stage4 = None
        self.post_switch_ratio = float(post_switch_ratio)
        self.post_step_ratio = float(post_step_ratio)
        self.step_schedule = None
        if step_schedule is not None:
            try:
                ss = []
                for it in list(step_schedule):
                    if not isinstance(it, dict):
                        continue
                    until = it.get('until', None)
                    step = it.get('step', None)
                    if until is None or step is None:
                        continue
                    ss.append(dict(until=float(until), step=float(step)))
                ss = sorted(ss, key=lambda x: float(x['until']))
                if ss:
                    self.step_schedule = ss
            except Exception:
                self.step_schedule = None
        self.round_to = int(round_to)
        self.branch_weight_stage3 = None
        self.branch_weight_stage4 = None
        try:
            if isinstance(branch_weight_stage3, (list, tuple)) and branch_weight_stage3:
                self.branch_weight_stage3 = [float(x) for x in branch_weight_stage3]
        except Exception:
            self.branch_weight_stage3 = None
        try:
            if isinstance(branch_weight_stage4, (list, tuple)) and branch_weight_stage4:
                self.branch_weight_stage4 = [float(x) for x in branch_weight_stage4]
        except Exception:
            self.branch_weight_stage4 = None
        self.min_ap_drop = float(min_ap_drop)
        self.recover_margin = float(recover_margin)
        self.recover_drop_tolerance = float(recover_drop_tolerance)
        self.recovery_schedule_base_ap = float(recovery_schedule_base_ap)
        self.recovery_schedule_drop_per_10p = float(recovery_schedule_drop_per_10p)
        self.recovery_schedule_ratio_unit = float(recovery_schedule_ratio_unit)
        self.use_ratio_based_recovery = bool(use_ratio_based_recovery)
        self.force_prune_gap = float(force_prune_gap)
        self.force_prune_max_steps = int(force_prune_max_steps)
        self.target_ap = float(target_ap)
        self.force_prune = bool(force_prune)
        self.force_eval_if_missing_ap = bool(force_eval_if_missing_ap)
        self.importance_criterion = str(importance_criterion)
        self.save_prune_ckpt = bool(save_prune_ckpt)
        self.prune_stages = tuple(int(x) for x in prune_stages)
        self.prune_branches_stage3 = tuple(int(x) for x in prune_branches_stage3)
        self.prune_branches_stage4 = tuple(int(x) for x in prune_branches_stage4)
        self.prune_protected_layers = tuple(str(x) for x in prune_protected_layers)

        self.base_extra = None
        self.base_student_params = None
        self.prune_step = 0
        self.target_ratio = 0.0
        self.recovering = False
        self.recover_target_ap = None
        self._forced_start_prune_done = False

    def _apply_branch_max_prune_ratio(self, ex):
        if not isinstance(ex, dict) or not isinstance(self.base_extra, dict):
            return
        rt = max(1, int(self.round_to))

        def _round_up(x):
            x = int(x)
            return int(((x + rt - 1) // rt) * rt)

        def _clamp_stage(stage_name, max_ratio_map):
            if not isinstance(max_ratio_map, dict) or not max_ratio_map:
                return
            base_stage = self.base_extra.get(stage_name, {})
            cur_stage = ex.get(stage_name, {})
            base_nc = list(base_stage.get('num_channels', ()))
            cur_nc = list(cur_stage.get('num_channels', ()))
            if not base_nc or not cur_nc:
                return
            changed = False
            for b, mr in max_ratio_map.items():
                if not isinstance(b, int) or b < 0 or b >= len(cur_nc) or b >= len(base_nc):
                    continue
                try:
                    mr = max(0.0, min(1.0, float(mr)))
                except Exception:
                    continue
                base_c = int(base_nc[b])
                min_keep = int(round(float(base_c) * (1.0 - mr)))
                min_keep = max(rt, _round_up(min_keep))
                if int(cur_nc[b]) < int(min_keep):
                    cur_nc[b] = int(min_keep)
                    changed = True
            if changed:
                cur_stage = dict(cur_stage)
                cur_stage['num_channels'] = tuple(int(x) for x in cur_nc)
                ex[stage_name] = cur_stage

        if 3 in self.prune_stages:
            _clamp_stage('stage3', self.max_prune_ratio_stage3)
        if 4 in self.prune_stages:
            _clamp_stage('stage4', self.max_prune_ratio_stage4)

    def _pick_step_ratio(self, cur_ratio):
        r = max(0.0, min(1.0, float(cur_ratio)))
        if isinstance(self.step_schedule, list) and self.step_schedule:
            for it in self.step_schedule:
                try:
                    if r <= float(it['until']) + 1.0e-12:
                        return float(it['step'])
                except Exception:
                    continue
            try:
                return float(self.step_schedule[-1]['step'])
            except Exception:
                return float(self.post_step_ratio)
        return float(self.prune_step_ratio) if r < float(self.post_switch_ratio) - 1.0e-9 else float(self.post_step_ratio)

    def before_run(self, runner):
        model = getattr(runner.model, 'module', runner.model)
        student_cfg = getattr(model, 'student_cfg', None)
        if student_cfg is None:
            return
        self.base_extra = copy.deepcopy(student_cfg.get('backbone', {}).get('extra', {}))
        try:
            if hasattr(model, 'student') and hasattr(model.student, 'parameters'):
                self.base_student_params = int(sum(int(p.numel()) for p in model.student.parameters()))
        except Exception:
            self.base_student_params = None

        meta = runner.meta if runner.meta is not None else {}
        state = meta.get('hrnet_prune_state', None)
        if isinstance(state, dict):
            try:
                self.prune_step = int(state.get('prune_step', 0))
            except Exception:
                self.prune_step = 0
            try:
                self.target_ratio = float(state.get('target_ratio', 0.0))
            except Exception:
                self.target_ratio = 0.0
            self.recovering = bool(state.get('recovering', False))
            self.recover_target_ap = state.get('recover_target_ap', None)
            base_extra = state.get('base_extra', None)
            if isinstance(base_extra, dict):
                self.base_extra = copy.deepcopy(base_extra)
            base_params = state.get('base_student_params', None)
            if base_params is not None:
                try:
                    self.base_student_params = int(base_params)
                except Exception:
                    pass

        if (
            (not self._forced_start_prune_done)
            and (self.force_prune_start_epoch is not None)
            and (int(self.prune_step) <= 0)
            and (not bool(self.recovering))
        ):
            try:
                current_epoch = runner.epoch + 1
                if int(current_epoch) == int(self.force_prune_start_epoch):
                    self._forced_start_prune_done = True
                    runner.logger.info(
                        f'Force prune on resume start: epoch {current_epoch}, prune_step_ratio={self.prune_step_ratio:.4f}'
                    )
                    self.after_train_epoch(runner)
            except Exception:
                pass

    def after_train_epoch(self, runner):
        current_epoch = runner.epoch + 1
        if self.base_extra is None:
            return
        if current_epoch < self.start_epoch:
            return
        immediate_once = (
            bool(self.immediate_prune_at_start)
            and int(self.prune_step) <= 0
            and int(current_epoch) == int(self.start_epoch)
        )
        if (current_epoch % self.interval != 0) and (not immediate_once):
            return

        ap_now = _logbuffer_get_last(runner.log_buffer, 'AP')
        if ap_now is None:
            if self.force_eval_if_missing_ap:
                eval_hook = None
                for h in getattr(runner, 'hooks', []):
                    if hasattr(h, '_do_evaluate'):
                        eval_hook = h
                        break
                if eval_hook is not None:
                    try:
                        eval_hook._do_evaluate(runner)
                    except Exception:
                        pass
                ap_now = _logbuffer_get_last(runner.log_buffer, 'AP')
            if ap_now is None:
                try:
                    runner.logger.info(f'Prune skipped at epoch {current_epoch}: missing AP in log_buffer')
                except Exception:
                    pass
                return
        ap_now = float(ap_now)

        if self.recovering:
            target = self.recover_target_ap
            if target is None:
                self.recovering = False
            else:
                target = float(target)
                if ap_now + self.recover_margin >= target:
                    self.recovering = False
                    runner.logger.info(
                        f'Recovery reached at epoch {current_epoch}: AP={ap_now:.4f} >= target={target:.4f}'
                    )
                else:
                    runner.logger.info(
                        f'Recovery ongoing at epoch {current_epoch}: AP={ap_now:.4f} < target={target:.4f}, skip pruning'
                    )
                    if runner.meta is None:
                        runner.meta = {}
                    runner.meta['hrnet_prune_state'] = dict(
                        prune_step=self.prune_step,
                        recovering=True,
                        recover_target_ap=target,
                        base_extra=copy.deepcopy(self.base_extra),
                        base_student_params=self.base_student_params,
                    )
                    return

        ap_before = ap_now
        runner.logger.info(f'Pre-prune eval epoch {current_epoch}: AP={ap_before:.4f}')

        next_step = int(self.prune_step) + 1
        cur_ratio = float(self.target_ratio)
        step_ratio = self._pick_step_ratio(cur_ratio)
        target_ratio = min(self.max_prune_ratio, cur_ratio + float(step_ratio))
        target_ratio = max(0.0, min(1.0, float(target_ratio)))
        self.target_ratio = float(target_ratio)

        model = getattr(runner.model, 'module', runner.model)
        current_extra = None
        try:
            cur_cfg = getattr(model, 'student_cfg', None)
            if isinstance(cur_cfg, dict):
                current_extra = copy.deepcopy(cur_cfg.get('backbone', {}).get('extra', None))
        except Exception:
            current_extra = None
        student_cfg = getattr(model, 'student_cfg', None)
        achieved_param_prune = None
        if isinstance(student_cfg, dict):
            try:
                new_extra, achieved_param_prune = _search_extra_for_param_prune_target(
                    student_cfg,
                    self.base_extra,
                    self.base_student_params,
                    target_ratio,
                    round_to=self.round_to,
                    prune_stages=self.prune_stages,
                    prune_branches_stage3=self.prune_branches_stage3,
                    prune_branches_stage4=self.prune_branches_stage4,
                    branch_weight_stage3=self.branch_weight_stage3,
                    branch_weight_stage4=self.branch_weight_stage4,
                )
            except Exception:
                new_extra = _candidate_extra_for_target_ratio(
                    self.base_extra,
                    target_ratio,
                    round_to=self.round_to,
                    prune_stages=self.prune_stages,
                    prune_branches_stage3=self.prune_branches_stage3,
                    prune_branches_stage4=self.prune_branches_stage4,
                    branch_weight_stage3=self.branch_weight_stage3,
                    branch_weight_stage4=self.branch_weight_stage4,
                    alpha=1.0,
                )
        else:
            new_extra = _candidate_extra_for_target_ratio(
                self.base_extra,
                target_ratio,
                round_to=self.round_to,
                prune_stages=self.prune_stages,
                prune_branches_stage3=self.prune_branches_stage3,
                prune_branches_stage4=self.prune_branches_stage4,
                branch_weight_stage3=self.branch_weight_stage3,
                branch_weight_stage4=self.branch_weight_stage4,
                alpha=1.0,
            )

        if isinstance(current_extra, dict) and isinstance(new_extra, dict):
            try:
                stage3_new = new_extra.get('stage3', {})
                stage4_new = new_extra.get('stage4', {})
                stage3_cur = current_extra.get('stage3', {})
                stage4_cur = current_extra.get('stage4', {})
                s3_new = list(stage3_new.get('num_channels', ()))
                s4_new = list(stage4_new.get('num_channels', ()))
                s3_cur = list(stage3_cur.get('num_channels', ()))
                s4_cur = list(stage4_cur.get('num_channels', ()))
                monotonic_changed = False
                if 3 in self.prune_stages:
                    for i in range(min(len(s3_new), len(s3_cur))):
                        if i in self.prune_branches_stage3 and int(s3_new[i]) > int(s3_cur[i]):
                            s3_new[i] = int(s3_cur[i])
                            monotonic_changed = True
                if 4 in self.prune_stages:
                    for i in range(min(len(s4_new), len(s4_cur))):
                        if i in self.prune_branches_stage4 and int(s4_new[i]) > int(s4_cur[i]):
                            s4_new[i] = int(s4_cur[i])
                            monotonic_changed = True
                if monotonic_changed:
                    stage3_new['num_channels'] = tuple(int(x) for x in s3_new)
                    stage4_new['num_channels'] = tuple(int(x) for x in s4_new)
                    new_extra['stage3'] = stage3_new
                    new_extra['stage4'] = stage4_new
                    runner.logger.info(f'Clamp prune extra to be monotonic at epoch {current_epoch}')
            except Exception:
                pass

        if isinstance(new_extra, dict):
            try:
                self._apply_branch_max_prune_ratio(new_extra)
            except Exception:
                pass

        def _force_one_step(ex):
            stage3 = ex.get('stage3', {})
            stage4 = ex.get('stage4', {})
            s3 = list(stage3.get('num_channels', ()))
            s4 = list(stage4.get('num_channels', ()))
            for b in [3, 2]:
                if b in self.prune_branches_stage4 and b < len(s4):
                    v = int(s4[b])
                    nv = max(self.round_to, v - self.round_to)
                    if nv < v:
                        s4[b] = nv
                        stage3['num_channels'] = tuple(int(x) for x in s3)
                        stage4['num_channels'] = tuple(int(x) for x in s4)
                        ex['stage3'] = stage3
                        ex['stage4'] = stage4
                        return True
            for b in [2]:
                if b in self.prune_branches_stage3 and b < len(s3):
                    v = int(s3[b])
                    nv = max(self.round_to, v - self.round_to)
                    if nv < v:
                        s3[b] = nv
                        stage3['num_channels'] = tuple(int(x) for x in s3)
                        stage4['num_channels'] = tuple(int(x) for x in s4)
                        ex['stage3'] = stage3
                        ex['stage4'] = stage4
                        return True
            return False

        if self.force_prune and isinstance(current_extra, dict) and isinstance(new_extra, dict):
            try:
                cur_s3 = tuple(current_extra.get('stage3', {}).get('num_channels', ()))
                cur_s4 = tuple(current_extra.get('stage4', {}).get('num_channels', ()))
                new_s3 = tuple(new_extra.get('stage3', {}).get('num_channels', ()))
                new_s4 = tuple(new_extra.get('stage4', {}).get('num_channels', ()))
                forced_steps = 0
                if cur_s3 == new_s3 and cur_s4 == new_s4 and (target_ratio > 0):
                    if _force_one_step(new_extra):
                        forced_steps += 1
                        runner.logger.info(f'Force prune at epoch {current_epoch}: new_extra adjusted to ensure change')

                planned = None if achieved_param_prune is None else float(achieved_param_prune)
                if (
                    isinstance(student_cfg, dict)
                    and (self.base_student_params is not None)
                    and (self.base_student_params > 0)
                    and (target_ratio > 0)
                    and (planned is not None)
                ):
                    cfg_tmp = copy.deepcopy(student_cfg)
                    cfg_tmp['pretrained'] = None
                    cfg_tmp.setdefault('backbone', {})
                    while (
                        forced_steps < max(0, int(self.force_prune_max_steps))
                        and planned + 1.0e-12 < float(target_ratio) - float(self.force_prune_gap)
                    ):
                        if not _force_one_step(new_extra):
                            break
                        forced_steps += 1
                        cfg_tmp['backbone']['extra'] = copy.deepcopy(new_extra)
                        p = _count_model_params_from_cfg(cfg_tmp)
                        planned = max(0.0, min(1.0, 1.0 - (float(p) / float(self.base_student_params))))
                    if forced_steps > 0:
                        achieved_param_prune = float(planned)
                        runner.logger.info(
                            f'Force prune extra steps at epoch {current_epoch}: forced_steps={forced_steps}, planned_param_prune_rate={float(planned):.4f}, target_ratio={float(target_ratio):.4f}'
                        )
            except Exception:
                pass

        if hasattr(model, 'prune_student_backbone_extra'):
            channel_map = None
            try:
                crit = str(self.importance_criterion).lower()
                if not (crit.startswith('pose') or crit.startswith('kpt')):
                    from experiments.DIST.distill_prune import _build_hrnet_channel_index_map

                    if hasattr(model, 'student') and hasattr(model.student, 'state_dict'):
                        channel_map = _build_hrnet_channel_index_map(
                            model.student.state_dict(),
                            new_extra,
                            criterion=self.importance_criterion,
                        )
            except Exception:
                channel_map = None

            model.prune_student_backbone_extra(
                new_extra,
                channel_map=channel_map,
                importance_criterion=self.importance_criterion,
            )
            model.set_prune_state(step=next_step, mid_ratio=target_ratio, high_ratio=target_ratio)

        runner.optimizer = build_optimizers(runner.model, runner.cfg.optimizer)
        self.prune_step = next_step

        prune_rate = _estimate_prune_rate(self.base_extra, new_extra)
        params_after = None
        try:
            if hasattr(model, 'student') and hasattr(model.student, 'parameters'):
                params_after = int(sum(int(p.numel()) for p in model.student.parameters()))
        except Exception:
            params_after = None

        param_prune_rate = None
        if self.base_student_params is not None and params_after is not None and self.base_student_params > 0:
            param_prune_rate = max(0.0, min(1.0, 1.0 - (float(params_after) / float(self.base_student_params))))

        try:
            b3 = tuple(self.base_extra.get('stage3', {}).get('num_channels', ()))
            b4 = tuple(self.base_extra.get('stage4', {}).get('num_channels', ()))
            n3 = tuple(new_extra.get('stage3', {}).get('num_channels', ()))
            n4 = tuple(new_extra.get('stage4', {}).get('num_channels', ()))
            msg = (
                f'Prune step {self.prune_step} target_ratio={target_ratio:.4f} epoch {current_epoch}: '
                f'prune_rate={prune_rate:.4f}, stage3 {b3}->{n3}, stage4 {b4}->{n4}'
            )
            if param_prune_rate is not None:
                msg += f', param_prune_rate={param_prune_rate:.4f}'
            if achieved_param_prune is not None:
                msg += f', planned_param_prune_rate={float(achieved_param_prune):.4f}'
            runner.logger.info(msg)
        except Exception:
            pass

        lb = {'prune_rate': prune_rate}
        if param_prune_rate is not None:
            lb['param_prune_rate'] = float(param_prune_rate)
        runner.log_buffer.update(lb)

        eval_hook = None
        for h in getattr(runner, 'hooks', []):
            if hasattr(h, '_do_evaluate'):
                eval_hook = h
                break
        if eval_hook is not None:
            try:
                eval_hook._do_evaluate(runner)
            except Exception:
                pass

        ap_after = _logbuffer_get_last(runner.log_buffer, 'AP')
        if ap_after is not None:
            ap_after = float(ap_after)
            runner.logger.info(
                f'Post-prune eval epoch {current_epoch}: AP={ap_after:.4f}, prune_step={self.prune_step}, prune_rate={prune_rate:.4f}'
            )
            drop = ap_before - ap_after
            if drop > self.min_ap_drop:
                self.recovering = True
                relative_target = float(ap_before) - float(self.recover_drop_tolerance)
                scheduled_target = float(self.target_ap)
                if self.use_ratio_based_recovery and self.recovery_schedule_ratio_unit > 0:
                    scheduled_target = float(self.recovery_schedule_base_ap) - float(self.recovery_schedule_drop_per_10p) * (
                        float(target_ratio) / float(self.recovery_schedule_ratio_unit)
                    )
                self.recover_target_ap = max(0.0, min(relative_target, scheduled_target))
                runner.logger.info(
                    f'Enter recovery mode: drop={drop:.4f} > {self.min_ap_drop:.4f}. '
                    f'No further pruning until AP >= {float(self.recover_target_ap):.4f} '
                    f'(relative={relative_target:.4f}, scheduled={scheduled_target:.4f}, target_ratio={float(target_ratio):.4f})'
                )

        if runner.meta is None:
            runner.meta = {}
        runner.meta['hrnet_prune_state'] = dict(
            prune_step=self.prune_step,
            recovering=bool(self.recovering),
            recover_target_ap=self.recover_target_ap,
            base_extra=copy.deepcopy(self.base_extra),
            student_backbone_extra=copy.deepcopy(new_extra),
            target_ratio=float(target_ratio),
            base_student_params=self.base_student_params,
            student_params=params_after,
        )
        runner.meta['hrnet_pruned_epoch'] = dict(epoch=current_epoch, ap_before=ap_before)

        if self.save_prune_ckpt:
            rank, _ = get_dist_info()
            if rank == 0:
                try:
                    filename_tmpl = f'prune_step_{self.prune_step}_epoch_{current_epoch}.pth'
                    runner.save_checkpoint(runner.work_dir, filename_tmpl=filename_tmpl, create_symlink=False)
                except Exception:
                    pass


@HOOKS.register_module()
class PruneAPDropStopHook(Hook):
    def __init__(self, min_ap_drop=0.003, interval=10, monitor='AP'):
        self.min_ap_drop = float(min_ap_drop)
        self.interval = int(interval)
        self.monitor = monitor
        self.last_prune_eval_ap = None
        self.best_ap = None

    def after_train_epoch(self, runner):
        current_epoch = runner.epoch + 1
        if current_epoch % self.interval != 0:
            return
        if runner.meta is None:
            return
        pruned = runner.meta.get('hrnet_pruned_epoch', None)
        if not isinstance(pruned, dict) or pruned.get('epoch') != current_epoch:
            return
        ap = _logbuffer_get_last(runner.log_buffer, self.monitor)
        if ap is None:
            return
        ap = float(ap)
        before = pruned.get('ap_before', None)
        if before is not None:
            try:
                if (float(before) - ap) > self.min_ap_drop:
                    runner._stop = True
            except Exception:
                pass
        self.last_prune_eval_ap = ap
        if self.best_ap is None:
            self.best_ap = ap
        else:
            self.best_ap = max(self.best_ap, ap)
        runner.meta.pop('hrnet_pruned_epoch', None)
