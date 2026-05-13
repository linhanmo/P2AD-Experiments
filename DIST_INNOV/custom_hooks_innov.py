import copy

from mmcv.runner import HOOKS, Hook


def _safe_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


@HOOKS.register_module()
class DistillScheduleStateHook(Hook):
    def __init__(self, prune_ratio_key='param_prune_rate', fallback_to_meta=True, try_estimate_from_model=True):
        self.prune_ratio_key = str(prune_ratio_key)
        self.fallback_to_meta = bool(fallback_to_meta)
        self.try_estimate_from_model = bool(try_estimate_from_model)

    def _get_prune_ratio(self, runner):
        try:
            out = getattr(runner.log_buffer, 'output', None)
            if isinstance(out, dict):
                v = out.get(self.prune_ratio_key, None)
                r = _safe_float(v, None)
                if r is not None:
                    return r
                v2 = out.get('prune_rate', None)
                r2 = _safe_float(v2, None)
                if r2 is not None:
                    return r2
        except Exception:
            pass

        if self.fallback_to_meta:
            try:
                meta = getattr(runner, 'meta', None)
                if isinstance(meta, dict) and 'hrnet_prune_state' in meta:
                    st = meta['hrnet_prune_state']
                    if isinstance(st, dict):
                        for k in ['param_prune_rate', 'prune_rate', 'target_ratio', 'mid_ratio', 'high_ratio']:
                            r = _safe_float(st.get(k, None), None)
                            if r is not None:
                                return r
            except Exception:
                pass

        if self.try_estimate_from_model:
            try:
                model = getattr(runner.model, 'module', runner.model)
                base_params = None
                meta = getattr(runner, 'meta', None)
                if isinstance(meta, dict) and 'hrnet_prune_state' in meta:
                    st = meta['hrnet_prune_state']
                    if isinstance(st, dict):
                        base_params = st.get('base_student_params', None)
                if base_params is None and hasattr(model, 'get_prune_state'):
                    st2 = model.get_prune_state()
                    if isinstance(st2, dict):
                        base_params = st2.get('base_student_params', None)
                if base_params is None:
                    return None
                base_params = int(base_params)
                if base_params <= 0 or (not hasattr(model, 'student')):
                    return None
                cur_params = int(sum(int(p.numel()) for p in model.student.parameters()))
                pr = max(0.0, min(1.0, 1.0 - (float(cur_params) / float(base_params))))
                return float(pr)
            except Exception:
                pass
        return None

    def _set_state(self, runner):
        model = getattr(runner.model, 'module', runner.model)
        if not hasattr(model, 'set_prune_state'):
            return
        state = dict(epoch=int(getattr(runner, 'epoch', 0)), max_epochs=int(getattr(runner, 'max_epochs', 1)))
        pr = self._get_prune_ratio(runner)
        if pr is not None:
            state[self.prune_ratio_key] = float(pr)
            state['prune_rate'] = float(pr)
        model.set_prune_state(**state)

    def before_train_epoch(self, runner):
        self._set_state(runner)

    def after_train_epoch(self, runner):
        self._set_state(runner)
