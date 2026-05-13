import argparse
import csv
import re


_RE_PRE = re.compile(r'Pre-prune eval epoch (?P<epoch>\d+): AP=(?P<ap>\d+\.\d+)')
_RE_PRUNE = re.compile(
    r'Prune step (?P<step>\d+)(?:/\d+)?(?: target_ratio=(?P<target>\d+\.\d+))? epoch (?P<epoch>\d+): '
    r'prune_rate=(?P<pr>\d+\.\d+), .*?param_prune_rate=(?P<ppr>\d+\.\d+)(?:, planned_param_prune_rate=(?P<ppp>\d+\.\d+))?'
)
_RE_POST = re.compile(
    r'Post-prune eval epoch (?P<epoch>\d+): AP=(?P<ap>\d+\.\d+), prune_step=(?P<step>\d+), prune_rate=(?P<pr>\d+\.\d+)'
)


def _f(x):
    try:
        return float(x)
    except Exception:
        return None


def parse_events(lines):
    events = []
    for ln, s in enumerate(lines, start=1):
        m = _RE_PRE.search(s)
        if m:
            events.append(
                dict(
                    type='pre',
                    line=ln,
                    epoch=int(m.group('epoch')),
                    ap=_f(m.group('ap')),
                )
            )
            continue
        m = _RE_PRUNE.search(s)
        if m:
            events.append(
                dict(
                    type='prune',
                    line=ln,
                    step=int(m.group('step')),
                    epoch=int(m.group('epoch')),
                    target=_f(m.group('target')),
                    prune_rate=_f(m.group('pr')),
                    param_prune_rate=_f(m.group('ppr')),
                    planned_param_prune_rate=_f(m.group('ppp')),
                )
            )
            continue
        m = _RE_POST.search(s)
        if m:
            events.append(
                dict(
                    type='post',
                    line=ln,
                    step=int(m.group('step')),
                    epoch=int(m.group('epoch')),
                    ap=_f(m.group('ap')),
                    prune_rate=_f(m.group('pr')),
                )
            )
            continue
    return events


def build_curve(events):
    by_step = {}
    ordered = []
    for e in events:
        if e['type'] == 'prune':
            step = e['step']
            rec = by_step.get(step)
            if rec is None:
                rec = dict(step=step)
                by_step[step] = rec
                ordered.append(rec)
            rec['epoch_prune'] = e['epoch']
            rec['target_ratio'] = e.get('target')
            rec['prune_rate'] = e.get('prune_rate')
            rec['param_prune_rate'] = e.get('param_prune_rate')
            rec['planned_param_prune_rate'] = e.get('planned_param_prune_rate')
            rec['line_prune'] = e['line']
        elif e['type'] == 'post':
            step = e['step']
            rec = by_step.get(step)
            if rec is None:
                rec = dict(step=step)
                by_step[step] = rec
                ordered.append(rec)
            rec['epoch_post'] = e['epoch']
            rec['ap_post'] = e['ap']
            rec['prune_rate_post'] = e.get('prune_rate')
            rec['line_post'] = e['line']

    for i, rec in enumerate(ordered):
        end_line = None
        if i + 1 < len(ordered):
            end_line = ordered[i + 1].get('line_prune')
        post_line = rec.get('line_post')
        if post_line is None:
            continue
        recovered = None
        recovered_epoch = None
        for e in events:
            if e['type'] != 'pre':
                continue
            if e['line'] <= post_line:
                continue
            if end_line is not None and e['line'] >= end_line:
                break
            recovered = e['ap']
            recovered_epoch = e['epoch']
            break
        rec['epoch_recovered'] = recovered_epoch
        rec['ap_recovered'] = recovered
    return ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.log, 'r', encoding='utf-8', errors='ignore') as f:
        events = parse_events(f)
    curve = build_curve(events)

    fieldnames = [
        'step',
        'target_ratio',
        'epoch_prune',
        'epoch_post',
        'epoch_recovered',
        'prune_rate',
        'param_prune_rate',
        'planned_param_prune_rate',
        'ap_post',
        'ap_recovered',
    ]
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in curve:
            w.writerow({k: r.get(k, None) for k in fieldnames})


if __name__ == '__main__':
    main()

