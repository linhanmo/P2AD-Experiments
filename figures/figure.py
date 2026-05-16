import csv
import os
import re
from dataclasses import dataclass
import logging
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    from matplotlib import font_manager as _fm
except Exception:
    _fm = None


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    label: str


def _en_label(name: str) -> str:
    s = (name or '').strip()
    if not s:
        return s
    mapping = {
        'HRNet-W48原论文': 'HRNet-W48 (Original)',
        'HRNet-W48微调后': 'HRNet-W48 (Fine-Tuning)',
        'HRNet-W48(PTQ-INT8)': 'HRNet-W48 (Fine-Tuning+PTQ-INT8)',
        'HRNet-W48(QAT-INT8)': 'HRNet-W48 (Fine-Tuning+QAT-INT8)',
        '完整框架严格无损': 'Ours (lossless)',
        '完整框架可接受损失': 'Ours (slight loss)',
        'SimpleBaseline (ResNet‑50)': 'ResNet-50',
        'SimpleBaseline (ResNet-50)': 'ResNet-50',
        'EEffPose‑P2 (EfficientNet)': 'EEffPose-P2',
        'EEffPose-P2 (EfficientNet)': 'EEffPose-P2',
        'HRNet‑W32': 'HRNet-W32',
        'HRNet-W32': 'HRNet-W32',
        'Lite‑HRNet‑30': 'Lite-HRNet-30',
        'Lite-HRNet-30': 'Lite-HRNet-30',
    }
    return mapping.get(s, s)


def _category(label: str) -> str:
    s = (label or '').strip()
    if not s:
        return 'General'
    if s.startswith('Ours'):
        return 'Ours'
    if 'HRNet' in s:
        return 'HRNet'
    if ('Lite' in s) or ('Mobile' in s) or ('Efficient' in s) or ('EEffPose' in s):
        return 'Lightweight'
    if ('ResNet' in s) or ('ResNeXt' in s) or ('SimpleBaseline' in s):
        return 'General'
    return 'General'


def _configure_matplotlib_for_paper():
    if plt is None or _fm is None:
        return
    try:
        plt.set_loglevel('warning')
    except Exception:
        pass
    try:
        # Avoid noisy font fallback logs.
        try:
            logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
        except Exception:
            pass

        preferred = [
            'Times New Roman',
            'Times',
            'Nimbus Roman',
            'TeX Gyre Termes',
            'Liberation Serif',
            'DejaVu Serif',
        ]
        available = {f.name for f in _fm.fontManager.ttflist}
        chosen = None
        for p in preferred:
            if p in available:
                chosen = p
                break
        if chosen is None:
            chosen = 'DejaVu Serif'

        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = [chosen]
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except Exception:
            pass
    except Exception:
        return


def _to_float(x) -> float:
    if x is None:
        raise ValueError('None')
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        raise ValueError('empty')
    s = s.replace('%', '').replace('×', 'x')
    s = re.sub(r'\s+', '', s)
    return float(s)


def _read_status_csv(path: str) -> List[Dict[str, str]]:
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        return [row for row in reader if any((v or '').strip() for v in row.values())]


def _series_from_status(rows: List[Dict[str, str]]) -> Dict[str, List[Point]]:
    by_method: Dict[str, List[Tuple[float, float]]] = {}
    ap_baseline = None
    for r in rows:
        method = (r.get('方法') or '').strip()
        pr = r.get('实际剪枝率PR(%)')
        ap = r.get('AP(%)')
        if not method or pr is None or ap is None:
            continue
        try:
            pr_f = _to_float(pr)
            ap_f = _to_float(ap)
        except Exception:
            continue
        by_method.setdefault(method, []).append((pr_f, ap_f))
        if method == 'HRNet-W48微调后' and abs(pr_f) < 1.0e-9:
            ap_baseline = ap_f

    if ap_baseline is None:
        for method, pts in by_method.items():
            if method.startswith('HRNet-W48') and any(abs(pr) < 1.0e-9 for pr, _ in pts):
                ap_baseline = next(ap for pr, ap in pts if abs(pr) < 1.0e-9)
                break
    if ap_baseline is None:
        ap_baseline = 0.0

    def build(name: str, method_key: str) -> List[Point]:
        pts = [(0.0, ap_baseline)]
        pts.extend(by_method.get(method_key, []))
        pts = sorted(set(pts), key=lambda t: t[0])
        return [Point(x=p, y=a, label=name) for p, a in pts]

    return {
        'v1': build('v1 BN-scale pruning', 'BN层缩放因子'),
        'v2': build('v2 Branch-wise pruning', '分支差异化剪枝策略'),
        'v3': build('v3 Pose-aware pruning', '姿态感知剪枝'),
        'v4': build('v4 Full framework', '完整框架'),
    }


def _nice_ticks(vmin: float, vmax: float, n: int = 6) -> List[float]:
    if vmax <= vmin:
        return [vmin]
    span = vmax - vmin
    raw = span / max(1, n - 1)
    base = 10 ** (int(round((len(str(int(raw))) - 1))) if raw >= 1 else -1)
    step = raw
    for m in [1, 2, 2.5, 5, 10]:
        cand = m * base
        if cand >= raw:
            step = cand
            break
    start = step * (vmin // step)
    if start > vmin:
        start -= step
    ticks = []
    x = start
    for _ in range(0, n + 6):
        if x >= vmin - 1.0e-9 and x <= vmax + 1.0e-9:
            ticks.append(float(x))
        x += step
    if not ticks:
        ticks = [vmin, vmax]
    return ticks


def _svg_escape(s: str) -> str:
    return (
        str(s)
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
        .replace("'", '&apos;')
    )


def _write_svg(path: str, width: int, height: int, elements: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    content = '\n'.join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
            *elements,
            '</svg>',
        ]
    )
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def _linear_map(v: float, vmin: float, vmax: float, pmin: float, pmax: float) -> float:
    if vmax <= vmin:
        return (pmin + pmax) * 0.5
    t = (v - vmin) / (vmax - vmin)
    return pmin + t * (pmax - pmin)


def _svg_marker(cat: str, cx: float, cy: float, color: str, r: float, opacity: float = 0.8) -> str:
    if cat == 'Ours':
        w = r * 2.0
        return f'<rect x="{cx - r:.2f}" y="{cy - r:.2f}" width="{w:.2f}" height="{w:.2f}" fill="{color}" opacity="{opacity:.3f}"/>'
    if cat == 'HRNet':
        pts = f'{cx:.2f},{cy - r:.2f} {cx - r:.2f},{cy + r:.2f} {cx + r:.2f},{cy + r:.2f}'
        return f'<polygon points="{pts}" fill="{color}" opacity="{opacity:.3f}"/>'
    return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{color}" opacity="{opacity:.3f}"/>'


def _svg_label_pos(label: str) -> Tuple[float, float, str]:
    if (label or '').strip() == 'Ours (slight loss)':
        return 0.0, 12.0, 'middle'
    if (label or '').strip() == 'HRNet-W48 (Original)':
        return 10.0, 12.0, 'start'
    if (label or '').strip() == 'HRNet-W48 (Fine-Tuning+PTQ-INT8)':
        return 10.0, 4.0, 'start'
    if (label or '').strip() == 'HRNet-W48 (Fine-Tuning+QAT-INT8)':
        return 10.0, 0.0, 'start'
    return 7.0, -7.0, 'start'


def _plot_pr_ap_curve_svg(series: Dict[str, List[Point]], out_dir: str) -> str:
    width, height = 980, 640
    ml, mr, mt, mb = 90, 30, 60, 80
    x0, x1 = ml, width - mr
    y0, y1 = height - mb, mt

    all_pts = [p for pts in series.values() for p in pts]
    xs = [p.x for p in all_pts] or [0.0, 1.0]
    ys = [p.y for p in all_pts] or [0.0, 1.0]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xmax = max(xmax, 1.0)
    pad_y = max(0.5, (ymax - ymin) * 0.12)
    ymin -= pad_y
    ymax += pad_y

    xt = _nice_ticks(xmin, xmax, n=6)
    yt = _nice_ticks(ymin, ymax, n=6)

    elements: List[str] = []
    elements.append(f'<text x="{width/2:.1f}" y="34" text-anchor="middle" font-size="18" font-family="Times New Roman, Times, serif">{_svg_escape("AP vs. Pruning Ratio (HRNet-W48)")}</text>')
    elements.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#222" stroke-width="1.2"/>')
    elements.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#222" stroke-width="1.2"/>')

    for v in xt:
        px = _linear_map(v, xmin, xmax, x0, x1)
        elements.append(f'<line x1="{px:.2f}" y1="{y0}" x2="{px:.2f}" y2="{y0+6}" stroke="#222" stroke-width="1"/>')
        elements.append(f'<line x1="{px:.2f}" y1="{y0}" x2="{px:.2f}" y2="{y1}" stroke="#999" stroke-width="0.8" opacity="0.35"/>')
        elements.append(f'<text x="{px:.2f}" y="{y0+22}" text-anchor="middle" font-size="12" font-family="Times New Roman, Times, serif">{v:g}</text>')

    for v in yt:
        py = _linear_map(v, ymin, ymax, y0, y1)
        elements.append(f'<line x1="{x0-6}" y1="{py:.2f}" x2="{x0}" y2="{py:.2f}" stroke="#222" stroke-width="1"/>')
        elements.append(f'<line x1="{x0}" y1="{py:.2f}" x2="{x1}" y2="{py:.2f}" stroke="#999" stroke-width="0.8" opacity="0.35"/>')
        elements.append(f'<text x="{x0-10}" y="{py+4:.2f}" text-anchor="end" font-size="12" font-family="Times New Roman, Times, serif">{v:g}</text>')

    elements.append(f'<text x="{(x0+x1)/2:.1f}" y="{height-22}" text-anchor="middle" font-size="14" font-family="Times New Roman, Times, serif">Pruning ratio PR (%)</text>')
    elements.append(f'<text x="22" y="{(y0+y1)/2:.1f}" text-anchor="middle" font-size="14" font-family="Times New Roman, Times, serif" transform="rotate(-90 22 {(y0+y1)/2:.1f})">AP (%)</text>')

    colors = {'v1': '#1f77b4', 'v2': '#ff7f0e', 'v3': '#2ca02c', 'v4': '#d62728'}
    markers = {'v1': 'circle', 'v2': 'rect', 'v3': 'tri', 'v4': 'diamond'}

    legend_x = x0 + 18
    legend_y = y1 + 12
    legend_gap = 20
    li = 0

    for key in ['v1', 'v2', 'v3', 'v4']:
        pts = series.get(key, [])
        if not pts:
            continue
        color = colors.get(key, '#000')
        path_pts = []
        for p in pts:
            px = _linear_map(p.x, xmin, xmax, x0, x1)
            py = _linear_map(p.y, ymin, ymax, y0, y1)
            path_pts.append((px, py))
        d = ' '.join([('M' if i == 0 else 'L') + f'{px:.2f},{py:.2f}' for i, (px, py) in enumerate(path_pts)])
        elements.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="3" opacity="0.95"/>')
        for (px, py) in path_pts:
            mk = markers.get(key, 'circle')
            if mk == 'circle':
                elements.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4.6" fill="{color}" opacity="0.95"/>')
            elif mk == 'rect':
                elements.append(f'<rect x="{px-4.2:.2f}" y="{py-4.2:.2f}" width="8.4" height="8.4" fill="{color}" opacity="0.95"/>')
            elif mk == 'tri':
                points = f'{px:.2f},{py-5.2:.2f} {px-4.8:.2f},{py+4.0:.2f} {px+4.8:.2f},{py+4.0:.2f}'
                elements.append(f'<polygon points="{points}" fill="{color}" opacity="0.95"/>')
            else:
                points = f'{px:.2f},{py-5.4:.2f} {px-5.0:.2f},{py:.2f} {px:.2f},{py+5.4:.2f} {px+5.0:.2f},{py:.2f}'
                elements.append(f'<polygon points="{points}" fill="{color}" opacity="0.95"/>')

        ly = legend_y + li * legend_gap
        elements.append(f'<rect x="{legend_x-6}" y="{ly-12}" width="240" height="18" fill="white" opacity="0.85"/>')
        elements.append(f'<line x1="{legend_x}" y1="{ly-3}" x2="{legend_x+20}" y2="{ly-3}" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<circle cx="{legend_x+10}" cy="{ly-3}" r="4.2" fill="{color}"/>')
        elements.append(f'<text x="{legend_x+28}" y="{ly}" font-size="12.5" font-family="Times New Roman, Times, serif">{_svg_escape(pts[0].label)}</text>')
        li += 1

    out_svg = os.path.join(out_dir, 'pr_ap_curve.svg')
    _write_svg(out_svg, width, height, elements)
    return out_svg


def plot_pr_ap_curve(status_csv: str, out_dir: str) -> str:
    rows = _read_status_csv(status_csv)
    series = _series_from_status(rows)

    if plt is None:
        return _plot_pr_ap_curve_svg(series, out_dir)

    _configure_matplotlib_for_paper()

    fig = plt.figure(figsize=(7.0, 4.6), dpi=160)
    ax = fig.add_subplot(1, 1, 1)

    colors = {'v1': '#1f77b4', 'v2': '#ff7f0e', 'v3': '#2ca02c', 'v4': '#d62728'}
    markers = {'v1': 'o', 'v2': 's', 'v3': '^', 'v4': 'D'}

    for k in ['v1', 'v2', 'v3', 'v4']:
        pts = series.get(k, [])
        if not pts:
            continue
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        ax.plot(
            xs,
            ys,
            color=colors.get(k, None),
            marker=markers.get(k, 'o'),
            linewidth=2.0,
            markersize=5.5,
            label=pts[0].label,
        )

    ax.set_xlabel('Pruning ratio PR (%)')
    ax.set_ylabel('AP (%)')
    ax.set_title('AP vs. Pruning Ratio (HRNet-W48)')
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.55)
    ax.legend(loc='best', frameon=True)

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, 'pr_ap_curve.png')
    out_pdf = os.path.join(out_dir, 'pr_ap_curve.pdf')
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    return out_png


def _read_comparison_csv(path: str) -> List[Dict[str, str]]:
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        return [row for row in reader if any((v or '').strip() for v in row.values())]


def _pareto_frontier(points: List[Point]) -> List[Point]:
    pts = sorted(points, key=lambda p: (p.x, -p.y))
    frontier = []
    best_y = float('-inf')
    for p in pts:
        if p.y > best_y:
            frontier.append(p)
            best_y = p.y
    return frontier


def _plot_ap_gflops_pareto_svg(points: List[Point], frontier: List[Point], out_dir: str) -> str:
    width, height = 1040, 660
    ml, mr, mt, mb = 90, 30, 60, 80
    x0, x1 = ml, width - mr
    y0, y1 = height - mb, mt

    xs = [p.x for p in points] or [0.0, 1.0]
    ys = [p.y for p in points] or [0.0, 1.0]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xmax = max(xmax, xmin + 1.0)
    pad_y = max(0.5, (ymax - ymin) * 0.12)
    ymin -= pad_y
    ymax += pad_y

    xt = _nice_ticks(xmin, xmax, n=6)
    yt = _nice_ticks(ymin, ymax, n=6)

    elements: List[str] = []
    elements.append(f'<text x="{width/2:.1f}" y="34" text-anchor="middle" font-size="18" font-family="Times New Roman, Times, serif">{_svg_escape("AP vs. GFLOPs")}</text>')
    elements.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#222" stroke-width="1.2"/>')
    elements.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#222" stroke-width="1.2"/>')

    for v in xt:
        px = _linear_map(v, xmin, xmax, x0, x1)
        elements.append(f'<line x1="{px:.2f}" y1="{y0}" x2="{px:.2f}" y2="{y0+6}" stroke="#222" stroke-width="1"/>')
        elements.append(f'<line x1="{px:.2f}" y1="{y0}" x2="{px:.2f}" y2="{y1}" stroke="#999" stroke-width="0.8" opacity="0.35"/>')
        elements.append(f'<text x="{px:.2f}" y="{y0+22}" text-anchor="middle" font-size="12" font-family="Times New Roman, Times, serif">{v:g}</text>')

    for v in yt:
        py = _linear_map(v, ymin, ymax, y0, y1)
        elements.append(f'<line x1="{x0-6}" y1="{py:.2f}" x2="{x0}" y2="{py:.2f}" stroke="#222" stroke-width="1"/>')
        elements.append(f'<line x1="{x0}" y1="{py:.2f}" x2="{x1}" y2="{py:.2f}" stroke="#999" stroke-width="0.8" opacity="0.35"/>')
        elements.append(f'<text x="{x0-10}" y="{py+4:.2f}" text-anchor="end" font-size="12" font-family="Times New Roman, Times, serif">{v:g}</text>')

    elements.append(f'<text x="{(x0+x1)/2:.1f}" y="{height-22}" text-anchor="middle" font-size="14" font-family="Times New Roman, Times, serif">{_svg_escape("GFLOPs (lower is better)")}</text>')
    elements.append(f'<text x="22" y="{(y0+y1)/2:.1f}" text-anchor="middle" font-size="14" font-family="Times New Roman, Times, serif" transform="rotate(-90 22 {(y0+y1)/2:.1f})">{_svg_escape("AP (%) (higher is better)")}</text>')

    palette = {
        'HRNet': '#4c78a8',
        'Ours': '#e45756',
        'Lightweight': '#54a24b',
        'General': '#9d9d9d',
    }
    marker_r = {'HRNet': 5.4, 'Ours': 5.8, 'Lightweight': 5.1, 'General': 4.9}

    label_set = set()
    for p in points:
        c = _category(p.label)
        if (c in ['HRNet', 'Ours', 'Lightweight', 'General']) or (p.label in label_set):
            label_set.add(p.label)

    for p in points:
        label = _en_label(p.label)
        cat = _category(p.label)
        color = palette.get(cat, palette['General'])
        r = marker_r.get(cat, 4.0)
        px = _linear_map(p.x, xmin, xmax, x0, x1)
        py = _linear_map(p.y, ymin, ymax, y0, y1)
        elements.append(_svg_marker(cat, px, py, color, r, opacity=0.78))
        if p.label in label_set:
            dx, dy, anchor = _svg_label_pos(label)
            elements.append(
                f'<text x="{px+dx:.2f}" y="{py+dy:.2f}" text-anchor="{anchor}" font-size="10.5" font-family="Times New Roman, Times, serif" fill="#111">{_svg_escape(label)}</text>'
            )

    lx = x0 + 18
    ly = y1 + 8
    legend = [
        ('HRNet', palette['HRNet']),
        ('Ours', palette['Ours']),
        ('Lightweight', palette['Lightweight']),
        ('General Backbone', palette['General']),
    ]
    for i, (name, color) in enumerate(legend):
        yy = ly + i * 18
        elements.append(f'<rect x="{lx-8}" y="{yy-12}" width="220" height="18" fill="white" opacity="0.82"/>')
        key = 'General' if name == 'General Backbone' else name
        elements.append(_svg_marker(key, lx, yy - 4, color, 5.2, opacity=0.9))
        elements.append(f'<text x="{lx+12}" y="{yy}" font-size="12" font-family="Times New Roman, Times, serif">{_svg_escape(name)}</text>')

    out_svg = os.path.join(out_dir, 'ap_gflops_pareto.svg')
    _write_svg(out_svg, width, height, elements)
    return out_svg


def plot_ap_gflops_pareto(comparison_csv: str, out_dir: str) -> str:
    rows = _read_comparison_csv(comparison_csv)
    pts: List[Point] = []
    for r in rows:
        name = (r.get('模型') or '').strip()
        gflops = r.get('GFLOPs')
        ap = r.get('AP (%)')
        if not name or gflops is None or ap is None:
            continue
        try:
            x = _to_float(gflops)
            y = _to_float(ap)
        except Exception:
            continue
        pts.append(Point(x=x, y=y, label=_en_label(name)))

    frontier = _pareto_frontier(pts)

    if plt is None:
        return _plot_ap_gflops_pareto_svg(pts, frontier, out_dir)

    _configure_matplotlib_for_paper()

    fig = plt.figure(figsize=(7.4, 5.0), dpi=220)
    ax = fig.add_subplot(1, 1, 1)

    palette = {
        'HRNet': '#4c78a8',
        'Ours': '#e45756',
        'Lightweight': '#54a24b',
        'General': '#9d9d9d',
    }
    markers = {'HRNet': '^', 'Ours': 's', 'Lightweight': 'o', 'General': 'o'}
    sizes = {'HRNet': 58, 'Ours': 74, 'Lightweight': 50, 'General': 44}
    alphas = {'HRNet': 0.86, 'Ours': 0.92, 'Lightweight': 0.82, 'General': 0.62}

    by_cat: Dict[str, List[Point]] = {'HRNet': [], 'Ours': [], 'Lightweight': [], 'General': []}
    for p in pts:
        by_cat[_category(p.label)].append(p)

    for cat in ['General', 'Lightweight', 'HRNet', 'Ours']:
        gp = by_cat.get(cat, [])
        if not gp:
            continue
        ax.scatter(
            [p.x for p in gp],
            [p.y for p in gp],
            s=sizes.get(cat, 30),
            alpha=alphas.get(cat, 0.7),
            c=palette.get(cat, '#777777'),
            marker=markers.get(cat, 'o'),
            edgecolors='white' if cat in ['HRNet', 'Ours', 'Lightweight'] else 'none',
            linewidths=0.7 if cat in ['HRNet', 'Ours', 'Lightweight'] else 0.0,
            label=cat,
            zorder=2 if cat != 'Ours' else 3,
        )

    label_set = set()
    for p in pts:
        c = _category(p.label)
        if (c in ['HRNet', 'Ours', 'Lightweight', 'General']) or (p.label in label_set):
            label_set.add(p.label)
    for p in pts:
        if p.label in label_set:
            if p.label == 'Ours (slight loss)':
                ax.annotate(p.label, (p.x, p.y), textcoords='offset points', xytext=(0, -12), ha='center', va='top', fontsize=8)
            elif p.label == 'HRNet-W48 (Original)':
                ax.annotate(p.label, (p.x, p.y), textcoords='offset points', xytext=(8, -12), ha='left', va='top', fontsize=8)
            elif p.label == 'HRNet-W48 (Fine-Tuning+PTQ-INT8)':
                ax.annotate(p.label, (p.x, p.y), textcoords='offset points', xytext=(8, 0), ha='left', va='center', fontsize=8)
            elif p.label == 'HRNet-W48 (Fine-Tuning)':
                ax.annotate(p.label, (p.x, p.y), textcoords='offset points', xytext=(6, -10), ha='left', va='top', fontsize=8)
            elif p.label == 'HRNet-W48 (Fine-Tuning+QAT-INT8)':
                ax.annotate(p.label, (p.x, p.y), textcoords='offset points', xytext=(8, 0), ha='left', va='center', fontsize=8)
            else:
                ax.annotate(p.label, (p.x, p.y), textcoords='offset points', xytext=(5, 4), fontsize=8)

    ax.set_xlabel('GFLOPs (lower is better)')
    ax.set_ylabel('AP (%) (higher is better)')
    ax.set_title('AP vs. GFLOPs')
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.35)
    try:
        from matplotlib.lines import Line2D

        handles = [
            Line2D([], [], linestyle='None', marker='^', markersize=8.5, markerfacecolor=palette['HRNet'], markeredgecolor='white', label='HRNet'),
            Line2D([], [], linestyle='None', marker='s', markersize=9.0, markerfacecolor=palette['Ours'], markeredgecolor='white', label='Ours'),
            Line2D([], [], linestyle='None', marker='o', markersize=8.5, markerfacecolor=palette['Lightweight'], markeredgecolor='white', label='Lightweight'),
            Line2D([], [], linestyle='None', marker='o', markersize=8.5, markerfacecolor=palette['General'], markeredgecolor='none', label='General Backbone'),
        ]
        ax.legend(handles=handles, loc='lower right', frameon=True, framealpha=0.92, fontsize=9)
    except Exception:
        ax.legend(loc='lower right', frameon=True, framealpha=0.92, fontsize=9)

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, 'ap_gflops_pareto.png')
    out_pdf = os.path.join(out_dir, 'ap_gflops_pareto.pdf')
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    return out_png


def main():
    root = '/root/rivermind-data/PoseBH/experiments'
    comparison_csv = os.path.join(root, 'docs', 'comparison.csv')
    out_dir = os.path.join(root, 'figures')

    b = plot_ap_gflops_pareto(comparison_csv, out_dir)
    print(b)


if __name__ == '__main__':
    main()
