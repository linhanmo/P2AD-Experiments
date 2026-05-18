import argparse
import csv
import json
import os
import re
from typing import Any, Dict, Optional


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _last_float_after(label: str, text: str) -> Optional[float]:
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(label)}\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)")
    matches = pattern.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def _extract_any_metrics(text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for m in re.finditer(r"([A-Za-z][A-Za-z0-9@_.\- ]{0,40})\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text):
        k = m.group(1).strip()
        v = float(m.group(2))
        if k in {"lr", "loss", "time", "eta"}:
            continue
        metrics[k] = v
    return metrics


def _parse_dataset_metrics(dataset: str, log_text: str) -> Dict[str, Any]:
    if dataset == "MPII":
        pckh = _last_float_after("PCKh", log_text)
        pckh_01 = _last_float_after("PCKh@0.1", log_text)
        metrics = {"PCKh": pckh, "PCKh@0.1": pckh_01}
        if pckh is None and pckh_01 is None:
            metrics.update(_extract_any_metrics(log_text))
        return metrics

    ap = _last_float_after("AP", log_text)
    ar = _last_float_after("AR", log_text)
    if isinstance(ap, (int, float)) and ap is not None and ap <= 1.5:
        ap = ap * 100.0
    if isinstance(ar, (int, float)) and ar is not None and ar <= 1.5:
        ar = ar * 100.0
    metrics = {"AP": ap, "AR": ar}
    if ap is None and ar is None:
        metrics.update(_extract_any_metrics(log_text))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    log_dir = os.path.abspath(args.log_dir)
    out_dir = os.path.abspath(args.out_dir)

    official_path = os.path.join(out_dir, "official_posebh_b_metrics.json")
    with open(official_path, "r", encoding="utf-8") as f:
        official = json.load(f)

    log_map = {
        "COCO val2017": os.path.join(log_dir, "06_coco_test.log"),
        "OCHuman test": os.path.join(log_dir, "03_ochuman_test.log"),
        "MPII val": os.path.join(log_dir, "02_mpii_test.log"),
        "AP-10K test (split1)": os.path.join(log_dir, "04_ap10k_test.log"),
        "COCO-WholeBody val": os.path.join(log_dir, "05_wholebody_test.log"),
    }

    measured: Dict[str, Any] = {"source_logs": log_map, "datasets": []}
    alignment_rows = []

    for ds in official["datasets"]:
        name = ds["name"]
        log_path = log_map.get(name)
        log_text = _read_text(log_path) if log_path and os.path.exists(log_path) else ""
        parsed = _parse_dataset_metrics("MPII" if "MPII" in name else "DEFAULT", log_text)
        measured_entry = {
            "name": name,
            "protocol": ds.get("protocol"),
            "log_path": log_path,
            "metrics": parsed,
        }
        measured["datasets"].append(measured_entry)

        official_metrics = ds.get("metrics", {})
        for k, official_v in official_metrics.items():
            measured_v = parsed.get(k)
            delta = None
            if isinstance(measured_v, (int, float)) and measured_v is not None:
                delta = measured_v - official_v
            alignment_rows.append(
                {
                    "dataset": name,
                    "metric": k,
                    "official": official_v,
                    "measured": measured_v,
                    "delta": delta,
                    "protocol": ds.get("protocol", ""),
                    "log_path": log_path or "",
                }
            )

    measured_path = os.path.join(out_dir, "measured_posebh_b_metrics.json")
    with open(measured_path, "w", encoding="utf-8") as f:
        json.dump(measured, f, ensure_ascii=False, indent=2)

    report_path = os.path.join(out_dir, "baseline_alignment_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"official": official, "measured": measured, "rows": alignment_rows}, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(out_dir, "baseline_alignment_report.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "metric", "official", "measured", "delta", "protocol", "log_path"],
        )
        writer.writeheader()
        for row in alignment_rows:
            writer.writerow(row)

    print(measured_path)
    print(report_path)
    print(csv_path)


if __name__ == "__main__":
    main()
