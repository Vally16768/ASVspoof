#!/usr/bin/env python3
import argparse
import csv
import importlib
import os
import pathlib
import sys


def parse_args():
    ap = argparse.ArgumentParser(description="Show top combo results from summary CSV.")
    ap.add_argument("--metric", choices=["accuracy", "min_tDCF"], default=None)
    ap.add_argument("--tie-break", choices=["accuracy", "min_tDCF"], default=None)
    ap.add_argument("--topk", type=int, default=20)
    return ap.parse_args()


def _sort(rows, metric, tie_break):
    def key(row):
        v_metric = row.get(metric, float("nan"))
        v_tie = row.get(tie_break, float("nan"))
        if metric == "accuracy":
            p = -v_metric
        else:
            p = v_metric
        if tie_break == "accuracy":
            t = -v_tie
        else:
            t = v_tie
        return (p, t)

    return sorted(rows, key=key)


def main():
    args = parse_args()
    sys.path.insert(0, os.getcwd())
    c = importlib.import_module("constants")

    metric = args.metric or getattr(c, "combo_primary_metric", "min_tDCF")
    tie_break = args.tie_break or getattr(c, "combo_tie_break_metric", "accuracy")

    results_dir = pathlib.Path(getattr(c, "results_folder", "results"))
    name = getattr(c, "save_combinations_file_name", "combinations_accuracy.txt")
    if not name.lower().endswith(".csv"):
        name = os.path.splitext(name)[0] + ".csv"
    csv_path = results_dir / name
    if not csv_path.exists():
        sys.exit(f"[!] {csv_path} not found (run training first).")

    rows = []
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                acc = float(row.get("accuracy", "nan"))
            except Exception:
                acc = float("nan")
            try:
                tdcf = float(row.get("min_tDCF", "nan"))
            except Exception:
                tdcf = float("nan")
            rows.append({"combo": row.get("combo", ""), "accuracy": acc, "min_tDCF": tdcf})

    rows = _sort(rows, metric, tie_break)
    print("combo,accuracy,min_tDCF")
    for row in rows[: args.topk]:
        print(f"{row['combo']},{row['accuracy']},{row['min_tDCF']}")


if __name__ == "__main__":
    main()
