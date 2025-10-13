# fill_accuracy.py
import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd


def parse_float_from_file(p: Path):
    """
    Deschide fișierul și încearcă să extragă primul număr float.
    Acceptă linii cu doar numărul sau cu text + număr.
    """
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"Nu pot citi {p}: {e}")

    # caută primul număr cu zecimale (acceptă și forma 1,234 -> 1.234)
    m = re.search(r"[-+]?\d+(?:[.,]\d+)?", text)
    if not m:
        raise ValueError(f"N-am găsit niciun număr în {p}")

    s = m.group(0).replace(",", ".")
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"Nu pot converti '{s}' din {p} la float")


def find_accuracy_file(results_dir_from_csv: str, combo: str):
    """
    Returnează Path către accuracy.txt dacă există, altfel None.
    1) results_dir/accuracy.txt
    2) <root>/combo_<combo>/accuracy.txt
    3) <root>/<combo>/accuracy.txt
    Unde <root> = parent-ul lui results_dir_from_csv (de obicei .../projects/ASVspoof/results)
    """
    candidates = []

    if results_dir_from_csv:
        rd = Path(results_dir_from_csv)
        candidates.append(rd / "accuracy.txt")
        root = rd.parent
    else:
        root = None

    if root and combo:
        candidates.append(root / f"combo_{combo}" / "accuracy.txt")
        candidates.append(root / combo / "accuracy.txt")

    for c in candidates:
        if c.is_file():
            return c
    return None


def main():
    ap = argparse.ArgumentParser(description="Completează accuracy din accuracy.txt pentru fiecare combinație.")
    ap.add_argument("--csv", required=True, help="Calea către combinations_accuracy.csv")
    ap.add_argument("--out", default=None, help="Fișierul de ieșire (default: *_filled.csv)")
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Suprascrie CSV-ul de intrare (atenție: recomandat doar după ce verifici rezultatul).",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"Eroare: nu găsesc fișierul CSV: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # normalizăm coloanele așteptate
    if "combo" not in df.columns:
        print("Eroare: CSV-ul trebuie să aibă coloana 'combo'.", file=sys.stderr)
        sys.exit(1)

    if "results_dir" not in df.columns:
        df["results_dir"] = ""

    if "accuracy" not in df.columns:
        df["accuracy"] = pd.NA

    if "error" not in df.columns:
        df["error"] = pd.NA

    if "accuracy_disp" not in df.columns:
        df["accuracy_disp"] = pd.NA

    filled, total = 0, len(df)

    for idx, row in df.iterrows():
        acc = row.get("accuracy")
        needs_fill = pd.isna(acc) or (isinstance(acc, str) and acc.strip().upper() in {"NA", ""})

        if not needs_fill:
            continue

        combo = str(row["combo"]).strip()
        results_dir = str(row.get("results_dir") or "").strip()

        acc_file = find_accuracy_file(results_dir, combo)
        if not acc_file:
            df.at[idx, "error"] = f"accuracy.txt lipsă pentru combo={combo}"
            continue

        try:
            value = parse_float_from_file(acc_file)
            df.at[idx, "accuracy"] = value
            # dacă vrei un display rotunjit:
            df.at[idx, "accuracy_disp"] = round(value, 6)
            df.at[idx, "error"] = pd.NA
            filled += 1
        except Exception as e:
            df.at[idx, "error"] = f"{type(e).__name__}: {e}"

    # unde accuracy e încă NA dar există error, lasă cum e; altfel curăță error
    # (deja făcut în buclă)

    if args.inplace:
        out_path = csv_path
    else:
        out_path = Path(args.out) if args.out else csv_path.with_name(csv_path.stem + "_filled.csv")

    df.to_csv(out_path, index=False)

    print(f"Completat {filled}/{total} rânduri cu accuracy. Scris în: {out_path}")


if __name__ == "__main__":
    main()
