#!/usr/bin/env python3
import sys, os, csv, pathlib, importlib
def main():
    sys.path.insert(0, os.getcwd())
    c = importlib.import_module('constants')
    results_dir = pathlib.Path(getattr(c,'results_folder','results'))
    name = getattr(c,'save_combinations_file_name','combinations_accuracy.txt')
    if not name.lower().endswith('.csv'):
        name = os.path.splitext(name)[0] + '.csv'
    csv_path = results_dir / name
    if not csv_path.exists():
        sys.exit(f"[!] {csv_path} not found (run training first).")

    rows = []
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                acc = float(row.get('accuracy','nan'))
            except Exception:
                continue
            rows.append((row.get('combo',''), acc))
    rows.sort(key=lambda x: x[1], reverse=True)
    print("combo,accuracy")
    for combo, acc in rows[:20]:
        print(f"{combo},{acc}")
if __name__ == "__main__":
    main()
