#!/usr/bin/env python3
from pathlib import Path
import os, sys

CANDIDATES = [
    os.getenv("ASVSPOOF_ROOT") or "",
    "data/asvspoof2019",
    "database/data/asvspoof2019",
]

def ok(p): 
    p = Path(p) if p else Path("")
    return p.exists() and (p/"index").exists()

def main():
    roots = []
    for c in CANDIDATES:
        if c and ok(c):
            roots.append(Path(c).resolve())
    print("[i] Candideți root (există index/):")
    for r in roots:
        print("   -", r)
        for name in ["train.csv","val.csv","test.csv","dev.csv","eval.csv","eval.list"]:
            exists = (r/"index"/name).exists()
            print(f"      index/{name}: {'OK' if exists else 'missing'}")
    if not roots:
        print("[!] Nu am găsit niciun <root>/index/. Verifică structura.")
        sys.exit(2)
    print("\nSugestie export:")
    print(f'   export ASVSPOOF_ROOT="{roots[0]}"')

if __name__ == "__main__":
    main()
