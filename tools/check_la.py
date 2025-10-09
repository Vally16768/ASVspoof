#!/usr/bin/env python3
import sys, os, pathlib, importlib, importlib.util

def main():
    sys.path.insert(0, os.getcwd())
    spec = importlib.util.find_spec('constants')
    if not spec:
        sys.exit("[!] constants.py not found in CWD")

    c = importlib.import_module('constants')
    root = pathlib.Path(getattr(c, 'directory', 'dataset')).resolve()
    print(f"[*] Checking dataset root: {root}")

    need = [
        root/'ASVspoof2019_LA_train'/'flac',
        root/'ASVspoof2019_LA_dev'/'flac',
    ]
    for d in need:
        if not d.is_dir():
            sys.exit(f"[!] Missing required directory: {d}")

    proto_ok = any(p.is_dir() for p in [
        root/'ASVspoof2019_LA_cm_protocols',
        root/'asvspoof2019_labelled'/'ASVspoof2019_LA_cm_protocols',
    ])
    if not proto_ok:
        sys.exit("[!] Could not find LA protocols folder under dataset root.")

    print("[✓] Structure OK")

if __name__ == "__main__":
    main()
