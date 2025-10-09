#!/usr/bin/env python3
import sys, os, shutil, pathlib, importlib
def main():
    sys.path.insert(0, os.getcwd())
    c = importlib.import_module('constants')
    temp_dir = pathlib.Path(getattr(c,'temp_data_folder_name','temp_data'))
    seq_dir  = pathlib.Path('features/seq')
    for p in [seq_dir, temp_dir]:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            print(f"[*] Removed {p}")
    print("[*] Clean done")
if __name__ == "__main__":
    main()
