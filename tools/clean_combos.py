#!/usr/bin/env python3
import sys, os, shutil, pathlib, importlib
def main():
    sys.path.insert(0, os.getcwd())
    c = importlib.import_module('constants')
    root = pathlib.Path(getattr(c,'directory','dataset'))
    idx  = pathlib.Path(getattr(c,'index_folder_name','index'))
    target = root / idx / 'combos'
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
        print(f"[*] Removed {target}")
    else:
        print(f"[i] {target} not found")
if __name__ == "__main__":
    main()
