import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import load_audio_mono, pad_or_trim, extract_basic_features, label_to_int

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Folder rădăcină pentru data/asvspoof2019")
    ap.add_argument("--csv", required=True, help="CSV cu (rel_path,label) sau (rel_path) pentru test")
    ap.add_argument("--out", required=True, help="Fișier .parquet de ieșire")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--dur", type=float, default=4.0, help="Durată fixă (s) pentru pad/trim")
    ap.add_argument("--no-labels", action="store_true", help="CSV fără label (set de test)")
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    df = pd.read_csv(args.csv, header=None)

    if args.no_labels:
        df.columns = ["rel_path"]
    else:
        df.columns = ["rel_path", "label"]

    for i, row in tqdm(df.iterrows(), total=len(df)):
        wav_path = root / row["rel_path"]
        y, sr = load_audio_mono(str(wav_path), target_sr=args.sr)
        y = pad_or_trim(y, sr, args.dur)
        feat = extract_basic_features(y, sr)
        rec = {"rel_path": row["rel_path"]}
        if not args.no_labels:
            rec["label"] = int(label_to_int(row["label"]))
        # expand features to columns f0..fN
        for j, v in enumerate(feat):
            rec[f"f{j}"] = float(v)
        rows.append(rec)

    out_df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"Scris {args.out} cu shape={out_df.shape}")

if __name__ == "__main__":
    main()
