import os

_LABELS = None

def _load_protocols():
    root = os.environ.get("ASVSPOOF_ROOT")
    if not root:
        raise RuntimeError("Setează ASVSPOOF_ROOT către rădăcina ASVspoof 2019 LA (unde ai directoarele ASVspoof2019_LA_*).")
    prot_dir = os.path.join(root, "ASVspoof2019_LA_cm_protocols")
    files = [
        "ASVspoof2019.LA.cm.train.trn.txt",
        "ASVspoof2019.LA.cm.dev.trl.txt",
        "ASVspoof2019.LA.cm.eval.trl.txt",
    ]
    mapping = {}
    for f in files:
        p = os.path.join(prot_dir, f)
        if not os.path.exists(p):
            continue
        with open(p, "r") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                utt_id = parts[0]
                label = parts[-1].lower()  # bonafide | spoof
                for sub in ("ASVspoof2019_LA_train/flac", "ASVspoof2019_LA_dev/flac", "ASVspoof2019_LA_eval/flac"):
                    cand = os.path.join(root, sub, utt_id + ".flac")
                    mapping[os.path.abspath(cand)] = label
    return mapping

def get_label_from_file_name(file_name):
    global _LABELS
    if _LABELS is None:
        _LABELS = _load_protocols()
    return _LABELS.get(os.path.abspath(file_name))
