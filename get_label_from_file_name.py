import os

# Cache simplu pentru maparea path -> label
_LABELS = None

def _load_protocols():
    root = os.environ.get("ASVSPOOF_ROOT")
    if not root:
        raise RuntimeError("Setează ASVSPOOF_ROOT către rădăcina ASVspoof 2019 LA.")
    protos = []
    prot_dir = os.path.join(root, "ASVspoof2019_LA_cm_protocols")
    for sub in ("ASVspoof2019.LA.cm.train.trn.txt",
                "ASVspoof2019.LA.cm.dev.trl.txt",
                "ASVspoof2019.LA.cm.eval.trl.txt"):
        p = os.path.join(prot_dir, sub)
        if os.path.exists(p): protos.append(p)
    mapping = {}
    # Format linie: <utt_id> <speaker> <system_id> <bonafide/spoof>
    # Ex: LA_0001  ...  bonafide
    for p in protos:
        with open(p, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4: 
                    continue
                utt_id = parts[0]              # ex: LA_0001
                label  = parts[-1].lower()     # bonafide | spoof
                # calculez posibile locații pentru fișiere (train/dev/eval)
                for subdir in ("ASVspoof2019_LA_train/flac",
                               "ASVspoof2019_LA_dev/flac",
                               "ASVspoof2019_LA_eval/flac"):
                    cand = os.path.join(root, subdir, utt_id + ".flac")
                    mapping[os.path.abspath(cand)] = label
    return mapping

def get_label_from_file_name(file_name):
    global _LABELS
    if _LABELS is None:
        _LABELS = _load_protocols()
    # normalizăm calea
    afn = os.path.abspath(file_name)
    return _LABELS.get(afn)
