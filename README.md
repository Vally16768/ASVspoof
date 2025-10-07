# ASVspoof 2019 LA — Features & Splits

Acest repo pregătește setul ASVspoof 2019 (Logical Access) și extrage un set bogat de
feature-uri clasice pentru fiecare fișier audio. Rezultatul este:
- un fișier Parquet/CSV cu **toate** feature-urile;
- split stratificat `train / val / test` (eval rămâne separat, fără etichete);
- **combinații** de feature-uri materializate în fișiere `.npz` (X, y, columns).

> Modelarea, evaluarea și centralizarea rezultatelor vin **în repo separat** (sau în subfoldere dedicate).

## Cerințe
- Python 3.9+ (recomandat un `venv`)
- `requirements.txt` din acest repo

## Instalare rapidă
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Date
1. Pune `LA.zip` în `database/` **sau** lasă scriptul să constate lipsa și să-ți spună ce să faci.
2. Rulează:
```bash
make dataset      # verificări + dezarhivare dacă e cazul
```

Structura finală a datelor trebuie să conțină:
```
database/data/asvspoof2019/
  ASVspoof2019_LA_train/flac/*.flac
  ASVspoof2019_LA_dev/flac/*.flac
  ASVspoof2019_LA_eval/flac/*.flac
  ASVspoof2019_LA_cm_protocols/*.txt
```

## Extrage feature-uri + splits
```bash
make extract           # rulează asvspoof_features_pipeline.py extract
```
Scrie:
```
database/data/asvspoof2019/index/
  features_all.parquet
  features_all.csv
  features_meta.json
```

## Combinații de feature-uri (X/y -> .npz)
- Toate combinațiile (2^15 - 1 = **32.767**; poate dura mult și ocupă spațiu):
```bash
make combos_all
```
- Doar anumite combinații (exemple: `AB`, `ABL`, `M`, `AEMNO`):
```bash
make combos codes="AB ABL M AEMNO"
```
Rezultatele apar în:
```
database/data/asvspoof2019/index/combos/{train,val,test}/<CODE>.npz
```

## Maparea feature-lor
Literele corespund grupurilor:
```
A: mfcc_mean          H: chroma_cens_mean   N: pitch_mean
B: mfcc_std           I: chroma_cqt_mean    O: pitch_std
C: spec_centroid_mean J: wavelet_mean
D: spec_bw_mean       K: chroma_stft_mean
E: spec_contrast_mean L: wavelet_std
F: spec_rolloff_mean  M: zcr_mean
G: rms_mean
```

## Comenzi utile
```bash
make help
make clean          # curăță fișierele generate
make clean_combos   # șterge numai combinațiile
```

## Structură recomandată
```
ASVspoof/
  asvspoof_features_pipeline.py
  Makefile
  README.md
  requirements.txt
  database/
    download_asvspoof_la.sh
    data/asvspoof2019/...
```

## Ce am eliminat / considerat inutil
- prototipuri vechi de feature extraction / notebook-uri nefolosite;
- scripturi ad-hoc din `scripts/` care duplică acum funcționalitatea din
  `asvspoof_features_pipeline.py`.

> Dacă ai fișiere vechi în `scripts/` pentru extragere/training, le poți șterge în siguranță.
Păstrăm strict: **download + verificare date** și **pipeline-ul unificat de feature-uri + splits**.