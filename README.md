# ASVspoof – Flow unificat (LA + CRNN + feature-mining)

Acest repo rulează două fluxuri:
1) **Flux clasic pe CSV** (pipeline existent): `make features`, `make train|eval|predict` (dacă ai scripturile tale `scripts/*.py`).
2) **Flux CRNN pe features secvențiale** (ASVspoof LA .flac + etichete din protocoale), plus **căutare exhaustivă de trăsături** (agregări tabulare).

## Setup rapid

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt  # dacă există
export ASVSPOOF_ROOT="$PWD/database/data/asvspoof2019"

1) Verificare & patch
make check_la       # verifică structura ASVspoof2019_LA_*
make patch          # .flac + constants.directory + main.py reshape + utilitare + scripturi lipsă


Patch-urile fac:

prepare_dataset.py acceptă .flac (nu doar .wav). Etichetele sunt citite din protocoalele LA via get_label_from_file_name.py.

constants.py citește dataset root din $ASVSPOOF_ROOT.

main.py folosește dataset/extracted_features și fixează reshape-ul la 4D (np.expand_dims(..., -1)).

utilities.py oferă BahdanauAttention, encode_combination, save_best_combination stabile.

2) Flow CRNN (secvențial)
make dataset   # creează X_train/X_val/X_test, y_*.npy + label_encoder.joblib în dataset/extracted_features
make train     # antrenează CRNN (Conv2D+BiLSTM+Bahdanau)


prepare_dataset.py extrage MFCC+delta etc. și aliniază frame-urile, apoi face split stratificat și padding pe lungimea maximă.
CRNN-ul este în build_CRNN_model.py.

3) Feature-mining (agregări + căutare)
make features_tabular   # exportă temp_data/{train,val,test}_tabular.csv
make search             # căutare (aproape) exhaustivă; scrie temp_data/combinations_accuracy.txt
make results            # afișează top 20 combinații ordonate


update_combinations.py ordonează combinations_accuracy.txt în combinations_ordered_by_accuracy.txt.

4) Fluxul pe CSV (opțional, dacă folosești indexele oficiale)
make index
make features
make eval
make predict




# 0) mediu
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt || true
export ASVSPOOF_ROOT="$PWD/database/data/asvspoof2019"

# 1) verifică & repară
make check_la
make patch
make doctor

# 2) pregătire + antrenare CRNN
make dataset
make train

# 3) feature-mining tabular
make features_tabular
make search
make results