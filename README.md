# ASVspoof LA – pipeline standard

## Pași
```bash
# 0) dependențe (minim)
pip install -U pip
pip install librosa soundfile numpy scipy pandas scikit-learn joblib tqdm matplotlib xgboost tensorflow==2.15.0

# 1) set root
export ASVSPOOF_ROOT="$PWD/database/data/asvspoof2019"

# 2) pipeline
make check_la            # verifică structura
make index               # CSV-uri: database/data/asvspoof2019/index/{train.csv,val.csv}
make features_seq        # features frame-wise -> features/seq/{train,dev}/*.npz
make pack                # X_train.npy, X_val.npy, y_train.npy, y_val.npy
make train               # folosește main.py (așteaptă dataset/extracted_features)
make features_tabular    # agregări tabulare (pt. căutări)
make search              # căutare "aproape" exhaustivă (AUC pe KFold)
make results             # top 20 combinații
