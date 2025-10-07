SHELL := /bin/bash
.ONESHELL:
.RECIPEPREFIX := >

# ===== Config dataset (păstrat) =====
PY               ?= python
DATA_ROOT        ?= database/data/asvspoof2019
INDEX_DIR        ?= $(DATA_ROOT)/index
DOWNLOAD_SCRIPT  ?= database/download_asvspoof_la.sh

# ===== Config pipeline existent (păstrat) =====
DATA_DIR         ?= $(DATA_ROOT)
FEAT_DIR         ?= features
MODEL_DIR        ?= models
RESULTS_DIR      ?= results
MODEL_NAME       ?= clf.joblib

TRAIN_CSV        ?= $(INDEX_DIR)/train.csv
DEV_CSV          ?= $(INDEX_DIR)/val.csv
TEST_CSV         ?= $(INDEX_DIR)/test.csv

# ===== Config CRNN + feature-mining =====
export ASVSPOOF_ROOT ?= $(CURDIR)/$(DATA_ROOT)
EXTRACTED_DIR   ?= dataset/extracted_features
TEMP_DIR        ?= temp_data
LOG_DIR         ?= logs

.PHONY: help check_la index clean_index setup dataset train eval predict features features_tabular search results clean clean_all quickstart doctor

help:
> echo "Targets:"
> echo "  check_la, index, clean_index"
> echo "  dataset, train (CRNN pe X_*.npy)"
> echo "  features_tabular, search, results (feature-mining)"
> echo "  features, eval, predict (pipeline-ul tău clasic, dacă există scripturile)"
> echo "  clean, clean_all, doctor, quickstart"

# ---------- Dataset (ASVspoof LA) ----------
check_la:
> set -e
> echo "[*] Verific ASVSPOOF_ROOT=$(ASVSPOOF_ROOT)"
> test -d "$(DATA_ROOT)/ASVspoof2019_LA_train/flac"
> test -d "$(DATA_ROOT)/ASVspoof2019_LA_dev/flac"
> test -d "$(DATA_ROOT)/ASVspoof2019_LA_cm_protocols"
> echo "[✓] ASVspoof 2019 LA găsit în $(DATA_ROOT)"

index: check_la
> set -e
> echo "[*] Generez indexurile (train/dev/test) în $(INDEX_DIR)..."
> mkdir -p "$(INDEX_DIR)"
> if [ -f database/index_official.py ]; then
>   if $(PY) -u database/index_official.py --root "$(DATA_ROOT)" --out "$(INDEX_DIR)" --path-mode rel --with-eval; then
>     echo "[OK] index_official cu --path-mode rel";
>   else
>     echo "[i] Reîncerc fără --path-mode rel...";
>     $(PY) -u database/index_official.py --root "$(DATA_ROOT)" --out "$(INDEX_DIR)" --with-eval;
>   fi
> else
>   echo "[i] Nu există database/index_official.py în repo (ok)."
> fi
> ls -lh "$(INDEX_DIR)"/train.csv "$(INDEX_DIR)"/val.csv 2>/dev/null || true
> ls -lh "$(INDEX_DIR)"/test.csv 2>/dev/null || echo "[i] test.csv nu este disponibil (ok)."

clean_index:
> rm -rf "$(INDEX_DIR)"
> echo "[*] Șters: $(INDEX_DIR)"

# ---------- Flow CRNN (secvențial) ----------
dataset: check_la
> $(PY) create_dataset.py

train: dataset
> $(PY) main.py

# ---------- Flow clasic pe CSV (opțional) ----------
setup:
> $(PY) -m pip install -U pip
> test -f requirements.txt && $(PY) -m pip install -r requirements.txt || true

features: check_la index
> set -e
> if [ -f "scripts/extract_features.py" ]; then
>   mkdir -p "$(FEAT_DIR)"
>   echo "[*] Extract features: train -> $(FEAT_DIR)/train_features.parquet"
>   $(PY) scripts/extract_features.py --root "$(DATA_DIR)" --csv "$(TRAIN_CSV)" --out "$(FEAT_DIR)/train_features.parquet"
>   echo "[*] Extract features: dev(val) -> $(FEAT_DIR)/dev_features.parquet"
>   $(PY) scripts/extract_features.py --root "$(DATA_DIR)" --csv "$(DEV_CSV)" --out "$(FEAT_DIR)/dev_features.parquet"
>   if [ -f "$(TEST_CSV)" ]; then
>     echo "[*] Extract features: test -> $(FEAT_DIR)/test_features.parquet (fără etichete)";
>     $(PY) scripts/extract_features.py --root "$(DATA_DIR)" --csv "$(TEST_CSV)" --out "$(FEAT_DIR)/test_features.parquet" --no-labels;
>   else
>     echo "[i] Nu există $(TEST_CSV); sar peste features pentru test.";
>   fi
> else
>   echo "[i] Omis 'features': scripts/extract_features.py lipsește (ok).";
> fi

eval:
> set -e
> if [ -f "scripts/evaluate.py" ]; then
>   mkdir -p "$(RESULTS_DIR)"
>   test -f "$(MODEL_DIR)/$(MODEL_NAME)" || (echo "[!] Lipsește modelul: $(MODEL_DIR)/$(MODEL_NAME). Rulează 'make train'." && exit 1)
>   echo "[*] Evaluez pe dev(val) -> $(RESULTS_DIR)"
>   $(PY) scripts/evaluate.py --features "$(FEAT_DIR)/dev_features.parquet" --model "$(MODEL_DIR)/$(MODEL_NAME)" --outdir "$(RESULTS_DIR)"
> else
>   echo "[i] Omis 'eval': scripts/evaluate.py lipsește (ok).";
> fi

predict:
> set -e
> if [ -f "scripts/evaluate.py" ]; then
>   mkdir -p "$(RESULTS_DIR)"
>   if [ -f "$(FEAT_DIR)/test_features.parquet" ]; then
>     echo "[*] Rulez predicții pe test -> $(RESULTS_DIR)";
>     $(PY) scripts/evaluate.py --features "$(FEAT_DIR)/test_features.parquet" --model "$(MODEL_DIR)/$(MODEL_NAME)" --outdir "$(RESULTS_DIR)" --test;
>   else
>     echo "Nu există features pentru test. Rulează 'make features' după ce ai $(TEST_CSV).";
>   fi
> else
>   echo "[i] Omis 'predict': scripts/evaluate.py lipsește (ok).";
> fi

# ---------- Feature-mining (tabular) ----------
features_tabular: dataset
> $(PY) scripts/extract_tabular_features.py --split train --out_csv $(TEMP_DIR)/train_tabular.csv
> $(PY) scripts/extract_tabular_features.py --split val   --out_csv $(TEMP_DIR)/val_tabular.csv
> $(PY) scripts/extract_tabular_features.py --split test  --out_csv $(TEMP_DIR)/test_tabular.csv

search: features_tabular
> $(PY) scripts/exhaustive_search.py --csv $(TEMP_DIR)/train_tabular.csv --out_txt $(TEMP_DIR)/combinations_accuracy.txt
> $(PY) update_combinations.py

results:
> echo "Top 20 combinații (după AUC mediu KFold):"
> head -n 20 $(TEMP_DIR)/combinations_ordered_by_accuracy.txt || true

# ---------- Diverse ----------
doctor:
> echo "[*] Verific CRLF și variabile..."
> if grep -n $$'\r' Makefile >/dev/null; then echo "[!] Are CRLF -> sed -i 's/\r$$//' Makefile"; else echo "[✓] Fără CRLF"; fi
> echo "[*] PY=$(PY)"
> echo "[*] ASVSPOOF_ROOT=$(ASVSPOOF_ROOT)"

clean:
> rm -rf "$(TEMP_DIR)" "$(LOG_DIR)"
> echo "[*] Curățat temporarele."

clean_all: clean
> rm -rf "$(EXTRACTED_DIR)" "$(MODEL_DIR)" "$(RESULTS_DIR)" "$(FEAT_DIR)"
> echo "[*] Curățat artefactele de antrenare."

quickstart: check_la dataset train features_tabular search results
> echo "[✓] Flow complet rulat."
