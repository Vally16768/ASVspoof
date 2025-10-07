SHELL := /bin/bash
.ONESHELL:
.RECIPEPREFIX := >

PY         ?= python
DATA_ROOT  ?= database/data/asvspoof2019
INDEX_DIR  ?= $(DATA_ROOT)/index

export ASVSPOOF_ROOT ?= $(CURDIR)/$(DATA_ROOT)

FEAT_SEQ_DIR ?= features/seq
PACKED_DIR   ?= dataset/extracted_features
TEMP_DIR     ?= temp_data

.PHONY: help check_la index features_seq pack dataset train features_tabular search results clean

help:
> echo "Targets:"
> echo "  check_la          - verifică structura ASVspoof LA"
> echo "  index             - face CSV-uri train/dev (etichete), ignoră eval"
> echo "  features_seq      - extrage features frame-wise în features/seq/{train,dev}"
> echo "  pack              - pad/trunc -> $(PACKED_DIR)/X_*.npy, y_*.npy"
> echo "  dataset           - index + features_seq + pack"
> echo "  train             - rulează main.py (folosește $(PACKED_DIR))"
> echo "  features_tabular  - agregă X_*.npy -> CSV-uri pentru căutare"
> echo "  search            - căutare aproape exhaustivă pe train_tabular.csv"
> echo "  results           - arată top 20 combinații"
> echo "  clean             - șterge features/seq și temp_data"

check_la:
> set -e
> echo "[*] Verific ASVSPOOF_ROOT=$(ASVSPOOF_ROOT)"
> test -d "$(DATA_ROOT)/ASVspoof2019_LA_train/flac"
> test -d "$(DATA_ROOT)/ASVspoof2019_LA_dev/flac"
> test -d "$(DATA_ROOT)/ASVspoof2019_LA_cm_protocols"
> echo "[✓] Structură OK"

index: check_la
> $(PY) scripts/make_index.py --root "$(DATA_ROOT)" --out "$(INDEX_DIR)" --splits train dev
> ls -lh "$(INDEX_DIR)"/train.csv "$(INDEX_DIR)"/val.csv

features_seq: index
> set -e
> mkdir -p "$(FEAT_SEQ_DIR)"
> $(PY) scripts/extract_features_seq.py --root "$(DATA_ROOT)" --csv "$(INDEX_DIR)/train.csv" --outdir "$(FEAT_SEQ_DIR)"
> $(PY) scripts/extract_features_seq.py --root "$(DATA_ROOT)" --csv "$(INDEX_DIR)/val.csv"   --outdir "$(FEAT_SEQ_DIR)"

pack: features_seq
> $(PY) scripts/pack_features.py --features_root "$(FEAT_SEQ_DIR)" --splits train dev --outdir "$(PACKED_DIR)" --frames 400

dataset: pack
> echo "[✓] Dataset pregătit în $(PACKED_DIR)"

train:
> $(PY) main.py

features_tabular:
> $(PY) scripts/extract_tabular_features.py --extracted_dir "$(PACKED_DIR)" --split train --out_csv $(TEMP_DIR)/train_tabular.csv
> $(PY) scripts/extract_tabular_features.py --extracted_dir "$(PACKED_DIR)" --split val   --out_csv $(TEMP_DIR)/val_tabular.csv

search:
> $(PY) scripts/exhaustive_search.py --csv $(TEMP_DIR)/train_tabular.csv --out_txt $(TEMP_DIR)/combinations_accuracy.txt
> $(PY) update_combinations.py

results:
> echo "Top 20 combinații (după AUC mediu KFold):"
> head -n 20 $(TEMP_DIR)/combinations_ordered_by_accuracy.txt || true

clean:
> rm -rf "$(FEAT_SEQ_DIR)" "$(TEMP_DIR)"
> echo "[*] Curățat features/seq și temp_data"
