# =========================
# ASVspoof Makefile (full)
# =========================

SHELL := /bin/bash
.ONESHELL:
.RECIPEPREFIX := >
# Fail fast în rețete
MAKEFLAGS += --no-builtin-rules

PY         ?= python
DATA_ROOT  ?= database/data/asvspoof2019
INDEX_DIR  ?= $(DATA_ROOT)/index
WORKERS    ?= 24

export ASVSPOOF_ROOT ?= $(CURDIR)/$(DATA_ROOT)

FEAT_SEQ_DIR ?= features/seq
PACKED_DIR   ?= dataset/extracted_features
TEMP_DIR     ?= temp_data

# ----- Helpers: existența scripturilor & directoare -----
.PHONY: check_la check_scripts check_combos_dir

check_la:
> set -euo pipefail
> echo "[*] Verific ASVSPOOF_ROOT=$(ASVSPOOF_ROOT)"
> test -d "$(DATA_ROOT)/ASVspoof2019_LA_train/flac"
> test -d "$(DATA_ROOT)/ASVspoof2019_LA_dev/flac"
> test -d "$(DATA_ROOT)/ASVspoof2019_LA_cm_protocols"
> echo "[✓] Structură OK"

check_scripts:
> set -euo pipefail
> test -f "asvspoof_features_pipeline.py" || { echo "[!] Lipsește asvspoof_features_pipeline.py în rădăcină"; exit 1; }
> test -f "scripts/train_all_combos.py"    || { echo "[!] Lipsește scripts/train_all_combos.py (mută-l în scripts/)"; exit 1; }
> echo "[✓] Scripturile există"

check_combos_dir:
> set -euo pipefail
> test -d "$(INDEX_DIR)/combos/train" || { echo "[!] Nu există $(INDEX_DIR)/combos/train. Rulează 'make combos_all' mai întâi."; exit 1; }
> echo "[✓] Găsit $(INDEX_DIR)/combos/train"

# ----- Ajutor -----
.PHONY: help
help:
> echo "Targets:"
> echo "  check_la            - verifică structura ASVspoof LA"
> echo "  index               - face CSV-uri train/dev (etichete), ignoră eval"
> echo "  features_seq        - extrage features frame-wise în features/seq/{train,dev}"
> echo "  pack                - pad/trunc -> $(PACKED_DIR)/X_*.npy, y_*.npy"
> echo "  dataset             - index + features_seq + pack"
> echo "  train               - rulează main.py (folosește $(PACKED_DIR))"
> echo "  features_tabular    - agregă X_*.npy -> CSV-uri pentru căutare"
> echo "  search              - căutare aproape exhaustivă pe train_tabular.csv"
> echo "  results             - arată top 20 combinații"
> echo "  clean               - șterge features/seq și temp_data"
> echo "  extract             - extrage toate feature-urile + splits + parquet/csv"
> echo "  combos_all          - materializează TOATE combinațiile (NPZ train/val/test)"
> echo "  combos_codes CODES='AB M AEMNO' - doar anumite combinații"
> echo "  train_all_combos    - antrenează pe toate combinațiile materializate"
> echo "  top_results         - top 20 după accuracy din results/combos_accuracy.csv"
> echo "  nohup_extract       - rulează extract cu nohup, log în logs/"
> echo "  nohup_combos_all    - rulează combos_all cu nohup, log în logs/"
> echo "  nohup_train_all     - rulează train_all_combos cu nohup, log în logs/"

# ----- Pipeline index + features_seq + pack -----
.PHONY: index features_seq pack dataset train features_tabular search results clean

index: check_la
> $(PY) scripts/make_index.py --root "$(DATA_ROOT)" --out "$(INDEX_DIR)" --splits train dev
> ls -lh "$(INDEX_DIR)"/train.csv "$(INDEX_DIR)"/val.csv

features_seq: index
> set -euo pipefail
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

# ----- Features + Combos (pipeline nou) -----
.PHONY: extract combos_all combos_codes

# Extrage features + splits + Parquet/CSV în $(INDEX_DIR)
extract: check_la check_scripts
> $(PY) asvspoof_features_pipeline.py extract \
>   --data-root "$(DATA_ROOT)" --workers $(WORKERS)

# Generează TOATE combinațiile (32767) -> $(INDEX_DIR)/combos/{train,val,test}
combos_all: extract check_scripts
> $(PY) asvspoof_features_pipeline.py combos \
>   --data-root "$(DATA_ROOT)" --all

# Generează doar anumite coduri (ex: AB AEMNO M)
combos_codes: extract check_scripts
> test -n "$(CODES)" || { echo "Setează CODES='AB AEMNO M'"; exit 1; }
> $(PY) asvspoof_features_pipeline.py combos \
>   --data-root "$(DATA_ROOT)" --codes $(CODES)

# ----- Training pe toate combinațiile materializate -----
.PHONY: train_all_combos top_results

train_all_combos: check_scripts check_combos_dir
> mkdir -p results
> $(PY) scripts/train_all_combos.py \
>   --data-root "$(DATA_ROOT)" \
>   --out results/combos_accuracy.csv \
>   --max-iter 200 --batch-size 256 --seed 42

# Afișează top 20 după acuratețe
top_results:
> test -f results/combos_accuracy.csv || { echo "[!] Lipsește results/combos_accuracy.csv (rulează make train_all_combos)"; exit 1; }
> (printf "combo,accuracy\n"; tail -n +2 results/combos_accuracy.csv | sort -t, -k2,2gr | head -n 20) | column -s, -t

# ----- Variante cu nohup + log în logs/ -----
.PHONY: nohup_extract nohup_combos_all nohup_train_all

nohup_extract: check_la check_scripts
> mkdir -p logs
> LOG="logs/extract_$$(date +%F_%H-%M-%S).log"; \
  nohup $(PY) asvspoof_features_pipeline.py extract \
    --data-root "$(DATA_ROOT)" --workers $(WORKERS) \
    > "$$LOG" 2>&1 & \
  echo "PID=$$!  log=$$LOG"

nohup_combos_all: extract check_scripts
> mkdir -p logs
> LOG="logs/combos_all_$$(date +%F_%H-%M-%S).log"; \
  nohup $(PY) asvspoof_features_pipeline.py combos \
    --data-root "$(DATA_ROOT)" --all \
    > "$$LOG" 2>&1 & \
  echo "PID=$$!  log=$$LOG"

nohup_train_all: check_scripts check_combos_dir
> mkdir -p logs results
> LOG="logs/train_all_$$(date +%F_%H-%M-%S).log"; \
  nohup $(PY) scripts/train_all_combos.py \
    --data-root "$(DATA_ROOT)" \
    --out results/combos_accuracy.csv \
    --max-iter 200 --batch-size 256 --seed 42 \
    > "$$LOG" 2>&1 & \
  echo "PID=$$!  log=$$LOG"

# Alias compatibil cu README: make combos codes="AB AEMNO M"
combos: extract check_scripts
> test -n "$(codes)" || { echo "Setează codes='AB AEMNO M'"; exit 1; }
> $(PY) asvspoof_features_pipeline.py combos \
>   --data-root "$(DATA_ROOT)" --codes $(codes)

# Curățare doar a combinațiilor
.PHONY: clean_combos
clean_combos:
> rm -rf "$(INDEX_DIR)/combos"
> echo "[*] Șters $(INDEX_DIR)/combos"
