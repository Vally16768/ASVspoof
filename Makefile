# =========================
# ASVspoof Makefile (full)
# =========================

SHELL := /bin/bash
.ONESHELL:
.RECIPEPREFIX := >
# Fail fast in recipes
MAKEFLAGS += --no-builtin-rules

PY         ?= python
DATA_ROOT  ?= database/data/asvspoof2019
INDEX_DIR  ?= $(DATA_ROOT)/index
WORKERS    ?= 36

export ASVSPOOF_ROOT ?= $(CURDIR)/$(DATA_ROOT)

FEAT_SEQ_DIR ?= features/seq
PACKED_DIR   ?= dataset/extracted_features
TEMP_DIR     ?= temp_data

# ----- Helpers: check scripts & directories -----
.PHONY: check_la check_scripts check_combos_dir

check_la:
> set -euo pipefail
> echo "[*] Checking ASVSPOOF_ROOT=$(ASVSPOOF_ROOT)"
> test -d "$(DATA_ROOT)/ASVspoof2019_LA_train/flac"
> test -d "$(DATA_ROOT)/ASVspoof2019_LA_dev/flac"
> test -d "$(DATA_ROOT)/ASVspoof2019_LA_cm_protocols"
> echo "[✓] Structure OK"

check_scripts:
> set -euo pipefail
> test -f "asvspoof_features_pipeline.py" || { echo "[!] Missing asvspoof_features_pipeline.py at repo root"; exit 1; }
> test -f "scripts/make_index_from_protocols.py" || { echo "[!] Missing scripts/make_index_from_protocols.py"; exit 1; }
> test -f "scripts/train_all_combos.py"          || { echo "[!] Missing scripts/train_all_combos.py"; exit 1; }
> echo "[✓] Required scripts found"

check_combos_dir:
> set -euo pipefail
> test -d "$(INDEX_DIR)/combos/train" || { echo "[!] $(INDEX_DIR)/combos/train does not exist. Run 'make combos_all' first."; exit 1; }
> echo "[✓] Found $(INDEX_DIR)/combos/train"

# ----- Help -----
.PHONY: help
help:
> echo "Targets:"
> echo "  check_la            - verify ASVspoof LA directory structure"
> echo "  check_scripts       - verify required scripts exist"
> echo "  index               - build train/dev CSVs (with labels), skips eval"
> echo "  results             - show top 20 combinations"
> echo "  clean               - remove features/seq and temp_data"
> echo "  extract             - extract all features + splits + parquet/csv"
> echo "  combos_all          - materialize ALL combinations (NPZ train/val/test)"
> echo "  combos_codes        - only given combinations (set CODES='AB M AEMNO')"
> echo "  train_all_combos    - train on all materialized combinations"
> echo "  top_results         - top 20 by accuracy from results/combos_accuracy.csv"
> echo "  nohup_extract       - run extract with nohup, log in logs/"
> echo "  nohup_combos_all    - run combos_all with nohup, log in logs/"
> echo "  nohup_train_all     - run train_all_combos with nohup, log in logs/"
> echo "  combos              - README-compatible alias: make combos codes='AB AEMNO M'"
> echo "  clean_combos        - remove only $(INDEX_DIR)/combos"

# ----- Index (protocols) -----
.PHONY: index features_seq pack dataset train features_tabular search results clean

index: check_la check_scripts
> $(PY) scripts/make_index_from_protocols.py --root "$(DATA_ROOT)" --out "$(INDEX_DIR)" --splits train dev
> ls -lh "$(INDEX_DIR)"/train.csv "$(INDEX_DIR)"/val.csv

dataset: pack
> echo "[✓] Dataset prepared in $(PACKED_DIR)"

train: check_scripts
> $(PY) main.py

results:
> echo "Top 20 combinations (by mean AUC KFold):"
> head -n 20 $(TEMP_DIR)/combinations_ordered_by_accuracy.txt || true

clean:
> rm -rf "$(FEAT_SEQ_DIR)" "$(TEMP_DIR)"
> echo "[*] Cleaned features/seq and temp_data"

# ----- Features + Combos (new pipeline) -----
.PHONY: extract combos_all combos_codes

# Extract features + splits + Parquet/CSV into $(INDEX_DIR)
extract: check_la check_scripts
> $(PY) asvspoof_features_pipeline.py extract \
>   --data-root "$(DATA_ROOT)" --workers $(WORKERS)

# Generate ALL combinations (32767) -> $(INDEX_DIR)/combos/{train,val,test}
combos_all: extract check_scripts
> $(PY) asvspoof_features_pipeline.py combos \
>   --data-root "$(DATA_ROOT)" --all

# Generate only specific codes (e.g., AB AEMNO M)
combos_codes: extract check_scripts
> test -n "$(CODES)" || { echo "Set CODES='AB AEMNO M'"; exit 1; }
> $(PY) asvspoof_features_pipeline.py combos \
>   --data-root "$(DATA_ROOT)" --codes $(CODES)

# ----- Training on all materialized combinations -----
.PHONY: train_all_combos top_results

train_all_combos: check_scripts check_combos_dir
> mkdir -p results
> $(PY) scripts/train_all_combos.py \
>   --data-root "$(DATA_ROOT)" \
>   --out results/combos_accuracy.csv \
>   --max-iter 200 --batch-size 256 --seed 42

# Show top 20 by accuracy
top_results:
> test -f results/combos_accuracy.csv || { echo "[!] results/combos_accuracy.csv missing (run make train_all_combos)"; exit 1; }
> (printf "combo,accuracy\n"; tail -n +2 results/combos_accuracy.csv | sort -t, -k2,2gr | head -n 20) | column -s, -t

# ----- nohup variants + logs/ -----
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

# README-compatible alias: make combos codes="AB AEMNO M"
combos: extract check_scripts
> test -n "$(codes)" || { echo "Set codes='AB AEMNO M'"; exit 1; }
> $(PY) asvspoof_features_pipeline.py combos \
>   --data-root "$(DATA_ROOT)" --codes $(codes)

# Remove only combinations
.PHONY: clean_combos
clean_combos:
> rm -rf "$(INDEX_DIR)/combos"
> echo "[*] Removed $(INDEX_DIR)/combos"
