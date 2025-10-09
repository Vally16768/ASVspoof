# =========================
# ASVspoof Makefile 
# =========================

SHELL := /bin/bash
.ONESHELL:
.RECIPEPREFIX := >
# Fail fast in recipes
MAKEFLAGS += --no-builtin-rules

PY ?= python

# ---------- Read config from constants.py ----------
# These shell expansions import constants.py once per var.

# Root of dataset (constants.directory)
DATA_ROOT := $(shell $(PY) -c "import sys,os; sys.path.insert(0, os.getcwd()); import constants; print(constants.directory)")
# Folder names from constants
INDEX_DIRNAME := $(shell $(PY) -c "import sys,os; sys.path.insert(0, os.getcwd()); import constants; print(constants.index_folder_name)")
RESULTS_DIRNAME := $(shell $(PY) -c "import sys,os; sys.path.insert(0, os.getcwd()); import constants; print(constants.results_folder)")
TEMP_DIRNAME := $(shell $(PY) -c "import sys,os; sys.path.insert(0, os.getcwd()); import constants; print(constants.temp_data_folder_name)")

# Derived paths
INDEX_DIR := $(DATA_ROOT)/$(INDEX_DIRNAME)
RESULTS_DIR := $(RESULTS_DIRNAME)
TEMP_DIR := $(TEMP_DIRNAME)

# File names from constants
SAVE_COMBOS_NAME_RAW := $(shell $(PY) -c "import sys,os; sys.path.insert(0, os.getcwd()); import constants; print(constants.save_combinations_file_name)")
# Ensure CSV extension (train_all_combos.py does the same conversion)
SAVE_COMBOS_CSV := $(shell $(PY) -c "import sys,os,os.path; sys.path.insert(0, os.getcwd()); import constants; name=constants.save_combinations_file_name; print(name if name.lower().endswith('.csv') else (os.path.splitext(name)[0]+'.csv'))")
BEST_COMBOS_TXT := $(shell $(PY) -c "import sys,os; sys.path.insert(0, os.getcwd()); import constants; print(constants.save_the_best_combination_file_name)")

# Optional: workers (fallback to CPU count if not defined in constants.py)
WORKERS := $(shell $(PY) -c "import sys,os; sys.path.insert(0, os.getcwd()); \
try: import constants; w=getattr(constants,'workers',None) \
except Exception: w=None; \
print(w if w is not None else (os.cpu_count() or 1))")

# You can still override these via the environment if you really need to.
export ASVSPOOF_ROOT ?= $(DATA_ROOT)

# Legacy paths kept as variables (not from constants)
FEAT_SEQ_DIR ?= features/seq

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
> test -f "asvspoof/cli.py"                      || { echo "[!] Missing asvspoof/cli.py (package entrypoint)"; exit 1; }
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
> echo "  extract             - extract all features + splits + parquet/csv (via asvspoof.cli)"
> echo "  list                - print feature groups and letter mapping"
> echo "  combos_all          - materialize ALL combinations (NPZ train/val/test)"
> echo "  combos_codes        - only given combinations (set CODES='AB M AEMNO')"
> echo "  train_all_combos    - train on all materialized combinations"
> echo "  top_results         - top 20 by accuracy from $(RESULTS_DIR)/$(SAVE_COMBOS_CSV)"
> echo "  nohup_extract       - run extract with nohup, log in logs/"
> echo "  nohup_combos_all    - run combos_all with nohup, log in logs/"
> echo "  nohup_train_all     - run train_all_combos with nohup, log in logs/"
> echo "  combos              - README-compatible alias: make combos codes='AB AEMNO M'"
> echo "  clean               - remove features/seq and $(TEMP_DIR)"
> echo "  clean_combos        - remove only $(INDEX_DIR)/combos"

# ----- Index (protocols) -----
.PHONY: index clean

index: check_la check_scripts
> $(PY) scripts/make_index_from_protocols.py --root "$(DATA_ROOT)" --out "$(INDEX_DIR)" --splits train dev
> ls -lh "$(INDEX_DIR)"/train.csv "$(INDEX_DIR)"/val.csv

clean:
> rm -rf "$(FEAT_SEQ_DIR)" "$(TEMP_DIR)"
> echo "[*] Cleaned $(FEAT_SEQ_DIR) and $(TEMP_DIR)"

# ----- Features + Combos (pipeline via CLI) -----
.PHONY: extract list combos_all combos_codes

# Extract features + splits + Parquet/CSV into $(INDEX_DIR)
extract: check_la check_scripts
> $(PY) -m asvspoof.cli extract \
>   --data-root "$(DATA_ROOT)" --workers $(WORKERS)

# Print mapping (letters -> feature groups) and total combo count
list: check_scripts
> $(PY) -m asvspoof.cli list

# Generate ALL combinations -> $(INDEX_DIR)/combos/{train,val,test}
combos_all: extract check_scripts
> $(PY) -m asvspoof.cli combos \
>   --data-root "$(DATA_ROOT)" --all

# Generate only specific codes (e.g., AB AEMNO M)
combos_codes: extract check_scripts
> test -n "$(CODES)" || { echo "Set CODES='AB AEMNO M'"; exit 1; }
> CODES_UP=$$(echo "$(CODES)" | tr '[:lower:]' '[:upper:]'); \
  $(PY) -m asvspoof.cli combos \
    --data-root "$(DATA_ROOT)" --codes $$CODES_UP

# ----- Training on all materialized combinations -----
.PHONY: train_all_combos top_results

# Output CSV and best-combos file resolved from constants
OUT_CSV := $(RESULTS_DIR)/$(SAVE_COMBOS_CSV)
BEST_COMBOS_PATH := $(TEMP_DIR)/$(BEST_COMBOS_TXT)

train_all_combos: check_scripts check_combos_dir
> mkdir -p "$(RESULTS_DIR)"
> $(PY) scripts/train_all_combos.py \
>   --data-root "$(DATA_ROOT)" \
>   --out "$(OUT_CSV)" \
>   --combos-file "$(BEST_COMBOS_PATH)"

# Show top 20 by accuracy
top_results:
> test -f "$(OUT_CSV)" || { echo "[!] $(OUT_CSV) missing (run 'make train_all_combos')"; exit 1; }
> (printf "combo,accuracy\n"; tail -n +2 "$(OUT_CSV)" | sort -t, -k2,2gr | head -n 20) | column -s, -t

# ----- nohup variants + logs/ -----
.PHONY: nohup_extract nohup_combos_all nohup_train_all

nohup_extract: check_la check_scripts
> mkdir -p logs
> LOG="logs/extract_$$(date +%F_%H-%M-%S).log"; \
  nohup $(PY) -m asvspoof.cli extract \
    --data-root "$(DATA_ROOT)" --workers $(WORKERS) \
    > "$$LOG" 2>&1 & \
  echo "PID=$$!  log=$$LOG"

nohup_combos_all: extract check_scripts
> mkdir -p logs
> LOG="logs/combos_all_$$(date +%F_%H-%M-%S).log"; \
  nohup $(PY) -m asvspoof.cli combos \
    --data-root "$(DATA_ROOT)" --all \
    > "$$LOG" 2>&1 & \
  echo "PID=$$!  log=$$LOG"

nohup_train_all: check_scripts check_combos_dir
> mkdir -p logs "$(RESULTS_DIR)"
> LOG="logs/train_all_$$(date +%F_%H-%M-%S).log"; \
  nohup $(PY) scripts/train_all_combos.py \
    --data-root "$(DATA_ROOT)" \
    --out "$(OUT_CSV)" \
    --combos-file "$(BEST_COMBOS_PATH)" \
    > "$$LOG" 2>&1 & \
  echo "PID=$$!  log=$$LOG"

# README-compatible alias: make combos codes="AB AEMNO M"
combos: extract check_scripts
> test -n "$(codes)" || { echo "Set codes='AB AEMNO M'"; exit 1; }
> CODES_UP=$$(echo "$(codes)" | tr '[:lower:]' '[:upper:]'); \
  $(PY) -m asvspoof.cli combos \
    --data-root "$(DATA_ROOT)" --codes $$CODES_UP

# Remove only combinations
.PHONY: clean_combos
clean_combos:
> rm -rf "$(INDEX_DIR)/combos"
> echo "[*] Removed $(INDEX_DIR)/combos"
