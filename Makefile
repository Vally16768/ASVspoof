# =========================
# ASVspoof Makefile
# =========================

SHELL := /bin/bash
.ONESHELL:
.RECIPEPREFIX := >
MAKEFLAGS += --no-builtin-rules

PY ?= python

# ----- Helpers (no constants in Make vars) -----
.PHONY: help check_scripts check_la

help:
> echo "Targets:"
> echo "  index               - build train/dev/test CSVs + eval.list (scripts read constants)"
> echo "  extract             - extract features + splits (scripts read constants)"
> echo "  list                - print feature groups and letter mapping"
> echo "  combos_all          - materialize ALL combinations"
> echo "  combos_codes        - materialize only given combinations (set CODES='AB M AEMNO')"
> echo "  train_all_combos    - train on all materialized combinations"
> echo "  top_results         - show top 20 by accuracy (paths resolved in Python)"
> echo "  clean               - remove feature cache + temp (paths resolved in Python)"
> echo "  clean_combos        - remove only index/combos (paths resolved in Python)"
> echo "  nohup_extract       - run extract with nohup (logs/)"
> echo "  nohup_combos_all    - run combos_all with nohup (logs/)"
> echo "  nohup_train_all     - run train_all_combos with nohup (logs/)"

check_scripts:
> set -euo pipefail
> test -f "asvspoof/cli.py"                      || { echo "[!] Missing asvspoof/cli.py"; exit 1; }
> test -f "scripts/make_index_from_protocols.py" || { echo "[!] Missing scripts/make_index_from_protocols.py"; exit 1; }
> test -f "scripts/train_all_combos.py"          || { echo "[!] Missing scripts/train_all_combos.py"; exit 1; }
> echo "[✓] Required scripts found"

# Validate dataset layout (logic in a separate Python file)
check_la:
> $(PY) tools/check_la.py

# ----- Index (protocols) -----
.PHONY: index
index: check_scripts check_la
> $(PY) scripts/make_index_from_protocols.py
> echo "[✓] Indices generated (paths decided by scripts/constants)"

# ----- Features + Combos (pipeline via CLI) -----
.PHONY: extract list combos_all combos_codes

extract: check_scripts check_la
> $(PY) -m asvspoof.cli extract

list: check_scripts
> $(PY) -m asvspoof.cli list

combos_all: check_scripts
> $(PY) -m asvspoof.cli combos --all

combos_codes: check_scripts
> test -n "$(CODES)" || { echo "Set CODES='AB AEMNO M'"; exit 1; }
> CODES_UP=$$(echo "$(CODES)" | tr '[:lower:]' '[:upper:]'); \
  $(PY) -m asvspoof.cli combos --codes $$CODES_UP

# ----- Training on all materialized combinations -----
.PHONY: train_all_combos top_results

train_all_combos: check_scripts
> $(PY) scripts/train_all_combos.py

top_results:
> $(PY) tools/top_results.py

# ----- Cleaning -----
.PHONY: clean clean_combos

clean:
> $(PY) tools/clean_paths.py

clean_combos:
> $(PY) tools/clean_combos.py

# ----- nohup + logs -----
.PHONY: nohup_extract nohup_combos_all nohup_train_all

nohup_extract: check_scripts
> mkdir -p logs
> LOG="logs/extract_$$(date +%F_%H-%M-%S).log"; \
  nohup $(PY) -m asvspoof.cli extract > "$$LOG" 2>&1 & \
  echo "PID=$$!  log=$$LOG"

nohup_combos_all: check_scripts
> mkdir -p logs
> LOG="logs/combos_all_$$(date +%F_%H-%M-%S).log"; \
  nohup $(PY) -m asvspoof.cli combos --all > "$$LOG" 2>&1 & \
  echo "PID=$$!  log=$$LOG"

nohup_train_all: check_scripts
> mkdir -p logs
> LOG="logs/train_all_$$(date +%F_%H-%M-%S).log"; \
  nohup $(PY) scripts/train_all_combos.py > "$$LOG" 2>&1 & \
  echo "PID=$$!  log=$$LOG"
