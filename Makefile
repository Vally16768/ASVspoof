# ==== Config ====
PY               ?= python
DATA_ROOT        ?= database/data/asvspoof2019
INDEX_DIR        ?= $(DATA_ROOT)/index
DOWNLOAD_SCRIPT  ?= database/download_asvspoof_la.sh

# ==== Targets ====
.PHONY: dataset check_la index clean clean_index

dataset: check_la index
	@echo "[✓] Dataset pregătit. CSV-urile sunt în $(INDEX_DIR)"

check_la:
	@set -e; \
	echo "[*] Verific ASVspoof 2019 LA..."; \
	if [ -d "$(DATA_ROOT)/ASVspoof2019_LA_train/flac" ] && \
	   [ -d "$(DATA_ROOT)/ASVspoof2019_LA_dev/flac" ]   && \
	   [ -d "$(DATA_ROOT)/ASVspoof2019_LA_cm_protocols" ]; then \
	  echo "[✓] Găsit: $(DATA_ROOT)"; \
	else \
	  echo "[!] Nu am găsit toate directoarele. Descarc LA.zip..."; \
	  chmod +x "$(DOWNLOAD_SCRIPT)" 2>/dev/null || true; \
	  bash "$(DOWNLOAD_SCRIPT)" "$(DATA_ROOT)"; \
	fi

index:
	@set -e; \
	echo "[*] Generez indexurile oficiale (train=LA_train, val=LA_dev)..."; \
	# Încerc întâi cu --path-mode (alias), apoi cu --path_mode, apoi fără flag (default=rel)
	$(PY) -u database/index_official.py \
	  --root "$(DATA_ROOT)" \
	  --out  "$(INDEX_DIR)" \
	  --path-mode rel \
	  --with-eval \
	|| $(PY) -u database/index_official.py \
	  --root "$(DATA_ROOT)" \
	  --out  "$(INDEX_DIR)" \
	  --path_mode rel \
	  --with-eval \
	|| $(PY) -u database/index_official.py \
	  --root "$(DATA_ROOT)" \
	  --out  "$(INDEX_DIR)" \
	  --with-eval; \
	echo "[*] Done."; \
	ls -lh "$(INDEX_DIR)"/train.csv "$(INDEX_DIR)"/val.csv 2>/dev/null || true

clean_index:
	@rm -rf "$(INDEX_DIR)"; \
	echo "[*] Șters: $(INDEX_DIR)"

clean:
	@rm -rf "$(INDEX_DIR)"; \
	echo "[*] Poți șterge manual și $(DATA_ROOT) dacă vrei să refaci totul."


# === Config ===
PY=python
DATA_DIR=data/asvspoof2019
FEAT_DIR=features
MODEL_DIR=models
RESULTS_DIR=results
MODEL_NAME=clf.joblib

TRAIN_CSV=$(DATA_DIR)/index/train.csv
DEV_CSV=$(DATA_DIR)/index/dev.csv
TEST_CSV=$(DATA_DIR)/index/test.csv

# === Phony targets ===
.PHONY: setup features train eval predict clean

setup:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

features:
	$(PY) scripts/extract_features.py --root $(DATA_DIR) --csv $(TRAIN_CSV) --out $(FEAT_DIR)/train_features.parquet
	$(PY) scripts/extract_features.py --root $(DATA_DIR) --csv $(DEV_CSV)   --out $(FEAT_DIR)/dev_features.parquet
	@if [ -f $(TEST_CSV) ]; then \
	$(PY) scripts/extract_features.py --root $(DATA_DIR) --csv $(TEST_CSV) --out $(FEAT_DIR)/test_features.parquet --no-labels ; \
	fi

train:
	$(PY) scripts/train.py --train $(FEAT_DIR)/train_features.parquet --dev $(FEAT_DIR)/dev_features.parquet --out $(MODEL_DIR)/$(MODEL_NAME)

eval:
	$(PY) scripts/evaluate.py --features $(FEAT_DIR)/dev_features.parquet --model $(MODEL_DIR)/$(MODEL_NAME) --outdir $(RESULTS_DIR)

predict:
	@if [ -f $(FEAT_DIR)/test_features.parquet ]; then \
	$(PY) scripts/evaluate.py --features $(FEAT_DIR)/test_features.parquet --model $(MODEL_DIR)/$(MODEL_NAME) --outdir $(RESULTS_DIR) --test; \
	else echo "Nu există features pentru test."; fi

clean:
	rm -f $(MODEL_DIR)/*.joblib
	rm -f $(RESULTS_DIR)/*.json $(RESULTS_DIR)/*.csv $(RESULTS_DIR)/*.png
