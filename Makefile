# ==== Config dataset ====
PY               ?= python
DATA_ROOT        ?= database/data/asvspoof2019
INDEX_DIR        ?= $(DATA_ROOT)/index
DOWNLOAD_SCRIPT  ?= database/download_asvspoof_la.sh

# ==== Config pipeline ====
DATA_DIR   ?= $(DATA_ROOT)
FEAT_DIR   ?= features
MODEL_DIR  ?= models
RESULTS_DIR?= results
MODEL_NAME ?= clf.joblib

TRAIN_CSV  ?= $(INDEX_DIR)/train.csv
DEV_CSV    ?= $(INDEX_DIR)/val.csv
TEST_CSV   ?= $(INDEX_DIR)/test.csv

# ==== Phony ====
.PHONY: dataset check_la index clean_index setup features train eval predict clean

# ------------------------
#   Dataset prep (ASVspoof LA)
# ------------------------
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
	echo "[*] Generez indexurile oficiale (train=LA_train, val=LA_dev, + optional test dacă e disponibil)..."; \
	mkdir -p "$(INDEX_DIR)"; \
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
	ls -lh "$(INDEX_DIR)"/train.csv "$(INDEX_DIR)"/val.csv 2>/dev/null || true; \
	ls -lh "$(INDEX_DIR)"/test.csv 2>/dev/null || echo "[i] test.csv nu este disponibil (ok)."

clean_index:
	@rm -rf "$(INDEX_DIR)"; \
	echo "[*] Șters: $(INDEX_DIR)"

# ------------------------
#   Pipeline (features / model / eval / predict)
# ------------------------
setup:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

features: dataset
	@set -e; \
	mkdir -p "$(FEAT_DIR)"; \
	echo "[*] Extract features: train -> $(FEAT_DIR)/train_features.parquet"; \
	$(PY) scripts/extract_features.py --root "$(DATA_DIR)" --csv "$(TRAIN_CSV)" --out "$(FEAT_DIR)/train_features.parquet"; \
	echo "[*] Extract features: dev(val) -> $(FEAT_DIR)/dev_features.parquet"; \
	$(PY) scripts/extract_features.py --root "$(DATA_DIR)" --csv "$(DEV_CSV)"   --out "$(FEAT_DIR)/dev_features.parquet"; \
	if [ -f "$(TEST_CSV)" ]; then \
	  echo "[*] Extract features: test -> $(FEAT_DIR)/test_features.parquet (fără etichete)"; \
	  $(PY) scripts/extract_features.py --root "$(DATA_DIR)" --csv "$(TEST_CSV)" --out "$(FEAT_DIR)/test_features.parquet" --no-labels; \
	else \
	  echo "[i] Nu există $(TEST_CSV); sar peste features pentru test."; \
	fi

train: features
	@set -e; \
	mkdir -p "$(MODEL_DIR)"; \
	echo "[*] Antrenez modelul -> $(MODEL_DIR)/$(MODEL_NAME)"; \
	$(PY) scripts/train.py --train "$(FEAT_DIR)/train_features.parquet" --dev "$(FEAT_DIR)/dev_features.parquet" --out "$(MODEL_DIR)/$(MODEL_NAME)"

eval:
	@set -e; \
	mkdir -p "$(RESULTS_DIR)"; \
	test -f "$(MODEL_DIR)/$(MODEL_NAME)" || (echo "[!] Lipsește modelul: $(MODEL_DIR)/$(MODEL_NAME). Rulează 'make train'." && exit 1); \
	echo "[*] Evaluez pe dev(val) -> $(RESULTS_DIR)"; \
	$(PY) scripts/evaluate.py --features "$(FEAT_DIR)/dev_features.parquet" --model "$(MODEL_DIR)/$(MODEL_NAME)" --outdir "$(RESULTS_DIR)"

predict:
	@set -e; \
	mkdir -p "$(RESULTS_DIR)"; \
	if [ -f "$(FEAT_DIR)/test_features.parquet" ]; then \
	  echo "[*] Rulez predicții pe test -> $(RESULTS_DIR)"; \
	  $(PY) scripts/evaluate.py --features "$(FEAT_DIR)/test_features.parquet" --model "$(MODEL_DIR)/$(MODEL_NAME)" --outdir "$(RESULTS_DIR)" --test; \
	else \
	  echo "Nu există features pentru test. Rulează 'make features' după ce ai $(TEST_CSV)."; \
	fi

clean:
	@set -e; \
	rm -f "$(MODEL_DIR)"/*.joblib 2>/dev/null || true; \
	rm -f "$(RESULTS_DIR)"/*.json "$(RESULTS_DIR)"/*.csv "$(RESULTS_DIR)"/*.png 2>/dev/null || true; \
	echo "[*] Curățate modele și rezultate."; \
	echo "[i] Poți șterge manual și $(DATA_ROOT) dacă vrei să refaci totul."
