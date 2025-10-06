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
