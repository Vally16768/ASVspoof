#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv/bin/python"
INDEX_DIR="$($PY - <<'PY'
from pathlib import Path
from constants import directory, index_folder_name
print((Path(directory).resolve() / index_folder_name).as_posix())
PY
)"

run_step() {
  echo "[STEP] $*"
  "$@"
}

merge_flag=()
if [[ -f "${INDEX_DIR}/features_all.parquet" ]]; then
  merge_flag=(--merge-existing)
fi

# 1) Low-memory non-SSL extraction (also LPCC/CQCC)
run_step "$PY" -m asvspoof.cli extract \
  --groups zcr_rms spectral_basic spectral_contrast chroma mfcc pitch wavelets lpcc cqcc \
  "${merge_flag[@]}"

# From now on we always merge incremental SSL features into existing table.
merge_flag=(--merge-existing)

# 2) wav2vec with memory-safe SSL PCA path (train-fit + full transform)
run_step "$PY" -m asvspoof.cli extract \
  --groups wav2vec \
  "${merge_flag[@]}"

# 3) WaveLM with memory-safe SSL PCA path
run_step "$PY" -m asvspoof.cli extract \
  --groups wavlm \
  "${merge_flag[@]}"

# 4) Benchmark combinations + final_model update (skip extract, now completed)
run_step "$PY" scripts/benchmark_extend_best.py --base-code AHKLMNO --skip-extract

echo "[DONE] $(date -Iseconds)"
