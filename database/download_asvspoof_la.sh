#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Unde vrei să stea datele (poți schimba)
DATA_DIR="${1:-${REPO_ROOT}/database/data/asvspoof2019}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
KEEP_ZIP="${KEEP_ZIP:-0}"

# Link direct spre LA.zip de pe Edinburgh DataShare (poate avea alt "sequence" în timp)
LA_URL="https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?isAllowed=y&sequence=3"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "[*] Destination: $(pwd)"
AVAIL_KB="$(df -Pk . | awk 'NR==2 {print $4}')"
AVAIL_GB="$(( AVAIL_KB / 1024 / 1024 ))"
echo "[*] Free space: ~${AVAIL_GB} GB (required >= ${MIN_FREE_GB} GB)"
if [ "$AVAIL_GB" -lt "$MIN_FREE_GB" ]; then
  echo "[!] Not enough free space to safely download+extract ASVspoof LA."
  echo "    Set ASVSPOOF_ROOT to a larger disk or lower MIN_FREE_GB if you accept the risk."
  exit 1
fi

if [ ! -f LA.zip ]; then
  echo "[*] Downloading ASVspoof2019 LA.zip (resume enabled)..."
  # wget first (more reliable in practice), then curl fallback
  (wget --content-disposition -c "$LA_URL" -O LA.zip) || (curl -L -C - -o LA.zip "$LA_URL")
fi

echo "[*] Extracting LA.zip..."
unzip -q -o LA.zip

# LA.zip se dezarhivează într-un folder 'LA/' care conține:
#  ASVspoof2019_LA_train/, _dev/, _eval/, și ASVspoof2019_LA_cm_protocols/
if [ -d LA ]; then
  shopt -s dotglob
  mv LA/* . || true
  rmdir LA || true
fi

if [ "$KEEP_ZIP" != "1" ] && [ -f LA.zip ]; then
  echo "[*] Removing LA.zip to save disk..."
  rm -f LA.zip
fi

echo "[*] Done. Found:"
ls -1d ASVspoof2019_LA_* ASVspoof2019_LA_cm_protocols | sed 's/^/   - /'

echo "[i] Codec: FLAC. License: ODC-By (atribution)."
