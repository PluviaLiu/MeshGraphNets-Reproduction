#!/bin/bash
# Faster drop-in replacement for the official meshgraphnets/download_dataset.sh.
#
# The official script uses `wget`, which on this host tops out around 67 KB/s
# against the dm-meshgraphnets GCS bucket. Plain `curl` hits ~9.5 MB/s on the
# same object, and ~14 MB/s aggregate across parallel range requests, so a
# 12 GB dataset goes from ~2 days to ~15 minutes.
#
# Usage: bash fetch_dataset.sh <dataset_name> <output_dir>
#   e.g. bash fetch_dataset.sh flag_simple /data/HOI/LiuSiyu/MeshGraphNets/data
#
# Adds resumability (`-C -`), retries, and parallel per-file download. Safe to
# re-run: already-complete files are detected via Content-Length and skipped.

set -u

DATASET_NAME="${1:?usage: fetch_dataset.sh <dataset_name> <output_dir>}"
OUTPUT_DIR="${2:?usage: fetch_dataset.sh <dataset_name> <output_dir>}/${DATASET_NAME}"
BASE_URL="https://storage.googleapis.com/dm-meshgraphnets/${DATASET_NAME}/"

mkdir -p "${OUTPUT_DIR}"

download_one() {
  local file="$1"
  local url="${BASE_URL}${file}"
  local dest="${OUTPUT_DIR}/${file}"

  local remote_size
  remote_size=$(curl -sI --max-time 60 "${url}" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r')
  local local_size=0
  [ -f "${dest}" ] && local_size=$(stat -c %s "${dest}")

  if [ -n "${remote_size}" ] && [ "${local_size}" = "${remote_size}" ]; then
    echo "[skip] ${file} already complete (${local_size} bytes)"
    return 0
  fi

  echo "[get ] ${file} (local=${local_size} remote=${remote_size})"
  # -C - resumes from the current file size; --retry handles the flaky GCS API.
  curl -L -C - --retry 10 --retry-delay 3 --retry-all-errors \
       --connect-timeout 30 --max-time 7200 \
       -o "${dest}" "${url}"

  local final_size
  final_size=$(stat -c %s "${dest}" 2>/dev/null || echo 0)
  if [ -n "${remote_size}" ] && [ "${final_size}" != "${remote_size}" ]; then
    echo "[FAIL] ${file}: got ${final_size}, expected ${remote_size}" >&2
    return 1
  fi
  echo "[ ok ] ${file} (${final_size} bytes)"
}

rc=0
for file in meta.json train.tfrecord valid.tfrecord test.tfrecord; do
  download_one "${file}" &
done
wait || rc=1

echo "--- ${OUTPUT_DIR} ---"
ls -lh "${OUTPUT_DIR}"
exit "${rc}"
