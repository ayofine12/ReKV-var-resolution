#!/usr/bin/env bash
set -euo pipefail

LV_ROOT="${LV_ROOT:-/mnt/ssd1/mwnoh/LVBench}"
META="${META:-${LV_ROOT}/data/video_info.meta.jsonl}"
VIDEO_DIR="${VIDEO_DIR:-${LV_ROOT}/scripts/videos/00000}"
LOG_DIR="${LOG_DIR:-/mnt/ssd1/mwnoh/download_logs}"
YTDLP="${YTDLP:-/root/mwnoh/anaconda3/envs/rekv/bin/yt-dlp}"
COOKIES="${COOKIES:-}"
SLEEP_INTERVAL="${SLEEP_INTERVAL:-45}"
MAX_SLEEP_INTERVAL="${MAX_SLEEP_INTERVAL:-120}"
LIMIT_RATE="${LIMIT_RATE:-2M}"

mkdir -p "${VIDEO_DIR}" "${LOG_DIR}"

MISSING_LIST="${LOG_DIR}/lvbench_missing_index_keys.txt"
RETRY_LOG="${LOG_DIR}/lvbench_retry_missing.log"
ARCHIVE="${LOG_DIR}/lvbench_yt_dlp_archive.txt"

python - "${META}" "${VIDEO_DIR}" > "${MISSING_LIST}" <<'PY'
import json
import sys
from pathlib import Path

meta = Path(sys.argv[1])
video_dir = Path(sys.argv[2])
video_exts = {".mp4", ".webm", ".mkv", ".mov", ".m4v"}

for idx, line in enumerate(meta.read_text().splitlines()):
    if not line.strip():
        continue
    item = json.loads(line)
    existing = [
        path
        for path in video_dir.glob(f"{idx:08d}.*")
        if path.suffix.lower() in video_exts
    ]
    if not existing:
        print(f"{idx}\t{item['key']}")
PY

echo "[setup] missing videos: $(wc -l < "${MISSING_LIST}")" | tee -a "${RETRY_LOG}"
echo "[setup] missing list: ${MISSING_LIST}" | tee -a "${RETRY_LOG}"

cookie_args=()
if [[ -n "${COOKIES}" ]]; then
  cookie_args=(--cookies "${COOKIES}")
  echo "[setup] using cookies: ${COOKIES}" | tee -a "${RETRY_LOG}"
else
  echo "[setup] no cookies provided; bot-check videos may still fail" | tee -a "${RETRY_LOG}"
fi

while IFS=$'\t' read -r idx key; do
  [[ -n "${idx}" && -n "${key}" ]] || continue
  padded="$(printf "%08d" "${idx}")"
  out="${VIDEO_DIR}/${padded}.%(ext)s"
  url="https://www.youtube.com/watch?v=${key}"

  if compgen -G "${VIDEO_DIR}/${padded}.mp4" > /dev/null || \
     compgen -G "${VIDEO_DIR}/${padded}.webm" > /dev/null || \
     compgen -G "${VIDEO_DIR}/${padded}.mkv" > /dev/null || \
     compgen -G "${VIDEO_DIR}/${padded}.mov" > /dev/null || \
     compgen -G "${VIDEO_DIR}/${padded}.m4v" > /dev/null; then
    echo "[skip] ${padded} ${key} already exists" | tee -a "${RETRY_LOG}"
    continue
  fi

  echo "[retry] ${padded} ${key}" | tee -a "${RETRY_LOG}"
  ytdlp_args=(
    --ignore-errors
    --no-playlist
    --retries 8
    --fragment-retries 8
    --download-archive "${ARCHIVE}"
    --merge-output-format mp4
    -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best"
    -o "${out}"
  )
  if [[ "${SLEEP_INTERVAL}" != "0" ]]; then
    ytdlp_args+=(
      --sleep-requests "${SLEEP_INTERVAL}"
      --min-sleep-interval "${SLEEP_INTERVAL}"
      --max-sleep-interval "${MAX_SLEEP_INTERVAL}"
    )
  fi
  if [[ -n "${LIMIT_RATE}" && "${LIMIT_RATE}" != "0" ]]; then
    ytdlp_args+=(--limit-rate "${LIMIT_RATE}")
  fi

  "${YTDLP}" "${ytdlp_args[@]}" "${cookie_args[@]}" "${url}" 2>&1 | tee -a "${RETRY_LOG}" || true
done < "${MISSING_LIST}"

echo "[done] retry pass finished" | tee -a "${RETRY_LOG}"
