#!/usr/bin/env bash
set -euo pipefail

LV_ROOT="${LV_ROOT:-/mnt/ssd1/mwnoh/LVBench}"
VIDEO_DIR="${VIDEO_DIR:-${LV_ROOT}/scripts/videos}"
LOG_DIR="${LOG_DIR:-/mnt/ssd1/mwnoh/download_logs}"
YTDLP="${YTDLP:-/root/mwnoh/anaconda3/envs/rekv/bin/yt-dlp}"
COOKIES="${COOKIES:-}"
SLEEP_INTERVAL="${SLEEP_INTERVAL:-0}"
LIMIT_RATE="${LIMIT_RATE:-0}"

mkdir -p "${VIDEO_DIR}" "${LOG_DIR}"

BAD_LIST="${BAD_LIST:-${LOG_DIR}/lvbench_bad_decord_keys.txt}"
REDOWNLOAD_LOG="${LOG_DIR}/lvbench_redownload_bad_h264.log"
BACKUP_DIR="${VIDEO_DIR}/bad_decord_backup"

cat > "${BAD_LIST}" <<'EOF'
1FsiZgGZU70
2LH3JCGkEBU
4LA_tH-VSnQ
9-gOCOu_KGU
9tBsMSDoDqk
AeEYQ62t8hA
Aiem1w_TvaA
Hf-n1yfd8II
JTa_Ue2MSwc
JlrzSvCsIjE
Va_9Q6ekm60
Z86xysw5Ncc
cWEnogdsW78
cXDT44zT8JY
hjoDzK0siaM
jp2M1hIEtsk
k2FIFQIYBvA
oZEVgDXJwCc
pe_LddfHAUU
rp4NKWb7dXk
vHlSoxg8WHo
EOF

if [[ ! -x "${YTDLP}" ]]; then
  echo "[error] yt-dlp not found or not executable: ${YTDLP}" | tee -a "${REDOWNLOAD_LOG}"
  exit 1
fi

mkdir -p "${BACKUP_DIR}"

cookie_args=()
if [[ -n "${COOKIES}" ]]; then
  cookie_args=(--cookies "${COOKIES}")
  echo "[setup] using cookies: ${COOKIES}" | tee -a "${REDOWNLOAD_LOG}"
else
  echo "[setup] no cookies provided" | tee -a "${REDOWNLOAD_LOG}"
fi

echo "[setup] bad list: ${BAD_LIST}" | tee -a "${REDOWNLOAD_LOG}"
echo "[setup] video dir: ${VIDEO_DIR}" | tee -a "${REDOWNLOAD_LOG}"
echo "[setup] backup dir: ${BACKUP_DIR}" | tee -a "${REDOWNLOAD_LOG}"

while IFS= read -r key; do
  [[ -n "${key}" ]] || continue

  src="${VIDEO_DIR}/${key}.mp4"
  backup="${BACKUP_DIR}/${key}.mp4"
  url="https://www.youtube.com/watch?v=${key}"

  if [[ -f "${src}" && ! -f "${backup}" ]]; then
    mv "${src}" "${backup}"
    echo "[backup] ${src} -> ${backup}" | tee -a "${REDOWNLOAD_LOG}"
  elif [[ -f "${src}" ]]; then
    rm -f "${src}"
    echo "[replace] removed previous retry output: ${src}" | tee -a "${REDOWNLOAD_LOG}"
  fi

  echo "[download] ${key}" | tee -a "${REDOWNLOAD_LOG}"
  ytdlp_args=(
    --no-playlist
    --retries 8
    --fragment-retries 8
    --no-part
    --force-overwrites
    -f "22/18/best[ext=mp4][vcodec^=avc1]/best[ext=mp4]"
    -o "${VIDEO_DIR}/${key}.%(ext)s"
  )
  if [[ "${SLEEP_INTERVAL}" != "0" ]]; then
    ytdlp_args+=(--sleep-requests "${SLEEP_INTERVAL}")
  fi
  if [[ -n "${LIMIT_RATE}" && "${LIMIT_RATE}" != "0" ]]; then
    ytdlp_args+=(--limit-rate "${LIMIT_RATE}")
  fi

  "${YTDLP}" "${ytdlp_args[@]}" "${cookie_args[@]}" "${url}" 2>&1 | tee -a "${REDOWNLOAD_LOG}" || true
done < "${BAD_LIST}"

echo "[done] redownload pass finished" | tee -a "${REDOWNLOAD_LOG}"
