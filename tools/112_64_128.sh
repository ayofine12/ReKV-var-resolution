#!/usr/bin/env bash
set -euo pipefail

cd /root/mwnoh/ReKV-var-resolution

export REKV_VIDEO_CACHE_DIR="/mnt/ssd1/mwnoh/qaego4d/cache"
export PYTHONPATH="/root/mwnoh/ReKV-var-resolution:/root/mwnoh/ReKV-var-resolution/model:/root/mwnoh/ReKV-var-resolution/model/longva:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="/root/mwnoh/anaconda3/envs/rekv/bin/python"
PROGRAM="/root/mwnoh/ReKV-var-resolution/video_qa/rekv_offline_vqa.py"
ANNO_PATH="/mnt/ssd1/mwnoh/LVBench/data/video_info.json"
BASE_SAVE_DIR="/mnt/ssd1/mwnoh/var-resolution-screen"
START_VIDEO_ID="${START_VIDEO_ID:-}"

# Budget-matched config:
# fs112 -> 16 tokens/frame
# local:    16 * 64  = 1024
# retrieve: 16 * 128 = 2048
declare -a CONFIGS=(
  "112 64 128"
)

for config in "${CONFIGS[@]}"; do
  read -r frame_size local_block_count retrieve_size <<<"${config}"
  save_dir="${BASE_SAVE_DIR}/fs${frame_size}_lb${local_block_count}_rs${retrieve_size}"
  extra_args=()
  if [[ -n "${START_VIDEO_ID}" ]]; then
    extra_args+=(--start_video_id "${START_VIDEO_ID}")
  fi

  echo "==== Running frame_size=${frame_size}, local_block_count=${local_block_count}, retrieve_size=${retrieve_size} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ===="
  if [[ -n "${START_VIDEO_ID}" ]]; then
    echo "==== Restarting from video_id=${START_VIDEO_ID} ===="
  fi

  "${PYTHON_BIN}" "${PROGRAM}" \
    --sample_fps 1 \
    --save_dir "${save_dir}" \
    --anno_path "${ANNO_PATH}" \
    --model qwen2_5_vl_7b \
    --frame_size "${frame_size}" \
    --local_block_count "${local_block_count}" \
    --retrieve_size "${retrieve_size}" \
    --retrieve_chunk_size 1 \
    --debug False \
    "${extra_args[@]}"
done
