#!/usr/bin/env bash
set -euo pipefail

cd /root/mwnoh/ReKV-var-resolution

export REKV_VIDEO_CACHE_DIR="/mnt/ssd1/mwnoh/qaego4d/cache"
export PYTHONPATH="/root/mwnoh/ReKV-var-resolution:/root/mwnoh/ReKV-var-resolution/model:/root/mwnoh/ReKV-var-resolution/model/longva:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PYTHON_BIN="/root/mwnoh/anaconda3/envs/rekv/bin/python"
PROGRAM="/root/mwnoh/ReKV-var-resolution/video_qa/rekv_offline_vqa.py"
ANNO_PATH="/mnt/ssd1/mwnoh/LVBench/data/video_info.json"
BASE_SAVE_DIR="/mnt/ssd1/mwnoh/var-resolution-screen"

# Approximate retrieval-token-budget-matched config:
# fs168 -> 36 tokens/frame -> rs57 => 2052 retrieved tokens
declare -a CONFIGS=(
  "168 32 57"
)

for config in "${CONFIGS[@]}"; do
  read -r frame_size local_block_count retrieve_size <<<"${config}"
  save_dir="${BASE_SAVE_DIR}/fs${frame_size}_lb${local_block_count}_rs${retrieve_size}"

  echo "==== Running frame_size=${frame_size}, local_block_count=${local_block_count}, retrieve_size=${retrieve_size} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ===="

  "${PYTHON_BIN}" "${PROGRAM}" \
    --sample_fps 1 \
    --save_dir "${save_dir}" \
    --anno_path "${ANNO_PATH}" \
    --model qwen2_5_vl_7b \
    --frame_size "${frame_size}" \
    --local_block_count "${local_block_count}" \
    --retrieve_size "${retrieve_size}" \
    --retrieve_chunk_size 1 \
    --debug False
done
