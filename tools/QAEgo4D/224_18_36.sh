#!/usr/bin/env bash
set -euo pipefail

cd /root/mwnoh/ReKV-var-resolution

export REKV_VIDEO_CACHE_DIR="${REKV_VIDEO_CACHE_DIR:-/mnt/ssd1/mwnoh/qaego4d/cache}"
export PYTHONPATH="/root/mwnoh/ReKV-var-resolution:/root/mwnoh/ReKV-var-resolution/model:/root/mwnoh/ReKV-var-resolution/model/longva:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

export PYTHON_BIN="${PYTHON_BIN:-/root/mwnoh/anaconda3/envs/rekv/bin/python}"
export PROGRAM="/root/mwnoh/ReKV-var-resolution/video_qa/rekv_offline_vqa.py"
export ANNO_SRC="${ANNO_SRC:-/mnt/ssd1/mwnoh/qaego4d/test_mc.json}"
export ANNO_ABS="${ANNO_ABS:-/tmp/qaego4d_test_mc_abs.json}"
export BASE_VIDEO_DIR="${BASE_VIDEO_DIR:-/mnt/ssd1/mwnoh/qaego4d/videos}"
export BASE_SAVE_DIR="${BASE_SAVE_DIR:-/mnt/ssd1/mwnoh/var-resolution-qaego4d}"
export MODEL="${MODEL:-qwen2_5_vl_7b}"
export SAMPLE_FPS="${SAMPLE_FPS:-1}"
export RETRIEVE_CHUNK_SIZE="${RETRIEVE_CHUNK_SIZE:-1}"
export DEBUG="${DEBUG:-False}"
export START_VIDEO_ID="${START_VIDEO_ID:-}"

"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

anno_src = Path(os.environ["ANNO_SRC"])
anno_abs = Path(os.environ["ANNO_ABS"])
base_video_dir = Path(os.environ["BASE_VIDEO_DIR"])

with anno_src.open() as fh:
    anno = json.load(fh)

for sample in anno:
    video_path = sample.get("video_path", "")
    if video_path and not os.path.isabs(video_path):
        sample["video_path"] = str(base_video_dir / Path(video_path).name)

anno_abs.parent.mkdir(parents=True, exist_ok=True)
with anno_abs.open("w") as fh:
    json.dump(anno, fh)

print(f"normalized annotation saved to {anno_abs}")
PY

# Exact budget-matched config:
# fs224 -> 64 tokens/frame
# local:    64 * 18 = 1152
# retrieve: 64 * 36 = 2304
declare -a CONFIGS=(
  "224 18 36"
)

for config in "${CONFIGS[@]}"; do
  read -r frame_size local_block_count retrieve_size <<<"${config}"
  save_dir="${BASE_SAVE_DIR}/fs${frame_size}_lb${local_block_count}_rs${retrieve_size}"
  extra_args=()
  if [[ -n "${START_VIDEO_ID}" ]]; then
    extra_args+=(--start_video_id "${START_VIDEO_ID}")
  fi

  echo "==== Running QAEgo4D frame_size=${frame_size}, local_block_count=${local_block_count}, retrieve_size=${retrieve_size} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ===="
  echo "==== anno_path=${ANNO_ABS} ===="
  if [[ -n "${START_VIDEO_ID}" ]]; then
    echo "==== Restarting from video_id=${START_VIDEO_ID} ===="
  fi

  "${PYTHON_BIN}" "${PROGRAM}" \
    --sample_fps "${SAMPLE_FPS}" \
    --save_dir "${save_dir}" \
    --anno_path "${ANNO_ABS}" \
    --model "${MODEL}" \
    --frame_size "${frame_size}" \
    --local_block_count "${local_block_count}" \
    --retrieve_size "${retrieve_size}" \
    --retrieve_chunk_size "${RETRIEVE_CHUNK_SIZE}" \
    --debug "${DEBUG}" \
    "${extra_args[@]}"
done
