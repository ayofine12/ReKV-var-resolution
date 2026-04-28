#!/usr/bin/env bash
set -euo pipefail

cd /root/mwnoh/ReKV-var-resolution

export PYTHONPATH="/root/mwnoh/ReKV-var-resolution:/root/mwnoh/ReKV-var-resolution/model:/root/mwnoh/ReKV-var-resolution/model/longva:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="/root/mwnoh/anaconda3/envs/rekv/bin/python"
PROGRAM="/root/mwnoh/ReKV-var-resolution/tools/benchmark_qwen_fs_latency.py"
ANNO_PATH="${ANNO_PATH:-/mnt/ssd1/mwnoh/LVBench/data/video_info.json}"
VIDEO_ID="${VIDEO_ID:-Cm73ma6Ibcs}"
SAMPLE_FPS="${SAMPLE_FPS:-1}"
QUESTION_LIMIT="${QUESTION_LIMIT:-8}"
REPEATS="${REPEATS:-3}"
LOCAL_BLOCK_COUNT="${LOCAL_BLOCK_COUNT:-32}"
RETRIEVE_SIZE="${RETRIEVE_SIZE:-128}"
RETRIEVE_CHUNK_SIZE="${RETRIEVE_CHUNK_SIZE:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/mwnoh/latency-benchmarks}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_PATH="${OUTPUT_DIR}/fs112_${VIDEO_ID}_lb${LOCAL_BLOCK_COUNT}_rs${RETRIEVE_SIZE}_fps${SAMPLE_FPS}_rep${REPEATS}_${TIMESTAMP}.txt"

mkdir -p "${OUTPUT_DIR}"

echo "==== Benchmarking fs112 on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ===="
echo "video_id=${VIDEO_ID}, sample_fps=${SAMPLE_FPS}, question_limit=${QUESTION_LIMIT}, repeats=${REPEATS}"
echo "saving_to=${OUTPUT_PATH}"

"${PYTHON_BIN}" "${PROGRAM}" \
  --anno_path "${ANNO_PATH}" \
  --video_id "${VIDEO_ID}" \
  --sample_fps "${SAMPLE_FPS}" \
  --frame_size 112 \
  --local_block_count "${LOCAL_BLOCK_COUNT}" \
  --retrieve_size "${RETRIEVE_SIZE}" \
  --retrieve_chunk_size "${RETRIEVE_CHUNK_SIZE}" \
  --question_limit "${QUESTION_LIMIT}" \
  --repeats "${REPEATS}" | tee "${OUTPUT_PATH}"
