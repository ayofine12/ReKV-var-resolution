#!/usr/bin/env bash
set -euo pipefail

cd /root/mwnoh/ReKV-var-resolution

export PYTHON_BIN="${PYTHON_BIN:-/root/mwnoh/anaconda3/envs/rekv/bin/python}"
export DATASETS="${DATASETS:-lvbench}"
export PER_CASE="${PER_CASE:-5}"
export SEED="${SEED:-2026}"
export SAMPLE_DIR="${SAMPLE_DIR:-/root/mwnoh/ReKV-var-resolution/final-results/latency_samples}"
export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/root/mwnoh/ReKV-var-resolution/final-results/retrieve_size_latency_sweep}"
export SAMPLE_FPS="${SAMPLE_FPS:-1}"
export REPEATS="${REPEATS:-3}"
export WARMUP="${WARMUP:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export REKV_BATCHED_RETRIEVAL_IO="${REKV_BATCHED_RETRIEVAL_IO:-1}"

export BASE_FS224_RETRIEVE_SIZE="${BASE_FS224_RETRIEVE_SIZE:-36}"
export BASE_FS112_RETRIEVE_SIZE="${BASE_FS112_RETRIEVE_SIZE:-144}"
export FS224_RETRIEVE_SIZES="${FS224_RETRIEVE_SIZES:-18 36 72 108 144}"
export FS112_RETRIEVE_SIZES="${FS112_RETRIEVE_SIZES:-72 144 288 432 576}"

mkdir -p "${BASE_OUTPUT_DIR}"

SUMMARY_CSV="${BASE_OUTPUT_DIR}/retrieve_size_latency_sweep_summary.csv"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  printf "dataset,sweep_fs,retrieve_size,fs224_retrieve_size,fs112_retrieve_size,per_case,seed,batched_io,weighted_video_side_ms,weighted_fs224_only_ms,relative_video_side_cost_vs_fs224,case1_fs224_mean_ms,case2_fs224_mean_ms,case2_fs112_mean_ms,case3_fs224_mean_ms,case3_fs112_mean_ms,summary_json\n" > "${SUMMARY_CSV}"
fi

append_summary() {
  local dataset="$1"
  local sweep_fs="$2"
  local retrieve_size="$3"
  local fs224_retrieve_size="$4"
  local fs112_retrieve_size="$5"
  local summary_json="$6"

  "${PYTHON_BIN}" - "${SUMMARY_CSV}" "${dataset}" "${sweep_fs}" "${retrieve_size}" \
    "${fs224_retrieve_size}" "${fs112_retrieve_size}" "${PER_CASE}" "${SEED}" \
    "${REKV_BATCHED_RETRIEVAL_IO}" "${summary_json}" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
dataset = sys.argv[2]
sweep_fs = sys.argv[3]
retrieve_size = sys.argv[4]
fs224_retrieve_size = sys.argv[5]
fs112_retrieve_size = sys.argv[6]
per_case = sys.argv[7]
seed = sys.argv[8]
batched_io = sys.argv[9]
summary_json = Path(sys.argv[10])

with summary_json.open(encoding="utf-8") as fh:
    summary = json.load(fh)

case_summary = summary.get("case_summary", {})

def case_value(case_name, key):
    value = case_summary.get(case_name, {}).get(key, "")
    return "" if value is None else value

row = {
    "dataset": dataset,
    "sweep_fs": sweep_fs,
    "retrieve_size": retrieve_size,
    "fs224_retrieve_size": fs224_retrieve_size,
    "fs112_retrieve_size": fs112_retrieve_size,
    "per_case": per_case,
    "seed": seed,
    "batched_io": batched_io,
    "weighted_video_side_ms": summary.get("weighted_video_side_ms", ""),
    "weighted_fs224_only_ms": summary.get("weighted_fs224_only_ms", ""),
    "relative_video_side_cost_vs_fs224": summary.get("relative_video_side_cost_vs_fs224", ""),
    "case1_fs224_mean_ms": case_value("case1_high_confidence", "fs224_mean_ms"),
    "case2_fs224_mean_ms": case_value("case2_low_confidence_agree", "fs224_mean_ms"),
    "case2_fs112_mean_ms": case_value("case2_low_confidence_agree", "fs112_mean_ms"),
    "case3_fs224_mean_ms": case_value("case3_low_confidence_disagree", "fs224_mean_ms"),
    "case3_fs112_mean_ms": case_value("case3_low_confidence_disagree", "fs112_mean_ms"),
    "summary_json": str(summary_json),
}

with summary_csv.open("a", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(row))
    writer.writerow(row)
PY
}

run_one() {
  local dataset="$1"
  local sweep_fs="$2"
  local retrieve_size="$3"
  local fs224_retrieve_size="$4"
  local fs112_retrieve_size="$5"
  local output_dir="${BASE_OUTPUT_DIR}/${dataset}/${sweep_fs}_rs${retrieve_size}"
  local summary_json="${output_dir}/${dataset}_case_latency_per${PER_CASE}_seed${SEED}_summary.json"

  echo "==== retrieve-size latency sweep ===="
  echo "dataset=${dataset} sweep_fs=${sweep_fs} retrieve_size=${retrieve_size}"
  echo "fs224_retrieve_size=${fs224_retrieve_size} fs112_retrieve_size=${fs112_retrieve_size}"
  echo "output_dir=${output_dir}"

  DATASETS="${dataset}" \
  OUTPUT_DIR="${output_dir}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  SAMPLE_DIR="${SAMPLE_DIR}" \
  PER_CASE="${PER_CASE}" \
  SEED="${SEED}" \
  SAMPLE_FPS="${SAMPLE_FPS}" \
  REPEATS="${REPEATS}" \
  WARMUP="${WARMUP}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  FS224_RETRIEVE_SIZE="${fs224_retrieve_size}" \
  FS112_RETRIEVE_SIZE="${fs112_retrieve_size}" \
  tools/run_case_latency_benchmark.sh

  append_summary "${dataset}" "${sweep_fs}" "${retrieve_size}" \
    "${fs224_retrieve_size}" "${fs112_retrieve_size}" "${summary_json}"
}

for dataset in ${DATASETS}; do
  for retrieve_size in ${FS224_RETRIEVE_SIZES}; do
    run_one "${dataset}" "fs224" "${retrieve_size}" \
      "${retrieve_size}" "${BASE_FS112_RETRIEVE_SIZE}"
  done

  for retrieve_size in ${FS112_RETRIEVE_SIZES}; do
    run_one "${dataset}" "fs112" "${retrieve_size}" \
      "${BASE_FS224_RETRIEVE_SIZE}" "${retrieve_size}"
  done
done

echo "==== summary ===="
echo "${SUMMARY_CSV}"
