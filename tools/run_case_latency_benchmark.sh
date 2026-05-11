#!/usr/bin/env bash
set -euo pipefail

cd /root/mwnoh/ReKV-var-resolution

export PYTHON_BIN="${PYTHON_BIN:-/root/mwnoh/anaconda3/envs/rekv/bin/python}"
export PYTHONPATH="/root/mwnoh/ReKV-var-resolution:/root/mwnoh/ReKV-var-resolution/model:/root/mwnoh/ReKV-var-resolution/model/longva:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export DATASETS="${DATASETS:-cgbench mlvu lvbench}"
export PER_CASE="${PER_CASE:-30}"
export SEED="${SEED:-2026}"
export SAMPLE_DIR="${SAMPLE_DIR:-/root/mwnoh/ReKV-var-resolution/final-results/latency_samples}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/mwnoh/ReKV-var-resolution/final-results/latency_measurements}"

export SAMPLE_FPS="${SAMPLE_FPS:-1}"
export REPEATS="${REPEATS:-3}"
export WARMUP="${WARMUP:-1}"
export SKIP_MISSING_VIDEOS="${SKIP_MISSING_VIDEOS:-0}"
export WARM_RETRIEVAL_CACHE="${WARM_RETRIEVAL_CACHE:-0}"
export MEASURE_VERIFIER="${MEASURE_VERIFIER:-0}"
export VERIFIER_MODEL="${VERIFIER_MODEL:-${LLM_ROUTER_MODEL:-}}"
export VERIFIER_RESPONSE_FORMAT_JSON="${VERIFIER_RESPONSE_FORMAT_JSON:-1}"
export INCLUDE_TASK="${INCLUDE_TASK:-0}"

export CGBENCH_ANNO_PATH="${CGBENCH_ANNO_PATH:-/root/mwnoh/ReKV-var-resolution/data/cgbench/full_mc.json}"
export MLVU_ANNO_PATH="${MLVU_ANNO_PATH:-/root/mwnoh/ReKV-var-resolution/data/mlvu/dev_debug_mc.json}"
export LVBENCH_ANNO_PATH="${LVBENCH_ANNO_PATH:-/mnt/ssd1/mwnoh/LVBench/data/video_info.json}"

# Optional path rewrites for relative annotation video paths.
# Example:
#   CGBENCH_VIDEO_ROOT='data/cgbench/videos=/mnt/ssd1/mwnoh/cgbench/videos'
#   MLVU_VIDEO_ROOT='data/mlvu/videos=/mnt/ssd1/mwnoh/mlvu/videos'
export CGBENCH_VIDEO_ROOT="${CGBENCH_VIDEO_ROOT:-}"
export MLVU_VIDEO_ROOT="${MLVU_VIDEO_ROOT:-}"
export LVBENCH_VIDEO_ROOT="${LVBENCH_VIDEO_ROOT:-}"

mkdir -p "${OUTPUT_DIR}"

truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "True" ]]
}

run_one() {
  local dataset="$1"
  local anno_path=""
  local video_root=""
  case "${dataset}" in
    cgbench)
      anno_path="${CGBENCH_ANNO_PATH}"
      video_root="${CGBENCH_VIDEO_ROOT}"
      ;;
    mlvu)
      anno_path="${MLVU_ANNO_PATH}"
      video_root="${MLVU_VIDEO_ROOT}"
      ;;
    lvbench)
      anno_path="${LVBENCH_ANNO_PATH}"
      video_root="${LVBENCH_VIDEO_ROOT}"
      ;;
    *)
      echo "Unknown dataset '${dataset}'. Use one or more of: cgbench mlvu lvbench." >&2
      exit 1
      ;;
  esac

  local samples_csv="${SAMPLE_DIR}/${dataset}_latency_cases_per${PER_CASE}_seed${SEED}.csv"
  local output_csv="${OUTPUT_DIR}/${dataset}_case_latency_per${PER_CASE}_seed${SEED}.csv"
  local summary_json="${OUTPUT_DIR}/${dataset}_case_latency_per${PER_CASE}_seed${SEED}_summary.json"

  declare -a extra_args=()
  if [[ -n "${video_root}" ]]; then
    extra_args+=(--video-root "${video_root}")
  fi
  if truthy "${SKIP_MISSING_VIDEOS}"; then
    extra_args+=(--skip-missing-videos)
  fi
  if truthy "${WARM_RETRIEVAL_CACHE}"; then
    extra_args+=(--warm-retrieval-cache)
  fi
  if truthy "${MEASURE_VERIFIER}"; then
    if [[ -z "${VERIFIER_MODEL}" ]]; then
      echo "MEASURE_VERIFIER=1 requires VERIFIER_MODEL or LLM_ROUTER_MODEL." >&2
      exit 1
    fi
    extra_args+=(--measure-verifier --verifier-model "${VERIFIER_MODEL}")
  fi
  if truthy "${VERIFIER_RESPONSE_FORMAT_JSON}"; then
    extra_args+=(--verifier-response-format-json)
  fi
  if truthy "${INCLUDE_TASK}"; then
    extra_args+=(--include-task)
  fi

  echo "==== benchmark dataset=${dataset} ===="
  echo "==== samples_csv=${samples_csv} ===="
  echo "==== anno_path=${anno_path} ===="
  echo "==== output_csv=${output_csv} ===="

  "${PYTHON_BIN}" tools/benchmark_routing_case_latency.py \
    --samples-csv "${samples_csv}" \
    --output-csv "${output_csv}" \
    --summary-json "${summary_json}" \
    --anno-path "${anno_path}" \
    --sample-fps "${SAMPLE_FPS}" \
    --repeats "${REPEATS}" \
    --warmup "${WARMUP}" \
    "${extra_args[@]}"
}

for dataset in ${DATASETS}; do
  run_one "${dataset}"
done
