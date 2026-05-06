#!/usr/bin/env bash
set -euo pipefail

cd /root/mwnoh/ReKV-var-resolution

export REKV_VIDEO_CACHE_DIR="${REKV_VIDEO_CACHE_DIR:-/mnt/ssd1/mwnoh/MLVU/cache}"
export PYTHONPATH="/root/mwnoh/ReKV-var-resolution:/root/mwnoh/ReKV-var-resolution/model:/root/mwnoh/ReKV-var-resolution/model/longva:${PYTHONPATH:-}"

STAGE="${STAGE:-all}" # all, scores, router, dry-run
SPLITS="${SPLITS:-front back}"
PYTHON_BIN="${PYTHON_BIN:-/root/mwnoh/anaconda3/envs/rekv/bin/python}"
ROUTER_PYTHON_BIN="${ROUTER_PYTHON_BIN:-/root/mwnoh/anaconda3/bin/python3}"
PROGRAM="${PROGRAM:-/root/mwnoh/ReKV-var-resolution/video_qa/rekv_offline_vqa.py}"
ANNO_PATH_FRONT="${ANNO_PATH_FRONT:-/mnt/ssd1/mwnoh/MLVU/MLVU/annotations/mlvu_front_rekv.json}"
ANNO_PATH_BACK="${ANNO_PATH_BACK:-/mnt/ssd1/mwnoh/MLVU/MLVU/annotations/mlvu_back_rekv.json}"
BASE_SAVE_DIR="${BASE_SAVE_DIR:-/mnt/ssd1/mwnoh/var-resolution-mlvu-confidence}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/mwnoh/ReKV-var-resolution/results}"

VQA_MODEL="${VQA_MODEL:-qwen2_5_vl_7b}"
SAMPLE_FPS="${SAMPLE_FPS:-1}"
RETRIEVE_CHUNK_SIZE="${RETRIEVE_CHUNK_SIZE:-1}"
DEBUG="${DEBUG:-False}"
SAVE_CHOICE_SCORES="${SAVE_CHOICE_SCORES:-True}"
START_VIDEO_ID="${START_VIDEO_ID:-}"
GPU_FS112="${GPU_FS112:-0}"
GPU_FS224="${GPU_FS224:-1}"

GATE_COLUMN="${GATE_COLUMN:-prob_margin}"
GATE_THRESHOLD="${GATE_THRESHOLD:-0.20}"
LOW_CONFIDENCE_WHEN="${LOW_CONFIDENCE_WHEN:-lt}"
VERIFIER="${VERIFIER:-confidence}"
CONFIDENCE_COMPARE_COLUMN="${CONFIDENCE_COMPARE_COLUMN:-prob_margin}"
DEFAULT_FS="${DEFAULT_FS:-224}"
ROUTER_MODEL="${ROUTER_MODEL:-${LLM_ROUTER_MODEL:-${MODEL:-}}}"
WORKERS="${WORKERS:-1}"
FLUSH_EVERY="${FLUSH_EVERY:-20}"
START="${START:-0}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-False}"
INCLUDE_TASK="${INCLUDE_TASK:-True}"
RESPONSE_FORMAT_JSON="${RESPONSE_FORMAT_JSON:-True}"

CSV_112_FRONT="${CSV_112_FRONT:-${BASE_SAVE_DIR}/fs112_lb72_rs144_front/1_0.csv}"
CSV_112_BACK="${CSV_112_BACK:-${BASE_SAVE_DIR}/fs112_lb72_rs144_back/1_0.csv}"
CSV_224_FRONT="${CSV_224_FRONT:-${BASE_SAVE_DIR}/fs224_lb18_rs36_front/1_0.csv}"
CSV_224_BACK="${CSV_224_BACK:-${BASE_SAVE_DIR}/fs224_lb18_rs36_back/1_0.csv}"
ROUTER_OUTPUT="${ROUTER_OUTPUT:-${OUTPUT_DIR}/selective_confidence_mlvu_${VERIFIER}_${GATE_COLUMN}_${GATE_THRESHOLD}.csv}"

flag_enabled() {
  case "${1:-}" in
    1|true|True|TRUE|yes|Yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

validate_stage() {
  case "${STAGE}" in
    all|scores|router|dry-run) ;;
    *)
      echo "STAGE must be one of: all, scores, router, dry-run. Got '${STAGE}'." >&2
      exit 1
      ;;
  esac
}

anno_for_split() {
  case "$1" in
    front) printf '%s\n' "${ANNO_PATH_FRONT}" ;;
    back) printf '%s\n' "${ANNO_PATH_BACK}" ;;
    *)
      echo "Unsupported MLVU split '$1'. SPLITS should contain front/back." >&2
      exit 1
      ;;
  esac
}

run_scores() {
  local split="$1"
  local cuda_devices="$2"
  local frame_size="$3"
  local local_block_count="$4"
  local retrieve_size="$5"
  local anno_path
  local save_dir="${BASE_SAVE_DIR}/fs${frame_size}_lb${local_block_count}_rs${retrieve_size}_${split}"
  local -a extra_args=()

  anno_path="$(anno_for_split "${split}")"
  if [[ ! -f "${anno_path}" ]]; then
    echo "Missing annotation for split '${split}': ${anno_path}" >&2
    echo "Set ANNO_PATH_FRONT/ANNO_PATH_BACK, or run STAGE=router with existing scored CSVs." >&2
    exit 1
  fi
  if [[ -n "${START_VIDEO_ID}" ]]; then
    extra_args+=(--start_video_id "${START_VIDEO_ID}")
  fi

  echo "==== MLVU ${split} score run fs${frame_size}_lb${local_block_count}_rs${retrieve_size} cuda=${cuda_devices} ===="
  echo "==== save_dir=${save_dir} ===="

  CUDA_VISIBLE_DEVICES="${cuda_devices}" "${PYTHON_BIN}" "${PROGRAM}" \
    --sample_fps "${SAMPLE_FPS}" \
    --save_dir "${save_dir}" \
    --anno_path "${anno_path}" \
    --model "${VQA_MODEL}" \
    --frame_size "${frame_size}" \
    --local_block_count "${local_block_count}" \
    --retrieve_size "${retrieve_size}" \
    --retrieve_chunk_size "${RETRIEVE_CHUNK_SIZE}" \
    --save_choice_scores "${SAVE_CHOICE_SCORES}" \
    --debug "${DEBUG}" \
    "${extra_args[@]}"
}

require_csv_column() {
  local csv_path="$1"
  local column="$2"
  local header

  if [[ ! -f "${csv_path}" ]]; then
    echo "Missing CSV: ${csv_path}" >&2
    echo "Run STAGE=scores first, or set CSV_* env vars to scored result CSVs." >&2
    exit 1
  fi

  IFS= read -r header < "${csv_path}"
  header="${header//$'\r'/}"
  if [[ ",${header}," != *",${column},"* ]]; then
    echo "CSV ${csv_path} does not contain column '${column}'." >&2
    echo "Re-run inference with --save_choice_scores True into a fresh save_dir." >&2
    exit 1
  fi
}

run_router() {
  local -a router_args=(
    --csv-112 "${CSV_112_FRONT}" "${CSV_112_BACK}"
    --csv-224 "${CSV_224_FRONT}" "${CSV_224_BACK}"
    --output "${ROUTER_OUTPUT}"
    --gate-column "${GATE_COLUMN}"
    --gate-threshold "${GATE_THRESHOLD}"
    --low-confidence-when "${LOW_CONFIDENCE_WHEN}"
    --verifier "${VERIFIER}"
    --confidence-compare-column "${CONFIDENCE_COMPARE_COLUMN}"
    --default-fs "${DEFAULT_FS}"
    --workers "${WORKERS}"
    --flush-every "${FLUSH_EVERY}"
    --start "${START}"
  )

  require_csv_column "${CSV_112_FRONT}" "${CONFIDENCE_COMPARE_COLUMN}"
  require_csv_column "${CSV_112_BACK}" "${CONFIDENCE_COMPARE_COLUMN}"
  require_csv_column "${CSV_224_FRONT}" "${GATE_COLUMN}"
  require_csv_column "${CSV_224_BACK}" "${GATE_COLUMN}"

  if [[ -n "${LIMIT}" ]]; then
    router_args+=(--limit "${LIMIT}")
  fi
  if [[ -n "${ROUTER_MODEL}" ]]; then
    router_args+=(--model "${ROUTER_MODEL}")
  fi
  if flag_enabled "${RESUME}"; then
    router_args+=(--resume)
  fi
  if flag_enabled "${INCLUDE_TASK}"; then
    router_args+=(--include-task)
  fi
  if flag_enabled "${RESPONSE_FORMAT_JSON}"; then
    router_args+=(--response-format-json)
  fi
  if [[ "${STAGE}" == "dry-run" ]]; then
    router_args+=(--dry-run)
  fi

  mkdir -p "$(dirname "${ROUTER_OUTPUT}")"
  echo "==== MLVU selective confidence router verifier=${VERIFIER} threshold=${GATE_THRESHOLD} ===="
  echo "==== output=${ROUTER_OUTPUT} ===="
  "${ROUTER_PYTHON_BIN}" tools/selective_confidence_router.py "${router_args[@]}"
}

validate_stage

if [[ "${STAGE}" == "all" || "${STAGE}" == "scores" ]]; then
  for split in ${SPLITS}; do
    run_scores "${split}" "${GPU_FS112}" 112 72 144
    run_scores "${split}" "${GPU_FS224}" 224 18 36
  done
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "router" || "${STAGE}" == "dry-run" ]]; then
  run_router
fi
