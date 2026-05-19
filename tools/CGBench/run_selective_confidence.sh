#!/usr/bin/env bash
set -euo pipefail

cd /root/mwnoh/ReKV-var-resolution

export REKV_VIDEO_CACHE_DIR="${REKV_VIDEO_CACHE_DIR:-/mnt/ssd1/mwnoh/CG-Bench/cache}"
export PYTHONPATH="/root/mwnoh/ReKV-var-resolution:/root/mwnoh/ReKV-var-resolution/model:/root/mwnoh/ReKV-var-resolution/model/longva:${PYTHONPATH:-}"

STAGE="${STAGE:-all}" # all, scores, router, dry-run
SPLIT="${SPLIT:-full}" # full, mini
PYTHON_BIN="${PYTHON_BIN:-/root/mwnoh/anaconda3/envs/rekv/bin/python}"
ROUTER_PYTHON_BIN="${ROUTER_PYTHON_BIN:-/root/mwnoh/anaconda3/bin/python3}"
PROGRAM="${PROGRAM:-/root/mwnoh/ReKV-var-resolution/video_qa/rekv_offline_vqa.py}"
CGBENCH_ROOT="${CGBENCH_ROOT:-/mnt/ssd1/mwnoh/CG-Bench}"
VIDEO_DIR="${VIDEO_DIR:-${CGBENCH_ROOT}/cg_videos_720p}"
VIDEO_EXT="${VIDEO_EXT:-.mp4}"
BASE_SAVE_ROOT="${BASE_SAVE_ROOT:-/mnt/ssd1/mwnoh/var-resolution-cgbench-confidence}"
BASE_SAVE_DIR="${BASE_SAVE_DIR:-${BASE_SAVE_ROOT}/${SPLIT}}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/mwnoh/ReKV-var-resolution/results}"

VQA_MODEL="${VQA_MODEL:-qwen2_5_vl_7b}"
SAMPLE_FPS="${SAMPLE_FPS:-}"
FS112_SAMPLE_FPS="${FS112_SAMPLE_FPS:-${SAMPLE_FPS:-2}}"
FS224_SAMPLE_FPS="${FS224_SAMPLE_FPS:-${SAMPLE_FPS:-0.5}}"
FS112_FPS_TAG="fps${FS112_SAMPLE_FPS//./p}"
FS224_FPS_TAG="fps${FS224_SAMPLE_FPS//./p}"
RETRIEVE_CHUNK_SIZE="${RETRIEVE_CHUNK_SIZE:-}"
FS112_RETRIEVE_CHUNK_SIZE="${FS112_RETRIEVE_CHUNK_SIZE:-${RETRIEVE_CHUNK_SIZE:-4}}"
FS224_RETRIEVE_CHUNK_SIZE="${FS224_RETRIEVE_CHUNK_SIZE:-${RETRIEVE_CHUNK_SIZE:-1}}"
INTERNAL_BLOCK_SIZE="${INTERNAL_BLOCK_SIZE:-}"
FS112_INTERNAL_BLOCK_SIZE="${FS112_INTERNAL_BLOCK_SIZE:-${INTERNAL_BLOCK_SIZE:-512}}"
FS224_INTERNAL_BLOCK_SIZE="${FS224_INTERNAL_BLOCK_SIZE:-${INTERNAL_BLOCK_SIZE:-128}}"
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
EXAMPLE_FILTER="${EXAMPLE_FILTER:-all}" # all, low-confidence, low-disagree, low-disagree-decisive
DUPLICATE_KEY_POLICY="${DUPLICATE_KEY_POLICY:-last}" # error, first, last
RESUME="${RESUME:-False}"
INCLUDE_TASK="${INCLUDE_TASK:-True}"
RESPONSE_FORMAT_JSON="${RESPONSE_FORMAT_JSON:-True}"
REBUILD_ANNOTATION="${REBUILD_ANNOTATION:-False}"
LIMIT_QUESTIONS="${LIMIT_QUESTIONS:-}"

case "${SPLIT}" in
  full)
    DEFAULT_ANNO_SRC="${CGBENCH_ROOT}/cgbench.json"
    DEFAULT_ANNO_PATH="${CGBENCH_ROOT}/rekv_annotations/full_mc.json"
    ;;
  mini)
    DEFAULT_ANNO_SRC="${CGBENCH_ROOT}/cgbench_mini.json"
    DEFAULT_ANNO_PATH="${CGBENCH_ROOT}/rekv_annotations/mini_mc.json"
    ;;
  *)
    echo "SPLIT must be one of: full, mini. Got '${SPLIT}'." >&2
    exit 1
    ;;
esac

ANNO_SRC="${ANNO_SRC:-${DEFAULT_ANNO_SRC}}"
ANNO_PATH="${ANNO_PATH:-${DEFAULT_ANNO_PATH}}"
CSV_112="${CSV_112:-${BASE_SAVE_DIR}/fs112_lb72_rs144_rcs${FS112_RETRIEVE_CHUNK_SIZE}_ibs${FS112_INTERNAL_BLOCK_SIZE}_${FS112_FPS_TAG}/1_0.csv}"
CSV_224="${CSV_224:-${BASE_SAVE_DIR}/fs224_lb18_rs36_rcs${FS224_RETRIEVE_CHUNK_SIZE}_ibs${FS224_INTERNAL_BLOCK_SIZE}_${FS224_FPS_TAG}/1_0.csv}"
OUTPUT_FILTER_SUFFIX=""
if [[ "${EXAMPLE_FILTER}" != "all" ]]; then
  OUTPUT_FILTER_SUFFIX="_${EXAMPLE_FILTER}"
fi
ROUTER_OUTPUT="${ROUTER_OUTPUT:-${OUTPUT_DIR}/selective_confidence_cgbench_${SPLIT}_fs112rcs${FS112_RETRIEVE_CHUNK_SIZE}_ibs${FS112_INTERNAL_BLOCK_SIZE}_${FS112_FPS_TAG}_fs224rcs${FS224_RETRIEVE_CHUNK_SIZE}_ibs${FS224_INTERNAL_BLOCK_SIZE}_${FS224_FPS_TAG}${OUTPUT_FILTER_SUFFIX}_${VERIFIER}_${GATE_COLUMN}_${GATE_THRESHOLD}.csv}"

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

prepare_annotation() {
  local -a annotation_args=(
    --anno_src "${ANNO_SRC}"
    --anno_abs "${ANNO_PATH}"
    --base_video_dir "${VIDEO_DIR}"
    --video_ext "${VIDEO_EXT}"
  )

  if [[ -n "${LIMIT_QUESTIONS}" ]]; then
    annotation_args+=(--limit_questions "${LIMIT_QUESTIONS}")
  fi

  if [[ -f "${ANNO_PATH}" ]] && ! flag_enabled "${REBUILD_ANNOTATION}"; then
    return
  fi
  if [[ ! -f "${ANNO_SRC}" ]]; then
    echo "Missing CG-Bench source annotation: ${ANNO_SRC}" >&2
    echo "Set ANNO_SRC, or set ANNO_PATH to an existing ReKV-formatted annotation." >&2
    exit 1
  fi
  if [[ ! -d "${VIDEO_DIR}" ]]; then
    echo "Missing CG-Bench video directory: ${VIDEO_DIR}" >&2
    echo "Set VIDEO_DIR to the directory containing CG-Bench videos." >&2
    exit 1
  fi

  echo "==== Preparing CGBench annotation split=${SPLIT} ===="
  "${PYTHON_BIN}" tools/CGBench/make_mc_annotation.py "${annotation_args[@]}"
}

run_scores() {
  local cuda_devices="$1"
  local frame_size="$2"
  local local_block_count="$3"
  local retrieve_size="$4"
  local retrieve_chunk_size="$5"
  local internal_block_size="$6"
  local sample_fps="$7"
  local fps_tag="fps${sample_fps//./p}"
  local save_dir="${BASE_SAVE_DIR}/fs${frame_size}_lb${local_block_count}_rs${retrieve_size}_rcs${retrieve_chunk_size}_ibs${internal_block_size}_${fps_tag}"
  local -a extra_args=()

  if [[ -n "${START_VIDEO_ID}" ]]; then
    extra_args+=(--start_video_id "${START_VIDEO_ID}")
  fi

  echo "==== CGBench ${SPLIT} score run fs${frame_size}_lb${local_block_count}_rs${retrieve_size}_rcs${retrieve_chunk_size}_ibs${internal_block_size} sample_fps=${sample_fps} cuda=${cuda_devices} ===="
  echo "==== anno_path=${ANNO_PATH} ===="
  echo "==== video_dir=${VIDEO_DIR} ===="
  echo "==== save_dir=${save_dir} ===="

  CUDA_VISIBLE_DEVICES="${cuda_devices}" "${PYTHON_BIN}" "${PROGRAM}" \
    --sample_fps "${sample_fps}" \
    --save_dir "${save_dir}" \
    --anno_path "${ANNO_PATH}" \
    --model "${VQA_MODEL}" \
    --frame_size "${frame_size}" \
    --local_block_count "${local_block_count}" \
    --retrieve_size "${retrieve_size}" \
    --retrieve_chunk_size "${retrieve_chunk_size}" \
    --internal_block_size "${internal_block_size}" \
    --save_choice_scores "${SAVE_CHOICE_SCORES}" \
    --debug "${DEBUG}" \
    "${extra_args[@]}"
}

run_score_pair() {
  local pid112
  local pid224
  local status112=0
  local status224=0

  prepare_annotation

  echo "==== Launching CGBench ${SPLIT} score runs in parallel ===="
  echo "==== fs112 sample_fps=${FS112_SAMPLE_FPS}; fs224 sample_fps=${FS224_SAMPLE_FPS} ===="
  run_scores "${GPU_FS112}" 112 72 144 "${FS112_RETRIEVE_CHUNK_SIZE}" "${FS112_INTERNAL_BLOCK_SIZE}" "${FS112_SAMPLE_FPS}" &
  pid112=$!
  run_scores "${GPU_FS224}" 224 18 36 "${FS224_RETRIEVE_CHUNK_SIZE}" "${FS224_INTERNAL_BLOCK_SIZE}" "${FS224_SAMPLE_FPS}" &
  pid224=$!

  wait "${pid112}" || status112=$?
  wait "${pid224}" || status224=$?

  if [[ "${status112}" -ne 0 || "${status224}" -ne 0 ]]; then
    echo "CGBench score run failed: fs112 status=${status112}, fs224 status=${status224}" >&2
    exit 1
  fi
}

require_csv_column() {
  local csv_path="$1"
  local column="$2"
  local header

  if [[ ! -f "${csv_path}" ]]; then
    echo "Missing CSV: ${csv_path}" >&2
    echo "Run STAGE=scores first, or set CSV_112/CSV_224 to scored result CSVs." >&2
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
    --csv-112 "${CSV_112}"
    --csv-224 "${CSV_224}"
    --output "${ROUTER_OUTPUT}"
    --duplicate-key-policy "${DUPLICATE_KEY_POLICY}"
    --gate-column "${GATE_COLUMN}"
    --gate-threshold "${GATE_THRESHOLD}"
    --low-confidence-when "${LOW_CONFIDENCE_WHEN}"
    --verifier "${VERIFIER}"
    --confidence-compare-column "${CONFIDENCE_COMPARE_COLUMN}"
    --default-fs "${DEFAULT_FS}"
    --workers "${WORKERS}"
    --flush-every "${FLUSH_EVERY}"
    --example-filter "${EXAMPLE_FILTER}"
    --start "${START}"
  )

  require_csv_column "${CSV_112}" "${CONFIDENCE_COMPARE_COLUMN}"
  require_csv_column "${CSV_224}" "${GATE_COLUMN}"

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
  echo "==== CGBench ${SPLIT} selective confidence router verifier=${VERIFIER} threshold=${GATE_THRESHOLD} ===="
  echo "==== output=${ROUTER_OUTPUT} ===="
  "${ROUTER_PYTHON_BIN}" tools/selective_confidence_router.py "${router_args[@]}"
}

validate_stage

if [[ "${STAGE}" == "all" || "${STAGE}" == "scores" ]]; then
  run_score_pair
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "router" || "${STAGE}" == "dry-run" ]]; then
  run_router
fi
