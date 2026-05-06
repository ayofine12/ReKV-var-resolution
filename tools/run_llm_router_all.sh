#!/usr/bin/env bash
set -euo pipefail

cd /root/mwnoh/ReKV-var-resolution

export PYTHON_BIN="${PYTHON_BIN:-/root/mwnoh/anaconda3/bin/python3}"
export MODEL="${MODEL:-gpt-5.4-mini}"
export INPUT_MODES="${INPUT_MODES:-question}"
export DATASETS="${DATASETS:-lvbench mlvu qaego4d}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/mwnoh/ReKV-var-resolution/results}"
export DEFAULT_FS="${DEFAULT_FS:-auto}"
export WORKERS="${WORKERS:-1}"
export FLUSH_EVERY="${FLUSH_EVERY:-20}"
export LIMIT="${LIMIT:-}"
export START="${START:-0}"
export RESPONSE_FORMAT_JSON="${RESPONSE_FORMAT_JSON:-False}"

export LVBENCH_RESULT_DIR="${LVBENCH_RESULT_DIR:-/mnt/ssd1/mwnoh/var-resolution-lvbench}"
export MLVU_RESULT_DIR="${MLVU_RESULT_DIR:-/mnt/ssd1/mwnoh/var-resolution-mlvu}"
export QAEGO4D_RESULT_DIR="${QAEGO4D_RESULT_DIR:-/mnt/ssd1/mwnoh/var-resolution-qaego4d}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Export a valid key before running this script." >&2
  exit 1
fi

declare -a common_extra_args=()
if [[ -n "${LIMIT}" ]]; then
  common_extra_args+=(--limit "${LIMIT}")
fi
if [[ "${RESPONSE_FORMAT_JSON}" == "True" || "${RESPONSE_FORMAT_JSON}" == "true" || "${RESPONSE_FORMAT_JSON}" == "1" ]]; then
  common_extra_args+=(--response-format-json)
fi

model_slug="$(echo "${MODEL}" | tr '/:' '__')"

run_router() {
  local dataset="$1"
  local input_mode="$2"
  shift 2

  if [[ "${input_mode}" != "question" && "${input_mode}" != "question_choices" ]]; then
    echo "INPUT_MODES entries must be 'question' or 'question_choices'; got '${input_mode}'." >&2
    exit 1
  fi

  local output="${OUTPUT_DIR}/llm_router_${dataset}_${input_mode}_${model_slug}.csv"

  echo "==== LLM routing dataset=${dataset} input_mode=${input_mode} model=${MODEL} ===="
  echo "==== output=${output} ===="

  "${PYTHON_BIN}" tools/llm_resolution_router.py \
    "$@" \
    --input-mode "${input_mode}" \
    --output "${output}" \
    --model "${MODEL}" \
    --default-fs "${DEFAULT_FS}" \
    --workers "${WORKERS}" \
    --flush-every "${FLUSH_EVERY}" \
    --start "${START}" \
    --resume \
    "${common_extra_args[@]}"
}

run_dataset() {
  local dataset="$1"
  local input_mode="$2"

  case "${dataset}" in
    lvbench)
      run_router "${dataset}" "${input_mode}" \
        --csv-112 "${LVBENCH_RESULT_DIR}/fs112_lb72_rs144/1_0.csv" \
        --csv-224 "${LVBENCH_RESULT_DIR}/fs224_lb18_rs36/1_0.csv"
      ;;
    mlvu)
      run_router "${dataset}" "${input_mode}" \
        --csv-112 "${MLVU_RESULT_DIR}/fs112_lb72_rs144_front/1_0.csv" \
                  "${MLVU_RESULT_DIR}/fs112_lb72_rs144_back/1_0.csv" \
        --csv-224 "${MLVU_RESULT_DIR}/fs224_lb18_rs36_front/1_0.csv" \
                  "${MLVU_RESULT_DIR}/fs224_lb18_rs36_back/1_0.csv"
      ;;
    qaego4d)
      run_router "${dataset}" "${input_mode}" \
        --csv-112 "${QAEGO4D_RESULT_DIR}/fs112_lb72_rs144/1_0.csv" \
        --csv-224 "${QAEGO4D_RESULT_DIR}/fs224_lb18_rs36/1_0.csv"
      ;;
    all)
      run_router "${dataset}" "${input_mode}" \
        --csv-112 "${LVBENCH_RESULT_DIR}/fs112_lb72_rs144/1_0.csv" \
                  "${MLVU_RESULT_DIR}/fs112_lb72_rs144_front/1_0.csv" \
                  "${MLVU_RESULT_DIR}/fs112_lb72_rs144_back/1_0.csv" \
                  "${QAEGO4D_RESULT_DIR}/fs112_lb72_rs144/1_0.csv" \
        --csv-224 "${LVBENCH_RESULT_DIR}/fs224_lb18_rs36/1_0.csv" \
                  "${MLVU_RESULT_DIR}/fs224_lb18_rs36_front/1_0.csv" \
                  "${MLVU_RESULT_DIR}/fs224_lb18_rs36_back/1_0.csv" \
                  "${QAEGO4D_RESULT_DIR}/fs224_lb18_rs36/1_0.csv"
      ;;
    *)
      echo "Unknown dataset '${dataset}'. Use one or more of: lvbench mlvu qaego4d all." >&2
      exit 1
      ;;
  esac
}

for input_mode in ${INPUT_MODES}; do
  for dataset in ${DATASETS}; do
    run_dataset "${dataset}" "${input_mode}"
  done
done
