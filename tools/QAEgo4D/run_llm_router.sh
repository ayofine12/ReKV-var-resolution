#!/usr/bin/env bash
set -euo pipefail

cd /root/mwnoh/ReKV-var-resolution

export PYTHON_BIN="${PYTHON_BIN:-/root/mwnoh/anaconda3/bin/python3}"
export MODEL="${MODEL:-gpt-5.4-mini}"
export INPUT_MODE="${INPUT_MODE:-question}"
export BASE_RESULT_DIR="${BASE_RESULT_DIR:-/mnt/ssd1/mwnoh/var-resolution-qaego4d}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/mwnoh/ReKV-var-resolution/results}"
export DEFAULT_FS="${DEFAULT_FS:-auto}"
export WORKERS="${WORKERS:-1}"
export FLUSH_EVERY="${FLUSH_EVERY:-20}"
export LIMIT="${LIMIT:-}"
export START="${START:-0}"
export RESPONSE_FORMAT_JSON="${RESPONSE_FORMAT_JSON:-False}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Export a valid key before running this script." >&2
  exit 1
fi

if [[ "${INPUT_MODE}" != "question" && "${INPUT_MODE}" != "question_choices" ]]; then
  echo "INPUT_MODE must be 'question' or 'question_choices'." >&2
  exit 1
fi

model_slug="$(echo "${MODEL}" | tr '/:' '__')"
output="${OUTPUT:-${OUTPUT_DIR}/llm_router_qaego4d_${INPUT_MODE}_${model_slug}.csv}"

declare -a extra_args=()
if [[ -n "${LIMIT}" ]]; then
  extra_args+=(--limit "${LIMIT}")
fi
if [[ "${RESPONSE_FORMAT_JSON}" == "True" || "${RESPONSE_FORMAT_JSON}" == "true" || "${RESPONSE_FORMAT_JSON}" == "1" ]]; then
  extra_args+=(--response-format-json)
fi

"${PYTHON_BIN}" tools/llm_resolution_router.py \
  --csv-112 "${BASE_RESULT_DIR}/fs112_lb72_rs144/1_0.csv" \
  --csv-224 "${BASE_RESULT_DIR}/fs224_lb18_rs36/1_0.csv" \
  --input-mode "${INPUT_MODE}" \
  --output "${output}" \
  --model "${MODEL}" \
  --default-fs "${DEFAULT_FS}" \
  --workers "${WORKERS}" \
  --flush-every "${FLUSH_EVERY}" \
  --start "${START}" \
  --resume \
  "${extra_args[@]}"
