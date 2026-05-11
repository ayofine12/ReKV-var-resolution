#!/usr/bin/env bash
set -euo pipefail

cd /root/mwnoh/ReKV-var-resolution

export PYTHON_BIN="${PYTHON_BIN:-/root/mwnoh/anaconda3/bin/python3}"
export MODEL="${MODEL:-${LLM_ROUTER_MODEL:-gpt-5.4-mini}}"
export LLM_ROUTER_MODEL="${MODEL}"

export DATASETS="${DATASETS:-cgbench mlvu lvbench}"
export THRESHOLDS="${THRESHOLDS:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9}"
export VERIFIER="${VERIFIER:-confidence}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/mwnoh/ReKV-var-resolution/results}"

export WORKERS="${WORKERS:-1}"
export FLUSH_EVERY="${FLUSH_EVERY:-20}"
export START="${START:-0}"
export LIMIT="${LIMIT:-}"
export RESUME="${RESUME:-1}"
export RESPONSE_FORMAT_JSON="${RESPONSE_FORMAT_JSON:-1}"

export GATE_COLUMN="${GATE_COLUMN:-prob_margin}"
export LOW_CONFIDENCE_WHEN="${LOW_CONFIDENCE_WHEN:-lt}"
export CONFIDENCE_COMPARE_COLUMN="${CONFIDENCE_COMPARE_COLUMN:-prob_margin}"
export FEATURE_COLUMNS="${FEATURE_COLUMNS:-top1_prob prob_margin normalized_choice_entropy}"

export DEFAULT_FS_CGBENCH="${DEFAULT_FS_CGBENCH:-224}"
export DEFAULT_FS_MLVU="${DEFAULT_FS_MLVU:-224}"
export DEFAULT_FS_LVBENCH="${DEFAULT_FS_LVBENCH:-224}"

export INCLUDE_TASK_CGBENCH="${INCLUDE_TASK_CGBENCH:-1}"
export INCLUDE_TASK_MLVU="${INCLUDE_TASK_MLVU:-1}"
export INCLUDE_TASK_LVBENCH="${INCLUDE_TASK_LVBENCH:-0}"
export INCLUDE_GATE_CONTEXT="${INCLUDE_GATE_CONTEXT:-0}"

export CGBENCH_CONF_DIR="${CGBENCH_CONF_DIR:-/mnt/ssd1/mwnoh/var-resolution-cgbench-confidence}"
export MLVU_CONF_DIR="${MLVU_CONF_DIR:-/mnt/ssd1/mwnoh/var-resolution-mlvu-confidence}"
export LVBENCH_CONF_DIR="${LVBENCH_CONF_DIR:-/mnt/ssd1/mwnoh/var-resolution-lvbench-confidence}"

if [[ "${VERIFIER}" == "llm" || "${VERIFIER}" == "llm_override" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not set. Export a valid key before running VERIFIER=${VERIFIER}." >&2
    exit 1
  fi
fi

mkdir -p "${OUTPUT_DIR}"

declare -a common_extra_args=()
if [[ -n "${LIMIT}" ]]; then
  common_extra_args+=(--limit "${LIMIT}")
fi
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "True" ]]; then
  common_extra_args+=(--resume)
fi
if [[ "${RESPONSE_FORMAT_JSON}" == "1" || "${RESPONSE_FORMAT_JSON}" == "true" || "${RESPONSE_FORMAT_JSON}" == "True" ]]; then
  common_extra_args+=(--response-format-json)
fi
if [[ "${INCLUDE_GATE_CONTEXT}" == "1" || "${INCLUDE_GATE_CONTEXT}" == "true" || "${INCLUDE_GATE_CONTEXT}" == "True" ]]; then
  common_extra_args+=(--include-gate-context)
else
  common_extra_args+=(--no-include-gate-context)
fi

threshold_slug() {
  local threshold="$1"
  awk -v t="${threshold}" 'BEGIN { printf "%03d", int(t * 100 + 0.5) }'
}

dataset_default_fs() {
  local dataset="$1"
  case "${dataset}" in
    cgbench) echo "${DEFAULT_FS_CGBENCH}" ;;
    mlvu) echo "${DEFAULT_FS_MLVU}" ;;
    lvbench) echo "${DEFAULT_FS_LVBENCH}" ;;
    *)
      echo "Unknown dataset '${dataset}'. Use one or more of: cgbench mlvu lvbench." >&2
      exit 1
      ;;
  esac
}

dataset_include_task() {
  local dataset="$1"
  case "${dataset}" in
    cgbench) echo "${INCLUDE_TASK_CGBENCH}" ;;
    mlvu) echo "${INCLUDE_TASK_MLVU}" ;;
    lvbench) echo "${INCLUDE_TASK_LVBENCH}" ;;
    *)
      echo "Unknown dataset '${dataset}'. Use one or more of: cgbench mlvu lvbench." >&2
      exit 1
      ;;
  esac
}

dataset_csv_args() {
  local dataset="$1"
  case "${dataset}" in
    cgbench)
      echo "--csv-112 ${CGBENCH_CONF_DIR}/fs112_lb72_rs144/1_0.csv --csv-224 ${CGBENCH_CONF_DIR}/fs224_lb18_rs36/1_0.csv"
      ;;
    mlvu)
      echo "--csv-112 ${MLVU_CONF_DIR}/fs112_lb72_rs144/1_0.csv --csv-224 ${MLVU_CONF_DIR}/fs224_lb18_rs36/1_0.csv"
      ;;
    lvbench)
      echo "--csv-112 ${LVBENCH_CONF_DIR}/fs112_lb72_rs144/1_0.csv --csv-224 ${LVBENCH_CONF_DIR}/fs224_lb18_rs36/1_0.csv"
      ;;
    *)
      echo "Unknown dataset '${dataset}'. Use one or more of: cgbench mlvu lvbench." >&2
      exit 1
      ;;
  esac
}

run_one() {
  local dataset="$1"
  local threshold="$2"
  local default_fs
  local include_task
  local slug
  local output

  default_fs="$(dataset_default_fs "${dataset}")"
  include_task="$(dataset_include_task "${dataset}")"
  slug="$(threshold_slug "${threshold}")"
  output="${OUTPUT_DIR}/selective_confidence_${dataset}_${VERIFIER}_fs${default_fs}default_t${slug}_no_gate_context_three_features.csv"

  declare -a dataset_args=()
  read -r -a dataset_args <<< "$(dataset_csv_args "${dataset}")"

  declare -a task_args=()
  if [[ "${include_task}" == "1" || "${include_task}" == "true" || "${include_task}" == "True" ]]; then
    task_args+=(--include-task)
  fi

  echo "==== dataset=${dataset} verifier=${VERIFIER} default_fs=${default_fs} threshold=${threshold} ===="
  echo "==== output=${output} ===="

  "${PYTHON_BIN}" tools/selective_confidence_router.py \
    "${dataset_args[@]}" \
    --duplicate-key-policy first \
    --output "${output}" \
    --gate-column "${GATE_COLUMN}" \
    --gate-threshold "${threshold}" \
    --low-confidence-when "${LOW_CONFIDENCE_WHEN}" \
    --verifier "${VERIFIER}" \
    --default-fs "${default_fs}" \
    --confidence-compare-column "${CONFIDENCE_COMPARE_COLUMN}" \
    --feature-columns ${FEATURE_COLUMNS} \
    "${task_args[@]}" \
    --model "${MODEL}" \
    --workers "${WORKERS}" \
    --flush-every "${FLUSH_EVERY}" \
    --start "${START}" \
    "${common_extra_args[@]}"
}

for dataset in ${DATASETS}; do
  for threshold in ${THRESHOLDS}; do
    run_one "${dataset}" "${threshold}"
  done
done
