#!/usr/bin/env bash
set -euo pipefail

cd /root/mwnoh/ReKV-var-resolution

export PYTHON_BIN="${PYTHON_BIN:-/root/mwnoh/anaconda3/bin/python3}"
export PER_CASE="${PER_CASE:-30}"
export SEED="${SEED:-2026}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/mwnoh/ReKV-var-resolution/final-results/latency_samples}"

export CGBENCH_ROUTED_CSV="${CGBENCH_ROUTED_CSV:-/root/mwnoh/ReKV-var-resolution/results/selective_confidence_cgbench_llm_t015_no_gate_context_three_features.csv}"
export MLVU_ROUTED_CSV="${MLVU_ROUTED_CSV:-/root/mwnoh/ReKV-var-resolution/results/selective_confidence_mlvu_llm_t050_no_gate_context_three_features.csv}"
export LVBENCH_ROUTED_CSV="${LVBENCH_ROUTED_CSV:-/root/mwnoh/ReKV-var-resolution/results/selective_confidence_lvbench_llm_fs224default_t030_no_gate_context_three_features.csv}"

mkdir -p "${OUTPUT_DIR}"

sample_one() {
  local dataset="$1"
  local routed_csv="$2"
  local output_csv="${OUTPUT_DIR}/${dataset}_latency_cases_per${PER_CASE}_seed${SEED}.csv"
  local summary_json="${OUTPUT_DIR}/${dataset}_latency_cases_per${PER_CASE}_seed${SEED}_summary.json"

  echo "==== sampling dataset=${dataset} ===="
  echo "==== routed_csv=${routed_csv} ===="
  echo "==== output_csv=${output_csv} ===="

  "${PYTHON_BIN}" tools/sample_routing_latency_cases.py \
    --routed-csv "${routed_csv}" \
    --output-csv "${output_csv}" \
    --summary-json "${summary_json}" \
    --per-case "${PER_CASE}" \
    --seed "${SEED}" \
    --allow-short
}

sample_one "cgbench" "${CGBENCH_ROUTED_CSV}"
sample_one "mlvu" "${MLVU_ROUTED_CSV}"
sample_one "lvbench" "${LVBENCH_ROUTED_CSV}"
