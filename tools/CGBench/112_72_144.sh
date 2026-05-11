#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Exact budget-matched config:
# fs112 -> 16 tokens/frame
# local:    16 * 72  = 1152
# retrieve: 16 * 144 = 2304
export FRAME_SIZE="${FRAME_SIZE:-112}"
export LOCAL_BLOCK_COUNT="${LOCAL_BLOCK_COUNT:-72}"
export RETRIEVE_SIZE="${RETRIEVE_SIZE:-144}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec "${SCRIPT_DIR}/run.sh"

