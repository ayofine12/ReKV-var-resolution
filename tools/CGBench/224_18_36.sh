#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Exact budget-matched config:
# fs224 -> 64 tokens/frame
# local:    64 * 18 = 1152
# retrieve: 64 * 36 = 2304
export FRAME_SIZE="${FRAME_SIZE:-224}"
export LOCAL_BLOCK_COUNT="${LOCAL_BLOCK_COUNT:-18}"
export RETRIEVE_SIZE="${RETRIEVE_SIZE:-36}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

exec "${SCRIPT_DIR}/run.sh"

