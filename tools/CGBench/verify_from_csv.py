#!/usr/bin/env python3
"""
Verify CGBench selective-confidence cases from existing score CSVs only.

This script does not run VQA inference and does not create fs112/fs224 scores.
It reads the CSVs produced by run_selective_confidence.sh, filters examples
such as fs224-low-confidence disagreements, then feeds each row's question,
choices, predictions, and confidence scores to the verifier.

Important: qa_acc/correct_choice are used only for offline evaluation in the
output summary. They are not included in the LLM verifier prompt.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools import selective_confidence_router as router  # noqa: E402


DEFAULT_CGBENCH_SAVE_DIR = Path("/mnt/ssd1/mwnoh/var-resolution-cgbench-confidence/full")
DEFAULT_OUTPUT = REPO_ROOT / "results" / "cgbench_low_disagree_verify_from_csv.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CSV-only verification over existing fs112/fs224 CGBench score files."
        )
    )
    parser.add_argument(
        "--csv-112",
        nargs="+",
        default=[str(DEFAULT_CGBENCH_SAVE_DIR / "fs112_lb72_rs144" / "1_0.csv")],
        help="One or more existing fs112 score CSVs.",
    )
    parser.add_argument(
        "--csv-224",
        nargs="+",
        default=[str(DEFAULT_CGBENCH_SAVE_DIR / "fs224_lb18_rs36" / "1_0.csv")],
        help="One or more existing fs224 score CSVs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to save verifier decisions.",
    )
    parser.add_argument(
        "--example-filter",
        choices=["all", "low-confidence", "low-disagree", "low-disagree-decisive"],
        default="low-disagree",
        help="Which CSV rows to verify after joining fs112/fs224 results.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Limit after --example-filter is applied. Default verifies 200 low-disagree rows.",
    )
    parser.add_argument(
        "--balanced-decisive",
        action="store_true",
        help=(
            "After filtering decisive examples, select an equal number of route_112 and route_224 labels. "
            "This uses qa_acc labels and is for offline diagnostics only."
        ),
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle selected examples with --seed before applying --limit or balanced sampling.",
    )
    parser.add_argument(
        "--duplicate-key-policy",
        choices=["error", "first", "last"],
        default="last",
        help="How to handle duplicated question rows in an input CSV.",
    )

    parser.add_argument("--gate-column", default="prob_margin")
    parser.add_argument("--gate-threshold", type=float, default=0.40)
    parser.add_argument(
        "--low-confidence-when",
        choices=["lt", "le", "gt", "ge"],
        default="lt",
    )
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        default=router.FEATURE_COLUMNS,
        help="Confidence columns shown to the verifier.",
    )
    parser.add_argument("--include-gate-context", action="store_true", default=True)
    parser.add_argument("--no-include-gate-context", dest="include_gate_context", action="store_false")
    parser.add_argument("--include-feature-deltas", action="store_true", default=True)
    parser.add_argument("--no-include-feature-deltas", dest="include_feature_deltas", action="store_false")
    parser.add_argument(
        "--verifier",
        choices=["llm", "confidence", "fs224", "fs112", "oracle"],
        default="llm",
        help="Verifier mode. llm sends prompts; confidence is a score-only baseline.",
    )
    parser.add_argument("--confidence-compare-column", default="prob_margin")
    parser.add_argument("--default-fs", choices=["112", "224", "auto"], default="224")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--model", default=os.environ.get("LLM_ROUTER_MODEL") or os.environ.get("MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--response-format-json", action="store_true", default=True)
    parser.add_argument("--no-response-format-json", dest="response_format_json", action="store_false")
    parser.add_argument("--min-verifier-confidence", type=float, default=0.0)
    parser.add_argument("--include-task", action="store_true", default=True)
    parser.add_argument("--no-include-task", dest="include_task", action="store_false")

    parser.add_argument("--cost-fs224", type=float, default=1.0)
    parser.add_argument("--cost-fs112", type=float, default=1.0)
    parser.add_argument("--cost-verifier", type=float, default=0.05)
    parser.set_defaults(workers=1)
    return parser.parse_args()


def select_balanced_decisive(
    examples: List[router.SelectiveExample],
    limit: int | None,
    seed: int,
    shuffle: bool,
) -> List[router.SelectiveExample]:
    route112 = [ex for ex in examples if router.gold_label(ex) == router.LABEL_112]
    route224 = [ex for ex in examples if router.gold_label(ex) == router.LABEL_224]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(route112)
        rng.shuffle(route224)

    if limit is None:
        per_side = min(len(route112), len(route224))
    else:
        if limit < 0:
            raise ValueError("--limit must be non-negative.")
        per_side = limit // 2
        if limit % 2:
            raise ValueError("--balanced-decisive requires an even --limit.")
        per_side = min(per_side, len(route112), len(route224))

    selected = route112[:per_side] + route224[:per_side]
    if shuffle:
        rng = random.Random(seed + 1)
        rng.shuffle(selected)
    else:
        selected = sorted(selected, key=lambda ex: ex.index)
    return selected


def main() -> None:
    args = parse_args()
    if args.verifier == "llm" and not args.model and not args.dry_run:
        raise ValueError("Provide --model or set LLM_ROUTER_MODEL/MODEL for --verifier llm.")

    examples = router.build_examples(args.csv_112, args.csv_224, args.duplicate_key_policy)
    default_fs = router.choose_default_fs(examples, args.default_fs)
    filtered_examples = router.filter_examples(examples, args)
    if args.balanced_decisive:
        if args.start != 0:
            raise ValueError("--balanced-decisive does not support --start; use --shuffle/--seed instead.")
        selected_examples = select_balanced_decisive(
            filtered_examples,
            args.limit,
            args.seed,
            args.shuffle,
        )
    else:
        if args.shuffle:
            rng = random.Random(args.seed)
            filtered_examples = list(filtered_examples)
            rng.shuffle(filtered_examples)
        selected_examples = router.select_examples(filtered_examples, args.start, args.limit)
    if not selected_examples:
        raise ValueError("No examples selected. Check --example-filter, --gate-threshold, and input CSV paths.")

    selected_gold_counts = {
        router.LABEL_112: sum(1 for ex in selected_examples if router.gold_label(ex) == router.LABEL_112),
        router.LABEL_224: sum(1 for ex in selected_examples if router.gold_label(ex) == router.LABEL_224),
        router.LABEL_TIE: sum(1 for ex in selected_examples if router.gold_label(ex) == router.LABEL_TIE),
    }
    print(
        f"[setup] joined={len(examples)} filter={args.example_filter} "
        f"filtered={len(filtered_examples)} selected={len(selected_examples)}"
    )
    print(f"[setup] selected_gold_counts={selected_gold_counts}")
    print(
        f"[setup] gate=fs224.{args.gate_column} {args.low_confidence_when} "
        f"{args.gate_threshold} verifier={args.verifier} default_fs={default_fs}"
    )

    if args.dry_run:
        router.print_dry_run(selected_examples, args, default_fs)
        return

    existing = router.load_existing_rows(args.output) if args.resume else {}
    rows, remaining = router.rows_from_existing(selected_examples, existing)
    print(f"[setup] reused={len(rows)} remaining={len(remaining)}")

    client = None
    if args.verifier == "llm":
        client = router.load_openai_client(args.api_key_env, args.base_url, args.timeout)

    rows = router.route_remaining(remaining, args, default_fs, client, rows)
    router.save_rows(args.output, rows)
    router.summarize(rows, args, default_fs)
    print(f"[saved_predictions] {args.output}")


if __name__ == "__main__":
    main()
