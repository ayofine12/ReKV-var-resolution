#!/usr/bin/env python3
import argparse
import ast
import csv
import gc
import importlib.util
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from decord import VideoReader, cpu

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.qwen2_5_vl_rekv import load_model


CASE_HIGH = "case1_high_confidence"
CASE_AGREE = "case2_low_confidence_agree"
CASE_DISAGREE = "case3_low_confidence_disagree"
CHOICE_LETTERS = "ABCDEFGH"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure stratified routing latency from a case-sampled manifest. "
            "Video-side latency measures question-time retrieval + QA prefill; video cache construction is "
            "reported separately and excluded from routed online cost."
        )
    )
    parser.add_argument("--samples-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--anno-path", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--video-root",
        action="append",
        default=[],
        help=(
            "Optional path rewrite OLD=NEW for annotation video_path. "
            "Example: data/mlvu/videos=/mnt/ssd1/mwnoh/mlvu/videos"
        ),
    )
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--warm-retrieval-cache",
        action="store_true",
        help="Keep retrieved blocks on GPU between questions for the same video. Default is cold retrieval.",
    )
    parser.add_argument("--skip-missing-videos", action="store_true")
    parser.add_argument(
        "--dry-run-resolve",
        action="store_true",
        help="Only check sample-to-video resolution and exit without loading models.",
    )

    parser.add_argument("--model-path", default="/mnt/models/qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--fs224-frame-size", type=int, default=224)
    parser.add_argument("--fs224-local-block-count", type=int, default=18)
    parser.add_argument("--fs224-retrieve-size", type=int, default=36)
    parser.add_argument("--fs112-frame-size", type=int, default=112)
    parser.add_argument("--fs112-local-block-count", type=int, default=72)
    parser.add_argument("--fs112-retrieve-size", type=int, default=144)
    parser.add_argument("--retrieve-chunk-size", type=int, default=1)
    parser.add_argument(
        "--measure-fs112-for-all",
        action="store_true",
        help="Also measure fs112 for high-confidence rows. Useful for always-both latency estimates.",
    )

    parser.add_argument("--measure-verifier", action="store_true")
    parser.add_argument("--verifier-model", default=None)
    parser.add_argument("--verifier-base-url", default=None)
    parser.add_argument("--verifier-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--verifier-response-format-json", action="store_true")
    parser.add_argument("--verifier-max-tokens", type=int, default=128)
    parser.add_argument("--verifier-temperature", type=float, default=0.0)
    parser.add_argument("--verifier-timeout", type=float, default=60.0)
    parser.add_argument("--verifier-max-retries", type=int, default=1)
    parser.add_argument("--verifier-retry-sleep", type=float, default=1.0)
    parser.add_argument("--verifier-seed", type=int, default=2024)
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        default=["top1_prob", "prob_margin", "normalized_choice_entropy"],
    )
    parser.add_argument("--include-task", action="store_true")
    parser.add_argument("--include-gate-context", action="store_true")
    parser.add_argument("--no-include-feature-deltas", dest="include_feature_deltas", action="store_false")
    parser.set_defaults(include_feature_deltas=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {path}")
        return [dict(row) for row in reader]


def parse_choices(raw: str) -> list[str]:
    if not raw:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(raw)
            if isinstance(value, list):
                return [str(item) for item in value]
        except Exception:
            pass
    return [raw]


def format_mcqa_prompt(question: str, choices: list[str], model) -> dict:
    formatted_choices = "\n".join(f"({CHOICE_LETTERS[i]}) {choice}" for i, choice in enumerate(choices))
    formatted_question = (
        f"Question: {question}\n"
        f"Options:\n{formatted_choices}\n"
        "Only give the best option."
    )
    return {
        "question": question,
        "formatted_question": formatted_question,
        "prompt": model.get_prompt(formatted_question, mc=True),
    }


def cuda_sync(device) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def timed_call(fn, device):
    cuda_sync(device)
    start = time.perf_counter()
    out = fn()
    cuda_sync(device)
    return out, time.perf_counter() - start


def load_video(video_path: Path, sample_fps: float):
    vr = VideoReader(str(video_path), ctx=cpu(0))
    fps = round(vr.get_avg_fps())
    frame_step = max(1, int(fps / sample_fps))
    frame_idx = [i for i in range(0, len(vr), frame_step)]
    video = vr.get_batch(frame_idx).asnumpy()
    return torch.from_numpy(video)


def normalize_lvbench_sample(sample: dict) -> dict:
    return {
        "video_id": sample.get("key", ""),
        "video_path": sample.get("downloaded_video_path", ""),
        "conversations": [],
    }


def index_key_variants(video_id: str, video_path: str) -> set[str]:
    variants = set()
    if video_id:
        variants.add(video_id)
        variants.add(Path(video_id).stem)
    if video_path:
        path = Path(video_path)
        variants.add(path.stem)
        if path.parent.name:
            variants.add(f"{path.parent.name}/{path.stem}")
    return {item for item in variants if item}


def apply_video_roots(path_str: str, video_roots: list[str], anno_path: Path) -> list[Path]:
    candidates = []
    path = Path(path_str)
    candidates.append(path)
    if not path.is_absolute():
        candidates.append(REPO_ROOT / path)
        candidates.append(anno_path.parent / path)

    for mapping in video_roots:
        if "=" in mapping:
            old, new = mapping.split("=", 1)
            if path_str.startswith(old):
                candidates.append(Path(new) / path_str[len(old):].lstrip("/"))
        else:
            root = Path(mapping)
            candidates.append(root / path_str)
            candidates.append(root / Path(path_str).name)
            parts = Path(path_str).parts
            if "videos" in parts:
                idx = parts.index("videos")
                candidates.append(root / Path(*parts[idx + 1:]))

    seen = set()
    out = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def build_video_index(anno_paths: list[Path], video_roots: list[str]) -> dict[str, Path]:
    index = {}
    for anno_path in anno_paths:
        data = json.load(anno_path.open(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected list annotation: {anno_path}")
        for item in data:
            if "downloaded_video_path" in item or "key" in item:
                sample = normalize_lvbench_sample(item)
            else:
                sample = item
            video_id = str(sample.get("video_id", ""))
            video_path = str(sample.get("video_path", ""))
            resolved = None
            for candidate in apply_video_roots(video_path, video_roots, anno_path):
                if candidate.exists():
                    resolved = candidate
                    break
            if resolved is None:
                resolved = apply_video_roots(video_path, video_roots, anno_path)[0]
            for variant in index_key_variants(video_id, video_path):
                index.setdefault(variant, resolved)
    return index


def video_path_for_row(row: dict, video_index: dict[str, Path]) -> Path | None:
    for key in index_key_variants(row.get("video_id", ""), row.get("video_id", "")):
        if key in video_index:
            return video_index[key]
    return None


def clear_retrieved_gpu_blocks(model) -> None:
    if getattr(model, "kv_cache", None) is None:
        return
    for layer_kv in model.kv_cache:
        if hasattr(layer_kv, "reset_retrieval"):
            layer_kv.reset_retrieval()
        cached_blocks = getattr(layer_kv, "cached_blocks", None)
        global_blocks = getattr(layer_kv, "global_blocks", None)
        if cached_blocks is None or global_blocks is None:
            continue
        for unit_idx, cache_map in enumerate(cached_blocks):
            for block_idx in list(cache_map.keys()):
                try:
                    memory_unit = global_blocks[unit_idx][block_idx]
                    if getattr(memory_unit, "gpu_data", None) is not None:
                        memory_unit.offload()
                except Exception:
                    pass
            cache_map.clear()
        if hasattr(layer_kv, "load_count"):
            layer_kv.load_count = 0


def stats_ms(values: list[float]) -> dict:
    if not values:
        return {"mean": "", "median": "", "p90": "", "std": ""}
    values = sorted(values)
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": values[min(len(values) - 1, int(0.9 * (len(values) - 1)))],
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def measure_question(model, row: dict, device, repeats: int, warmup: int, cold_retrieval: bool) -> dict:
    choices = parse_choices(row.get("choices", ""))
    if not choices:
        raise ValueError(f"No choices for row video_id={row.get('video_id')} question={row.get('question')!r}")
    input_text = format_mcqa_prompt(row.get("question", ""), choices, model)
    num_choices = min(len(choices), len(CHOICE_LETTERS))

    for _ in range(warmup):
        if cold_retrieval:
            clear_retrieved_gpu_blocks(model)
        model.multiple_choice_answering(input_text, num_choices=num_choices, return_scores=False)
        cuda_sync(device)

    latencies_ms = []
    for _ in range(repeats):
        if cold_retrieval:
            clear_retrieved_gpu_blocks(model)
        _, elapsed = timed_call(
            lambda: model.multiple_choice_answering(input_text, num_choices=num_choices, return_scores=False),
            device,
        )
        latencies_ms.append(elapsed * 1000.0)
    stat = stats_ms(latencies_ms)
    return {
        "mean_ms": stat["mean"],
        "median_ms": stat["median"],
        "p90_ms": stat["p90"],
        "std_ms": stat["std"],
        "repeats_ms": json.dumps(latencies_ms),
    }


def measure_fs(args: argparse.Namespace, rows: list[dict], fs_name: str, video_index: dict[str, Path]) -> dict[str, dict]:
    if fs_name == "224":
        frame_size = args.fs224_frame_size
        local_block_count = args.fs224_local_block_count
        retrieve_size = args.fs224_retrieve_size
    else:
        frame_size = args.fs112_frame_size
        local_block_count = args.fs112_local_block_count
        retrieve_size = args.fs112_retrieve_size

    print(
        f"[load_model] fs{fs_name} frame_size={frame_size} "
        f"local_block_count={local_block_count} retrieve_size={retrieve_size}",
        flush=True,
    )
    model, _ = load_model(
        model_path=args.model_path,
        local_block_count=local_block_count,
        topk=retrieve_size,
        chunk_size=args.retrieve_chunk_size,
        frame_size=frame_size,
    )
    device = next(model.parameters()).device
    cold_retrieval = not args.warm_retrieval_cache

    rows_by_video = defaultdict(list)
    skipped = {}
    for row in rows:
        video_path = video_path_for_row(row, video_index)
        if video_path is None or not video_path.exists():
            message = f"missing video for video_id={row.get('video_id')} resolved={video_path}"
            if args.skip_missing_videos:
                skipped[row["sample_uid"]] = {"error": message}
                continue
            raise FileNotFoundError(message)
        rows_by_video[str(video_path)].append(row)

    results = dict(skipped)
    for video_path_str, group in rows_by_video.items():
        video_path = Path(video_path_str)
        print(f"[video] fs{fs_name} {video_path} rows={len(group)}", flush=True)
        model.clear_cache()
        video_tensor = load_video(video_path, args.sample_fps)
        _, build_elapsed = timed_call(
            lambda: (model.encode_init_prompt(), model.encode_video(video_tensor)),
            device,
        )
        build_ms = build_elapsed * 1000.0
        for row in group:
            measured = measure_question(model, row, device, args.repeats, args.warmup, cold_retrieval)
            measured["cache_build_ms_for_video"] = build_ms
            results[row["sample_uid"]] = measured

    model.clear_cache()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def row_key(row: dict) -> tuple[str, str, str, str]:
    return (row["video_id"], row["question"], row.get("choices", ""), row.get("correct_choice", ""))


def load_side_csvs(paths: set[str]) -> dict[tuple[str, str, str, str], dict]:
    out = {}
    for path_str in sorted(path for path in paths if path):
        path = Path(path_str)
        if not path.exists():
            continue
        for row in read_csv(path):
            out.setdefault(row_key(row), row)
    return out


def load_selective_router_module():
    path = REPO_ROOT / "tools" / "selective_confidence_router.py"
    spec = importlib.util.spec_from_file_location("selective_confidence_router_for_latency", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def measure_verifier(args: argparse.Namespace, rows: list[dict]) -> dict[str, dict]:
    if not args.measure_verifier:
        return {}
    if not args.verifier_model:
        raise ValueError("--measure-verifier requires --verifier-model")

    scr = load_selective_router_module()
    client = scr.load_openai_client(args.verifier_api_key_env, args.verifier_base_url, args.verifier_timeout)
    side112 = load_side_csvs({row.get("source_112", "") for row in rows})
    side224 = load_side_csvs({row.get("source_224", "") for row in rows})
    verifier_args = SimpleNamespace(
        feature_columns=args.feature_columns,
        include_task=args.include_task,
        include_gate_context=args.include_gate_context,
        include_feature_deltas=args.include_feature_deltas,
    )

    out = {}
    for row in rows:
        key = row_key(row)
        row112 = side112.get(key)
        row224 = side224.get(key)
        if row112 is None or row224 is None:
            out[row["sample_uid"]] = {"error": "missing source side row for verifier prompt"}
            continue
        ex = scr.SelectiveExample(
            index=int(row.get("_routed_row_index", 0) or 0),
            key=row.get("key", ""),
            video_id=row.get("video_id", ""),
            question=row.get("question", ""),
            choices=row.get("choices", ""),
            correct_choice=row.get("correct_choice", ""),
            task=row.get("task", ""),
            row112=row112,
            row224=row224,
            acc112=float(row.get("acc112", 0) or 0),
            acc224=float(row.get("acc224", 0) or 0),
            pred112=row.get("pred112", ""),
            pred224=row.get("pred224", ""),
        )
        default_fs = row.get("default_fs", "224") or "224"
        messages = scr.build_verifier_messages(
            ex,
            verifier_args,
            scr.randomized_candidates(ex, args.verifier_seed),
            default_fs,
        )
        latencies_ms = []
        raw_response = ""
        error = ""
        for _ in range(args.repeats):
            start = time.perf_counter()
            raw_response, request_error = scr.request_llm(
                client=client,
                model=args.verifier_model,
                messages=messages,
                temperature=args.verifier_temperature,
                max_tokens=args.verifier_max_tokens,
                response_format_json=args.verifier_response_format_json,
                max_retries=args.verifier_max_retries,
                retry_sleep=args.verifier_retry_sleep,
            )
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            if request_error is not None:
                error = request_error
        stat = stats_ms(latencies_ms)
        out[row["sample_uid"]] = {
            "mean_ms": stat["mean"],
            "median_ms": stat["median"],
            "p90_ms": stat["p90"],
            "std_ms": stat["std"],
            "repeats_ms": json.dumps(latencies_ms),
            "raw_response": raw_response,
            "error": error,
        }
    return out


def numeric(value) -> float | None:
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def build_summary(rows: list[dict], output_rows: list[dict], verifier_measured: bool) -> dict:
    by_case = defaultdict(list)
    full_ratios = {}
    for row in output_rows:
        case = row["latency_case"]
        by_case[case].append(row)
        full_ratios[case] = float(row.get("case_full_ratio", 0.0) or 0.0)

    case_summary = {}
    weighted_video_side = 0.0
    weighted_with_verifier = 0.0
    weighted_fs224 = 0.0
    for case, group in by_case.items():
        t224 = [numeric(row.get("fs224_mean_ms")) for row in group]
        t112 = [numeric(row.get("fs112_mean_ms")) for row in group]
        tv = [numeric(row.get("verifier_mean_ms")) for row in group]
        t224 = [v for v in t224 if v is not None]
        t112 = [v for v in t112 if v is not None]
        tv = [v for v in tv if v is not None]

        if case == CASE_HIGH:
            video_side_costs = t224
            routed_costs = t224
        elif case == CASE_AGREE:
            video_side_costs = [
                numeric(row.get("fs224_mean_ms")) + numeric(row.get("fs112_mean_ms"))
                for row in group
                if numeric(row.get("fs224_mean_ms")) is not None and numeric(row.get("fs112_mean_ms")) is not None
            ]
            routed_costs = list(video_side_costs)
        else:
            video_side_costs = [
                numeric(row.get("fs224_mean_ms")) + numeric(row.get("fs112_mean_ms"))
                for row in group
                if numeric(row.get("fs224_mean_ms")) is not None and numeric(row.get("fs112_mean_ms")) is not None
            ]
            routed_costs = [
                numeric(row.get("fs224_mean_ms"))
                + numeric(row.get("fs112_mean_ms"))
                + (numeric(row.get("verifier_mean_ms")) or 0.0)
                for row in group
                if numeric(row.get("fs224_mean_ms")) is not None and numeric(row.get("fs112_mean_ms")) is not None
            ]

        ratio = full_ratios.get(case, 0.0)
        video_side_case_mean = mean(video_side_costs)
        case_mean = mean(routed_costs)
        fs224_mean = mean(t224)
        weighted_video_side += ratio * video_side_case_mean
        weighted_with_verifier += ratio * case_mean
        weighted_fs224 += ratio * fs224_mean
        case_summary[case] = {
            "full_ratio": ratio,
            "sample_count": len(group),
            "fs224_mean_ms": fs224_mean,
            "fs112_mean_ms": mean(t112),
            "verifier_mean_ms": mean(tv),
            "video_side_case_mean_ms": video_side_case_mean,
            "routed_case_mean_ms": case_mean,
        }

    return {
        "input_sample_count": len(rows),
        "measured_sample_count": len(output_rows),
        "verifier_measured": verifier_measured,
        "case_summary": case_summary,
        "weighted_video_side_ms": weighted_video_side,
        "weighted_with_measured_verifier_ms": weighted_with_verifier,
        "weighted_routed_ms": weighted_with_verifier,
        "weighted_fs224_only_ms": weighted_fs224,
        "relative_video_side_cost_vs_fs224": (weighted_video_side / weighted_fs224) if weighted_fs224 else 0.0,
        "relative_with_measured_verifier_cost_vs_fs224": (
            weighted_with_verifier / weighted_fs224
        ) if weighted_fs224 else 0.0,
        "relative_cost_vs_fs224": (weighted_with_verifier / weighted_fs224) if weighted_fs224 else 0.0,
    }


def main() -> None:
    args = parse_args()
    rows = read_csv(args.samples_csv)
    for idx, row in enumerate(rows):
        row["sample_uid"] = f"{row.get('latency_case', 'case')}_{idx}"

    video_index = build_video_index(args.anno_path, args.video_root)
    if args.dry_run_resolve:
        resolved = 0
        existing = 0
        missing_examples = []
        for row in rows:
            path = video_path_for_row(row, video_index)
            if path is not None:
                resolved += 1
                if path.exists():
                    existing += 1
                elif len(missing_examples) < 5:
                    missing_examples.append(
                        {
                            "video_id": row.get("video_id", ""),
                            "resolved_path": str(path),
                            "latency_case": row.get("latency_case", ""),
                        }
                    )
        summary = {
            "samples": len(rows),
            "resolved": resolved,
            "existing": existing,
            "missing": len(rows) - existing,
            "missing_examples": missing_examples,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    rows_fs224 = rows
    rows_fs112 = rows if args.measure_fs112_for_all else [
        row for row in rows if row.get("latency_case") != CASE_HIGH
    ]
    rows_verifier = [row for row in rows if row.get("latency_case") == CASE_DISAGREE]

    fs224_results = measure_fs(args, rows_fs224, "224", video_index)
    fs112_results = measure_fs(args, rows_fs112, "112", video_index)
    verifier_results = measure_verifier(args, rows_verifier)

    output_rows = []
    for row in rows:
        uid = row["sample_uid"]
        out = dict(row)
        for prefix, results in [
            ("fs224", fs224_results),
            ("fs112", fs112_results),
            ("verifier", verifier_results),
        ]:
            result = results.get(uid, {})
            out[f"{prefix}_mean_ms"] = result.get("mean_ms", "")
            out[f"{prefix}_median_ms"] = result.get("median_ms", "")
            out[f"{prefix}_p90_ms"] = result.get("p90_ms", "")
            out[f"{prefix}_std_ms"] = result.get("std_ms", "")
            out[f"{prefix}_repeats_ms"] = result.get("repeats_ms", "")
            out[f"{prefix}_error"] = result.get("error", "")
        out["fs224_cache_build_ms_for_video"] = fs224_results.get(uid, {}).get("cache_build_ms_for_video", "")
        out["fs112_cache_build_ms_for_video"] = fs112_results.get(uid, {}).get("cache_build_ms_for_video", "")
        output_rows.append(out)

    extra_fields = [
        "sample_uid",
        "fs224_mean_ms",
        "fs224_median_ms",
        "fs224_p90_ms",
        "fs224_std_ms",
        "fs224_repeats_ms",
        "fs224_error",
        "fs224_cache_build_ms_for_video",
        "fs112_mean_ms",
        "fs112_median_ms",
        "fs112_p90_ms",
        "fs112_std_ms",
        "fs112_repeats_ms",
        "fs112_error",
        "fs112_cache_build_ms_for_video",
        "verifier_mean_ms",
        "verifier_median_ms",
        "verifier_p90_ms",
        "verifier_std_ms",
        "verifier_repeats_ms",
        "verifier_error",
    ]
    fieldnames = list(rows[0].keys()) + [field for field in extra_fields if field not in rows[0]]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    summary = build_summary(rows, output_rows, args.measure_verifier)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
