#!/usr/bin/env python3
"""Build per-question labels for adaptive frame-size routing experiments."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


CONFIGS = {
    "fs112": "fs112_lb32_rs128",
    "fs168": "fs168_lb32_rs57",
    "fs224": "fs224_lb32_rs32",
}
REPO_ROOT = Path(__file__).resolve().parents[1]


def read_result(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="") as f:
        return {(r["video_id"], r["question"]): r for r in csv.DictReader(f)}


def load_video_meta(
    mapping_csv: Path | None,
    video_info_json: Path | None,
    probe_resolution: bool,
) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = defaultdict(dict)
    if mapping_csv and mapping_csv.exists():
        with mapping_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                key = row.get("key")
                if not key:
                    continue
                meta[key].update(
                    {
                        "key": key,
                        "video_type": row.get("type", ""),
                        "duration_minutes": row.get("duration_minutes", ""),
                        "mp4_exists": row.get("mp4_exists", ""),
                        "mp4_path": row.get("mp4_path", ""),
                    }
                )
    if video_info_json and video_info_json.exists():
        with video_info_json.open() as f:
            for item in json.load(f):
                key = item.get("key")
                if key:
                    meta[key].setdefault("video_type", item.get("type", ""))
    if probe_resolution:
        add_video_resolution(meta)
    return dict(meta)


def acc(row: dict[str, str]) -> float:
    return float(row["qa_acc"])


def add_video_resolution(meta: dict[str, dict[str, str]]) -> None:
    if shutil.which("ffprobe") is None:
        return
    for row in meta.values():
        mp4_path = row.get("mp4_path", "")
        if not mp4_path or not Path(mp4_path).exists():
            video_key = row.get("key", "")
            fallback_path = Path("/mnt/ssd1/mwnoh/LVBench/scripts/videos") / f"{video_key}.mp4"
            mp4_path = str(fallback_path) if fallback_path.exists() else ""
        if not mp4_path:
            continue
        try:
            out = subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "csv=p=0:s=x",
                    mp4_path,
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        except (subprocess.SubprocessError, OSError):
            continue
        if "x" not in out:
            continue
        width, height = out.split("x", 1)
        row["source_width"] = width
        row["source_height"] = height
        row["source_resolution_bin"] = resolution_bin(height)


def resolution_bin(height: str) -> str:
    try:
        h = int(height)
    except (TypeError, ValueError):
        return "unknown"
    if h <= 360:
        return "<=360p"
    if h <= 720:
        return "361-720p"
    return ">720p"


def duration_bin(duration_minutes: str) -> str:
    try:
        minutes = float(duration_minutes)
    except (TypeError, ValueError):
        return "unknown"
    if minutes < 30:
        return "<30m"
    if minutes < 60:
        return "30-60m"
    if minutes < 90:
        return "60-90m"
    return ">=90m"


def exact_task_router(task: str, task_best: dict[str, str]) -> str:
    return task_best.get(task, "fs168")


def group_router(
    task: str,
    meta: dict[str, str],
    group_best: dict[tuple[str, str], str],
    group_name: str,
    task_best: dict[str, str],
) -> str:
    if group_name == "video_type":
        group_value = meta.get("video_type", "")
    elif group_name == "duration_bin":
        group_value = duration_bin(meta.get("duration_minutes", ""))
    elif group_name == "resolution_bin":
        group_value = meta.get("source_resolution_bin", "")
    else:
        group_value = ""
    return group_best.get((task, group_value), task_best.get(task, "fs168"))


def heuristic_router(task: str, question: str, meta: dict[str, str]) -> str:
    task_set = {part.strip() for part in task.split(",") if part.strip()}
    q = question.lower()
    video_type = meta.get("video_type", "")
    dur_bin = duration_bin(meta.get("duration_minutes", ""))
    res_bin = meta.get("source_resolution_bin", "")

    detail_words = {
        "color",
        "number",
        "many",
        "how many",
        "wearing",
        "written",
        "text",
        "sign",
        "where",
        "which",
        "what object",
        "what is on",
    }
    flow_words = {
        "after",
        "before",
        "then",
        "next",
        "happen",
        "why",
        "how does",
        "what does",
        "what happens",
    }

    has_detail_word = any(word in q for word in detail_words)
    has_flow_word = any(word in q for word in flow_words)

    if res_bin == "<=360p" and {"entity recognition", "key information retrieval"} & task_set:
        return "fs224"
    if res_bin == "361-720p" and "event understanding" in task_set and "entity recognition" not in task_set:
        return "fs112"
    if video_type == "cartoon" and "entity recognition" in task_set:
        return "fs224"
    if dur_bin == ">=90m" and "event understanding" in task_set and "entity recognition" not in task_set:
        return "fs112"
    if "key information retrieval" in task_set and "entity recognition" in task_set:
        return "fs224"
    if "entity recognition" in task_set and "reasoning" in task_set:
        return "fs224"
    if "event understanding" in task_set and not {"entity recognition", "key information retrieval"} & task_set:
        return "fs112"
    if task_set <= {"reasoning"} or task_set <= {"summarization"}:
        return "fs112"
    if has_detail_word and not has_flow_word:
        return "fs224"
    if has_flow_word and not has_detail_word:
        return "fs112"
    return "fs168"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=Path("/mnt/ssd1/mwnoh/var-resolution-screen"))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/resolution_routing")
    parser.add_argument("--mapping-csv", type=Path, default=Path("/mnt/ssd1/mwnoh/LVBench/video_file_mapping.csv"))
    parser.add_argument("--video-info-json", type=Path, default=Path("/mnt/ssd1/mwnoh/LVBench/data/video_info.json"))
    parser.add_argument("--probe-resolution", action="store_true")
    args = parser.parse_args()

    results = {
        name: read_result(args.result_root / dirname / "1_0.csv")
        for name, dirname in CONFIGS.items()
    }
    common_keys = sorted(set.intersection(*(set(rows) for rows in results.values())))
    if not common_keys:
        raise SystemExit("No common questions found.")

    video_meta = load_video_meta(args.mapping_csv, args.video_info_json, args.probe_resolution)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_task: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in common_keys:
        task = results["fs168"][key]["task"]
        by_task[task].append(key)

    task_best: dict[str, str] = {}
    for task, keys in by_task.items():
        task_scores = {
            name: mean([acc(results[name][key]) for key in keys])
            for name in CONFIGS
        }
        task_best[task] = max(task_scores, key=task_scores.get)

    group_bests: dict[str, dict[tuple[str, str], str]] = {
        "video_type": {},
        "duration_bin": {},
        "resolution_bin": {},
    }
    grouped_keys: dict[str, dict[tuple[str, str], list[tuple[str, str]]]] = {
        "video_type": defaultdict(list),
        "duration_bin": defaultdict(list),
        "resolution_bin": defaultdict(list),
    }
    for key in common_keys:
        video_id, _ = key
        task = results["fs168"][key]["task"]
        meta = video_meta.get(video_id, {})
        grouped_keys["video_type"][(task, meta.get("video_type", ""))].append(key)
        grouped_keys["duration_bin"][(task, duration_bin(meta.get("duration_minutes", "")))].append(key)
        grouped_keys["resolution_bin"][(task, meta.get("source_resolution_bin", ""))].append(key)

    for group_name, buckets in grouped_keys.items():
        for group_key, keys in buckets.items():
            if len(keys) < 5:
                continue
            task_scores = {
                name: mean([acc(results[name][key]) for key in keys])
                for name in CONFIGS
            }
            group_bests[group_name][group_key] = max(task_scores, key=task_scores.get)

    router_scores: dict[str, list[float]] = defaultdict(list)
    rows: list[dict[str, object]] = []

    for key in common_keys:
        video_id, question = key
        base = results["fs168"][key]
        scores = {name: acc(results[name][key]) for name in CONFIGS}
        correct = [name for name, value in scores.items() if value > 0]
        best_any = max(scores.values())
        oracle_choices = [name for name, value in scores.items() if value == best_any]

        meta = video_meta.get(video_id, {})
        exact_choice = exact_task_router(base["task"], task_best)
        task_type_choice = group_router(base["task"], meta, group_bests["video_type"], "video_type", task_best)
        task_duration_choice = group_router(base["task"], meta, group_bests["duration_bin"], "duration_bin", task_best)
        task_resolution_choice = group_router(base["task"], meta, group_bests["resolution_bin"], "resolution_bin", task_best)
        heuristic_choice = heuristic_router(base["task"], question, meta)

        router_scores["fs112"].append(scores["fs112"])
        router_scores["fs168"].append(scores["fs168"])
        router_scores["fs224"].append(scores["fs224"])
        router_scores["oracle_any"].append(best_any)
        router_scores["exact_task_router"].append(scores[exact_choice])
        router_scores["task_video_type_router"].append(scores[task_type_choice])
        router_scores["task_duration_router"].append(scores[task_duration_choice])
        router_scores["task_resolution_router"].append(scores[task_resolution_choice])
        router_scores["heuristic_router"].append(scores[heuristic_choice])

        rows.append(
            {
                "video_id": video_id,
                "video_type": meta.get("video_type", ""),
                "duration_minutes": meta.get("duration_minutes", ""),
                "duration_bin": duration_bin(meta.get("duration_minutes", "")),
                "source_width": meta.get("source_width", ""),
                "source_height": meta.get("source_height", ""),
                "source_resolution_bin": meta.get("source_resolution_bin", ""),
                "question": question,
                "task": base["task"],
                "answer": base["answer"],
                "correct_choice": base["correct_choice"],
                "fs112_acc": scores["fs112"],
                "fs168_acc": scores["fs168"],
                "fs224_acc": scores["fs224"],
                "correct_configs": "|".join(correct),
                "oracle_choices": "|".join(oracle_choices),
                "exact_task_choice": exact_choice,
                "task_video_type_choice": task_type_choice,
                "task_duration_choice": task_duration_choice,
                "task_resolution_choice": task_resolution_choice,
                "heuristic_choice": heuristic_choice,
                "exact_task_acc": scores[exact_choice],
                "task_video_type_acc": scores[task_type_choice],
                "task_duration_acc": scores[task_duration_choice],
                "task_resolution_acc": scores[task_resolution_choice],
                "heuristic_acc": scores[heuristic_choice],
                "fs112_pred_choice": results["fs112"][key]["pred_choice"],
                "fs168_pred_choice": results["fs168"][key]["pred_choice"],
                "fs224_pred_choice": results["fs224"][key]["pred_choice"],
            }
        )

    out_csv = args.output_dir / "per_question_resolution_labels.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.output_dir / "summary.md"
    with summary_path.open("w") as f:
        f.write("# Resolution Routing Summary\n\n")
        f.write(f"- Common questions: {len(common_keys)}\n")
        f.write("- Compared configs: `fs112_lb32_rs128`, `fs168_lb32_rs57`, `fs224_lb32_rs32`\n\n")
        f.write("## Overall\n\n")
        router_order = [
            "fs112",
            "fs168",
            "fs224",
            "oracle_any",
            "exact_task_router",
            "task_video_type_router",
            "task_duration_router",
            "task_resolution_router",
            "heuristic_router",
        ]
        for name in router_order:
            f.write(f"- {name}: {mean(router_scores[name]):.2f}\n")
        f.write(
            "\n`exact_*` and `task_*` routers choose the best config from these same "
            "results, so treat them as analysis upper bounds rather than deployable scores.\n"
        )
        if not args.probe_resolution:
            f.write(
                "`task_resolution_router` falls back to task-only routing unless "
                "`--probe-resolution` is used.\n"
            )
        f.write("\n## Best Config By Task\n\n")
        f.write("| task | n | fs112 | fs168 | fs224 | best |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | --- |\n")
        for task, keys in sorted(by_task.items(), key=lambda item: (-len(item[1]), item[0])):
            task_scores = {
                name: mean([acc(results[name][key]) for key in keys])
                for name in CONFIGS
            }
            best = max(task_scores, key=task_scores.get)
            f.write(
                f"| {task} | {len(keys)} | {task_scores['fs112']:.2f} | "
                f"{task_scores['fs168']:.2f} | {task_scores['fs224']:.2f} | {best} |\n"
            )

    print(f"Wrote {out_csv}")
    print(f"Wrote {summary_path}")
    for name in [
        "fs112",
        "fs168",
        "fs224",
        "oracle_any",
        "exact_task_router",
        "task_video_type_router",
        "task_duration_router",
        "task_resolution_router",
        "heuristic_router",
    ]:
        print(f"{name}: {mean(router_scores[name]):.2f}")


if __name__ == "__main__":
    main()
