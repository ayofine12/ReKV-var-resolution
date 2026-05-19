#!/usr/bin/env python3
import argparse
import gc
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from decord import VideoReader, cpu

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
for root in (REPO_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import prototype_interleaved_video_encoding as proto


@dataclass(frozen=True)
class BranchSpec:
    name: str
    requested_fps: float
    frame_size: int
    local_block_count: int
    retrieve_size: int
    retrieve_chunk_size: int
    encode_chunk_size: int
    internal_block_size: int | None


CASE_DEFINITIONS = {
    "A": ("fs224@1.0 only baseline", (("fs224", 1.0),)),
    "B": ("fs112@2.0 + fs224@0.5 temporal-heavy", (("fs112", 2.0), ("fs224", 0.5))),
    "C": ("fs112@1.0 + fs224@0.75 balanced-highres", (("fs112", 1.0), ("fs224", 0.75))),
    "D": ("fs112@0.5 + fs224@0.875 spatial-heavy", (("fs112", 0.5), ("fs224", 0.875))),
    "E": ("fs112@4.0 only low-res dense", (("fs112", 4.0),)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep fixed-budget resolution/fps allocations and measure video-cache encoding latency. "
            "This intentionally skips retrieval and QA."
        )
    )
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--model-path", default="/mnt/models/qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--cases", nargs="+", default=["A", "B", "C", "D", "E"], choices=sorted(CASE_DEFINITIONS))
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)

    parser.add_argument("--offload-granularity", choices=["block", "chunk"], default=os.getenv("REKV_OFFLOAD_GRANULARITY", "block"))
    parser.add_argument("--batched-offload-copy", action="store_true", default=os.getenv("REKV_BATCHED_OFFLOAD_COPY", "0") == "1")
    parser.add_argument("--max-pending-chunks-per-branch", type=int, default=1)
    parser.add_argument("--branch-order", choices=["listed", "fs224-first", "fs112-first"], default="listed")

    parser.add_argument("--fs112-frame-size", type=int, default=112)
    parser.add_argument("--fs112-local-block-count", type=int, default=72)
    parser.add_argument("--fs112-retrieve-size", type=int, default=144)
    parser.add_argument("--fs112-retrieve-chunk-size", type=int, default=4)
    parser.add_argument("--fs112-encode-chunk-size", type=int, default=64)
    parser.add_argument("--fs112-internal-block-size", type=int, default=0)

    parser.add_argument("--fs224-frame-size", type=int, default=224)
    parser.add_argument("--fs224-local-block-count", type=int, default=18)
    parser.add_argument("--fs224-retrieve-size", type=int, default=36)
    parser.add_argument("--fs224-retrieve-chunk-size", type=int, default=1)
    parser.add_argument("--fs224-encode-chunk-size", type=int, default=16)
    parser.add_argument("--fs224-internal-block-size", type=int, default=0)
    return parser.parse_args()


def stats_ms(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean_ms": 0.0, "median_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "std_ms": 0.0, "values_ms": []}
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "std_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "values_ms": values,
    }


def load_sampled_video_with_meta(video_path: Path, requested_fps: float):
    start = time.perf_counter()
    vr = VideoReader(str(video_path), ctx=cpu(0))
    fps = round(vr.get_avg_fps())
    frame_step = max(1, int(fps / requested_fps))
    frame_indices = list(range(0, len(vr), frame_step))
    video = vr.get_batch(frame_indices).asnumpy()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    effective_fps = fps / frame_step
    return video, {
        "original_fps_rounded": int(fps),
        "sampling_step": int(frame_step),
        "requested_fps": float(requested_fps),
        "effective_fps": float(effective_fps),
        "sampled_frames": int(video.shape[0]),
        "decode_wall_ms": elapsed_ms,
    }


def make_branch_spec(args: argparse.Namespace, branch_name: str, requested_fps: float) -> BranchSpec:
    if branch_name == "fs112":
        return BranchSpec(
            name=branch_name,
            requested_fps=requested_fps,
            frame_size=args.fs112_frame_size,
            local_block_count=args.fs112_local_block_count,
            retrieve_size=args.fs112_retrieve_size,
            retrieve_chunk_size=args.fs112_retrieve_chunk_size,
            encode_chunk_size=args.fs112_encode_chunk_size,
            internal_block_size=args.fs112_internal_block_size or None,
        )
    if branch_name == "fs224":
        return BranchSpec(
            name=branch_name,
            requested_fps=requested_fps,
            frame_size=args.fs224_frame_size,
            local_block_count=args.fs224_local_block_count,
            retrieve_size=args.fs224_retrieve_size,
            retrieve_chunk_size=args.fs224_retrieve_chunk_size,
            encode_chunk_size=args.fs224_encode_chunk_size,
            internal_block_size=args.fs224_internal_block_size or None,
        )
    raise ValueError(f"Unknown branch: {branch_name}")


def make_branch(model, args: argparse.Namespace, spec: BranchSpec):
    load_start = time.perf_counter()
    source_video, sample_meta = load_sampled_video_with_meta(args.video_path, spec.requested_fps)
    resize_start = time.perf_counter()
    resized_video = proto.resize_video_to_square(source_video, spec.frame_size)
    resize_ms = (time.perf_counter() - resize_start) * 1000.0

    device = next(model.parameters()).device
    processor = proto.make_processor(args.model_path, spec.frame_size)
    n_frame_tokens = proto.frame_tokens_for_processor(processor, spec.frame_size)
    n_local = spec.local_block_count * n_frame_tokens
    init_prompt_ids = proto.make_init_prompt_ids(processor, device)
    config = proto.BranchConfig(
        name=spec.name,
        frame_size=spec.frame_size,
        local_block_count=spec.local_block_count,
        retrieve_size=spec.retrieve_size,
        retrieve_chunk_size=spec.retrieve_chunk_size,
        encode_chunk_size=spec.encode_chunk_size,
        internal_block_size=spec.internal_block_size,
    )
    branch = proto.BranchState(
        config=config,
        processor=processor,
        init_prompt_ids=init_prompt_ids,
        n_frame_tokens=n_frame_tokens,
        n_local=n_local,
        video=torch.from_numpy(resized_video),
        compute_stream=torch.cuda.Stream(),
        offload_stream=torch.cuda.Stream(),
    )
    branch_meta = {
        **sample_meta,
        "prepare_video_wall_ms": (time.perf_counter() - load_start) * 1000.0,
        "resize_wall_ms": resize_ms,
        "n_frame_tokens": int(n_frame_tokens),
        "total_video_tokens": int(n_frame_tokens * sample_meta["sampled_frames"]),
        "effective_tokens_per_second": float(n_frame_tokens * sample_meta["effective_fps"]),
        "n_local": int(n_local),
        "local_tokens": int(n_local),
        "frame_size": int(spec.frame_size),
        "local_block_count": int(spec.local_block_count),
        "retrieve_size": int(spec.retrieve_size),
        "retrieve_chunk_size": int(spec.retrieve_chunk_size),
        "encode_chunk_size": int(spec.encode_chunk_size),
        "resolved_internal_block_size": int(proto.branch_internal_block_size(branch)),
    }
    return branch, branch_meta


def order_branches(branches: list[proto.BranchState], branch_order: str) -> list[proto.BranchState]:
    if branch_order == "listed":
        return branches
    preferred = "fs224" if branch_order == "fs224-first" else "fs112"
    return sorted(branches, key=lambda branch: branch.config.name != preferred)


def run_branch_sequence(model, branches: list[proto.BranchState], max_pending: int) -> tuple[float, dict[str, float]]:
    proto.reset_branches(model, branches)
    proto.encode_init_prompts(model, branches)
    proto.clear_kv_profile_records(branches)

    proto.cuda_sync()
    total_start = time.perf_counter()
    branch_ms = {}
    for branch in branches:
        branch_start = time.perf_counter()
        pending = []
        while branch.has_next_chunk():
            proto.wait_oldest_if_needed(pending, max_pending, branch.config.name)
            pending.append(proto.encode_video_chunk(model, branch))
        for event in pending:
            event.synchronize()
        proto.sync_branch(branch)
        branch_ms[branch.config.name] = (time.perf_counter() - branch_start) * 1000.0
    proto.cuda_sync()
    total_ms = (time.perf_counter() - total_start) * 1000.0
    return total_ms, branch_ms


def run_case(model, args: argparse.Namespace, case_name: str) -> dict:
    description, branch_defs = CASE_DEFINITIONS[case_name]
    branches = []
    branch_meta = {}
    for branch_name, requested_fps in branch_defs:
        spec = make_branch_spec(args, branch_name, requested_fps)
        branch, meta = make_branch(model, args, spec)
        branches.append(branch)
        branch_meta[branch_name] = meta

    branches_by_name = {branch.config.name: branch for branch in branches}
    proto.install_dynamic_branch_attention(model, branches_by_name)
    branches = order_branches(branches, args.branch_order)

    for _ in range(args.warmup):
        run_branch_sequence(model, branches, args.max_pending_chunks_per_branch)

    total_values = []
    per_branch_values = {branch.config.name: [] for branch in branches}
    parallel_estimate_values = []
    for _ in range(args.repeats):
        total_ms, branch_ms = run_branch_sequence(model, branches, args.max_pending_chunks_per_branch)
        total_values.append(total_ms)
        if branch_ms:
            parallel_estimate_values.append(max(branch_ms.values()))
        for branch_name, elapsed_ms in branch_ms.items():
            per_branch_values[branch_name].append(elapsed_ms)

    for branch in branches:
        proto.sync_branch(branch)
    proto.reset_branches(model, branches)
    del branches
    gc.collect()
    torch.cuda.empty_cache()

    effective_tokens_per_second = sum(meta["effective_tokens_per_second"] for meta in branch_meta.values())
    total_video_tokens = sum(meta["total_video_tokens"] for meta in branch_meta.values())
    return {
        "case": case_name,
        "description": description,
        "branch_order": [name for name, _ in branch_defs] if args.branch_order == "listed" else [branch.config.name for branch in order_branches(list(branches_by_name.values()), args.branch_order)],
        "branches": branch_meta,
        "effective_tokens_per_second": effective_tokens_per_second,
        "total_video_tokens": int(total_video_tokens),
        "single_gpu_sequential": stats_ms(total_values),
        "two_gpu_parallel_estimate": stats_ms(parallel_estimate_values),
        "per_branch_sequential_segments": {
            name: stats_ms(values) for name, values in per_branch_values.items()
        },
    }


def print_table(results: list[dict]) -> None:
    baseline = next((item for item in results if item["case"] == "A"), None)
    baseline_seq = baseline["single_gpu_sequential"]["mean_ms"] if baseline else 0.0
    baseline_par = baseline["two_gpu_parallel_estimate"]["mean_ms"] if baseline else 0.0
    header = (
        "case", "tok/s", "tokens", "seq_ms", "seq_vs_A", "2gpu_est_ms", "2gpu_vs_A", "branches"
    )
    print("\t".join(header))
    for item in results:
        seq = item["single_gpu_sequential"]["mean_ms"]
        par = item["two_gpu_parallel_estimate"]["mean_ms"]
        branch_desc = ", ".join(
            f"{name}:fps={meta['requested_fps']}=>{meta['effective_fps']:.3g},frames={meta['sampled_frames']},rcs={meta['retrieve_chunk_size']}"
            for name, meta in item["branches"].items()
        )
        print("\t".join([
            item["case"],
            f"{item['effective_tokens_per_second']:.2f}",
            str(item["total_video_tokens"]),
            f"{seq:.2f}",
            f"{seq / baseline_seq:.3f}" if baseline_seq else "",
            f"{par:.2f}",
            f"{par / baseline_par:.3f}" if baseline_par else "",
            branch_desc,
        ]))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    if not args.video_path.exists():
        raise FileNotFoundError(args.video_path)
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    os.environ.setdefault("REKV_DEFER_OFFLOAD_WAIT", "1")
    os.environ["REKV_OFFLOAD_GRANULARITY"] = args.offload_granularity
    os.environ["REKV_BATCHED_OFFLOAD_COPY"] = "1" if args.batched_offload_copy else "0"

    model = proto.load_shared_model(args)
    results = []
    for case_name in args.cases:
        print(f"[case {case_name}] {CASE_DEFINITIONS[case_name][0]}", file=sys.stderr, flush=True)
        results.append(run_case(model, args, case_name))

    summary = {
        "video_path": str(args.video_path),
        "retrieval_or_qa_executed": False,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "rekv_defer_offload_wait": os.getenv("REKV_DEFER_OFFLOAD_WAIT", "0"),
        "rekv_offload_granularity": os.getenv("REKV_OFFLOAD_GRANULARITY", "block"),
        "rekv_batched_offload_copy": os.getenv("REKV_BATCHED_OFFLOAD_COPY", "0"),
        "max_pending_chunks_per_branch": args.max_pending_chunks_per_branch,
        "results": results,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print_table(results)


if __name__ == "__main__":
    main()
