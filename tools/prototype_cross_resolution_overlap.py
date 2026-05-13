#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from dataclasses import dataclass

import torch


@dataclass
class BranchConfig:
    name: str
    compute_dim: int
    compute_iters: int
    offload_mb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic prototype for cross-resolution compute/offload overlap. "
            "It models each branch as GPU compute followed by GPU-to-host KV offload."
        )
    )
    parser.add_argument("--chunks", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--fs112-compute-dim", type=int, default=2048)
    parser.add_argument("--fs112-compute-iters", type=int, default=4)
    parser.add_argument("--fs112-offload-mb", type=float, default=128.0)
    parser.add_argument("--fs224-compute-dim", type=int, default=3072)
    parser.add_argument("--fs224-compute-iters", type=int, default=6)
    parser.add_argument("--fs224-offload-mb", type=float, default=128.0)
    parser.add_argument(
        "--max-pending-events",
        type=int,
        default=2,
        help="Maximum unresolved offload events per branch in the overlapped schedule.",
    )
    parser.add_argument("--json", action="store_true", help="Print only JSON summary.")
    return parser.parse_args()


def cuda_sync() -> None:
    torch.cuda.synchronize()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def make_branch_tensors(config: BranchConfig, dtype: torch.dtype) -> dict:
    compute_a = torch.randn((config.compute_dim, config.compute_dim), device="cuda", dtype=dtype)
    compute_b = torch.randn((config.compute_dim, config.compute_dim), device="cuda", dtype=dtype)
    n_elements = max(1, int(config.offload_mb * 1024 * 1024 / torch.tensor([], dtype=dtype).element_size()))
    offload_src = torch.empty((n_elements,), device="cuda", dtype=dtype)
    offload_dst = torch.empty((n_elements,), device="cpu", dtype=dtype, pin_memory=True)
    return {
        "a": compute_a,
        "b": compute_b,
        "offload_src": offload_src,
        "offload_dst": offload_dst,
    }


def branch_compute(tensors: dict, iters: int) -> torch.cuda.Event:
    out = tensors["a"]
    for _ in range(iters):
        out = out @ tensors["b"]
    # The output is intentionally synthetic; the event models "KV is ready to offload".
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream())
    return ready


def branch_offload(
    tensors: dict,
    stream: torch.cuda.Stream,
    ready_event: torch.cuda.Event | None = None,
) -> torch.cuda.Event:
    with torch.cuda.stream(stream):
        if ready_event is not None:
            stream.wait_event(ready_event)
        tensors["offload_dst"].copy_(tensors["offload_src"], non_blocking=True)
        event = torch.cuda.Event()
        event.record(stream)
    return event


def wait_pending(pending: list[torch.cuda.Event], max_pending: int) -> None:
    while pending and len(pending) >= max_pending:
        event = pending.pop(0)
        event.synchronize()


def finish_pending(pending: list[torch.cuda.Event]) -> None:
    for event in pending:
        event.synchronize()
    pending.clear()


def run_sequential(branches: list[tuple[BranchConfig, dict]], chunks: int, streams: dict) -> float:
    cuda_sync()
    start = time.perf_counter()
    for config, tensors in branches:
        pending: list[torch.cuda.Event] = []
        for _ in range(chunks):
            ready = branch_compute(tensors, config.compute_iters)
            pending.append(branch_offload(tensors, streams[config.name], ready))
            finish_pending(pending)
    cuda_sync()
    return (time.perf_counter() - start) * 1000.0


def run_interleaved(
    branches: list[tuple[BranchConfig, dict]],
    chunks: int,
    streams: dict,
    max_pending_events: int,
) -> float:
    cuda_sync()
    pending = {config.name: [] for config, _ in branches}
    start = time.perf_counter()
    for _ in range(chunks):
        for config, tensors in branches:
            wait_pending(pending[config.name], max_pending_events)
            ready = branch_compute(tensors, config.compute_iters)
            pending[config.name].append(branch_offload(tensors, streams[config.name], ready))
    for events in pending.values():
        finish_pending(events)
    cuda_sync()
    return (time.perf_counter() - start) * 1000.0


def run_compute_only(branches: list[tuple[BranchConfig, dict]], chunks: int) -> float:
    cuda_sync()
    start = time.perf_counter()
    for config, tensors in branches:
        for _ in range(chunks):
            branch_compute(tensors, config.compute_iters)
    cuda_sync()
    return (time.perf_counter() - start) * 1000.0


def run_offload_only(branches: list[tuple[BranchConfig, dict]], chunks: int, streams: dict) -> float:
    cuda_sync()
    start = time.perf_counter()
    pending = []
    for config, tensors in branches:
        for _ in range(chunks):
            pending.append(branch_offload(tensors, streams[config.name]))
            pending[-1].synchronize()
    cuda_sync()
    return (time.perf_counter() - start) * 1000.0


def summarize(values: list[float]) -> dict:
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "std_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "values_ms": values,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this prototype.")

    dtype = dtype_from_name(args.dtype)
    configs = [
        BranchConfig("fs112", args.fs112_compute_dim, args.fs112_compute_iters, args.fs112_offload_mb),
        BranchConfig("fs224", args.fs224_compute_dim, args.fs224_compute_iters, args.fs224_offload_mb),
    ]
    branches = [(config, make_branch_tensors(config, dtype)) for config in configs]
    streams = {config.name: torch.cuda.Stream() for config in configs}

    for _ in range(args.warmup):
        run_sequential(branches, args.chunks, streams)
        run_interleaved(branches, args.chunks, streams, args.max_pending_events)

    sequential = [run_sequential(branches, args.chunks, streams) for _ in range(args.repeats)]
    interleaved = [
        run_interleaved(branches, args.chunks, streams, args.max_pending_events)
        for _ in range(args.repeats)
    ]
    compute_only = [run_compute_only(branches, args.chunks) for _ in range(args.repeats)]
    offload_only = [run_offload_only(branches, args.chunks, streams) for _ in range(args.repeats)]

    sequential_mean = statistics.mean(sequential)
    interleaved_mean = statistics.mean(interleaved)
    summary = {
        "chunks": args.chunks,
        "dtype": args.dtype,
        "max_pending_events": args.max_pending_events,
        "branches": [config.__dict__ for config in configs],
        "sequential": summarize(sequential),
        "interleaved": summarize(interleaved),
        "compute_only": summarize(compute_only),
        "offload_only": summarize(offload_only),
        "speedup_interleaved_vs_sequential": sequential_mean / interleaved_mean if interleaved_mean else 0.0,
        "overlap_reduction_ratio": 1.0 - interleaved_mean / sequential_mean if sequential_mean else 0.0,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print(json.dumps(summary, indent=2))
    print()
    print(
        "Interpretation: interleaved < sequential means cross-resolution scheduling hides "
        "some branch-local offload stalls. compute_only/offload_only are synthetic lower/upper context, "
        "not ReKV accuracy measurements."
    )


if __name__ == "__main__":
    main()
