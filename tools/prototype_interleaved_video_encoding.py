#!/usr/bin/env python3
import argparse
import gc
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.attention import kv_cache_manager
from model.patch import patch_hf
from model.qwen2_5_vl_rekv import load_model


@dataclass
class BranchConfig:
    name: str
    frame_size: int
    local_block_count: int
    retrieve_size: int
    retrieve_chunk_size: int
    encode_chunk_size: int


@dataclass
class BranchState:
    config: BranchConfig
    processor: object
    init_prompt_ids: torch.Tensor
    n_frame_tokens: int
    n_local: int
    video: torch.Tensor
    compute_stream: torch.cuda.Stream
    offload_stream: torch.cuda.Stream
    kv_cache: object = None
    start_idx: int = 0
    effective_chunk_size: int = 0

    def __post_init__(self) -> None:
        self.effective_chunk_size = self.config.encode_chunk_size

    def reset(self) -> None:
        self.kv_cache = None
        self.start_idx = 0
        self.effective_chunk_size = self.config.encode_chunk_size

    def has_next_chunk(self) -> bool:
        return self.start_idx < self.video.shape[0]

    def current_chunk(self) -> torch.Tensor:
        end_idx = min(self.start_idx + self.effective_chunk_size, self.video.shape[0])
        return self.video[self.start_idx:end_idx]

    def advance_chunk(self, chunk: torch.Tensor) -> None:
        self.start_idx += chunk.shape[0]

    def shrink_chunk_size(self, failed_chunk_size: int) -> None:
        if failed_chunk_size <= 1:
            raise AssertionError(
                f"{self.config.name}: single-frame chunk still exceeds n_local={self.n_local}"
            )
        self.effective_chunk_size = max(1, failed_chunk_size // 2)


class ChunkTooLarge(Exception):
    def __init__(self, branch_name: str, n_local: int, feature_tokens: int, chunk_size: int):
        super().__init__(
            f"{branch_name}: n_local={n_local}, video_features={feature_tokens}, chunk_size={chunk_size}"
        )
        self.chunk_size = chunk_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shared-model prototype for interleaved fs112/fs224 ReKV video encoding. "
            "It loads model weights once, keeps branch-local processors/KV caches/streams, "
            "and intentionally skips retrieval and QA."
        )
    )
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--model-path", default="/mnt/models/qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", action="store_true")

    parser.add_argument("--fs112-frame-size", type=int, default=112)
    parser.add_argument("--fs112-local-block-count", type=int, default=72)
    parser.add_argument("--fs112-retrieve-size", type=int, default=144)
    parser.add_argument("--fs112-retrieve-chunk-size", type=int, default=1)
    parser.add_argument("--fs112-encode-chunk-size", type=int, default=64)

    parser.add_argument("--fs224-frame-size", type=int, default=224)
    parser.add_argument("--fs224-local-block-count", type=int, default=18)
    parser.add_argument("--fs224-retrieve-size", type=int, default=36)
    parser.add_argument("--fs224-retrieve-chunk-size", type=int, default=1)
    parser.add_argument("--fs224-encode-chunk-size", type=int, default=32)

    parser.add_argument(
        "--max-pending-chunks-per-branch",
        type=int,
        default=2,
        help="Throttle unresolved chunks per branch during interleaving. Use 0 for no explicit throttle.",
    )
    parser.add_argument(
        "--sequential-order",
        choices=["fs224-first", "fs112-first"],
        default="fs224-first",
    )
    return parser.parse_args()


def cuda_sync() -> None:
    torch.cuda.synchronize()


def stats_ms(values: list[float]) -> dict:
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "std_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "values_ms": values,
    }


def load_sampled_video(video_path: Path, sample_fps: float) -> tuple[np.ndarray, int, int]:
    vr = VideoReader(str(video_path), ctx=cpu(0))
    fps = round(vr.get_avg_fps())
    frame_step = max(1, int(fps / sample_fps))
    frame_indices = [idx for idx in range(0, len(vr), frame_step)]
    video = vr.get_batch(frame_indices).asnumpy()
    return video, fps, frame_step


def resize_video_to_square(video: np.ndarray, frame_size: int) -> np.ndarray:
    if video.shape[1] == frame_size and video.shape[2] == frame_size:
        return video

    resized = np.empty((video.shape[0], frame_size, frame_size, video.shape[3]), dtype=video.dtype)
    batch_size = 256
    for st in range(0, video.shape[0], batch_size):
        ed = min(st + batch_size, video.shape[0])
        frames = torch.from_numpy(video[st:ed]).permute(0, 3, 1, 2).float()
        frames = F.interpolate(frames, size=(frame_size, frame_size), mode="bilinear", align_corners=False)
        frames = frames.round().clamp(0, 255).to(torch.uint8)
        resized[st:ed] = frames.permute(0, 2, 3, 1).numpy()
    return resized


def make_branch_config(args: argparse.Namespace, name: str) -> BranchConfig:
    if name == "fs112":
        return BranchConfig(
            name=name,
            frame_size=args.fs112_frame_size,
            local_block_count=args.fs112_local_block_count,
            retrieve_size=args.fs112_retrieve_size,
            retrieve_chunk_size=args.fs112_retrieve_chunk_size,
            encode_chunk_size=args.fs112_encode_chunk_size,
        )
    if name == "fs224":
        return BranchConfig(
            name=name,
            frame_size=args.fs224_frame_size,
            local_block_count=args.fs224_local_block_count,
            retrieve_size=args.fs224_retrieve_size,
            retrieve_chunk_size=args.fs224_retrieve_chunk_size,
            encode_chunk_size=args.fs224_encode_chunk_size,
        )
    raise ValueError(f"Unknown branch: {name}")


def make_processor(model_path: str, frame_size: int):
    return AutoProcessor.from_pretrained(
        model_path,
        min_pixels=frame_size * frame_size,
        max_pixels=frame_size * frame_size,
    )


def frame_tokens_for_processor(processor, frame_size: int) -> int:
    spatial_unit = processor.video_processor.patch_size * processor.video_processor.merge_size
    if frame_size % spatial_unit != 0:
        raise ValueError(
            f"frame_size must be divisible by patch_size * merge_size ({spatial_unit}), got {frame_size}"
        )
    return (frame_size // spatial_unit) ** 2


def make_init_prompt_ids(processor, device: torch.device) -> torch.Tensor:
    init_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    return processor.tokenizer(init_prompt, return_tensors="pt").input_ids.to(device)


def branch_attn_config(branch: BranchState) -> dict:
    return {
        "n_init": branch.init_prompt_ids.shape[1],
        "n_local": branch.n_local,
        "fattn": True,
        "block_size": branch.n_frame_tokens,
        "topk": branch.config.retrieve_size,
        "chunk_size": branch.config.retrieve_chunk_size,
        "max_cached_block": max(128, branch.config.retrieve_size),
        "exc_block_size": branch.n_frame_tokens,
        "pin_memory": True,
    }


def patch_model_for_branch(model, branch: BranchState) -> None:
    model.language_model = patch_hf(model.model.language_model, **branch_attn_config(branch))


def apply_branch_state(model, branch: BranchState) -> None:
    model.processor = branch.processor
    model.init_prompt_ids = branch.init_prompt_ids
    model.n_frame_tokens = branch.n_frame_tokens
    model.n_local = branch.n_local
    model.topk = branch.config.retrieve_size
    model.chunk_size = branch.config.retrieve_chunk_size
    model.kv_cache = branch.kv_cache


@contextmanager
def branch_cuda_context(branch: BranchState):
    previous_stream = kv_cache_manager.GLOBAL_STREAM
    kv_cache_manager.GLOBAL_STREAM = branch.offload_stream
    try:
        with torch.cuda.stream(branch.compute_stream):
            yield
    finally:
        kv_cache_manager.GLOBAL_STREAM = previous_stream


def sync_branch(branch: BranchState) -> None:
    branch.compute_stream.synchronize()
    branch.offload_stream.synchronize()


def reset_branches(model, branches: list[BranchState]) -> None:
    for branch in branches:
        sync_branch(branch)
        branch.reset()
    model.kv_cache = None
    gc.collect()
    torch.cuda.empty_cache()
    cuda_sync()


def encode_init_prompt(model, branch: BranchState) -> None:
    patch_model_for_branch(model, branch)
    apply_branch_state(model, branch)
    with branch_cuda_context(branch):
        output = model.language_model(
            input_ids=branch.init_prompt_ids,
            use_cache=True,
            return_dict=True,
        )
        branch.kv_cache = output.past_key_values
        model.kv_cache = branch.kv_cache


def encode_init_prompts(model, branches: list[BranchState]) -> None:
    for branch in branches:
        encode_init_prompt(model, branch)
    for branch in branches:
        sync_branch(branch)


def encode_video_chunk(model, branch: BranchState) -> torch.cuda.Event:
    while True:
        chunk = branch.current_chunk()
        patch_model_for_branch(model, branch)
        apply_branch_state(model, branch)
        try:
            with branch_cuda_context(branch):
                pixel_values_videos, video_grid_thw = model._prepare_video_inputs(chunk)
                video_features = model._get_video_features(pixel_values_videos, video_grid_thw)
                spatial_tokens = int(
                    video_grid_thw[0, 1].item()
                    * video_grid_thw[0, 2].item()
                    // (branch.processor.video_processor.merge_size ** 2)
                )
                if spatial_tokens != branch.n_frame_tokens:
                    raise AssertionError(
                        f"{branch.config.name}: expected {branch.n_frame_tokens} tokens per temporal block, "
                        f"got {spatial_tokens}; video_grid_thw={video_grid_thw.tolist()}"
                    )
                if branch.n_local < video_features.shape[1]:
                    raise ChunkTooLarge(
                        branch.config.name,
                        branch.n_local,
                        int(video_features.shape[1]),
                        int(chunk.shape[0]),
                    )
                output = model.language_model(
                    inputs_embeds=video_features,
                    past_key_values=branch.kv_cache,
                    use_cache=True,
                    return_dict=True,
                )
                branch.kv_cache = output.past_key_values
                model.kv_cache = branch.kv_cache
                branch.advance_chunk(chunk)
                event = torch.cuda.Event()
                event.record(branch.compute_stream)
            return event
        except ChunkTooLarge as exc:
            sync_branch(branch)
            branch.shrink_chunk_size(exc.chunk_size)


def wait_oldest_if_needed(pending: list[torch.cuda.Event], max_pending: int) -> None:
    if max_pending <= 0:
        return
    while len(pending) >= max_pending:
        pending.pop(0).synchronize()


def run_sequential(model, branches_by_name: dict[str, BranchState], order: str) -> float:
    branches = [branches_by_name["fs224"], branches_by_name["fs112"]]
    if order == "fs112-first":
        branches = list(reversed(branches))

    reset_branches(model, list(branches_by_name.values()))
    encode_init_prompts(model, branches)

    cuda_sync()
    start = time.perf_counter()
    for branch in branches:
        while branch.has_next_chunk():
            encode_video_chunk(model, branch)
        sync_branch(branch)
    cuda_sync()
    return (time.perf_counter() - start) * 1000.0


def run_interleaved(
    model,
    branches_by_name: dict[str, BranchState],
    max_pending_chunks_per_branch: int,
) -> float:
    branches = [branches_by_name["fs112"], branches_by_name["fs224"]]
    reset_branches(model, list(branches_by_name.values()))
    encode_init_prompts(model, branches)

    pending = {branch.config.name: [] for branch in branches}
    cuda_sync()
    start = time.perf_counter()
    while any(branch.has_next_chunk() for branch in branches):
        for branch in branches:
            if not branch.has_next_chunk():
                continue
            branch_pending = pending[branch.config.name]
            wait_oldest_if_needed(branch_pending, max_pending_chunks_per_branch)
            branch_pending.append(encode_video_chunk(model, branch))

    for branch_events in pending.values():
        for event in branch_events:
            event.synchronize()
    for branch in branches:
        sync_branch(branch)
    cuda_sync()
    return (time.perf_counter() - start) * 1000.0


def build_branches(args: argparse.Namespace, model, source_video: np.ndarray) -> dict[str, BranchState]:
    device = next(model.parameters()).device
    branches = {}
    for name in ("fs112", "fs224"):
        config = make_branch_config(args, name)
        processor = make_processor(args.model_path, config.frame_size)
        n_frame_tokens = frame_tokens_for_processor(processor, config.frame_size)
        n_local = config.local_block_count * n_frame_tokens
        init_prompt_ids = make_init_prompt_ids(processor, device)
        video = torch.from_numpy(resize_video_to_square(source_video, config.frame_size))
        branches[name] = BranchState(
            config=config,
            processor=processor,
            init_prompt_ids=init_prompt_ids,
            n_frame_tokens=n_frame_tokens,
            n_local=n_local,
            video=video,
            compute_stream=torch.cuda.Stream(),
            offload_stream=torch.cuda.Stream(),
        )
    return branches


def load_shared_model(args: argparse.Namespace):
    # load_model is used only to load one set of weights and install the ReKV patch once.
    # Branch-specific patch closures are installed before each branch initializes its KV cache.
    model, _ = load_model(
        model_path=args.model_path,
        local_block_count=args.fs224_local_block_count,
        topk=args.fs224_retrieve_size,
        chunk_size=args.fs224_retrieve_chunk_size,
        frame_size=args.fs224_frame_size,
    )
    cuda_sync()
    return model


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this prototype.")
    if not args.video_path.exists():
        raise FileNotFoundError(args.video_path)

    source_video, fps, frame_step = load_sampled_video(args.video_path, args.sample_fps)
    model = load_shared_model(args)
    branches_by_name = build_branches(args, model, source_video)

    for _ in range(args.warmup):
        run_sequential(model, branches_by_name, args.sequential_order)
        run_interleaved(model, branches_by_name, args.max_pending_chunks_per_branch)

    sequential = [
        run_sequential(model, branches_by_name, args.sequential_order)
        for _ in range(args.repeats)
    ]
    interleaved = [
        run_interleaved(model, branches_by_name, args.max_pending_chunks_per_branch)
        for _ in range(args.repeats)
    ]

    sequential_mean = statistics.mean(sequential)
    interleaved_mean = statistics.mean(interleaved)
    summary = {
        "video_path": str(args.video_path),
        "shared_model": True,
        "retrieval_or_qa_executed": False,
        "rekv_batched_retrieval_io": os.getenv("REKV_BATCHED_RETRIEVAL_IO", "0"),
        "sample_fps": args.sample_fps,
        "original_fps_rounded": fps,
        "sampling_step": frame_step,
        "sampled_frames": int(source_video.shape[0]),
        "sequential_order": args.sequential_order,
        "max_pending_chunks_per_branch": args.max_pending_chunks_per_branch,
        "branches": [
            {
                **branch.config.__dict__,
                "n_frame_tokens": int(branch.n_frame_tokens),
                "n_local": int(branch.n_local),
                "local_tokens": int(branch.n_local),
                "retrieval_tokens_budget_unused": int(branch.n_frame_tokens * branch.config.retrieve_size),
            }
            for branch in branches_by_name.values()
        ],
        "sequential": stats_ms(sequential),
        "interleaved": stats_ms(interleaved),
        "speedup_interleaved_vs_sequential": (
            sequential_mean / interleaved_mean if interleaved_mean else 0.0
        ),
        "overlap_reduction_ratio": (
            1.0 - interleaved_mean / sequential_mean if sequential_mean else 0.0
        ),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.json:
        print()
        print(
            "Interpretation: this uses one shared model instance and branch-local "
            "processors/KV caches/CUDA streams. It measures video encoding only."
        )


if __name__ == "__main__":
    main()
