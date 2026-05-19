#!/usr/bin/env python3
import argparse
import gc
import json
import os
import statistics
import sys
import time
from collections import defaultdict
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
from model.attention import rekv_attention_forward
from model.attention.kv_cache_manager import ContextManager
from model.qwen2_5_vl_rekv import load_model


@dataclass
class BranchConfig:
    name: str
    frame_size: int
    local_block_count: int
    retrieve_size: int
    retrieve_chunk_size: int
    encode_chunk_size: int
    internal_block_size: int | None


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


@dataclass
class PreparedVideoChunk:
    branch: BranchState
    chunk: torch.Tensor
    video_features: torch.Tensor
    prepare_start_event: torch.cuda.Event | None
    prepare_end_event: torch.cuda.Event | None
    vision_start_event: torch.cuda.Event | None
    vision_end_event: torch.cuda.Event | None
    prepare_start_wall: float
    prepare_end_wall: float


@dataclass
class PreparedVideoInputs:
    branch: BranchState
    chunk: torch.Tensor
    pixel_values_videos: torch.Tensor
    video_grid_thw: torch.Tensor
    prepare_start_event: torch.cuda.Event | None
    prepare_end_event: torch.cuda.Event | None
    prepare_start_wall: float
    prepare_end_wall: float


class DynamicBranchRuntime:
    def __init__(self):
        self.active_name = None
        self.attention_forwards = {}


class ProfileCollector:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.cuda_intervals = []
        self.wall_sections = defaultdict(list)
        self.chunks = defaultdict(lambda: {"count": 0, "frames": 0, "tokens": 0})
        self.internal_cuda_intervals = []
        self.internal_wall_sections = defaultdict(list)
        self.internal_value_sections = defaultdict(list)
        self.internal_blocks = defaultdict(
            lambda: {
                "count": 0,
                "tokens": 0,
                "offload_blocks": 0,
                "offload_bytes": 0,
                "nonzero_offload_count": 0,
                "deferred_post_offload_wait_count": 0,
            }
        )

    def cuda_event(self):
        if not self.enabled:
            return None
        return torch.cuda.Event(enable_timing=True)

    def add_cuda_interval(self, branch_name: str, section: str, start_event, end_event) -> None:
        if self.enabled and start_event is not None and end_event is not None:
            self.cuda_intervals.append((branch_name, section, start_event, end_event))

    def add_wall_ms(self, branch_name: str, section: str, elapsed_ms: float) -> None:
        if self.enabled:
            self.wall_sections[(branch_name, section)].append(elapsed_ms)

    def add_chunk(self, branch_name: str, frames: int, tokens: int) -> None:
        if not self.enabled:
            return
        entry = self.chunks[branch_name]
        entry["count"] += 1
        entry["frames"] += frames
        entry["tokens"] += tokens

    @staticmethod
    def _summarize(values: list[float]) -> dict:
        if not values:
            return {"count": 0, "sum_ms": 0.0, "mean_ms": 0.0, "max_ms": 0.0}
        return {
            "count": len(values),
            "sum_ms": sum(values),
            "mean_ms": statistics.mean(values),
            "max_ms": max(values),
        }

    @staticmethod
    def _summarize_values(values: list[float]) -> dict:
        if not values:
            return {"count": 0, "sum": 0.0, "mean": 0.0, "max": 0.0}
        return {
            "count": len(values),
            "sum": sum(values),
            "mean": statistics.mean(values),
            "max": max(values),
        }

    def collect_append_global_events(self, branches: list["BranchState"]) -> None:
        if not self.enabled:
            return
        if os.getenv("REKV_PROFILE_INTERNAL_BLOCKS", "0") == "1":
            return
        for branch in branches:
            if branch.kv_cache is None:
                continue
            for layer_kv in branch.kv_cache:
                for start_event, end_event in getattr(layer_kv, "profile_append_global_events", []):
                    self.add_cuda_interval(
                        branch.config.name,
                        "append_global_gpu",
                        start_event,
                        end_event,
                    )

    def collect_internal_block_records(self, branches: list["BranchState"]) -> None:
        if not self.enabled:
            return
        for branch in branches:
            if branch.kv_cache is None:
                continue
            branch_name = branch.config.name
            for layer_idx, layer_kv in enumerate(branch.kv_cache):
                for record in getattr(layer_kv, "profile_internal_block_records", []):
                    self.add_internal_block_record(branch_name, layer_idx, record)

    def add_internal_block_record(self, branch_name: str, layer_idx: int, record: dict) -> None:
        meta = self.internal_blocks[branch_name]
        offload_blocks = int(record.get("offload_blocks", 0))
        offload_bytes = int(record.get("offload_bytes", 0))
        meta["count"] += 1
        meta["tokens"] += int(record.get("tokens", 0))
        meta["offload_blocks"] += offload_blocks
        meta["offload_bytes"] += offload_bytes
        if offload_blocks > 0:
            meta["nonzero_offload_count"] += 1
        if record.get("post_offload_wait_deferred", False):
            meta["deferred_post_offload_wait_count"] += 1

        self.internal_value_sections[(branch_name, "tokens_per_internal_block")].append(
            float(record.get("tokens", 0))
        )
        self.internal_value_sections[(branch_name, "offload_blocks_per_internal_block")].append(
            float(offload_blocks)
        )
        self.internal_value_sections[(branch_name, "offload_bytes_per_internal_block")].append(
            float(offload_bytes)
        )
        self.internal_value_sections[(branch_name, "layer_idx")].append(float(layer_idx))

        offload_wall_ms = record.get("offload_enqueue_wall_ms")
        if offload_wall_ms is not None:
            self.internal_wall_sections[(branch_name, "offload_enqueue_wall")].append(
                float(offload_wall_ms)
            )

        for section, events in record.get("cuda_events", {}).items():
            start_event, end_event = events
            self.internal_cuda_intervals.append((branch_name, section, start_event, end_event))
            if section == "offload_gpu":
                self.add_cuda_interval(branch_name, "append_global_gpu", start_event, end_event)

    def summarize(self) -> dict:
        if not self.enabled:
            return {}
        cuda_sync()
        cuda_sections = defaultdict(list)
        for branch_name, section, start_event, end_event in self.cuda_intervals:
            cuda_sections[(branch_name, section)].append(start_event.elapsed_time(end_event))
        internal_cuda_sections = defaultdict(list)
        for branch_name, section, start_event, end_event in self.internal_cuda_intervals:
            internal_cuda_sections[(branch_name, section)].append(start_event.elapsed_time(end_event))
        internal_branch_names = (
            set(self.internal_blocks.keys())
            | {branch for branch, _ in self.internal_wall_sections.keys()}
            | {branch for branch, _ in internal_cuda_sections.keys()}
            | {branch for branch, _ in self.internal_value_sections.keys()}
        )
        return {
            "cuda_sections": {
                branch_name: {
                    section: self._summarize(values)
                    for (candidate_branch, section), values in cuda_sections.items()
                    if candidate_branch == branch_name
                }
                for branch_name in sorted({branch for branch, _ in cuda_sections.keys()})
            },
            "wall_sections": {
                branch_name: {
                    section: self._summarize(values)
                    for (candidate_branch, section), values in self.wall_sections.items()
                    if candidate_branch == branch_name
                }
                for branch_name in sorted({branch for branch, _ in self.wall_sections.keys()})
            },
            "chunks": dict(self.chunks),
            "internal_blocks": {
                branch_name: {
                    **dict(self.internal_blocks[branch_name]),
                    "cuda_sections": {
                        section: self._summarize(values)
                        for (candidate_branch, section), values in internal_cuda_sections.items()
                        if candidate_branch == branch_name
                    },
                    "wall_sections": {
                        section: self._summarize(values)
                        for (candidate_branch, section), values in self.internal_wall_sections.items()
                        if candidate_branch == branch_name
                    },
                    "value_sections": {
                        section: self._summarize_values(values)
                        for (candidate_branch, section), values in self.internal_value_sections.items()
                        if candidate_branch == branch_name
                    },
                }
                for branch_name in sorted(internal_branch_names)
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shared-model prototype for interleaved fs112/fs224 ReKV video encoding. "
            "It loads model weights once, keeps branch-local processors/KV caches/offload streams, "
            "and intentionally skips retrieval and QA."
        )
    )
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--model-path", default="/mnt/models/qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect CUDA-event timing for vision/LM/offload and wall-clock scheduler waits.",
    )
    parser.add_argument(
        "--offload-granularity",
        choices=["block", "chunk"],
        default=os.getenv("REKV_OFFLOAD_GRANULARITY", "block"),
        help="block keeps current per-internal-block offload; chunk defers offload until each append call finishes.",
    )
    parser.add_argument(
        "--batched-offload-copy",
        action="store_true",
        default=os.getenv("REKV_BATCHED_OFFLOAD_COPY", "0") == "1",
        help="Copy all currently offloadable blocks in one batched CPU transfer per unit.",
    )
    parser.add_argument(
        "--batched-co-encoding",
        action="store_true",
        default=os.getenv("REKV_BATCHED_CO_ENCODING", "0") == "1",
        help=(
            "Experimental layer-wise packed co-encoding: pack fs112/fs224 tokens together "
            "for QKV/o_proj/MLP while keeping branch-local ReKV ContextManagers. "
            "Single-branch tail chunks fall back to normal encoding."
        ),
    )
    parser.add_argument(
        "--batched-vision-co-encoding",
        action="store_true",
        default=os.getenv("REKV_BATCHED_VISION_CO_ENCODING", "0") == "1",
        help=(
            "When --batched-co-encoding is enabled, concatenate fs112/fs224 "
            "pixel_values_videos and video_grid_thw so the Qwen vision encoder runs once "
            "for both branches."
        ),
    )

    parser.add_argument("--fs112-frame-size", type=int, default=112)
    parser.add_argument("--fs112-local-block-count", type=int, default=72)
    parser.add_argument("--fs112-retrieve-size", type=int, default=144)
    parser.add_argument("--fs112-retrieve-chunk-size", type=int, default=1)
    parser.add_argument("--fs112-encode-chunk-size", type=int, default=64)
    parser.add_argument(
        "--fs112-internal-block-size",
        type=int,
        default=0,
        help="ReKV attention microblock size in visual tokens. 0 uses fs112 n_frame_tokens.",
    )

    parser.add_argument("--fs224-frame-size", type=int, default=224)
    parser.add_argument("--fs224-local-block-count", type=int, default=18)
    parser.add_argument("--fs224-retrieve-size", type=int, default=36)
    parser.add_argument("--fs224-retrieve-chunk-size", type=int, default=1)
    parser.add_argument("--fs224-encode-chunk-size", type=int, default=32)
    parser.add_argument(
        "--fs224-internal-block-size",
        type=int,
        default=0,
        help="ReKV attention microblock size in visual tokens. 0 uses fs224 n_frame_tokens.",
    )

    parser.add_argument(
        "--max-pending-chunks-per-branch",
        type=int,
        default=1,
        help="Throttle unresolved offload events per branch. Use 0 for no explicit throttle.",
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
            internal_block_size=args.fs112_internal_block_size or None,
        )
    if name == "fs224":
        return BranchConfig(
            name=name,
            frame_size=args.fs224_frame_size,
            local_block_count=args.fs224_local_block_count,
            retrieve_size=args.fs224_retrieve_size,
            retrieve_chunk_size=args.fs224_retrieve_chunk_size,
            encode_chunk_size=args.fs224_encode_chunk_size,
            internal_block_size=args.fs224_internal_block_size or None,
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


def branch_internal_block_size(branch: BranchState) -> int:
    internal_block_size = branch.config.internal_block_size or branch.n_frame_tokens
    if internal_block_size < branch.n_frame_tokens:
        raise ValueError(
            f"{branch.config.name}: internal_block_size={internal_block_size} must be >= "
            f"n_frame_tokens={branch.n_frame_tokens}"
        )
    if internal_block_size % branch.n_frame_tokens != 0:
        raise ValueError(
            f"{branch.config.name}: internal_block_size={internal_block_size} must be a "
            f"multiple of n_frame_tokens={branch.n_frame_tokens}"
        )
    if internal_block_size > branch.n_local:
        raise ValueError(
            f"{branch.config.name}: internal_block_size={internal_block_size} must be <= "
            f"n_local={branch.n_local}"
        )
    return internal_block_size


def branch_attn_config(branch: BranchState) -> dict:
    return {
        "n_init": branch.init_prompt_ids.shape[1],
        "n_local": branch.n_local,
        "fattn": True,
        "block_size": branch.n_frame_tokens,
        "topk": branch.config.retrieve_size,
        "chunk_size": branch.config.retrieve_chunk_size,
        "max_cached_block": max(128, branch.config.retrieve_size),
        "exc_block_size": branch_internal_block_size(branch),
        "pin_memory": True,
    }


def make_dynamic_hf_attention_forward(runtime: DynamicBranchRuntime):
    def hf_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        assert not output_attentions
        active_name = runtime.active_name
        if active_name is None:
            raise RuntimeError("No active ReKV branch selected.")
        forward = runtime.attention_forwards[active_name]
        ret = forward(
            self,
            hidden_states,
            hidden_states,
            position_ids,
            use_cache,
            past_key_value,
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.head_dim,
            self.num_heads,
            self.num_key_value_heads,
        )
        if use_cache:
            o, pkv = ret
        else:
            o = ret
            pkv = None
        return o, None, pkv

    return hf_forward


def install_dynamic_branch_attention(model, branches_by_name: dict[str, BranchState]) -> None:
    runtime = DynamicBranchRuntime()
    runtime.attention_forwards = {
        name: rekv_attention_forward(**branch_attn_config(branch))
        for name, branch in branches_by_name.items()
    }
    decoder_root = model.model.language_model
    attention_cls = decoder_root.layers[0].self_attn.__class__
    dynamic_forward = make_dynamic_hf_attention_forward(runtime)

    def set_forward(module):
        if isinstance(module, attention_cls):
            module.forward = dynamic_forward.__get__(module, attention_cls)

    decoder_root.apply(set_forward)
    model._rekv_dynamic_branch_runtime = runtime


def apply_branch_state(model, branch: BranchState) -> None:
    runtime = getattr(model, "_rekv_dynamic_branch_runtime", None)
    if runtime is not None:
        runtime.active_name = branch.config.name
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


@torch.inference_mode()
def encode_init_prompt(model, branch: BranchState) -> None:
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


def clear_kv_profile_records(branches: list[BranchState]) -> None:
    for branch in branches:
        if branch.kv_cache is None:
            continue
        for layer_kv in branch.kv_cache:
            getattr(layer_kv, "profile_append_global_events", []).clear()
            getattr(layer_kv, "profile_internal_block_records", []).clear()
            if hasattr(layer_kv, "_profile_append_call_idx"):
                layer_kv._profile_append_call_idx = 0


def make_context_manager_for_branch(model, branch: BranchState) -> ContextManager:
    config = branch_attn_config(branch)
    return ContextManager(
        model.model.language_model.position_bias,
        config["n_init"],
        config["n_local"],
        config["block_size"],
        config["max_cached_block"],
        config["topk"],
        config["chunk_size"],
        config["exc_block_size"],
        config["fattn"],
        True,
        config["pin_memory"],
    )


def _video_grid_token_counts(branch: BranchState, video_grid_thw: torch.Tensor) -> tuple[int, int]:
    spatial_tokens = int(
        video_grid_thw[0, 1].item()
        * video_grid_thw[0, 2].item()
        // (branch.processor.video_processor.merge_size ** 2)
    )
    feature_tokens = int(video_grid_thw[0].prod().item() // (branch.processor.video_processor.merge_size ** 2))
    return spatial_tokens, feature_tokens


def _validate_video_grid(
    branch: BranchState,
    video_grid_thw: torch.Tensor,
    feature_tokens: int,
    chunk_size: int,
) -> None:
    spatial_tokens, _ = _video_grid_token_counts(branch, video_grid_thw)
    if spatial_tokens != branch.n_frame_tokens:
        raise AssertionError(
            f"{branch.config.name}: expected {branch.n_frame_tokens} tokens per temporal block, "
            f"got {spatial_tokens}; video_grid_thw={video_grid_thw.tolist()}"
        )
    if branch.n_local < feature_tokens:
        raise ChunkTooLarge(
            branch.config.name,
            branch.n_local,
            int(feature_tokens),
            int(chunk_size),
        )


def _validate_video_features(branch: BranchState, video_grid_thw: torch.Tensor, video_features: torch.Tensor) -> None:
    _, feature_tokens = _video_grid_token_counts(branch, video_grid_thw)
    if int(video_features.shape[1]) != feature_tokens:
        raise AssertionError(
            f"{branch.config.name}: expected {feature_tokens} total video tokens, "
            f"got {int(video_features.shape[1])}; video_grid_thw={video_grid_thw.tolist()}"
        )
    _validate_video_grid(
        branch,
        video_grid_thw,
        int(video_features.shape[1]),
        int(branch.current_chunk().shape[0]),
    )


def prepare_video_chunk_inputs(
    model,
    branch: BranchState,
    profile: ProfileCollector | None = None,
) -> PreparedVideoInputs:
    while True:
        chunk = branch.current_chunk()
        apply_branch_state(model, branch)
        try:
            with branch_cuda_context(branch):
                prepare_start_event = profile.cuda_event() if profile is not None else None
                if prepare_start_event is not None:
                    prepare_start_event.record(branch.compute_stream)
                prepare_start_wall = time.perf_counter()
                pixel_values_videos, video_grid_thw = model._prepare_video_inputs(chunk)
                prepare_end_wall = time.perf_counter()
                prepare_end_event = profile.cuda_event() if profile is not None else None
                if prepare_end_event is not None:
                    prepare_end_event.record(branch.compute_stream)

                _, feature_tokens = _video_grid_token_counts(branch, video_grid_thw)
                _validate_video_grid(
                    branch,
                    video_grid_thw,
                    feature_tokens,
                    int(chunk.shape[0]),
                )
                return PreparedVideoInputs(
                    branch=branch,
                    chunk=chunk,
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                    prepare_start_event=prepare_start_event,
                    prepare_end_event=prepare_end_event,
                    prepare_start_wall=prepare_start_wall,
                    prepare_end_wall=prepare_end_wall,
                )
        except ChunkTooLarge as exc:
            sync_branch(branch)
            branch.shrink_chunk_size(exc.chunk_size)


def prepare_video_chunk_features(
    model,
    branch: BranchState,
    profile: ProfileCollector | None = None,
) -> PreparedVideoChunk:
    prepared_inputs = prepare_video_chunk_inputs(model, branch, profile)
    branch = prepared_inputs.branch
    apply_branch_state(model, branch)
    with branch_cuda_context(branch):
        vision_start_event = profile.cuda_event() if profile is not None else None
        if vision_start_event is not None:
            vision_start_event.record(branch.compute_stream)
        video_features = model._get_video_features(
            prepared_inputs.pixel_values_videos,
            prepared_inputs.video_grid_thw,
        )
        vision_end_event = profile.cuda_event() if profile is not None else None
        if vision_end_event is not None:
            vision_end_event.record(branch.compute_stream)

    _validate_video_features(branch, prepared_inputs.video_grid_thw, video_features)
    return PreparedVideoChunk(
        branch=branch,
        chunk=prepared_inputs.chunk,
        video_features=video_features,
        prepare_start_event=prepared_inputs.prepare_start_event,
        prepare_end_event=prepared_inputs.prepare_end_event,
        vision_start_event=vision_start_event,
        vision_end_event=vision_end_event,
        prepare_start_wall=prepared_inputs.prepare_start_wall,
        prepare_end_wall=prepared_inputs.prepare_end_wall,
    )


def prepare_batched_video_chunk_features(
    model,
    branches: list[BranchState],
    profile: ProfileCollector | None = None,
) -> list[PreparedVideoChunk]:
    prepared_inputs = [
        prepare_video_chunk_inputs(model, branch, profile)
        for branch in branches
    ]
    compute_stream = branches[0].compute_stream
    for branch in branches[1:]:
        if branch.compute_stream is not compute_stream:
            raise RuntimeError("Batched vision co-encoding requires branches to share one compute stream.")

    with torch.cuda.stream(compute_stream):
        vision_start_event = profile.cuda_event() if profile is not None else None
        if vision_start_event is not None:
            vision_start_event.record(compute_stream)
        pixel_values_videos = torch.cat(
            [prepared.pixel_values_videos for prepared in prepared_inputs],
            dim=0,
        )
        video_grid_thw = torch.cat(
            [prepared.video_grid_thw for prepared in prepared_inputs],
            dim=0,
        )
        video_feature_list = model.get_video_features(
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        ).pooler_output
        vision_end_event = profile.cuda_event() if profile is not None else None
        if vision_end_event is not None:
            vision_end_event.record(compute_stream)

    if len(video_feature_list) != len(prepared_inputs):
        raise RuntimeError(
            f"Batched vision returned {len(video_feature_list)} feature groups "
            f"for {len(prepared_inputs)} branches."
        )

    prepared_chunks = []
    for prepared, video_features in zip(prepared_inputs, video_feature_list):
        video_features = video_features.unsqueeze(0)
        _validate_video_features(prepared.branch, prepared.video_grid_thw, video_features)
        prepared_chunks.append(
            PreparedVideoChunk(
                branch=prepared.branch,
                chunk=prepared.chunk,
                video_features=video_features,
                prepare_start_event=prepared.prepare_start_event,
                prepare_end_event=prepared.prepare_end_event,
                vision_start_event=vision_start_event,
                vision_end_event=vision_end_event,
                prepare_start_wall=prepared.prepare_start_wall,
                prepare_end_wall=prepared.prepare_end_wall,
            )
        )
    return prepared_chunks


def add_prepared_chunk_profile(
    prepared: PreparedVideoChunk,
    profile: ProfileCollector | None,
) -> None:
    if profile is None or not profile.enabled:
        return
    branch = prepared.branch
    profile.add_wall_ms(
        branch.config.name,
        "prepare_inputs_wall",
        (prepared.prepare_end_wall - prepared.prepare_start_wall) * 1000.0,
    )
    profile.add_cuda_interval(
        branch.config.name,
        "prepare_inputs_gpu",
        prepared.prepare_start_event,
        prepared.prepare_end_event,
    )
    profile.add_cuda_interval(
        branch.config.name,
        "vision_gpu",
        prepared.vision_start_event,
        prepared.vision_end_event,
    )
    profile.add_chunk(
        branch.config.name,
        int(prepared.chunk.shape[0]),
        int(prepared.video_features.shape[1]),
    )


@torch.inference_mode()
def encode_prepared_video_chunk(
    model,
    prepared: PreparedVideoChunk,
    profile: ProfileCollector | None = None,
) -> torch.cuda.Event:
    branch = prepared.branch
    apply_branch_state(model, branch)
    with branch_cuda_context(branch):
        lm_start_wall = time.perf_counter()
        output = model.language_model(
            inputs_embeds=prepared.video_features,
            past_key_values=branch.kv_cache,
            use_cache=True,
            return_dict=True,
        )
        lm_end_wall = time.perf_counter()
        lm_end_event = profile.cuda_event() if profile is not None else None
        if lm_end_event is not None:
            lm_end_event.record(branch.compute_stream)

        branch.kv_cache = output.past_key_values
        model.kv_cache = branch.kv_cache
        branch.advance_chunk(prepared.chunk)
        add_prepared_chunk_profile(prepared, profile)
        if profile is not None and profile.enabled:
            profile.add_wall_ms(
                branch.config.name,
                "language_model_enqueue_wall",
                (lm_end_wall - lm_start_wall) * 1000.0,
            )
            profile.add_cuda_interval(
                branch.config.name,
                "language_model_compute_gpu",
                prepared.vision_end_event,
                lm_end_event,
            )
        event = torch.cuda.Event()
        event.record(branch.offload_stream)
    return event


def run_branch_context_attention(
    model,
    branch: BranchState,
    layer_idx: int,
    hidden_states: torch.Tensor,
    h_q: torch.Tensor,
    h_k: torch.Tensor,
    h_v: torch.Tensor,
) -> torch.Tensor:
    if branch.kv_cache is None:
        branch.kv_cache = tuple(
            make_context_manager_for_branch(model, branch)
            for _ in model.model.language_model.layers
        )
    layer_cache = branch.kv_cache[layer_idx]
    previous_stream = kv_cache_manager.GLOBAL_STREAM
    kv_cache_manager.GLOBAL_STREAM = branch.offload_stream
    try:
        score = layer_cache.append(h_q, h_k, h_v, h_q, h_k, h_v)
    finally:
        kv_cache_manager.GLOBAL_STREAM = previous_stream
    _, num_heads, len_q, dim_head = score.shape
    score = score.view(1, num_heads, len_q, dim_head).permute(0, 2, 1, 3)
    return score.reshape(1, len_q, hidden_states.shape[-1])


@torch.inference_mode()
def run_layerwise_batched_language_model(
    model,
    prepared_chunks: list[PreparedVideoChunk],
) -> None:
    branches = [prepared.branch for prepared in prepared_chunks]
    token_spans = []
    offset = 0
    for prepared in prepared_chunks:
        if prepared.video_features.shape[0] != 1:
            raise RuntimeError(
                "Packed co-encoding expects one sample per branch, "
                f"got batch={prepared.video_features.shape[0]} for {prepared.branch.config.name}."
            )
        token_count = int(prepared.video_features.shape[1])
        token_spans.append((prepared, offset, offset + token_count))
        offset += token_count

    hidden_states = torch.cat([prepared.video_features for prepared in prepared_chunks], dim=1)
    decoder_root = model.model.language_model
    present_by_branch = {branch.config.name: [] for branch in branches}

    for layer_idx, decoder_layer in enumerate(decoder_root.layers):
        residual = hidden_states
        hidden_states = decoder_layer.input_layernorm(hidden_states)
        attn = decoder_layer.self_attn

        batch_size, len_q, _ = hidden_states.shape
        if batch_size != 1:
            raise RuntimeError(f"Packed co-encoding expects batch=1, got batch={batch_size}.")
        h_q = attn.q_proj(hidden_states)
        h_k = attn.k_proj(hidden_states)
        h_v = attn.v_proj(hidden_states)
        h_q = h_q.view(batch_size, len_q, attn.num_heads, attn.head_dim).permute(0, 2, 1, 3).contiguous()
        h_k = h_k.view(batch_size, len_q, attn.num_key_value_heads, attn.head_dim).permute(0, 2, 1, 3).contiguous()
        h_v = h_v.view(batch_size, len_q, attn.num_key_value_heads, attn.head_dim).permute(0, 2, 1, 3).contiguous()

        branch_outputs = []
        for prepared, token_st, token_ed in token_spans:
            branch = prepared.branch
            branch_output = run_branch_context_attention(
                model,
                branch,
                layer_idx,
                hidden_states[:, token_st:token_ed, :],
                h_q[:, :, token_st:token_ed, :],
                h_k[:, :, token_st:token_ed, :],
                h_v[:, :, token_st:token_ed, :],
            )
            present_by_branch[branch.config.name].append(branch.kv_cache[layer_idx])
            branch_outputs.append(branch_output)

        attn_output = torch.cat(branch_outputs, dim=1)
        hidden_states = residual + attn.o_proj(attn_output)

        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

    decoder_root.norm(hidden_states)
    for branch in branches:
        branch.kv_cache = tuple(present_by_branch[branch.config.name])


@torch.inference_mode()
def encode_batched_prepared_video_chunks(
    model,
    prepared_chunks: list[PreparedVideoChunk],
    profile: ProfileCollector | None = None,
) -> dict[str, torch.cuda.Event]:
    branches = [prepared.branch for prepared in prepared_chunks]
    compute_stream = branches[0].compute_stream
    for branch in branches[1:]:
        if branch.compute_stream is not compute_stream:
            raise RuntimeError("Batched co-encoding requires branches to share one compute stream.")

    with torch.cuda.stream(compute_stream):
        lm_start_event = profile.cuda_event() if profile is not None else None
        if lm_start_event is not None:
            lm_start_event.record(compute_stream)
        lm_start_wall = time.perf_counter()
        run_layerwise_batched_language_model(model, prepared_chunks)
        lm_end_wall = time.perf_counter()
        lm_end_event = profile.cuda_event() if profile is not None else None
        if lm_end_event is not None:
            lm_end_event.record(compute_stream)

        for prepared in prepared_chunks:
            prepared.branch.advance_chunk(prepared.chunk)
            add_prepared_chunk_profile(prepared, profile)
        if profile is not None and profile.enabled:
            profile.add_wall_ms(
                "batched_co_encoding",
                "language_model_enqueue_wall",
                (lm_end_wall - lm_start_wall) * 1000.0,
            )
            profile.add_cuda_interval(
                "batched_co_encoding",
                "language_model_compute_gpu",
                lm_start_event,
                lm_end_event,
            )

    events = {}
    for branch in branches:
        event = torch.cuda.Event()
        event.record(branch.offload_stream)
        events[branch.config.name] = event
    return events


@torch.inference_mode()
def encode_video_chunk(
    model,
    branch: BranchState,
    profile: ProfileCollector | None = None,
) -> torch.cuda.Event:
    while True:
        try:
            prepared = prepare_video_chunk_features(model, branch, profile)
            return encode_prepared_video_chunk(model, prepared, profile)
        except ChunkTooLarge as exc:
            sync_branch(branch)
            branch.shrink_chunk_size(exc.chunk_size)


def wait_oldest_if_needed(
    pending: list[torch.cuda.Event],
    max_pending: int,
    branch_name: str | None = None,
    profile: ProfileCollector | None = None,
    section: str = "scheduler_offload_wait",
) -> None:
    if max_pending <= 0:
        return
    while len(pending) >= max_pending:
        event = pending.pop(0)
        start = time.perf_counter()
        event.synchronize()
        if profile is not None and branch_name is not None:
            profile.add_wall_ms(branch_name, section, (time.perf_counter() - start) * 1000.0)


def run_sequential(
    model,
    branches_by_name: dict[str, BranchState],
    order: str,
    max_pending_chunks_per_branch: int,
    profile: ProfileCollector | None = None,
) -> float:
    branches = [branches_by_name["fs224"], branches_by_name["fs112"]]
    if order == "fs112-first":
        branches = list(reversed(branches))

    reset_branches(model, list(branches_by_name.values()))
    encode_init_prompts(model, branches)
    clear_kv_profile_records(branches)

    cuda_sync()
    start = time.perf_counter()
    for branch in branches:
        branch_pending = []
        while branch.has_next_chunk():
            wait_oldest_if_needed(
                branch_pending,
                max_pending_chunks_per_branch,
                branch.config.name,
                profile,
            )
            branch_pending.append(encode_video_chunk(model, branch, profile))
        for event in branch_pending:
            start_wait = time.perf_counter()
            event.synchronize()
            if profile is not None:
                profile.add_wall_ms(
                    branch.config.name,
                    "final_offload_wait",
                    (time.perf_counter() - start_wait) * 1000.0,
                )
        sync_branch(branch)
    cuda_sync()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if profile is not None:
        profile.collect_internal_block_records(branches)
        profile.collect_append_global_events(branches)
    return elapsed_ms


def run_interleaved(
    model,
    branches_by_name: dict[str, BranchState],
    max_pending_chunks_per_branch: int,
    profile: ProfileCollector | None = None,
) -> float:
    branches = [branches_by_name["fs112"], branches_by_name["fs224"]]
    reset_branches(model, list(branches_by_name.values()))
    encode_init_prompts(model, branches)
    clear_kv_profile_records(branches)

    pending = {branch.config.name: [] for branch in branches}
    cuda_sync()
    start = time.perf_counter()
    while any(branch.has_next_chunk() for branch in branches):
        for branch in branches:
            if not branch.has_next_chunk():
                continue
            branch_pending = pending[branch.config.name]
            wait_oldest_if_needed(
                branch_pending,
                max_pending_chunks_per_branch,
                branch.config.name,
                profile,
            )
            branch_pending.append(encode_video_chunk(model, branch, profile))

    for branch in branches:
        branch_events = pending[branch.config.name]
        for event in branch_events:
            start_wait = time.perf_counter()
            event.synchronize()
            if profile is not None:
                profile.add_wall_ms(
                    branch.config.name,
                    "final_offload_wait",
                    (time.perf_counter() - start_wait) * 1000.0,
                )
    for branch in branches:
        sync_branch(branch)
    cuda_sync()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if profile is not None:
        profile.collect_internal_block_records(branches)
        profile.collect_append_global_events(branches)
    return elapsed_ms


def run_batched_co_encoding(
    model,
    branches_by_name: dict[str, BranchState],
    max_pending_chunks_per_branch: int,
    batched_vision_co_encoding: bool = False,
    profile: ProfileCollector | None = None,
) -> float:
    branches = [branches_by_name["fs112"], branches_by_name["fs224"]]
    reset_branches(model, list(branches_by_name.values()))
    encode_init_prompts(model, branches)
    clear_kv_profile_records(branches)

    pending = {branch.config.name: [] for branch in branches}
    cuda_sync()
    start = time.perf_counter()
    while any(branch.has_next_chunk() for branch in branches):
        ready_branches = [branch for branch in branches if branch.has_next_chunk()]
        for branch in ready_branches:
            wait_oldest_if_needed(
                pending[branch.config.name],
                max_pending_chunks_per_branch,
                branch.config.name,
                profile,
            )

        if len(ready_branches) == 2:
            if batched_vision_co_encoding:
                prepared_chunks = prepare_batched_video_chunk_features(
                    model,
                    ready_branches,
                    profile,
                )
            else:
                prepared_chunks = [
                    prepare_video_chunk_features(model, branch, profile)
                    for branch in ready_branches
                ]
            hidden_sizes = {int(prepared.video_features.shape[2]) for prepared in prepared_chunks}
            if len(hidden_sizes) == 1:
                events = encode_batched_prepared_video_chunks(model, prepared_chunks, profile)
                for branch_name, event in events.items():
                    pending[branch_name].append(event)
            else:
                for prepared in prepared_chunks:
                    event = encode_prepared_video_chunk(model, prepared, profile)
                    pending[prepared.branch.config.name].append(event)
        else:
            branch = ready_branches[0]
            pending[branch.config.name].append(encode_video_chunk(model, branch, profile))

    for branch in branches:
        branch_events = pending[branch.config.name]
        for event in branch_events:
            start_wait = time.perf_counter()
            event.synchronize()
            if profile is not None:
                profile.add_wall_ms(
                    branch.config.name,
                    "final_offload_wait",
                    (time.perf_counter() - start_wait) * 1000.0,
                )
    for branch in branches:
        sync_branch(branch)
    cuda_sync()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if profile is not None:
        profile.collect_internal_block_records(branches)
        profile.collect_append_global_events(branches)
    return elapsed_ms


def build_branches(args: argparse.Namespace, model, source_video: np.ndarray) -> dict[str, BranchState]:
    device = next(model.parameters()).device
    branches = {}
    shared_compute_stream = torch.cuda.Stream()
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
            compute_stream=shared_compute_stream,
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
    os.environ.setdefault("REKV_DEFER_OFFLOAD_WAIT", "1")
    os.environ["REKV_OFFLOAD_GRANULARITY"] = args.offload_granularity
    os.environ["REKV_BATCHED_OFFLOAD_COPY"] = "1" if args.batched_offload_copy else "0"
    os.environ["REKV_BATCHED_CO_ENCODING"] = "1" if args.batched_co_encoding else "0"
    os.environ["REKV_BATCHED_VISION_CO_ENCODING"] = "1" if args.batched_vision_co_encoding else "0"
    if args.profile:
        os.environ["REKV_PROFILE_INTERNAL_BLOCKS"] = "1"
        os.environ.setdefault("REKV_PROFILE_APPEND_GLOBAL", "0")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this prototype.")
    if not args.video_path.exists():
        raise FileNotFoundError(args.video_path)

    source_video, fps, frame_step = load_sampled_video(args.video_path, args.sample_fps)
    model = load_shared_model(args)
    branches_by_name = build_branches(args, model, source_video)
    install_dynamic_branch_attention(model, branches_by_name)

    for _ in range(args.warmup):
        run_sequential(
            model,
            branches_by_name,
            args.sequential_order,
            args.max_pending_chunks_per_branch,
        )
        run_interleaved(model, branches_by_name, args.max_pending_chunks_per_branch)
        if args.batched_co_encoding:
            run_batched_co_encoding(
                model,
                branches_by_name,
                args.max_pending_chunks_per_branch,
                batched_vision_co_encoding=args.batched_vision_co_encoding,
            )

    sequential = []
    sequential_profiles = []
    for _ in range(args.repeats):
        profile = ProfileCollector(args.profile)
        sequential.append(run_sequential(
            model,
            branches_by_name,
            args.sequential_order,
            args.max_pending_chunks_per_branch,
            profile,
        ))
        if args.profile:
            sequential_profiles.append(profile.summarize())

    interleaved = []
    interleaved_profiles = []
    for _ in range(args.repeats):
        profile = ProfileCollector(args.profile)
        interleaved.append(run_interleaved(
            model,
            branches_by_name,
            args.max_pending_chunks_per_branch,
            profile,
        ))
        if args.profile:
            interleaved_profiles.append(profile.summarize())

    batched_co_encoding = []
    batched_co_encoding_profiles = []
    if args.batched_co_encoding:
        for _ in range(args.repeats):
            profile = ProfileCollector(args.profile)
            batched_co_encoding.append(run_batched_co_encoding(
                model,
                branches_by_name,
                args.max_pending_chunks_per_branch,
                batched_vision_co_encoding=args.batched_vision_co_encoding,
                profile=profile,
            ))
            if args.profile:
                batched_co_encoding_profiles.append(profile.summarize())

    sequential_mean = statistics.mean(sequential)
    interleaved_mean = statistics.mean(interleaved)
    batched_co_encoding_mean = (
        statistics.mean(batched_co_encoding) if batched_co_encoding else None
    )
    summary = {
        "video_path": str(args.video_path),
        "shared_model": True,
        "retrieval_or_qa_executed": False,
        "rekv_defer_offload_wait": os.getenv("REKV_DEFER_OFFLOAD_WAIT", "0"),
        "rekv_profile_internal_blocks": os.getenv("REKV_PROFILE_INTERNAL_BLOCKS", "0"),
        "rekv_profile_append_global": os.getenv("REKV_PROFILE_APPEND_GLOBAL", "0"),
        "rekv_batched_retrieval_io": os.getenv("REKV_BATCHED_RETRIEVAL_IO", "0"),
        "rekv_offload_granularity": os.getenv("REKV_OFFLOAD_GRANULARITY", "block"),
        "rekv_batched_offload_copy": os.getenv("REKV_BATCHED_OFFLOAD_COPY", "0"),
        "rekv_batched_co_encoding": os.getenv("REKV_BATCHED_CO_ENCODING", "0"),
        "rekv_batched_vision_co_encoding": os.getenv("REKV_BATCHED_VISION_CO_ENCODING", "0"),
        "batched_co_encoding_mode": "packed_variable_length" if args.batched_co_encoding else None,
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
                "resolved_internal_block_size": int(branch_internal_block_size(branch)),
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
    if args.batched_co_encoding:
        summary["batched_co_encoding"] = stats_ms(batched_co_encoding)
        summary["speedup_batched_co_encoding_vs_sequential"] = (
            sequential_mean / batched_co_encoding_mean if batched_co_encoding_mean else 0.0
        )
        summary["batched_co_encoding_reduction_ratio"] = (
            1.0 - batched_co_encoding_mean / sequential_mean if sequential_mean else 0.0
        )
    if args.profile:
        summary["profile"] = {
            "sequential": sequential_profiles,
            "interleaved": interleaved_profiles,
        }
        if args.batched_co_encoding:
            summary["profile"]["batched_co_encoding"] = batched_co_encoding_profiles

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.json:
        print()
        print(
            "Interpretation: this uses one shared model instance and branch-local "
            "processors/KV caches/CUDA streams. It measures video encoding only."
        )


if __name__ == "__main__":
    main()
