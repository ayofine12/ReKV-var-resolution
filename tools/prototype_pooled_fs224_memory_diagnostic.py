#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

TOOLS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import prototype_interleaved_video_encoding as proto


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic for replacing native fs112 memory with spatially pooled fs224 "
            "representations. It compares native fs112 vision/layer K/V tensors against "
            "pooled fs224 tensors on the same sampled prefix."
        )
    )
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--model-path", default="/mnt/models/qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--max-layers", type=int, default=0, help="0 means all decoder layers.")
    parser.add_argument("--json", action="store_true")

    parser.add_argument("--fs112-frame-size", type=int, default=112)
    parser.add_argument("--fs112-local-block-count", type=int, default=72)
    parser.add_argument("--fs112-retrieve-size", type=int, default=144)
    parser.add_argument("--fs112-retrieve-chunk-size", type=int, default=1)
    parser.add_argument("--fs112-encode-chunk-size", type=int, default=16)
    parser.add_argument("--fs112-internal-block-size", type=int, default=128)

    parser.add_argument("--fs224-frame-size", type=int, default=224)
    parser.add_argument("--fs224-local-block-count", type=int, default=18)
    parser.add_argument("--fs224-retrieve-size", type=int, default=36)
    parser.add_argument("--fs224-retrieve-chunk-size", type=int, default=1)
    parser.add_argument("--fs224-encode-chunk-size", type=int, default=16)
    parser.add_argument("--fs224-internal-block-size", type=int, default=128)
    return parser.parse_args()


def make_proto_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_path=args.model_path,
        video_path=args.video_path,
        sample_fps=args.sample_fps,
        fs112_frame_size=args.fs112_frame_size,
        fs112_local_block_count=args.fs112_local_block_count,
        fs112_retrieve_size=args.fs112_retrieve_size,
        fs112_retrieve_chunk_size=args.fs112_retrieve_chunk_size,
        fs112_encode_chunk_size=args.fs112_encode_chunk_size,
        fs112_internal_block_size=args.fs112_internal_block_size,
        fs224_frame_size=args.fs224_frame_size,
        fs224_local_block_count=args.fs224_local_block_count,
        fs224_retrieve_size=args.fs224_retrieve_size,
        fs224_retrieve_chunk_size=args.fs224_retrieve_chunk_size,
        fs224_encode_chunk_size=args.fs224_encode_chunk_size,
        fs224_internal_block_size=args.fs224_internal_block_size,
    )


def load_sampled_video_prefix(video_path: Path, sample_fps: float, frames: int):
    vr = VideoReader(str(video_path), ctx=cpu(0))
    fps = round(vr.get_avg_fps())
    frame_step = max(1, int(fps / sample_fps))
    frame_indices = [idx for idx in range(0, len(vr), frame_step)][:frames]
    if not frame_indices:
        raise ValueError(f"No frames selected from {video_path}")
    return vr.get_batch(frame_indices).asnumpy(), fps, frame_step, frame_indices


def merged_grid(branch, video_grid_thw: torch.Tensor) -> tuple[int, int, int]:
    merge_size = int(branch.processor.video_processor.merge_size)
    t = int(video_grid_thw[0, 0].item())
    h = int(video_grid_thw[0, 1].item())
    w = int(video_grid_thw[0, 2].item())
    if h % merge_size or w % merge_size:
        raise ValueError(f"{branch.config.name}: grid is not divisible by merge_size={merge_size}")
    return t, h // merge_size, w // merge_size


def pool_hidden_spatial(
    hidden_states: torch.Tensor,
    src_grid: tuple[int, int, int],
    dst_grid: tuple[int, int, int],
) -> torch.Tensor:
    batch_size, length, dim = hidden_states.shape
    src_t, src_h, src_w = src_grid
    dst_t, dst_h, dst_w = dst_grid
    if src_t != dst_t:
        raise ValueError(f"Temporal grids differ: src_t={src_t}, dst_t={dst_t}")
    if src_h % dst_h or src_w % dst_w:
        raise ValueError(f"Cannot integer-pool {src_grid} to {dst_grid}")
    if length != src_t * src_h * src_w:
        raise ValueError(f"Expected {src_t * src_h * src_w} tokens, got {length}")
    ratio_h = src_h // dst_h
    ratio_w = src_w // dst_w
    x = hidden_states.reshape(batch_size, src_t, dst_h, ratio_h, dst_w, ratio_w, dim)
    return x.mean(dim=(3, 5)).reshape(batch_size, dst_t * dst_h * dst_w, dim)


def pool_kv_spatial(
    kv_states: torch.Tensor,
    src_grid: tuple[int, int, int],
    dst_grid: tuple[int, int, int],
) -> torch.Tensor:
    batch_size, num_heads, length, dim = kv_states.shape
    src_t, src_h, src_w = src_grid
    dst_t, dst_h, dst_w = dst_grid
    if src_t != dst_t:
        raise ValueError(f"Temporal grids differ: src_t={src_t}, dst_t={dst_t}")
    if src_h % dst_h or src_w % dst_w:
        raise ValueError(f"Cannot integer-pool {src_grid} to {dst_grid}")
    if length != src_t * src_h * src_w:
        raise ValueError(f"Expected {src_t * src_h * src_w} tokens, got {length}")
    ratio_h = src_h // dst_h
    ratio_w = src_w // dst_w
    x = kv_states.reshape(
        batch_size,
        num_heads,
        src_t,
        dst_h,
        ratio_h,
        dst_w,
        ratio_w,
        dim,
    )
    return x.mean(dim=(4, 6)).reshape(batch_size, num_heads, dst_t * dst_h * dst_w, dim)


def tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: reference={reference.shape}, candidate={candidate.shape}")
    ref = reference.float()
    cand = candidate.float()
    ref_vectors = ref.reshape(-1, ref.shape[-1])
    cand_vectors = cand.reshape(-1, cand.shape[-1])
    cosine = F.cosine_similarity(ref_vectors, cand_vectors, dim=-1, eps=1e-6)
    diff = cand - ref
    ref_norm = torch.linalg.vector_norm(ref)
    rel_l2 = torch.linalg.vector_norm(diff) / (ref_norm + 1e-6)
    return {
        "shape": list(reference.shape),
        "vector_count": int(cosine.numel()),
        "cosine_mean": float(cosine.mean().item()),
        "cosine_min": float(cosine.min().item()),
        "cosine_p05": float(torch.quantile(cosine, 0.05).item()),
        "mse": float((diff * diff).mean().item()),
        "relative_l2": float(rel_l2.item()),
    }


def mean_metric(layer_metrics: list[dict], metric_name: str, tensor_name: str) -> float:
    values = [layer[tensor_name][metric_name] for layer in layer_metrics if tensor_name in layer]
    return float(sum(values) / len(values)) if values else 0.0


def summarize_layers(layer_metrics: list[dict]) -> dict:
    summary = {"layer_count": len(layer_metrics)}
    for tensor_name in ("hidden_in", "key", "value", "hidden_out"):
        summary[tensor_name] = {
            "cosine_mean_avg": mean_metric(layer_metrics, "cosine_mean", tensor_name),
            "cosine_p05_avg": mean_metric(layer_metrics, "cosine_p05", tensor_name),
            "relative_l2_avg": mean_metric(layer_metrics, "relative_l2", tensor_name),
        }
    return summary


@torch.inference_mode()
def prepare_video_features_with_grid(model, branch):
    prepared_inputs = proto.prepare_video_chunk_inputs(model, branch, None)
    proto.apply_branch_state(model, branch)
    with proto.branch_cuda_context(branch):
        video_features = model._get_video_features(
            prepared_inputs.pixel_values_videos,
            prepared_inputs.video_grid_thw,
        )
    proto.sync_branch(branch)
    proto._validate_video_features(branch, prepared_inputs.video_grid_thw, video_features)
    return prepared_inputs, video_features


@torch.inference_mode()
def collect_branch_layer_records(
    model,
    branch,
    video_features: torch.Tensor,
    max_layers: int,
) -> list[dict]:
    records = []
    proto.apply_branch_state(model, branch)
    with proto.branch_cuda_context(branch):
        hidden_states = video_features
        decoder_root = model.model.language_model
        layers = decoder_root.layers
        if max_layers > 0:
            layers = layers[:max_layers]
        for layer_idx, decoder_layer in enumerate(layers):
            record = {
                "layer": layer_idx,
                "hidden_in": hidden_states.detach().cpu(),
            }

            residual = hidden_states
            normed = decoder_layer.input_layernorm(hidden_states)
            attn = decoder_layer.self_attn

            batch_size, length, _ = normed.shape
            h_q = attn.q_proj(normed)
            h_k = attn.k_proj(normed)
            h_v = attn.v_proj(normed)
            h_q = h_q.view(batch_size, length, attn.num_heads, attn.head_dim)
            h_q = h_q.permute(0, 2, 1, 3).contiguous()
            h_k = h_k.view(batch_size, length, attn.num_key_value_heads, attn.head_dim)
            h_k = h_k.permute(0, 2, 1, 3).contiguous()
            h_v = h_v.view(batch_size, length, attn.num_key_value_heads, attn.head_dim)
            h_v = h_v.permute(0, 2, 1, 3).contiguous()

            record["query"] = h_q.detach().cpu()
            record["key"] = h_k.detach().cpu()
            record["value"] = h_v.detach().cpu()

            attn_output = proto.run_branch_context_attention(
                model,
                branch,
                layer_idx,
                normed,
                h_q,
                h_k,
                h_v,
            )
            hidden_states = residual + attn.o_proj(attn_output)

            residual = hidden_states
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            record["hidden_out"] = hidden_states.detach().cpu()
            records.append(record)

    proto.sync_branch(branch)
    return records


def compare_layer_records(
    fs112_records: list[dict],
    fs224_records: list[dict],
    fs112_grid: tuple[int, int, int],
    fs224_grid: tuple[int, int, int],
) -> list[dict]:
    if len(fs112_records) != len(fs224_records):
        raise ValueError(f"Layer count mismatch: {len(fs112_records)} vs {len(fs224_records)}")
    layer_metrics = []
    for low, high in zip(fs112_records, fs224_records):
        pooled_hidden_in = pool_hidden_spatial(high["hidden_in"], fs224_grid, fs112_grid)
        pooled_hidden_out = pool_hidden_spatial(high["hidden_out"], fs224_grid, fs112_grid)
        pooled_key = pool_kv_spatial(high["key"], fs224_grid, fs112_grid)
        pooled_value = pool_kv_spatial(high["value"], fs224_grid, fs112_grid)
        layer_metrics.append(
            {
                "layer": int(low["layer"]),
                "hidden_in": tensor_metrics(low["hidden_in"], pooled_hidden_in),
                "key": tensor_metrics(low["key"], pooled_key),
                "value": tensor_metrics(low["value"], pooled_value),
                "hidden_out": tensor_metrics(low["hidden_out"], pooled_hidden_out),
            }
        )
    return layer_metrics


def main() -> None:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")

    os.environ.setdefault("REKV_DEFER_OFFLOAD_WAIT", "1")
    os.environ.setdefault("REKV_OFFLOAD_GRANULARITY", "chunk")
    os.environ.setdefault("REKV_BATCHED_OFFLOAD_COPY", "1")
    os.environ.setdefault("REKV_PROFILE_INTERNAL_BLOCKS", "0")
    os.environ.setdefault("REKV_PROFILE_APPEND_GLOBAL", "0")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this diagnostic.")
    if not args.video_path.exists():
        raise FileNotFoundError(args.video_path)

    proto_args = make_proto_args(args)
    source_video, fps, frame_step, frame_indices = load_sampled_video_prefix(
        args.video_path,
        args.sample_fps,
        args.frames,
    )
    model = proto.load_shared_model(proto_args)
    branches_by_name = proto.build_branches(proto_args, model, source_video)
    proto.install_dynamic_branch_attention(model, branches_by_name)
    fs112 = branches_by_name["fs112"]
    fs224 = branches_by_name["fs224"]

    fs112_inputs, fs112_features = prepare_video_features_with_grid(model, fs112)
    fs224_inputs, fs224_features = prepare_video_features_with_grid(model, fs224)
    fs112_grid = merged_grid(fs112, fs112_inputs.video_grid_thw)
    fs224_grid = merged_grid(fs224, fs224_inputs.video_grid_thw)
    pooled_fs224_features = pool_hidden_spatial(fs224_features.cpu(), fs224_grid, fs112_grid)
    vision_metrics = tensor_metrics(fs112_features.cpu(), pooled_fs224_features)

    proto.reset_branches(model, [fs112, fs224])
    proto.encode_init_prompts(model, [fs112, fs224])
    fs112_records = collect_branch_layer_records(model, fs112, fs112_features, args.max_layers)
    fs224_records = collect_branch_layer_records(model, fs224, fs224_features, args.max_layers)
    layer_metrics = compare_layer_records(fs112_records, fs224_records, fs112_grid, fs224_grid)

    summary = {
        "video_path": str(args.video_path),
        "sample_fps": args.sample_fps,
        "original_fps_rounded": fps,
        "sampling_step": frame_step,
        "sampled_frame_indices": frame_indices,
        "frames": int(source_video.shape[0]),
        "fs112": {
            "frame_size": fs112.config.frame_size,
            "n_frame_tokens": int(fs112.n_frame_tokens),
            "n_local": int(fs112.n_local),
            "local_block_count": fs112.config.local_block_count,
            "internal_block_size": int(proto.branch_internal_block_size(fs112)),
            "grid_t_h_w_after_merge": list(fs112_grid),
            "feature_shape": list(fs112_features.shape),
        },
        "fs224": {
            "frame_size": fs224.config.frame_size,
            "n_frame_tokens": int(fs224.n_frame_tokens),
            "n_local": int(fs224.n_local),
            "local_block_count": fs224.config.local_block_count,
            "internal_block_size": int(proto.branch_internal_block_size(fs224)),
            "grid_t_h_w_after_merge": list(fs224_grid),
            "feature_shape": list(fs224_features.shape),
        },
        "pooling": {
            "type": "spatial_mean",
            "source": "fs224",
            "target": "fs112",
        },
        "vision_feature_metrics": vision_metrics,
        "layer_summary": summarize_layers(layer_metrics),
        "layer_metrics": layer_metrics,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.json:
        print()
        print(
            "Interpretation: low cosine or high relative_l2 for layer key/value means "
            "pooled fs224 memory is not a drop-in replacement for native fs112 memory."
        )


if __name__ == "__main__":
    main()
