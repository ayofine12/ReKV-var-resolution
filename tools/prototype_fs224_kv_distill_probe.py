#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

TOOLS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import prototype_interleaved_video_encoding as proto
import prototype_pooled_fs224_memory_diagnostic as diag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Feasibility probe for replacing the fs224 decoder pass with a learned "
            "KV generator. It builds teacher fs224 K/V tensors, then trains a tiny "
            "adapter from fs224 vision features plus fs112 context to those K/V tensors."
        )
    )
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--model-path", default="/mnt/models/qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 13, 27],
        help="Decoder layers to probe. Default samples early/middle/late layers.",
    )
    parser.add_argument(
        "--context-tensor",
        choices=["hidden_in", "hidden_out"],
        default="hidden_out",
        help="Which fs112 layer tensor conditions the fs224 K/V adapter.",
    )
    parser.add_argument("--adapter", choices=["linear", "mlp"], default="mlp")
    parser.add_argument(
        "--objective",
        choices=["raw-kv", "attention-output"],
        default="raw-kv",
        help=(
            "raw-kv directly matches teacher K/V. attention-output matches the "
            "causal attention result read by teacher fs224 queries."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-ratio", type=float, default=1.0)
    parser.add_argument("--key-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--direct-key-weight", type=float, default=0.1)
    parser.add_argument("--direct-value-weight", type=float, default=0.0)
    parser.add_argument("--cosine-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
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


class KVAdapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, adapter: str, hidden_dim: int):
        super().__init__()
        if adapter == "linear":
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def upsample_hidden_spatial(
    hidden_states: torch.Tensor,
    src_grid: tuple[int, int, int],
    dst_grid: tuple[int, int, int],
) -> torch.Tensor:
    batch_size, length, dim = hidden_states.shape
    src_t, src_h, src_w = src_grid
    dst_t, dst_h, dst_w = dst_grid
    if src_t != dst_t:
        raise ValueError(f"Temporal grids differ: src_t={src_t}, dst_t={dst_t}")
    if dst_h % src_h or dst_w % src_w:
        raise ValueError(f"Cannot integer-upsample {src_grid} to {dst_grid}")
    if length != src_t * src_h * src_w:
        raise ValueError(f"Expected {src_t * src_h * src_w} tokens, got {length}")
    ratio_h = dst_h // src_h
    ratio_w = dst_w // src_w
    x = hidden_states.reshape(batch_size, src_t, src_h, src_w, dim)
    x = x.repeat_interleave(ratio_h, dim=2).repeat_interleave(ratio_w, dim=3)
    return x.reshape(batch_size, dst_t * dst_h * dst_w, dim)


def flatten_kv(key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if key.shape != value.shape:
        raise ValueError(f"K/V shape mismatch: {key.shape} vs {value.shape}")
    _, num_heads, num_tokens, head_dim = key.shape
    key_flat = key.permute(0, 2, 1, 3).reshape(num_tokens, num_heads * head_dim)
    value_flat = value.permute(0, 2, 1, 3).reshape(num_tokens, num_heads * head_dim)
    return torch.cat([key_flat, value_flat], dim=-1)


def unflatten_kv(flat: torch.Tensor, template_key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, num_heads, num_tokens, head_dim = template_key.shape
    per_tensor = num_heads * head_dim
    key = flat[:, :per_tensor].reshape(1, num_tokens, num_heads, head_dim)
    value = flat[:, per_tensor:].reshape(1, num_tokens, num_heads, head_dim)
    key = key.permute(0, 2, 1, 3).contiguous()
    value = value.permute(0, 2, 1, 3).contiguous()
    return key, value


def repeat_kv_for_query_heads(kv_states: torch.Tensor, query_heads: int) -> torch.Tensor:
    kv_heads = kv_states.shape[1]
    if kv_heads == query_heads:
        return kv_states
    if query_heads % kv_heads:
        raise ValueError(f"Cannot expand {kv_heads} KV heads to {query_heads} query heads")
    return kv_states.repeat_interleave(query_heads // kv_heads, dim=1)


def causal_attention_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    query = query.float()
    key = key.float()
    value = value.float()
    key = repeat_kv_for_query_heads(key, query.shape[1])
    value = repeat_kv_for_query_heads(value, query.shape[1])
    scores = torch.matmul(query, key.transpose(-2, -1)) * (query.shape[-1] ** -0.5)
    q_len, k_len = query.shape[-2], key.shape[-2]
    if q_len == k_len:
        causal_mask = torch.ones((q_len, k_len), dtype=torch.bool, device=query.device).tril()
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    return torch.matmul(probs, value)


def vector_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cosine_weight: float,
    mse_weight: float,
) -> torch.Tensor:
    pred_vectors = pred.reshape(-1, pred.shape[-1])
    target_vectors = target.reshape(-1, target.shape[-1])
    cosine = 1.0 - F.cosine_similarity(pred_vectors, target_vectors, dim=-1, eps=1e-6).mean()
    mse = F.mse_loss(pred, target)
    return cosine_weight * cosine + mse_weight * mse


def kv_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    key_loss_weight: float,
    value_loss_weight: float,
    cosine_weight: float,
    mse_weight: float,
) -> torch.Tensor:
    pred_pairs = pred.reshape(pred.shape[0], 2, -1)
    target_pairs = target.reshape(target.shape[0], 2, -1)
    pair_weights = pred_pairs.new_tensor([key_loss_weight, value_loss_weight])
    pair_weights = pair_weights / pair_weights.sum().clamp_min(1e-6)
    cosine_per_pair = 1.0 - F.cosine_similarity(pred_pairs, target_pairs, dim=-1, eps=1e-6)
    mse_per_pair = ((pred_pairs - target_pairs) ** 2).mean(dim=-1)
    cosine = (cosine_per_pair * pair_weights).sum(dim=-1).mean()
    mse = (mse_per_pair * pair_weights).sum(dim=-1).mean()
    return cosine_weight * cosine + mse_weight * mse


def attention_output_loss(
    pred_flat: torch.Tensor,
    target_flat: torch.Tensor,
    query: torch.Tensor,
    key_template: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    pred_key, pred_value = unflatten_kv(pred_flat, key_template)
    target_key, target_value = unflatten_kv(target_flat, key_template)
    pred_out = causal_attention_output(query, pred_key, pred_value)
    target_out = causal_attention_output(query, target_key, target_value)
    loss = vector_loss(pred_out, target_out, args.cosine_weight, args.mse_weight)
    if args.direct_key_weight:
        loss = loss + args.direct_key_weight * vector_loss(
            pred_key,
            target_key,
            args.cosine_weight,
            args.mse_weight,
        )
    if args.direct_value_weight:
        loss = loss + args.direct_value_weight * vector_loss(
            pred_value,
            target_value,
            args.cosine_weight,
            args.mse_weight,
        )
    return loss


def adapter_metrics(
    adapter: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    key_template: torch.Tensor,
    query_template: torch.Tensor,
    indices: torch.Tensor,
) -> dict:
    if indices.numel() == 0:
        return {}
    adapter.eval()
    with torch.no_grad():
        pred = adapter(x[indices]).detach().cpu()
    target = y[indices].detach().cpu()
    pred_key, pred_value = unflatten_kv(pred, key_template[:, :, indices.cpu(), :])
    target_key, target_value = unflatten_kv(target, key_template[:, :, indices.cpu(), :])
    query = query_template[:, :, indices.cpu(), :]
    pred_attention = causal_attention_output(query, pred_key, pred_value)
    target_attention = causal_attention_output(query, target_key, target_value)
    return {
        "count": int(indices.numel()),
        "key": diag.tensor_metrics(target_key, pred_key),
        "value": diag.tensor_metrics(target_value, pred_value),
        "attention_output": diag.tensor_metrics(target_attention, pred_attention),
    }


def make_layer_dataset(
    fs112_record: dict,
    fs224_record: dict,
    fs224_features: torch.Tensor,
    fs112_grid: tuple[int, int, int],
    fs224_grid: tuple[int, int, int],
    context_tensor: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    fs112_context = upsample_hidden_spatial(
        fs112_record[context_tensor],
        fs112_grid,
        fs224_grid,
    )
    x = torch.cat([fs224_features.cpu(), fs112_context], dim=-1).reshape(
        fs224_features.shape[1],
        -1,
    )
    y = flatten_kv(fs224_record["key"], fs224_record["value"])
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Input/target token count mismatch: {x.shape[0]} vs {y.shape[0]}")
    return x.float(), y.float()


def train_one_layer(
    args: argparse.Namespace,
    layer_idx: int,
    x_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    key_template: torch.Tensor,
    query_template: torch.Tensor,
) -> dict:
    device = torch.device("cuda")
    num_tokens = x_cpu.shape[0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + layer_idx)
    perm = torch.randperm(num_tokens, generator=generator)
    train_count = int(round(num_tokens * args.train_ratio))
    train_count = min(max(train_count, 1), num_tokens)
    train_idx = perm[:train_count].sort().values.to(device)
    val_idx = perm[train_count:].sort().values.to(device)

    x = x_cpu.to(device)
    y = y_cpu.to(device)
    key_template_device = key_template.to(device)
    query_template_device = query_template.to(device)
    adapter = KVAdapter(x.shape[-1], y.shape[-1], args.adapter, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    losses = []
    for step in range(args.steps):
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        pred = adapter(x[train_idx])
        if args.objective == "attention-output":
            loss = attention_output_loss(
                pred,
                y[train_idx],
                query_template_device[:, :, train_idx, :],
                key_template_device[:, :, train_idx, :],
                args,
            )
        else:
            loss = kv_loss(
                pred,
                y[train_idx],
                args.key_loss_weight,
                args.value_loss_weight,
                args.cosine_weight,
                args.mse_weight,
            )
        loss.backward()
        optimizer.step()
        if step == 0 or step == args.steps - 1:
            losses.append(float(loss.detach().item()))

    train_metrics = adapter_metrics(adapter, x, y, key_template, query_template, train_idx)
    val_metrics = adapter_metrics(adapter, x, y, key_template, query_template, val_idx)
    return {
        "layer": int(layer_idx),
        "tokens": int(num_tokens),
        "train_tokens": int(train_idx.numel()),
        "val_tokens": int(val_idx.numel()),
        "adapter": args.adapter,
        "objective": args.objective,
        "hidden_dim": int(args.hidden_dim) if args.adapter == "mlp" else 0,
        "steps": int(args.steps),
        "loss_first": losses[0] if losses else 0.0,
        "loss_last": losses[-1] if losses else 0.0,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }


def main() -> None:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if not 0 < args.train_ratio <= 1:
        raise ValueError("--train-ratio must be in (0, 1]")
    if not args.layers:
        raise ValueError("--layers must contain at least one layer index")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("REKV_DEFER_OFFLOAD_WAIT", "1")
    os.environ.setdefault("REKV_OFFLOAD_GRANULARITY", "chunk")
    os.environ.setdefault("REKV_BATCHED_OFFLOAD_COPY", "1")
    os.environ.setdefault("REKV_PROFILE_INTERNAL_BLOCKS", "0")
    os.environ.setdefault("REKV_PROFILE_APPEND_GLOBAL", "0")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this probe.")
    if not args.video_path.exists():
        raise FileNotFoundError(args.video_path)

    proto_args = diag.make_proto_args(args)
    source_video, fps, frame_step, frame_indices = diag.load_sampled_video_prefix(
        args.video_path,
        args.sample_fps,
        args.frames,
    )
    model = proto.load_shared_model(proto_args)
    branches_by_name = proto.build_branches(proto_args, model, source_video)
    proto.install_dynamic_branch_attention(model, branches_by_name)
    fs112 = branches_by_name["fs112"]
    fs224 = branches_by_name["fs224"]

    fs112_inputs, fs112_features = diag.prepare_video_features_with_grid(model, fs112)
    fs224_inputs, fs224_features = diag.prepare_video_features_with_grid(model, fs224)
    fs112_grid = diag.merged_grid(fs112, fs112_inputs.video_grid_thw)
    fs224_grid = diag.merged_grid(fs224, fs224_inputs.video_grid_thw)

    max_layer = max(args.layers)
    proto.reset_branches(model, [fs112, fs224])
    proto.encode_init_prompts(model, [fs112, fs224])
    fs112_records = diag.collect_branch_layer_records(model, fs112, fs112_features, max_layer + 1)
    fs224_records = diag.collect_branch_layer_records(model, fs224, fs224_features, max_layer + 1)

    layer_results = []
    for layer_idx in args.layers:
        if layer_idx < 0 or layer_idx >= len(fs224_records):
            raise ValueError(f"Layer {layer_idx} is out of range for collected records.")
        x_cpu, y_cpu = make_layer_dataset(
            fs112_records[layer_idx],
            fs224_records[layer_idx],
            fs224_features,
            fs112_grid,
            fs224_grid,
            args.context_tensor,
        )
        layer_results.append(
            train_one_layer(
                args,
                layer_idx,
                x_cpu,
                y_cpu,
                fs224_records[layer_idx]["key"],
                fs224_records[layer_idx]["query"],
            )
        )
        torch.cuda.empty_cache()

    summary = {
        "video_path": str(args.video_path),
        "sample_fps": args.sample_fps,
        "original_fps_rounded": fps,
        "sampling_step": frame_step,
        "sampled_frame_indices": frame_indices,
        "frames": int(source_video.shape[0]),
        "layers": [int(layer) for layer in args.layers],
        "context_tensor": args.context_tensor,
        "adapter": args.adapter,
        "objective": args.objective,
        "key_loss_weight": args.key_loss_weight,
        "value_loss_weight": args.value_loss_weight,
        "direct_key_weight": args.direct_key_weight,
        "direct_value_weight": args.direct_value_weight,
        "train_ratio": args.train_ratio,
        "input": {
            "fs224_vision_features": list(fs224_features.shape),
            "fs112_context_grid_t_h_w": list(fs112_grid),
            "fs224_target_grid_t_h_w": list(fs224_grid),
        },
        "target": "teacher_fs224_layer_key_value",
        "layer_results": layer_results,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.json:
        print()
        print(
            "Interpretation: this is a feasibility probe. Overfit success only says "
            "the adapter has enough capacity on this prefix; held-out tokens/videos "
            "are needed before treating it as a real replacement for the fs224 decoder."
        )


if __name__ == "__main__":
    main()
