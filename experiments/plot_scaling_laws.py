#!/usr/bin/env python3
"""Plot loss and accuracy against log training tokens and transformer compute."""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


RUN_RE = re.compile(
    r"md(?P<model_dim>\d+)_ctx(?P<context_len>\d+)"
    r"(?:_bs(?P<batch_size>\d+))?"
    r"_bps(?P<batches_per_step>\d+)"
    r"(?:_sp(?P<sp_size>\d+)_dp(?P<dp_size>\d+))?$"
)
LOG_RE = re.compile(
    r"\[Step\]\s+(?P<step>\d+):\s+"
    r"(?P<split>Train w/o Grad|Train w/ Grad|Eval|Test)\s+"
    r"Loss:\s+(?P<loss>[0-9.]+)"
    r"(?:,\s+(?P=split)\s+Accuracy:\s+(?P<accuracy>[0-9.]+))?"
)

SPLIT_ORDER = ["Train w/ Grad", "Test", "Eval", "Train w/o Grad"]
X_AXES = [
    ("log10_tokens", "log10 training tokens"),
    ("log10_compute", "log10 training FLOPs"),
]
X_TITLES = {
    "log10_tokens": "training tokens",
    "log10_compute": "training FLOPs",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot scaling-law traces from experiments/log/<run>/log.txt."
    )
    parser.add_argument("--log-root", type=Path, default=Path("log"))
    parser.add_argument("--out-dir", type=Path, default=Path("plots"))
    parser.add_argument(
        "--series",
        choices=["train", "eval", "test", "train_test"],
        default="train_test",
        help="Series to plot. train_test writes separate train and test curve plots.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Fallback per-GPU microbatch size for old run names without _bs... in them.",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=64,
        help="Fallback data parallel size for old run names without _dp... in them.",
    )
    parser.add_argument(
        "--sp-size",
        type=int,
        default=1,
        help="Fallback sequence parallel size for old run names without _sp... in them.",
    )
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=6)
    parser.add_argument("--max-loss", type=float, default=None, help="Optional loss filter.")
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.9,
        help="EMA history factor in [0, 1]. Higher means smoother (e.g., 0.99). Use 0 to disable.",
    )
    parser.add_argument(
        "--ema-unbiased",
        action="store_true",
        help="Apply bias-corrected (unbiased) EMA instead of standard EMA.",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        default=500,
        help="Drop metric points before this training step. Use 1 to remove initialization anchors.",
    )
    return parser.parse_args()


def parse_run_name(run_dir):
    match = RUN_RE.match(run_dir.name)
    if match is None:
        return None
    return {
        key: int(value) if value is not None else None
        for key, value in match.groupdict().items()
    }


def parse_log(log_path):
    rows = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = LOG_RE.search(line)
            if match is None:
                continue
            rows.append(
                {
                    "step": int(match.group("step")),
                    "split": match.group("split"),
                    "loss": float(match.group("loss")),
                    "accuracy": (
                        float(match.group("accuracy"))
                        if match.group("accuracy") is not None
                        else None
                    ),
                }
            )
    return rows


def forward_flops_per_sample(context_len, model_dim, num_layers, vocab_size):
    """Dense pre-norm transformer forward FLOPs for one sequence.

    Per layer, this counts:
    - QKV projection: 6 * S * D^2
    - attention output projection: 2 * S * D^2
    - two-layer 4D MLP: 16 * S * D^2
    - full non-causal attention QK and AV matmuls: 4 * S^2 * D

    LayerNorm, RoPE, activation, softmax, dropout, and communication are ignored.
    Training compute uses 3x forward FLOPs as a standard forward+backward estimate.
    """
    s = context_len
    d = model_dim
    transformer = num_layers * (24 * s * d * d + 4 * s * s * d)
    unembed = 2 * s * d * vocab_size
    return transformer + unembed


def model_num_parameters(model_dim, num_layers, num_heads, vocab_size):
    """Parameter count for the MLM transformer architecture used in training."""
    if model_dim % num_heads != 0:
        raise ValueError(f"model_dim={model_dim} must be divisible by num_heads={num_heads}")

    d = model_dim
    head_dim = d // num_heads

    token_embedding = vocab_size * d
    attention = (
        3 * d * d + 3 * d  # qkv
        + 2 * head_dim     # q_norm
        + 2 * head_dim     # k_norm
        + d * d + d        # out_proj
    )
    mlp = d * (4 * d) + (4 * d) + (4 * d) * d + d
    block_norms = 4 * d
    block = attention + mlp + block_norms
    output_norm = 2 * d
    return token_embedding + num_layers * block + output_norm


def build_points(args):
    split_names = {
        "train": {"Train w/ Grad"},
        "eval": {"Eval"},
        "test": {"Test"},
        "train_test": {"Train w/ Grad", "Test"},
    }[args.series]

    points = []
    for log_path in sorted(args.log_root.glob("*/log.txt")):
        run_dir = log_path.parent
        run = parse_run_name(run_dir)
        if run is None:
            continue
        batch_size = run["batch_size"] if run["batch_size"] is not None else args.batch_size
        dp_size = run["dp_size"] if run["dp_size"] is not None else args.dp_size
        sp_size = run["sp_size"] if run["sp_size"] is not None else args.sp_size

        tokens_per_step = (
            run["context_len"]
            * batch_size
            * run["batches_per_step"]
            * dp_size
        )
        train_flops_per_step = (
            3
            * forward_flops_per_sample(
                context_len=run["context_len"],
                model_dim=run["model_dim"],
                num_layers=args.num_layers,
                vocab_size=args.vocab_size,
            )
            * batch_size
            * run["batches_per_step"]
            * dp_size
        )
        num_parameters = model_num_parameters(
            model_dim=run["model_dim"],
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            vocab_size=args.vocab_size,
        )

        for row in parse_log(log_path):
            if row["split"] not in split_names:
                continue
            if row["step"] < args.min_step:
                continue
            if args.max_loss is not None and row["loss"] > args.max_loss:
                continue

            # Step 0 eval/test losses are useful curve anchors, but log axes cannot
            # represent zero tokens/compute. Plot them at the first-step position.
            plot_step = max(row["step"], 1)
            tokens = plot_step * tokens_per_step
            compute = plot_step * train_flops_per_step
            points.append(
                {
                    **run,
                    "batch_size": batch_size,
                    "dp_size": dp_size,
                    "sp_size": sp_size,
                    "model_num_parameters": num_parameters,
                    "run_name": run_dir.name,
                    "step": row["step"],
                    "split": row["split"],
                    "loss": row["loss"],
                    "accuracy": row["accuracy"],
                    "tokens": tokens,
                    "compute_flops": compute,
                    "log10_tokens": math.log10(tokens),
                    "log10_compute": math.log10(compute),
                }
            )
    return points


def write_csv(points, path):
    if not points:
        return
    fieldnames = list(points[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(points)


def _style_maps(points):
    model_num_parameters = sorted({point["model_num_parameters"] for point in points})
    context_lens = sorted({point["context_len"] for point in points})

    cmap = plt.get_cmap("tab10")
    color_by_params = {
        num_parameters: cmap(idx % cmap.N)
        for idx, num_parameters in enumerate(model_num_parameters)
    }
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    marker_by_context = {
        context_len: markers[idx % len(markers)]
        for idx, context_len in enumerate(context_lens)
    }
    return color_by_params, marker_by_context


def _format_num_parameters(num_parameters):
    if num_parameters >= 1_000_000_000:
        return f"{num_parameters / 1_000_000_000:.2f}B"
    if num_parameters >= 1_000_000:
        return f"{num_parameters / 1_000_000:.1f}M"
    if num_parameters >= 1_000:
        return f"{num_parameters / 1_000:.1f}K"
    return str(num_parameters)


def _add_style_legends(ax, color_by_params, marker_by_context, metric_key):
    if metric_key == "accuracy":
        param_loc = "upper left"
        context_loc = "lower right"
    else:
        param_loc = "upper right"
        context_loc = "lower left"

    param_handles = [
        Line2D([0], [0], color=color, lw=3, label=_format_num_parameters(num_parameters))
        for num_parameters, color in color_by_params.items()
    ]
    context_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            lw=2.0,
            linestyle="-",
            marker=marker,
            markersize=8,
            label=f"ctx {context_len}",
        )
        for context_len, marker in marker_by_context.items()
    ]

    param_legend = ax.legend(
        handles=param_handles,
        title="Model parameters",
        fontsize=11,
        title_fontsize=12,
        loc=param_loc,
    )
    ax.add_artist(param_legend)
    ax.legend(
        handles=context_handles,
        title="Context length",
        fontsize=11,
        title_fontsize=12,
        loc=context_loc,
    )


def apply_ema(values, alpha, unbiased=False):
    if not values or alpha <= 0:
        return list(values)
    if alpha >= 1:
        return [values[0]] * len(values)

    beta = alpha
    one_minus_beta = 1.0 - beta
    if not unbiased:
        # Use the first observation as the EMA initial state to avoid a
        # misleadingly low first plotted point from zero-state warm start.
        avg = values[0]
        smoothed = [avg]
        for value in values[1:]:
            avg = beta * avg + one_minus_beta * value
            smoothed.append(avg)
        return smoothed

    avg = 0.0
    beta_power = 1.0
    smoothed = []
    for value in values:
        avg = beta * avg + one_minus_beta * value
        beta_power *= beta
        correction = 1.0 - beta_power
        smoothed.append(avg / correction if correction > 0 else value)
    return smoothed


def _plot_split(
    ax,
    points,
    split,
    x_key,
    x_label,
    metric_key,
    metric_label,
    color_by_params,
    marker_by_context,
    ema_alpha,
    ema_unbiased,
):
    grouped = defaultdict(list)
    for point in points:
        if point["split"] == split and point[metric_key] is not None:
            grouped[point["run_name"]].append(point)

    for run_points in grouped.values():
        run_points = sorted(run_points, key=lambda point: point["step"])
        first = run_points[0]
        ax.plot(
            [point[x_key] for point in run_points],
            apply_ema([point[metric_key] for point in run_points], ema_alpha, unbiased=ema_unbiased),
            color=color_by_params[first["model_num_parameters"]],
            linestyle="-",
            marker=marker_by_context[first["context_len"]],
            markersize=2.5 if split == "Train w/ Grad" else 3.5,
            linewidth=1.2,
            alpha=0.9,
        )

    ax.set_title(f"{metric_label} on {split} vs {X_TITLES[x_key]}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric_label)
    ax.grid(True, alpha=0.3)


def plot_points(points, x_key, x_label, metric_key, metric_label, path, ema_alpha, ema_unbiased):
    splits = [
        split
        for split in SPLIT_ORDER
        if any(point["split"] == split and point[metric_key] is not None for point in points)
    ]
    if not splits:
        return False

    color_by_params, marker_by_context = _style_maps(points)

    fig, axes = plt.subplots(
        len(splits),
        1,
        figsize=(10, 4.5 * len(splits)),
        sharex=True,
        squeeze=False,
    )

    for ax, split in zip(axes[:, 0], splits):
        _plot_split(
            ax,
            points,
            split,
            x_key,
            x_label,
            metric_key,
            metric_label,
            color_by_params,
            marker_by_context,
            ema_alpha,
            ema_unbiased,
        )

    _add_style_legends(axes[0, 0], color_by_params, marker_by_context, metric_key)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def plot_single_curve(points, split, x_key, x_label, metric_key, metric_label, path, ema_alpha, ema_unbiased):
    split_points = [
        point
        for point in points
        if point["split"] == split and point[metric_key] is not None
    ]
    if not split_points:
        return False

    color_by_params, marker_by_context = _style_maps(points)
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_split(
        ax,
        split_points,
        split,
        x_key,
        x_label,
        metric_key,
        metric_label,
        color_by_params,
        marker_by_context,
        ema_alpha,
        ema_unbiased,
    )
    _add_style_legends(ax, color_by_params, marker_by_context, metric_key)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    if args.ema_alpha < 0 or args.ema_alpha > 1:
        raise SystemExit("--ema-alpha must be between 0 and 1.")
    args.log_root = args.log_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    points = build_points(args)
    if not points:
        raise SystemExit(f"No {args.series} points found under {args.log_root}")

    csv_path = args.out_dir / f"{args.series}_scaling_points.csv"
    write_csv(points, csv_path)
    print(f"Wrote {csv_path}")

    metrics = [
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
    ]
    if args.series == "train_test":
        for split, label in [("Train w/ Grad", "train"), ("Test", "test")]:
            for x_key, x_label in X_AXES:
                for metric_key, metric_label in metrics:
                    path = args.out_dir / f"{label}_{metric_key}_vs_{x_key.replace('log10_', 'log_')}.png"
                    if plot_single_curve(
                        points,
                        split,
                        x_key,
                        x_label,
                        metric_key,
                        metric_label,
                        path,
                        args.ema_alpha,
                        args.ema_unbiased,
                    ):
                        print(f"Wrote {path}")
    else:
        for x_key, x_label in X_AXES:
            for metric_key, metric_label in metrics:
                path = args.out_dir / f"{args.series}_{metric_key}_vs_{x_key.replace('log10_', 'log_')}.png"
                if plot_points(
                    points,
                    x_key,
                    x_label,
                    metric_key,
                    metric_label,
                    path,
                    args.ema_alpha,
                    args.ema_unbiased,
                ):
                    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
