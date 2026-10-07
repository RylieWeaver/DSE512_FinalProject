"""Epoch history export and loss/R² plots shared by all genome-regression models."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch


COLORS = {
    "train": "#1f77b4",
    "val": "#ff7f0e",
    "test": "#2ca02c",
}


def train_and_plot(trainer, datasets, epochs, resume=False):
    """Run the existing trainer, save full-precision epoch metrics, and plot them."""
    variances = {
        split: torch.cat([record["labels"].reshape(-1).double() for record in dataset])
                    .var(correction=0).item()
        for split, dataset in datasets.items()
    }
    history_path = trainer.cfg.checkpoint_dir / "history.csv"
    previous = []
    if resume and history_path.exists():
        with history_path.open(newline="") as handle:
            previous = [row for row in csv.DictReader(handle) if int(row["epoch"]) <= trainer.last_epoch]
    fields = ["epoch", "learning_rate"] + [
        f"{split}_{metric}" for split in datasets for metric in ("mse", "r2")
    ]
    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(previous)
        for _ in range(epochs):
            trainer.train(epochs=1)
            if trainer.last_epoch % trainer.cfg.eval_every:
                continue
            row = {"epoch": trainer.last_epoch, "learning_rate": trainer.optimizer.param_groups[0]["lr"]}
            for split, desc in zip(datasets, trainer.descriptors[1:]):
                measured = trainer.cumulative_metrics[desc]
                mse = measured["loss"] / measured["count"]
                row[f"{split}_mse"] = mse
                row[f"{split}_r2"] = 1 - mse / variances[split] if variances[split] > 0 else None
            writer.writerow(row)
            handle.flush()
    plot_history(
        pd.read_csv(history_path, float_precision="round_trip"),
        history_path.parent / "plots",
        f"ae{trainer.model.cfg.input_dim}_{getattr(trainer.model.cfg, 'model_type', 'transformer')}",
    )


def plot_history(history, output, name):
    """Loss and R² curves in the original plot_this_*.png style."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if history.empty or history.epoch.duplicated().any() or not history.epoch.is_monotonic_increasing:
        raise ValueError("Expected a nonempty, ordered epoch history")
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.labelcolor": "black",
        "text.color": "black",
        "axes.edgecolor": "black",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        files = []
        for metric, suffix, title, ylabel in (
            ("mse", "loss", "Loss", "MSE Loss"),
            ("r2", "r2", r"$R^2$", r"$R^2$"),
        ):
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for split, split_label in (
                ("train", "Train"),
                ("val", "Validation"),
                ("test", "Test"),
            ):
                if f"{split}_{metric}" not in history:
                    continue
                ax.plot(
                    history.epoch,
                    history[f"{split}_{metric}"],
                    label=split_label,
                    color=COLORS[split],
                    linewidth=1.7,
                )
            if metric == "r2":
                ax.axhline(0, color="#777777", linewidth=1)
            ax.set(
                title=f"Genome Embedding Regression {title}",
                xlabel="Epoch",
                ylabel=ylabel,
            )
            ax.grid(color="#b0b0b0", alpha=0.3)
            ax.set_axisbelow(True)
            ax.legend(loc="best", frameon=True, fontsize=10)
            fig.tight_layout()
            for extension in ("png", "pdf"):
                path = output / f"{name}_{suffix}.{extension}"
                fig.savefig(path, dpi=180)
                files.append(path.name)
            plt.close(fig)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--name", default="embedding_regression")
    args = parser.parse_args()
    plot_history(
        pd.read_csv(args.history, float_precision="round_trip"),
        args.output_dir or args.history.parent / "plots",
        args.name,
    )


if __name__ == "__main__":
    main()
