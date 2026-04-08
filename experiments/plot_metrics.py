import argparse
import re
from pathlib import Path

LINE_RE = re.compile(
    r"\[Step\]\s+(?P<step>\d+):\s+"
    r"(?P<desc>.+?)\s+Loss:\s+(?P<loss>[0-9]*\.?[0-9]+),\s+"
    r"(?P=desc)\s+Accuracy:\s+(?P<acc>[0-9]*\.?[0-9]+)"
)


def parse_metrics(path: Path):
    metrics = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = LINE_RE.search(line)
            if match is None:
                continue
            desc = match.group("desc")
            metrics.setdefault(desc, {"step": [], "loss": [], "acc": []})
            metrics[desc]["step"].append(int(match.group("step")))
            metrics[desc]["loss"].append(float(match.group("loss")))
            metrics[desc]["acc"].append(float(match.group("acc")))
    if not metrics:
        raise ValueError(f"No metric lines found in {path}")
    return metrics


def plot_metrics(metrics, title: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required to plot metrics. Install it in your environment first.") from exc

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for desc, values in metrics.items():
        axes[0].plot(values["step"], values["loss"], label=desc)
        axes[1].plot(values["step"], values["acc"], label=desc)

    axes[0].set_ylabel("Loss")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot loss/accuracy vs step from trainer logs.")
    parser.add_argument(
        "log_path",
        nargs="?",
        default=str(Path(__file__).resolve().parent / "log" / "log.txt"),
        help="Path to output.txt or log.txt",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the plot as an image instead of showing it.",
    )
    args = parser.parse_args()

    log_path = Path(args.log_path).resolve()
    metrics = parse_metrics(log_path)
    fig = plot_metrics(metrics, title=log_path.name)

    if args.save:
        save_path = Path(args.save).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
