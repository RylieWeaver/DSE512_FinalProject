"""Evaluate simple doubling-time baselines on the project's fixed splits.

Only the training split is used to fit the mean and temperature models. gRodon
predictions, when provided, are joined by assembly ID and scored on the same
normalized log-doubling-time target as the sequence regression trainer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean


SPLITS = ("train", "val", "test")
METHODS = ("train_mean", "temperature_linear", "grodon")
PREDICTION_FIELDS = (
    "split", "assembly_id", "method", "observed_log_hours",
    "predicted_log_hours", "observed_hours", "predicted_hours",
    "observed_normalized", "predicted_normalized",
)
METRIC_FIELDS = (
    "cohort", "split", "method", "n", "mse_normalized",
    "rmse_normalized", "mae_normalized", "rmse_log_hours", "mae_log_hours",
)


@dataclass(frozen=True)
class Example:
    split: str
    assembly_id: str
    log_hours: float
    temperature: float
    y: float
    x: float


def finite_number(value: str, name: str, location: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name} at {location}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {name} at {location}: {value!r}")
    return number


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {path}")
        return list(reader)


def read_keyed_rows(path: Path, required: set[str]) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"No data rows in {path}")
    missing = required - rows[0].keys()
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")
    keyed = {}
    for row in rows:
        assembly_id = row["assembly_id"].strip()
        if not assembly_id or assembly_id in keyed:
            raise ValueError(f"Missing or duplicate assembly_id {assembly_id!r} in {path}")
        keyed[assembly_id] = row
    return keyed


def load_data(data_dir: Path) -> tuple[dict[str, list[Example]], dict]:
    stats_path = data_dir / "normalization_stats_std_norm.json"
    with stats_path.open(encoding="utf-8") as handle:
        stats = json.load(handle)
    target_mean = finite_number(str(stats["log_dob_h"]["mean"]), "target mean", str(stats_path))
    target_std = finite_number(str(stats["log_dob_h"]["std"]), "target std", str(stats_path))
    temp_mean = finite_number(str(stats["growth_tmp"]["mean"]), "temperature mean", str(stats_path))
    temp_std = finite_number(str(stats["growth_tmp"]["std"]), "temperature std", str(stats_path))
    if target_std <= 0 or temp_std <= 0:
        raise ValueError("Normalization standard deviations must be positive")

    splits: dict[str, list[Example]] = {}
    all_ids: set[str] = set()
    required = {"assembly_id", "log_dob_h", "growth_tmp"}
    for split in SPLITS:
        raw_path = data_dir / f"iso_rib_temp_mod_{split}.csv"
        norm_path = data_dir / f"iso_rib_temp_mod_{split}_std_norm.csv"
        raw = read_keyed_rows(raw_path, required)
        norm = read_keyed_rows(norm_path, required)
        if raw.keys() != norm.keys():
            raise ValueError(f"Raw and normalized IDs differ for {split}")
        if all_ids.intersection(raw):
            raise ValueError(f"Assembly IDs overlap between splits at {split}")
        all_ids.update(raw)
        examples = []
        for assembly_id, row in raw.items():
            location = f"{raw_path}:{assembly_id}"
            log_hours = finite_number(row["log_dob_h"], "log_dob_h", location)
            temperature = finite_number(row["growth_tmp"], "growth_tmp", location)
            norm_row = norm[assembly_id]
            y = finite_number(norm_row["log_dob_h"], "normalized log_dob_h", location)
            x = finite_number(norm_row["growth_tmp"], "normalized growth_tmp", location)
            if not math.isclose(y, (log_hours - target_mean) / target_std, abs_tol=1e-9):
                raise ValueError(f"Target normalization mismatch at {location}")
            if not math.isclose(x, (temperature - temp_mean) / temp_std, abs_tol=1e-9):
                raise ValueError(f"Temperature normalization mismatch at {location}")
            examples.append(Example(split, assembly_id, log_hours, temperature, y, x))
        splits[split] = examples
    return splits, stats


def fit_temperature_linear(train: list[Example]) -> tuple[float, float]:
    mean_x = fmean(row.x for row in train)
    mean_y = fmean(row.y for row in train)
    numerator = sum((row.x - mean_x) * (row.y - mean_y) for row in train)
    denominator = sum((row.x - mean_x) ** 2 for row in train)
    if denominator <= 0:
        raise ValueError("Training temperatures have no variance")
    slope = numerator / denominator
    return mean_y - slope * mean_x, slope


def load_grodon(path: Path, splits: dict[str, list[Example]]) -> tuple[dict[str, float], dict]:
    rows = read_csv(path)
    if not rows or not {"split", "assembly_id", "status", "doubling_hours"}.issubset(rows[0]):
        raise ValueError("gRodon CSV needs split, assembly_id, status, doubling_hours columns")
    expected = {row.assembly_id: row.split for split in SPLITS for row in splits[split]}
    predictions: dict[str, float] = {}
    seen: set[str] = set()
    status_counts = Counter()
    over_five = Counter()
    saturation_warnings = Counter()
    for row in rows:
        assembly_id = row["assembly_id"].strip()
        split = row["split"].strip()
        status = row["status"].strip().lower()
        if assembly_id not in expected or expected[assembly_id] != split:
            raise ValueError(f"Unknown or wrong-split gRodon assembly: {assembly_id}, {split}")
        if assembly_id in seen:
            raise ValueError(f"Duplicate gRodon assembly: {assembly_id}")
        seen.add(assembly_id)
        status_counts[status or "missing_status"] += 1
        if status == "ok":
            hours = finite_number(row["doubling_hours"], "gRodon doubling_hours", assembly_id)
            if hours <= 0:
                raise ValueError(f"gRodon doubling_hours must be positive for {assembly_id}")
            predictions[assembly_id] = hours
            if hours > 5:
                over_five[split] += 1
            if "CUB signal saturates" in row.get("warning", ""):
                saturation_warnings[split] += 1
    status_counts["not_in_file"] = len(expected) - len(seen)
    coverage = {
        split: {"available": sum(row.assembly_id in predictions for row in splits[split]),
                "total": len(splits[split])}
        for split in SPLITS
    }
    return predictions, {
        "status_counts": dict(status_counts), "coverage": coverage,
        "predicted_over_five_hours": {split: over_five[split] for split in SPLITS},
        "saturation_warnings": {split: saturation_warnings[split] for split in SPLITS},
    }


def metric_row(cohort: str, split: str, method: str, pairs: list[tuple[Example, float]], target_std: float) -> dict:
    if not pairs:
        raise ValueError(f"No predictions for {cohort}/{split}/{method}")
    normalized_errors = [prediction - row.y for row, prediction in pairs]
    mse = fmean(error * error for error in normalized_errors)
    mae = fmean(abs(error) for error in normalized_errors)
    return {
        "cohort": cohort, "split": split, "method": method, "n": len(pairs),
        "mse_normalized": mse, "rmse_normalized": math.sqrt(mse),
        "mae_normalized": mae, "rmse_log_hours": math.sqrt(mse) * target_std,
        "mae_log_hours": mae * target_std,
    }


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def display_path(path: Path, repo_dir: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_dir.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def write_report(path: Path, metrics: list[dict], split_sizes: dict[str, int], grodon_info: dict | None) -> None:
    lines = [
        "# Doubling-time baseline results",
        "",
        "The fixed assembly-level split has "
        f"{split_sizes['train']} train, {split_sizes['val']} validation, and "
        f"{split_sizes['test']} test examples. Mean and temperature regression "
        "are fitted on train only. Lower MSE is better.",
        "",
        "MSE uses the training-standardized `log_dob_h` target, matching the "
        "sequence regression trainer. Each assembly is scored once.",
        "",
        "| Cohort | Split | Method | N | MSE | RMSE (log hours) |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in metrics:
        if row["split"] in ("val", "test"):
            lines.append(
                f"| {row['cohort']} | {row['split']} | {row['method']} | "
                f"{row['n']} | {row['mse_normalized']:.4f} | "
                f"{row['rmse_log_hours']:.4f} |"
            )
    lines += [""]
    lines += [
        "The train mean predicts one constant log doubling time. Temperature "
        "regression fits an intercept and slope using only training rows. "
        "gRodon uses genome-wide annotated coding genes and ribosomal-protein "
        "labels with `mode=full`, the `madin` training set, and no temperature "
        "correction.",
        "",
    ]
    if grodon_info is None:
        lines += [
            "gRodon has not yet been run. Its scores require annotated CDS FASTA "
            "files and the R package; see `../README.md`.", "",
        ]
    else:
        coverage = grodon_info["coverage"]
        mse = {(row["cohort"], row["split"], row["method"]): row["mse_normalized"]
               for row in metrics}
        lines += [
            "gRodon coverage (valid predictions): "
            + ", ".join(f"{split} {coverage[split]['available']}/{coverage[split]['total']}"
                        for split in SPLITS) + ".",
            "The `grodon_matched` cohort compares all methods on those same "
            "assemblies. Review `summary.json` and the gRodon status CSV for failures.",
        ]
        matched_keys = [
            ("grodon_matched", split, method)
            for split in ("val", "test")
            for method in ("grodon", "temperature_linear")
        ]
        if all(key in mse for key in matched_keys):
            lines.append(
                f"On validation, gRodon MSE was {mse['grodon_matched', 'val', 'grodon']:.4f} "
                f"versus {mse['grodon_matched', 'val', 'temperature_linear']:.4f} for "
                "temperature regression. On the covered test assemblies, the "
                f"scores were {mse['grodon_matched', 'test', 'grodon']:.4f} and "
                f"{mse['grodon_matched', 'test', 'temperature_linear']:.4f}, "
                "respectively. gRodon uses additional genome-wide annotation, so "
                "this comparison describes predictive performance with different "
                "input information."
            )
        lines += [
            "gRodon predictions above five hours: "
            + ", ".join(f"{split} {grodon_info['predicted_over_five_hours'][split]}"
                        for split in SPLITS) + ". Its codon-usage signal can saturate "
            "for slow growers, so errors on those rows need careful interpretation.",
            "Question for the team: Are the project's `log_dob_h` labels the "
            "natural log of doubling time in hours, and do they represent a "
            "quantity comparable to gRodon's minimum doubling time estimate?",
            "",
        ]
    lines += [
        "For the final paper comparison, use model predictions from the "
        "validation-selected checkpoint, with each unique assembly counted once. "
        "The Frontier distributed sampler can repeat validation and test rows.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=script_dir.parent.parent / "dse/data/ribosomal")
    parser.add_argument("--out-dir", type=Path, default=script_dir / "results")
    parser.add_argument("--grodon-predictions", type=Path, help="CSV produced by run_grodon.R")
    args = parser.parse_args()

    splits, stats = load_data(args.data_dir)
    target_mean = stats["log_dob_h"]["mean"]
    target_std = stats["log_dob_h"]["std"]
    train_mean = fmean(row.y for row in splits["train"])
    intercept, slope = fit_temperature_linear(splits["train"])
    grodon, grodon_info = ({}, None)
    if args.grodon_predictions:
        grodon, grodon_info = load_grodon(args.grodon_predictions, splits)

    predictions = []
    metrics = []
    for split in SPLITS:
        examples = splits[split]
        by_method = {
            "train_mean": [(row, train_mean) for row in examples],
            "temperature_linear": [(row, intercept + slope * row.x) for row in examples],
        }
        if args.grodon_predictions:
            by_method["grodon"] = [
                (row, (math.log(grodon[row.assembly_id]) - target_mean) / target_std)
                for row in examples if row.assembly_id in grodon
            ]
        for method, pairs in by_method.items():
            for row, predicted in pairs:
                predicted_log = predicted * target_std + target_mean
                predictions.append({
                    "split": split, "assembly_id": row.assembly_id, "method": method,
                    "observed_log_hours": row.log_hours, "predicted_log_hours": predicted_log,
                    "observed_hours": math.exp(row.log_hours), "predicted_hours": math.exp(predicted_log),
                    "observed_normalized": row.y, "predicted_normalized": predicted,
                })
            if method != "grodon":
                metrics.append(metric_row("all", split, method, pairs, target_std))
        if args.grodon_predictions and by_method["grodon"]:
            matched = {row.assembly_id for row, _ in by_method["grodon"]}
            for method in METHODS:
                pairs = [(row, value) for row, value in by_method[method] if row.assembly_id in matched]
                metrics.append(metric_row("grodon_matched", split, method, pairs, target_std))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "predictions.csv", PREDICTION_FIELDS, predictions)
    write_csv(args.out_dir / "metrics.csv", METRIC_FIELDS, metrics)
    summary = {
        "data_dir": display_path(args.data_dir, script_dir.parent.parent),
        "grodon_predictions": display_path(args.grodon_predictions, script_dir.parent.parent)
        if args.grodon_predictions else None,
        "target": "log_dob_h", "primary_metric": "mse_normalized",
        "split_sizes": {split: len(splits[split]) for split in SPLITS},
        "train_mean_normalized": train_mean,
        "temperature_linear": {"intercept_normalized": intercept, "slope_normalized": slope},
        "normalization": stats,
        "grodon": grodon_info,
    }
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    write_report(args.out_dir / "REPORT.md", metrics, summary["split_sizes"], grodon_info)
    for row in metrics:
        if row["split"] in ("val", "test"):
            print(f"{row['cohort']:14} {row['split']:5} {row['method']:18} "
                  f"n={row['n']:3} MSE={row['mse_normalized']:.4f}")
    print(f"Wrote metrics.csv, predictions.csv, summary.json, and REPORT.md to {args.out_dir}")


if __name__ == "__main__":
    main()
