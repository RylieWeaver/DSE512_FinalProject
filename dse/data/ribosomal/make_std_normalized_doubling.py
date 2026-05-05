import json
from pathlib import Path
import pandas as pd


DATA_DIR = Path(__file__).parent.resolve()
SPLITS = ("train", "val", "test")
TARGET_COLS = ("log_dob_h", "growth_tmp")
OUTPUT_SUFFIX = "_std_norm"


def main() -> None:
    input_paths = {split: DATA_DIR / f"iso_rib_temp_mod_{split}.csv" for split in SPLITS}
    dfs = {split: pd.read_csv(path) for split, path in input_paths.items()}

    train_df = dfs["train"]
    stats = {}
    for col in TARGET_COLS:
        values = pd.to_numeric(train_df[col], errors="coerce").dropna()
        mean = float(values.mean())
        std = float(values.std())
        if std == 0.0:
            raise ValueError(f"Standard deviation is zero for '{col}'")
        stats[col] = {"mean": mean, "std": std}

    for split, df in dfs.items():
        out = df.copy()
        for col in TARGET_COLS:
            values = pd.to_numeric(out[col], errors="coerce")
            out[col] = (values - stats[col]["mean"]) / stats[col]["std"]
        out_path = DATA_DIR / f"iso_rib_temp_mod_{split}{OUTPUT_SUFFIX}.csv"
        out.to_csv(out_path, index=False)

    stats_path = DATA_DIR / f"normalization_stats{OUTPUT_SUFFIX}.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(f"Wrote normalized files to {DATA_DIR}")


if __name__ == "__main__":
    main()
