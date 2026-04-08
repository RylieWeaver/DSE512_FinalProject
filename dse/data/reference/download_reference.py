# General
import argparse
import subprocess
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import shutil
import zipfile


# Commands
# - python download_reference.py --csv_dir path/to/csvs --target_root path/to/downloaded/genomes


SPLITS = ("train", "val", "test")


def log(message: str) -> None:
    print(f"[download_reference] {message}")


def read_assembly_ids(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path)
    ids = df["assembly_id"].tolist()
    return ids


def extract_fasta_from_zip(zip_path: Path, output_fasta: Path) -> bool:
    # Open ZIP
    with zipfile.ZipFile(zip_path) as archive:
        # Search for FASTA within ZIP
        fasta_path = next((name for name in archive.namelist() if name.endswith(".fna")), None)
        if fasta_path is None:
            return False
        # Setup
        output_fasta.parent.mkdir(parents=True, exist_ok=True)
        # Extract FASTA to output directory
        with archive.open(fasta_path) as source, output_fasta.open("wb") as target:
            shutil.copyfileobj(source, target)
    return True


def build_assembly_list(csv_dir: Path, output_path: Path, keywords: list[str]) -> list[str]:
    # Setup paths
    keywords = [keyword.lower() for keyword in keywords]

    # Iterate over all CSV files
    all_paths = list(csv_dir.rglob("*.csv"))  # All CSVs
    print(f"Found paths: {all_paths}")
    matched_paths = [p for p in all_paths if any(keyword in p.name.lower() for keyword in keywords)]
    print(f"Matched paths: {matched_paths} to keywords {keywords}")

    # Aggregate assembly IDs from matching CSVs
    assembly_ids: set[str] = set()  # set to avoid dupes
    for csv_path in matched_paths:
        assembly_ids.update(read_assembly_ids(csv_path))
    assembly_ids = list(assembly_ids)  # convert to list

    # Write assembly IDs to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for assembly_id in assembly_ids:
            handle.write(f"{assembly_id}\n")
    print(f"Wrote {len(assembly_ids)} assembly IDs to {output_path}")
    return assembly_ids


def download_split(csv_dir: Path, target_root: Path, split: str, delete_zip: bool = True) -> None:
    # Setup paths
    split_dir = target_root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # Build assembly ID list
    log(f"Building assembly ID list for {split} from {csv_dir}")
    ids_path = split_dir / "assembly_ids.txt"
    assembly_ids = build_assembly_list(csv_dir, output_path=ids_path, keywords=[split])

    # Download FASTA files
    for assembly_id in tqdm(assembly_ids, desc=f"Downloading {split} FASTA files"):
        fasta_path = split_dir / f"{assembly_id}.fa"
        zip_path = split_dir / f"{assembly_id}.zip"

        # Skip if FASTA already exists
        if fasta_path.exists():
            log(f"Skipping existing FASTA for {assembly_id} because it already exists at {fasta_path}")
            continue
        
        # Download ZIP from NCBI datasets
        try:
            subprocess.run(
                [
                    "datasets",
                    "download",
                    "genome",
                    "accession",
                    assembly_id,
                    "--include",
                    "genome",
                    "--filename",
                    str(zip_path),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            log(f"Error downloading {assembly_id}: {e}")
            continue
        extracted = extract_fasta_from_zip(zip_path, fasta_path)
        if not extracted:
            log(f"No FASTA payload found in {zip_path}; skipping")
            if delete_zip:
                zip_path.unlink(missing_ok=True)
            continue
        if delete_zip:
            zip_path.unlink(missing_ok=True)


if __name__ == "__main__":
    # Read args
    parser = argparse.ArgumentParser()
    default_csv_dir = Path(__file__).resolve().parent.parent / "ribosomal"
    default_target_root = Path(__file__).resolve().parent / "Microbial"
    parser.add_argument(
        "--csv_dir",
        default=default_csv_dir,
        help="Directory containing the ribosomal split CSV files.",
    )
    parser.add_argument(
        "--target_root",
        default=default_target_root,
        help="Directory where the genomes will be downloaded.",
    )
    parser.add_argument(
        "--delete-zip",
        default=True,
        help="Delete downloaded zip files after FASTA extraction.",
    )
    args = parser.parse_args()
    csv_dir = Path(args.csv_dir).resolve()
    target_root = Path(args.target_root).resolve()

    # Setup
    if not csv_dir.is_dir():
        raise SystemExit(f"Error: CSV directory does not exist: {csv_dir}")
    target_root.mkdir(parents=True, exist_ok=True)

    # Download
    for split_name in SPLITS:
        download_split(csv_dir, target_root, split_name, delete_zip=args.delete_zip)
    log(f"Finished writing genomes to {target_root}")
