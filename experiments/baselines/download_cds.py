"""Download annotated CDS FASTA files for the project's assembly IDs.

Requires the NCBI Datasets CLI. Downloads only CDS, one assembly at a time,
so interrupted runs can resume without repeating successful downloads.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


SPLITS = ("train", "val", "test")
MANIFEST_FIELDS = (
    "split", "assembly_id", "assembly_status", "annotated", "status",
    "n_cds", "n_ribosomal", "error",
)
RIBOSOMAL = re.compile(r"\b[35]0S ribosomal protein\b", re.IGNORECASE)


class CDSNotFound(ValueError):
    pass


def assemblies(data_dir: Path) -> list[tuple[str, str]]:
    result = []
    seen = set()
    for split in SPLITS:
        path = data_dir / f"iso_rib_temp_mod_{split}.csv"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "assembly_id" not in reader.fieldnames:
                raise ValueError(f"Missing assembly_id in {path}")
            for row in reader:
                assembly_id = row["assembly_id"].strip()
                if not assembly_id or assembly_id in seen:
                    raise ValueError(f"Missing or duplicate assembly_id {assembly_id!r} in {path}")
                seen.add(assembly_id)
                result.append((split, assembly_id))
    return result


def count_annotations(path: Path) -> tuple[int, int]:
    n_cds = n_ribosomal = 0
    with path.open("rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                n_cds += 1
                if (RIBOSOMAL.search(line)
                        and "methyl" not in line.lower()
                        and "hydroxy" not in line.lower()):
                    n_ribosomal += 1
    return n_cds, n_ribosomal


def assembly_metadata(datasets_bin: str, requested: list[tuple[str, str]],
                      cds_dir: Path, timeout: int) -> dict[str, dict]:
    with tempfile.TemporaryDirectory(prefix="ncbi_metadata_", dir=cds_dir) as temp_dir:
        ids_path = Path(temp_dir) / "ids.txt"
        ids_path.write_text("\n".join(assembly_id for _, assembly_id in requested) + "\n",
                            encoding="utf-8")
        command = [datasets_bin, "summary", "genome", "accession", "--inputfile",
                   str(ids_path), "--as-json-lines"]
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=timeout, check=False)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout).strip()[-500:]
                           or f"datasets summary exited {completed.returncode}")
    metadata = {}
    for line in completed.stdout.splitlines():
        if line.strip():
            record = json.loads(line)
            metadata[record["accession"]] = record
    return metadata


def extract_cds(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        candidates = [
            name for name in archive.namelist()
            if "cds" in Path(name).name.lower()
            and name.lower().endswith((".fna", ".fna.gz"))
        ]
        if len(candidates) != 1:
            raise CDSNotFound(f"Expected one CDS FASTA, found {len(candidates)}: {candidates[:5]}")
        candidate = candidates[0]
        temp_path = destination.with_suffix(".fna.part")
        try:
            with archive.open(candidate) as source, temp_path.open("wb") as target:
                if candidate.lower().endswith(".gz"):
                    with gzip.GzipFile(fileobj=source) as decompressed:
                        shutil.copyfileobj(decompressed, target)
                else:
                    shutil.copyfileobj(source, target)
            temp_path.replace(destination)
        finally:
            temp_path.unlink(missing_ok=True)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    local_datasets = script_dir / "bin" / "datasets.exe"
    default_datasets = str(local_datasets) if local_datasets.exists() else "datasets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=script_dir.parent.parent / "dse/data/ribosomal")
    parser.add_argument("--cds-dir", type=Path, default=script_dir / "cds")
    parser.add_argument("--datasets-bin", default=default_datasets,
                        help="NCBI Datasets executable")
    parser.add_argument("--limit", type=int, help="Download only the first N assemblies for a pilot")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Audit annotation availability without downloading new CDS files")
    parser.add_argument("--manifest", type=Path, help="Output manifest CSV path")
    parser.add_argument("--timeout", type=int, default=600, help="Seconds per assembly download")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not shutil.which(args.datasets_bin):
        parser.error(f"NCBI Datasets executable not found: {args.datasets_bin}")
    args.cds_dir.mkdir(parents=True, exist_ok=True)

    requested = assemblies(args.data_dir)
    if args.limit is not None:
        requested = requested[:args.limit]
    metadata = assembly_metadata(args.datasets_bin, requested, args.cds_dir, args.timeout)
    manifest = []
    for index, (split, assembly_id) in enumerate(requested, start=1):
        record = metadata.get(assembly_id, {})
        assembly_status = record.get("assembly_info", {}).get("assembly_status", "unknown")
        annotated = bool(record.get("annotation_info"))
        destination = args.cds_dir / f"{assembly_id}.fna"
        status = "ok"
        error = ""
        n_cds = n_ribosomal = 0
        try:
            if destination.exists():
                pass  # A manually supplied CDS FASTA can be used even without NCBI annotation.
            elif not record:
                status = "no_metadata"
            elif not annotated:
                status = "not_annotated"
            elif args.metadata_only:
                status = "ready_for_download"
            else:
                with tempfile.TemporaryDirectory(prefix="ncbi_", dir=args.cds_dir) as temp_dir:
                    archive_path = Path(temp_dir) / "package.zip"
                    command = [args.datasets_bin, "download", "genome", "accession",
                               assembly_id, "--include", "cds", "--no-progressbar",
                               "--filename", str(archive_path)]
                    completed = subprocess.run(command, capture_output=True, text=True,
                                               timeout=args.timeout, check=False)
                    if completed.returncode != 0:
                        raise RuntimeError((completed.stderr or completed.stdout).strip()[-500:]
                                           or f"datasets exited {completed.returncode}")
                    extract_cds(archive_path, destination)
            if status == "ok":
                n_cds, n_ribosomal = count_annotations(destination)
                if n_cds == 0:
                    status = "no_cds"
                elif n_ribosomal == 0:
                    status = "no_ribosomal_annotations"
        except CDSNotFound as exc:
            status = "no_cds"
            error = str(exc).replace("\n", " ")[:500]
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, zipfile.BadZipFile) as exc:
            status = "failed"
            error = str(exc).replace("\n", " ")[:500]
        manifest.append({"split": split, "assembly_id": assembly_id,
                         "assembly_status": assembly_status, "annotated": annotated,
                         "status": status,
                         "n_cds": n_cds, "n_ribosomal": n_ribosomal, "error": error})
        print(f"[{index}/{len(requested)}] {split} {assembly_id}: {status} "
              f"({n_cds} CDS, {n_ribosomal} ribosomal)", flush=True)

    manifest_path = args.manifest or args.cds_dir / "download_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
