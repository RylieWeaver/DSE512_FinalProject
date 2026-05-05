#!/usr/bin/env python3
from pathlib import Path
import gzip
import sys


FASTA_SUFFIXES = {".fa", ".fasta", ".fna"}


def open_maybe_gzip(path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "r")


def is_fasta(path):
    if path.suffix == ".gz":
        return path.with_suffix("").suffix.lower() in FASTA_SUFFIXES
    return path.suffix.lower() in FASTA_SUFFIXES


def count_bp(path):
    total = 0
    with open_maybe_gzip(path) as f:
        for line in f:
            if line.startswith(">"):
                continue
            total += len(line.strip())
    return total


def main(directory):
    directory = Path(directory)
    grand_total = 0

    for path in sorted(directory.iterdir()):
        if not path.is_file() or not is_fasta(path):
            continue

        grand_total += count_bp(path)

    print(grand_total)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_fasta_bp.py /path/to/fasta_dir")
        sys.exit(1)

    main(sys.argv[1])
