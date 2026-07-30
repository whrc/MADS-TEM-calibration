#!/usr/bin/env python3
"""
Download site run files from GCS into local sites_runs/<site_name>.

Usage:
  python fetch_site_run.py --site-name CMT02
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


FILES_TO_COPY = ("sample_matrix.csv", "results.csv", "targets.csv")


def copy_from_gcs(site_name: str) -> None:
    destination = Path("sites_runs") / site_name
    destination.mkdir(parents=True, exist_ok=True)

    for filename in FILES_TO_COPY:
        source = f"gs://vb-tem/Sample_Runs/{site_name}/{filename}"
        print(f"Copying {source} -> {destination}")
        subprocess.run(
            ["gsutil", "cp", source, str(destination / filename)],
            check=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create sites_runs/<site_name> (if needed) and copy "
            "sample_matrix.csv, results.csv, targets.csv from GCS."
        )
    )
    parser.add_argument(
        "--site-name",
        default="CMT02",
        help="Site name under gs://vb-tem/Sample_Runs/ (default: CMT02).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        copy_from_gcs(args.site_name)
    except FileNotFoundError:
        print(
            "Error: gsutil is not installed or not on PATH. "
            "Install Google Cloud SDK and try again.",
            file=sys.stderr,
        )
        return 1
    except subprocess.CalledProcessError as error:
        print(f"Error: failed to copy files ({error}).", file=sys.stderr)
        return error.returncode or 1

    print(f"Done. Files are in sites_runs/{args.site_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
