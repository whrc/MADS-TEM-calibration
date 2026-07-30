#!/usr/bin/env python3
"""
Run fetch + plot pipeline for multiple sites.

Example:
  python3 run_sites_pipeline.py --sites CMT02,CA_OBS
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List


BUCKET_URI = "gs://vb-tem/Sample_Runs/"

# FALLBACK_SITES: [(site_id, [pft0, pft1, ...]), ...] from Excel PFTS sheet
# Matched by CMT# or Site Reference. Use site_pfts.build_fallback_sites() to build.
import site_pfts as _sp

FALLBACK_SITES = _sp.build_fallback_sites()
FALLBACK_SITE_IDS = [s[0] for s in FALLBACK_SITES]  # flat list for pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Loop through site names and run: "
            "'python3 fetch_site_run.py --site-name <site>' then "
            "'python3 plot_site.py sites_runs/<site>'."
        )
    )
    parser.add_argument(
        "--sites",
        default=None,
        help=(
            "Comma-separated site names. If omitted, sites are discovered from "
            f"{BUCKET_URI} (fallback to a predefined list)."
        ),
    )
    return parser.parse_args()


def list_sites_from_bucket(bucket_uri: str = BUCKET_URI) -> List[str]:
    result = subprocess.run(
        ["gsutil", "ls", bucket_uri],
        check=True,
        capture_output=True,
        text=True,
    )

    sites: List[str] = []
    for line in result.stdout.splitlines():
        uri = line.strip()
        if not uri or uri == bucket_uri:
            continue
        if not uri.endswith("/"):
            continue
        site_name = uri.rstrip("/").split("/")[-1]
        if site_name:
            sites.append(site_name)
    return sites


def run_for_site(site_name: str) -> None:
    fetch_cmd = ["python3", "fetch_site_run.py", "--site-name", site_name]
    plot_cmd = ["python3", "plot_site.py", f"sites_runs/{site_name}"]

    print(f"\n=== {site_name}: fetch ===")
    subprocess.run(fetch_cmd, check=True)

    print(f"=== {site_name}: plot ===")
    subprocess.run(plot_cmd, check=True)


def main() -> int:
    args = parse_args()
    if args.sites:
        sites = [site.strip() for site in args.sites.split(",") if site.strip()]
    else:
        try:
            sites = list_sites_from_bucket(BUCKET_URI)
            print(f"Discovered {len(sites)} sites from {BUCKET_URI}")
        except subprocess.CalledProcessError as error:
            print(
                f"Warning: could not list {BUCKET_URI} ({error}). "
                "Using fallback site list.",
                file=sys.stderr,
            )
            sites = FALLBACK_SITE_IDS.copy()

    if not sites:
        print("Error: no sites provided.", file=sys.stderr)
        return 1

    for site in sites:
        try:
            run_for_site(site)
        except subprocess.CalledProcessError as error:
            print(f"Pipeline failed for {site}: {error}", file=sys.stderr)
            return error.returncode or 1

    print("\nAll requested sites completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
