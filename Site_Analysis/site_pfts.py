"""
Load site-to-PFT mapping from Sentinel_Sites_Table PFTS sheet.
Matches sites by CMT# or Site Reference columns.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

_EXCEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "Sentinel_Sites_Table_with_Soil_and_Descriptions_IKW.xlsx",
)
_PFTS_SHEET = "PFTS"

# Hardcoded fallback when Excel/openpyxl unavailable (e.g. Python 3.7)
FALLBACK_MAPPING: Dict[str, List[str]] = {
    "CA_OBS": ["EverTree", "DecidTree", "Shrub", "Sphag", "Feather", "Lichen"],
    "CMT02": ["EverTree", "DecidShrub", "DecidTree", "Moss"],
    "CMT04": ["Salix", "Betula", "DecidTree", "EverTree", "Sedge", "Forbs", "Grasses", "Lichen", "Feather"],
    "CMT05": ["Betula", "DecidTree", "EverTree", "Sedge", "Forbs", "Lichen", "Feather", "Sphag"],
    "CMT06": ["DecidShrub", "Sedge", "Grasses", "Forbs", "Lichen", "Feather", "Sphag"],
    "CMT07": ["DecidShrub", "EverTree", "Forbs", "Lichen", "Grasses", "Moss"],
    "CMT20": ["Betnan", "Carex", "Ericoid", "Feather", "Lichen", "OthMoss", "Rubcha"],
    "CMT44": ["Salix", "Betula", "DecidShrub", "EverShrub", "Sedge", "Forbs", "Grasses", "Lichen", "Feather"],
    "CMT65": ["DecidTree", "DecidShrub"],
    "CMT67": ["DecidTree", "EverTree", "Moss", "Forbs"],
    "CMT70": ["DecidShrub", "EverShrub", "Gram", "Othmoss"],
    "CMT71_Yakutsk": ["Larch", "DecidTree", "Shrub", "Gram"],
    "CMT73": ["Sedge", "Shrub", "Moss"],
    "CMT74": ["EverTree", "Lichen", "DwarfShub", "Moss"],
    "CMT75": ["EverShrub", "DecidShrub", "Moss", "Lichem", "Gram"],
    "CMT76": ["DecidShrub", "Betula", "Salix", "Gram", "SphagMoss"],
    "CMT82": ["ScotPine", "Spruce", "EverShrub", "Orthmoss", "Forbs"],
    "CMT90": ["DwarfShub", "Sedge", "Cryptogam"],
    "CMT92": ["EverShrub", "DecidShrub", "Gram", "Lichen", "Moss"],
    "CMT93": ["Orthmoss", "Salix", "Forbs", "Grasses"],
    "DL55": ["Sphag", "EverShrub"],
    "DL56": ["Lichen", "Sphag", "EverShrub", "DecidShrub", "Sedge"],
    "DL57": ["Lichen", "Othmoss", "EverShrub", "Gram", "DecidShrub"],
    "EML21": ["DecidShrub", "EverShrub", "Sedge", "Forbs", "Lichen", "Othmoss", "Sphag"],
    "MD1": ["EverTree", "DecidShrub", "DecidTree", "Moss"],
    "MD3": ["EverTree", "DecidShrub", "DecidTree", "Moss"],
    "OJP": ["EverTree", "DecidShrub", "Lichen", "Feather"],
    "SCB": ["Sphag", "EverTree", "Sedge", "EricShrub", "Forbs"],
    "SCC": ["EverTree", "EricShrub", "Lichen"],
    "TVC50": ["DecidShrub", "EverShrub", "Forbs", "Sedge", "Moss", "Lichen"],
    "TVC51": ["Sedge", "Sphag", "Lichen"],
    "TVC52": ["Lichen"],
    "US-Prr": ["BlacSprus", "Moss", "Shrub", "Lichen", "Sedge"],
}


def _load_pft_mapping() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Load CMT and Site Reference -> PFTs from Excel PFTS sheet."""
    if not os.path.isfile(_EXCEL_PATH):
        return {}, {}
    try:
        df = pd.read_excel(_EXCEL_PATH, sheet_name=_PFTS_SHEET, engine="openpyxl")
    except Exception:
        return {}, {}
    pft_cols = [c for c in df.columns if str(c).startswith("Unnamed")]

    def get_pfts(row: pd.Series) -> List[str]:
        pfts = []
        for c in pft_cols:
            v = row[c]
            if pd.notna(v) and str(v).strip():
                pfts.append(str(v).strip())
        return pfts

    def normalize(s: str) -> str:
        return str(s).replace("-", "_").replace(" ", "").upper()

    cmt_to_pfts: Dict[str, List[str]] = {}
    ref_to_pfts: Dict[str, List[str]] = {}

    for _, row in df.iterrows():
        pfts = get_pfts(row)
        if not pfts:
            continue

        cmt = row.get("CMT #")
        if pd.notna(cmt) and str(cmt).strip():
            cmt_clean = str(cmt).strip().split()[0]  # "CMT77 (TBD)" -> "CMT77"
            cmt_to_pfts[cmt_clean] = pfts

        site_ref = row.get("Site Reference")
        if pd.notna(site_ref) and str(site_ref).strip():
            ref_norm = normalize(str(site_ref))
            ref_to_pfts[ref_norm] = pfts
            # Also map suffix (e.g. OJP from CA-Ojp, SCB from CA-SCB)
            if "-" in str(site_ref):
                suffix = str(site_ref).split("-")[-1]
                ref_to_pfts[normalize(suffix)] = pfts

    return cmt_to_pfts, ref_to_pfts


_CMT_TO_PFTS: Optional[Dict[str, List[str]]] = None
_REF_TO_PFTS: Optional[Dict[str, List[str]]] = None


def _ensure_loaded() -> None:
    global _CMT_TO_PFTS, _REF_TO_PFTS
    if _CMT_TO_PFTS is None:
        _CMT_TO_PFTS, _REF_TO_PFTS = _load_pft_mapping()
        # Use hardcoded fallback when Excel failed to load
        if not _CMT_TO_PFTS and not _REF_TO_PFTS:
            _CMT_TO_PFTS = FALLBACK_MAPPING.copy()


# Override Excel/fallback PFT lists for specific sites (checked first in get_site_pfts).
SITE_PFT_OVERRIDES: Dict[str, List[str]] = {
    "CMT20": ["Betnan", "Carex", "Ericoid", "Feather", "Lichen", "OthMoss", "Rubcha"],
}


def _pft_sort_key(pft_id: str) -> int:
    m = re.search(r"pft([0-9]+)", pft_id)
    return int(m.group(1)) if m else 999


def get_limited_pft_list(site_folder: str, pft_indices: List[str]) -> List[str]:
    """Return pft indices truncated to the number of PFTs defined for the site."""
    pfts = get_site_pfts(site_folder)
    ordered = sorted(pft_indices, key=_pft_sort_key)
    if pfts:
        return ordered[: len(pfts)]
    return ordered


def _pft_id_from_column(column: str) -> Optional[str]:
    m = re.search(r"(pft[0-9]+)", column)
    return m.group(1) if m else None


def filter_columns_for_site(site_folder: str, columns: List[str], allowed_pfts: List[str]) -> List[str]:
    """Keep only model columns belonging to the allowed PFT indices."""
    allowed = set(allowed_pfts)
    return [c for c in columns if _pft_id_from_column(c) in allowed]


def get_site_pfts(site_folder: str) -> Optional[List[str]]:
    """
    Return PFT names for a site folder, or None if not found.
    Matches by CMT# or Site Reference from the Excel PFTS sheet.
    """
    if site_folder in SITE_PFT_OVERRIDES:
        return SITE_PFT_OVERRIDES[site_folder]

    _ensure_loaded()

    # Direct lookup (works for both Excel and hardcoded fallback)
    if site_folder in _CMT_TO_PFTS:
        return _CMT_TO_PFTS[site_folder]

    # CMT match (CMT71_Yakutsk -> CMT71)
    if site_folder.startswith("CMT"):
        cmt_part = site_folder.split("_")[0]
        if cmt_part in SITE_PFT_OVERRIDES:
            return SITE_PFT_OVERRIDES[cmt_part]
        if cmt_part in _CMT_TO_PFTS:
            return _CMT_TO_PFTS[cmt_part]

    # Site Reference (CA_OBS -> CA-OBS, OJP -> CA-Ojp suffix)
    ref_norm = str(site_folder).replace("_", "-").replace(" ", "").upper()
    if ref_norm in _REF_TO_PFTS:
        return _REF_TO_PFTS[ref_norm]
    ref_norm_alt = str(site_folder).replace("-", "_").replace(" ", "").upper()
    if ref_norm_alt in _REF_TO_PFTS:
        return _REF_TO_PFTS[ref_norm_alt]

    # TVC50, TVC51, TVC52 -> CMT50, CMT51, CMT52
    if site_folder.startswith("TVC"):
        num = site_folder[3:]
        if num.isdigit():
            cmt = "CMT" + num
            if cmt in _CMT_TO_PFTS:
                return _CMT_TO_PFTS[cmt]

    # DL55, DL56, DL57 -> CMT55, CMT56, CMT57
    if site_folder.startswith("DL") and len(site_folder) >= 4:
        num = site_folder[2:]
        if num.isdigit():
            cmt = "CMT" + num
            if cmt in _CMT_TO_PFTS:
                return _CMT_TO_PFTS[cmt]

    # EML21 -> CMT21
    if site_folder == "EML21" and "CMT21" in _CMT_TO_PFTS:
        return _CMT_TO_PFTS["CMT21"]

    return None


def build_fallback_sites(sites_dir: str = "sites_runs") -> List[Tuple[str, List[str]]]:
    """
    Build FALLBACK_SITES as [(site_id, [pft0, pft1, ...]), ...]
    for all site folders that have a PFT mapping.
    """
    _ensure_loaded()
    base = os.path.dirname(__file__)
    sites_path = os.path.join(base, sites_dir)
    if not os.path.isdir(sites_path):
        return []

    result = []
    for name in sorted(os.listdir(sites_path)):
        path = os.path.join(sites_path, name)
        if not os.path.isdir(path):
            continue
        pfts = get_site_pfts(name)
        if pfts:
            result.append((name, pfts))
    return result


def get_fallback_sites_list() -> List[str]:
    """Return flat list of site IDs for run_sites_pipeline (backward compat)."""
    return [s[0] for s in build_fallback_sites()]


# Pre-built FALLBACK_SITES with PFTs for use in scripts.
# Format: [(site_id, [EverTree, DecidTree, ...]), ...]
FALLBACK_SITES: List[Tuple[str, List[str]]] = []


def _init_fallback() -> None:
    global FALLBACK_SITES
    if not FALLBACK_SITES:
        FALLBACK_SITES = build_fallback_sites()


def format_fallback_sites_python() -> str:
    """
    Return FALLBACK_SITES as a Python list string for copy-paste:
    [("CA_OBS", ["EverTree", "DecidTree", ...]), ...]
    """
    _ensure_loaded()
    items = build_fallback_sites()
    lines = ["FALLBACK_SITES = ["]
    for site_id, pfts in items:
        pft_str = ", ".join(f'"{p}"' for p in pfts)
        lines.append(f'    ("{site_id}", [{pft_str}]),')
    lines.append("]")
    return "\n".join(lines)


def get_pft_names_for_site(site_folder: str, pft_indices: List[str]) -> List[str]:
    """
    Map pft0, pft1, ... to actual PFT names for a site.
    Truncates to the number of PFTs defined for the site when model data has extra slots.
    """
    pfts = get_site_pfts(site_folder)
    ordered = sorted(pft_indices, key=_pft_sort_key)
    if not pfts:
        return ordered
    ordered = ordered[: len(pfts)]
    return [pfts[i] for i in range(len(ordered))]


if __name__ == "__main__":
    print(format_fallback_sites_python())
