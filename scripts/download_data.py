"""
download_data.py — Fetch public source datasets for the replication package.

Downloads the three publicly available datasets used in the paper:
  1. EO 13960 Federal AI Use Case Inventory (ai.gov)
  2. AI Incident Database incidents (AIID API / GitHub)
  3. MITRE ATLAS case studies (atlas.mitre.org)

Usage:
    python scripts/download_data.py

Output:
    data/raw/eo13960/   — EO 13960 inventory CSV
    data/raw/aiid/      — AIID incidents JSON
    data/raw/atlas/     — ATLAS case studies YAML/JSON
"""

import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW  = ROOT / "data" / "raw"

# ── Source URLs ───────────────────────────────────────────────────────────────
SOURCES = {
    "eo13960": {
        "dir": RAW / "eo13960",
        "files": [
            {
                "url": "https://ai.gov/wp-content/uploads/2024/11/2024_consolidated_ai_inventory_raw.csv",
                "filename": "2024_consolidated_ai_inventory_raw.csv",
                "description": "EO 13960 Federal AI Use Case Inventory (2024 consolidated)",
            }
        ],
    },
    "aiid": {
        "dir": RAW / "aiid",
        "files": [
            {
                "url": "https://raw.githubusercontent.com/responsible-ai-collaborative/aiid/main/site/gatsby-site/api-server/openapi.yaml",
                "filename": "README_API.yaml",
                "description": "AI Incident Database API schema (see note below for full export)",
            }
        ],
        "note": (
            "The full AIID incident corpus is available at https://incidentdatabase.ai/research/snapshots/. "
            "Download the latest JSON snapshot and place it in data/raw/aiid/incidents.json."
        ),
    },
    "atlas": {
        "dir": RAW / "atlas",
        "files": [
            {
                "url": "https://raw.githubusercontent.com/mitre-atlas/atlas-data/main/dist/ATLAS.yaml",
                "filename": "ATLAS.yaml",
                "description": "MITRE ATLAS adversarial threat knowledge base (full YAML)",
            }
        ],
    },
}


def download(url: str, dest: Path, description: str) -> bool:
    """Download a file with progress reporting. Returns True on success."""
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return True
    print(f"  Downloading {description} ...")
    print(f"    from: {url}")
    print(f"    to:   {dest}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "governance-readiness-replication/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 65536
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                out.write(block)
                downloaded += len(block)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r    {downloaded:,} / {total:,} bytes ({pct:.0f}%)", end="", flush=True)
            print()
        print(f"    OK ({dest.stat().st_size:,} bytes)")
        return True
    except urllib.error.HTTPError as e:
        print(f"    FAILED: HTTP {e.code} — {e.reason}")
        print(f"    Please download manually from: {url}")
        return False
    except Exception as e:
        print(f"    FAILED: {e}")
        print(f"    Please download manually from: {url}")
        return False


def main() -> None:
    print("=" * 60)
    print("governance-readiness: data download")
    print("=" * 60)
    print()

    success_count = 0
    skip_count = 0
    fail_count = 0

    for source_name, source in SOURCES.items():
        dest_dir: Path = source["dir"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{source_name.upper()}]")

        if "note" in source:
            print(f"  NOTE: {source['note']}")

        for f in source["files"]:
            dest = dest_dir / f["filename"]
            if dest.exists():
                skip_count += 1
                print(f"  [skip] {f['filename']} already exists ({dest.stat().st_size:,} bytes)")
            else:
                ok = download(f["url"], dest, f["description"])
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
        print()

    # Summary
    print("-" * 60)
    print(f"Done — {success_count} downloaded, {skip_count} skipped, {fail_count} failed")
    print()
    if fail_count > 0:
        print("For failed downloads, see manual instructions above.")
        print("The analysis scripts in scripts/ will fail until all raw data is present.")
        sys.exit(1)
    else:
        print("All data files are present. You can now run the analysis pipeline:")
        print("  python scripts/02_prepare_datasets.py")
        print("  python scripts/09_pathway_model.py")


if __name__ == "__main__":
    main()
