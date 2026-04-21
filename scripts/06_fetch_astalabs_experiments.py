"""
06_fetch_astalabs_experiments.py
================================
Poll the AstaLabs AutoDiscovery API for experiment results and save
each batch to data/astalabs_exports/<timestamp>_experiments.json.

Usage:
    python scripts/06_fetch_astalabs_experiments.py

    # Custom run ID:
    python scripts/06_fetch_astalabs_experiments.py --run-id <UUID>

    # Poll until job completes (checks every 30s):
    python scripts/06_fetch_astalabs_experiments.py --poll

    # One-shot fetch:
    python scripts/06_fetch_astalabs_experiments.py --no-poll
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ── Configuration ───────────────────────────────────────────────────────────
# Extracted from the browser DevTools network tab
BASE_URL = "https://autodiscovery.allen.ai/api/runs"
USER_ID = "google-oauth2%7C110653986973039891181"
DEFAULT_RUN_ID = "c685374c-9ac3-43d2-83e7-061f30f747cd"

ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = ROOT / "data" / "astalabs_exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Auth token ──────────────────────────────────────────────────────────────
# AstaLabs uses Auth0 Bearer tokens (NOT cookies).
#
# To get your token:
#   1. Open DevTools → Network tab
#   2. Click any "experiments" request
#   3. In Request Headers, find: Authorization: Bearer eyJ...
#   4. Copy the token (everything after "Bearer ")
#   5. Set it:  $env:ASTALABS_TOKEN = "eyJ..."
#
#   OR: right-click a request → Copy → Copy as cURL, and find the
#       -H 'Authorization: Bearer eyJ...' part.
TOKEN_ENV = os.environ.get("ASTALABS_TOKEN", "")

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json",
    "Origin": "https://autodiscovery.allen.ai",
    "Referer": "https://autodiscovery.allen.ai/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
}


def build_session() -> requests.Session:
    """Create a requests session with auth token."""
    s = requests.Session()
    s.headers.update(HEADERS)

    if TOKEN_ENV:
        s.headers["Authorization"] = f"Bearer {TOKEN_ENV}"
    else:
        print(
            "⚠  No ASTALABS_TOKEN env var set.\n"
            "   The request may fail if auth is required.\n"
            "   Set it with:  $env:ASTALABS_TOKEN = 'eyJ...'\n"
            "   (Get it from DevTools → Network → any experiments request\n"
            "    → Request Headers → Authorization: Bearer <token>)\n"
        )
    return s


def fetch_experiments(
    session: requests.Session,
    run_id: str,
    cursor: str | None = None,
) -> dict:
    """Fetch one page of experiments from the API."""
    url = f"{BASE_URL}/{USER_ID}/{run_id}/experiments"

    # The API uses POST; body may carry pagination cursor
    payload = {}
    if cursor:
        payload["cursor"] = cursor

    resp = session.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def save_snapshot(data: dict, label: str = "experiments") -> Path:
    """Save a JSON snapshot with timestamp."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = EXPORT_DIR / f"{ts}_{label}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def fetch_all_experiments(session: requests.Session, run_id: str) -> dict:
    """
    Fetch all experiment pages, handling pagination if present.
    AstaLabs may use next_cursor, offset, page, or similar.
    """
    all_experiments = []
    cursor = None
    page = 0

    while True:
        page += 1
        print(f"  Fetching page {page}…", end=" ")
        data = fetch_experiments(session, run_id, cursor=cursor)

        experiments = data.get("experiments", [])
        all_experiments.extend(experiments)
        print(f"got {len(experiments)} experiments (total: {len(all_experiments)})")

        # Check for pagination — try common cursor patterns
        next_cursor = (
            data.get("next_cursor")
            or data.get("nextCursor")
            or data.get("cursor")
            or data.get("next_page_token")
        )

        if next_cursor and next_cursor != cursor:
            cursor = next_cursor
        else:
            break  # No more pages

    # Reassemble full response with all experiments merged
    data["experiments"] = all_experiments
    return data


def main():
    parser = argparse.ArgumentParser(description="Fetch AstaLabs experiment results")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="Run UUID")
    parser.add_argument(
        "--poll",
        action="store_true",
        default=True,
        help="Poll until has_job_completed is true (default)",
    )
    parser.add_argument(
        "--no-poll",
        dest="poll",
        action="store_false",
        help="Fetch once and exit",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Seconds between polls (default: 30)",
    )
    args = parser.parse_args()

    session = build_session()
    print(f"Run ID : {args.run_id}")
    print(f"Export : {EXPORT_DIR}\n")

    iteration = 0
    while True:
        iteration += 1
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Iteration {iteration}")

        try:
            data = fetch_all_experiments(session, args.run_id)
        except requests.HTTPError as e:
            print(f"  ✗ HTTP error: {e}")
            if not args.poll:
                raise
            print(f"  Retrying in {args.interval}s…")
            time.sleep(args.interval)
            continue

        n_exp = len(data.get("experiments", []))
        completed = data.get("has_job_completed", False)

        # Save snapshot
        path = save_snapshot(data)
        print(f"  ✓ Saved {n_exp} experiments → {path.name}")
        print(f"  Job completed: {completed}")

        # Print experiment summary
        for exp in data.get("experiments", []):
            eid = exp.get("experiment_id", "?")
            status = exp.get("status", "?")
            surprise = exp.get("surprise")
            s_str = f"{surprise:+.3f}" if surprise is not None else "  n/a "
            hyp = (exp.get("hypothesis", "") or "")[:80]
            print(f"    [{status:>9}] {eid:12s}  surprise={s_str}  {hyp}…")

        if not args.poll or completed:
            break

        print(f"\n  Waiting {args.interval}s before next poll…\n")
        time.sleep(args.interval)

    # Final consolidated save
    if iteration > 1:
        final_path = save_snapshot(data, "experiments_final")
        print(f"\n✅ Final consolidated save: {final_path.name}")

    print(f"\n✅ Done. {n_exp} experiments collected.")
    print(f"   Files in: {EXPORT_DIR}")


if __name__ == "__main__":
    main()
