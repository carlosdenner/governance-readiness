"""
07_archive_astalabs_experiments.py
===================================
Parse AstaLabs AutoDiscovery experiment exports into a structured,
reproducible local archive.

For each experiment creates:
  experiments/EXP_NN/
    ├── experiment.json          (raw JSON for this experiment)
    ├── report.md                (human-readable full report)
    ├── code.py                  (executable Python code)
    ├── code_output.txt          (original execution output)
    └── figures/                 (regenerated figures, if code produces them)

Also creates:
  experiments/
    ├── summary.csv              (one row per experiment)
    ├── summary.md               (master narrative)
    └── tree.json                (hypothesis tree with parent-child relations)

Usage:
    python scripts/07_archive_astalabs_experiments.py
    python scripts/07_archive_astalabs_experiments.py --source data/astalabs_exports/20260222T013214Z_experiments.json
    python scripts/07_archive_astalabs_experiments.py --fetch   # fetch fresh from API first
    python scripts/07_archive_astalabs_experiments.py --rerun   # also re-execute code locally
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DIR = ROOT / "data" / "astalabs_exports"
ARCHIVE_DIR = ROOT / "data" / "astalabs_experiments"

# ── Helpers ─────────────────────────────────────────────────────────────────

def latest_export() -> Path:
    """Find the most recent experiments JSON export."""
    jsons = sorted(EXPORTS_DIR.glob("*_experiments*.json"))
    if not jsons:
        raise FileNotFoundError(f"No exports found in {EXPORTS_DIR}")
    return jsons[-1]


def belief_label(mean: float | None) -> str:
    """Convert a belief mean to a human-readable label."""
    if mean is None:
        return "Unknown"
    if mean >= 0.9:
        return "Definitely True"
    if mean >= 0.7:
        return "Likely True"
    if mean >= 0.55:
        return "Maybe True"
    if mean >= 0.45:
        return "Uncertain"
    if mean >= 0.3:
        return "Maybe False"
    if mean >= 0.1:
        return "Likely False"
    return "Definitely False"


def surprise_label(surprise: float | None) -> str:
    """Interpret surprise magnitude."""
    if surprise is None:
        return "N/A"
    if surprise > 0.3:
        return "Strong Positive (hypothesis strengthened)"
    if surprise > 0.05:
        return "Mild Positive (hypothesis somewhat supported)"
    if surprise > -0.05:
        return "Neutral (no significant belief shift)"
    if surprise > -0.3:
        return "Mild Negative (hypothesis weakened)"
    return "Strong Negative (hypothesis contradicted)"


def wrap(text: str | None, width: int = 80) -> str:
    """Wrap text for readability."""
    if not text:
        return ""
    return "\n".join(textwrap.fill(line, width=width) for line in text.split("\n"))


# ── Per-experiment archiver ─────────────────────────────────────────────────

def archive_experiment(exp: dict, idx: int, out_dir: Path) -> dict:
    """
    Archive a single experiment into its own directory.
    Returns a summary dict for the master CSV.
    """
    eid = exp.get("experiment_id", f"unknown_{idx}")
    exp_num = exp.get("id_in_run", idx)
    folder_name = f"EXP_{exp_num:03d}_{eid}"
    exp_dir = out_dir / folder_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Raw JSON ────────────────────────────────────────────────────
    (exp_dir / "experiment.json").write_text(
        json.dumps(exp, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── 2. Extract code ────────────────────────────────────────────────
    code = exp.get("code", "")
    if code:
        (exp_dir / "code.py").write_text(code, encoding="utf-8")

    # ── 3. Code output ─────────────────────────────────────────────────
    code_output = exp.get("code_output", "")
    if code_output:
        (exp_dir / "code_output.txt").write_text(code_output, encoding="utf-8")

    # ── 4. Rich outputs / figures ──────────────────────────────────────
    rich = exp.get("rich_outputs")
    if rich:
        fig_dir = exp_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        (fig_dir / "rich_outputs_raw.json").write_text(
            json.dumps(rich, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── 5. Build detailed report.md ────────────────────────────────────
    status = exp.get("status", "UNKNOWN")
    hypothesis = exp.get("hypothesis", "")
    analysis = exp.get("analysis", "")
    review = exp.get("review", "")
    surprise = exp.get("surprise")
    prior = exp.get("prior")
    posterior = exp.get("posterior")
    prior_belief = exp.get("prior_belief", {}) or {}
    posterior_belief = exp.get("posterior_belief", {}) or {}
    plan = exp.get("experiment_plan", {}) or {}
    parent_id = exp.get("parent_id", "")
    child_ids = exp.get("child_ids", [])
    created_at = exp.get("created_at", "")
    runtime_ms = exp.get("runtime_ms")
    is_surprising = exp.get("is_surprising", False)
    creation_idx = exp.get("creation_idx", "")

    prior_mean = prior_belief.get("mean", prior)
    post_mean = posterior_belief.get("mean", posterior)

    # Belief distribution details
    def belief_distribution_table(b: dict) -> str:
        if not b:
            return "N/A"
        rows = []
        for key in ["definitely_true", "maybe_true", "uncertain", "maybe_false", "definitely_false"]:
            val = b.get(key)
            if val is not None:
                label = key.replace("_", " ").title()
                rows.append(f"| {label} | {val} |")
        if not rows:
            return "N/A"
        return "| Category | Count |\n|---|---|\n" + "\n".join(rows)

    # Steps
    steps_text = plan.get("steps", "")
    if steps_text:
        # Format numbered steps properly
        steps_lines = re.split(r'(?<=\.)\s*(?=\d+\.)', steps_text)
        steps_md = "\n".join(f"- {s.strip()}" for s in steps_lines if s.strip())
    else:
        steps_md = "N/A"

    # Deliverables
    deliverables = plan.get("deliverables", "")
    if deliverables:
        deliv_items = re.split(r'(?<=\.)\s*(?=\d+\.)', deliverables)
        deliv_md = "\n".join(f"- {d.strip()}" for d in deliv_items if d.strip())
    else:
        deliv_md = "N/A"

    runtime_str = f"{runtime_ms/1000:.1f}s" if runtime_ms else "N/A"

    report = f"""# Experiment {exp_num}: {eid}

| Property | Value |
|---|---|
| **Experiment ID** | `{eid}` |
| **ID in Run** | {exp_num} |
| **Status** | {status} |
| **Created** | {created_at} |
| **Runtime** | {runtime_str} |
| **Parent** | `{parent_id}` |
| **Children** | {', '.join(f'`{c}`' for c in child_ids) if child_ids else 'None'} |
| **Creation Index** | {creation_idx} |

---

## Hypothesis

> {wrap(hypothesis)}

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | {f'{prior_mean:.4f}' if prior_mean is not None else 'N/A'} ({belief_label(prior_mean)}) |
| **Posterior** | {f'{post_mean:.4f}' if post_mean is not None else 'N/A'} ({belief_label(post_mean)}) |
| **Surprise** | {f'{surprise:+.4f}' if surprise is not None else 'N/A'} |
| **Surprise Interpretation** | {surprise_label(surprise)} |
| **Is Surprising?** | {'Yes' if is_surprising else 'No'} |

### Prior Belief Distribution
{belief_distribution_table(prior_belief)}

### Posterior Belief Distribution
{belief_distribution_table(posterior_belief)}

---

## Experiment Plan

**Objective:** {plan.get('objective', 'N/A')}

### Steps
{steps_md}

### Deliverables
{deliv_md}

---

## Analysis

{wrap(analysis) if analysis else 'N/A'}

---

## Review

{wrap(review) if review else 'N/A'}

---

## Code

```python
{code if code else '# No code available'}
```

## Code Output

```
{code_output if code_output else 'No output available'}
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
"""

    (exp_dir / "report.md").write_text(report, encoding="utf-8")

    # ── 6. Return summary row ─────────────────────────────────────────
    return {
        "exp_num": exp_num,
        "experiment_id": eid,
        "status": status,
        "hypothesis": hypothesis,
        "prior": round(prior_mean, 4) if prior_mean is not None else None,
        "prior_label": belief_label(prior_mean),
        "posterior": round(post_mean, 4) if post_mean is not None else None,
        "posterior_label": belief_label(post_mean),
        "surprise": round(surprise, 4) if surprise is not None else None,
        "surprise_label": surprise_label(surprise),
        "is_surprising": is_surprising,
        "parent_id": parent_id,
        "n_children": len(child_ids),
        "runtime_s": round(runtime_ms / 1000, 1) if runtime_ms else None,
        "created_at": created_at,
        "objective": plan.get("objective", ""),
        "analysis_snippet": (analysis or "")[:200],
        "folder": folder_name,
    }


# ── Re-run experiment code locally ──────────────────────────────────────────

def rerun_experiment(exp_dir: Path) -> bool:
    """Re-execute experiment code.py locally to regenerate figures."""
    code_path = exp_dir / "code.py"
    if not code_path.exists():
        return False

    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Inject matplotlib savefig override so figures are saved to disk
    wrapper = f'''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_orig_show = plt.show
_fig_counter = [0]

def _save_show(*args, **kwargs):
    _fig_counter[0] += 1
    fig = plt.gcf()
    fig.savefig(r"{fig_dir / "figure_"}" + str(_fig_counter[0]) + ".png",
                dpi=150, bbox_inches="tight")
    print(f"[SAVED] figure_{{_fig_counter[0]}}.png")

plt.show = _save_show

# Change working directory to where the data lives
import os
os.chdir(r"{ROOT / "data" / "astalabs"}")

# Execute original code
exec(open(r"{code_path}").read())

# Save any remaining figures
for i in plt.get_fignums():
    _fig_counter[0] += 1
    plt.figure(i).savefig(
        r"{fig_dir / "figure_"}" + str(_fig_counter[0]) + ".png",
        dpi=150, bbox_inches="tight"
    )
    print(f"[SAVED] figure_{{_fig_counter[0]}}.png (remaining)")
'''

    wrapper_path = exp_dir / "_runner.py"
    wrapper_path.write_text(wrapper, encoding="utf-8")

    try:
        result = subprocess.run(
            [sys.executable, str(wrapper_path)],
            capture_output=True, text=True, timeout=120,
            cwd=str(ROOT / "data" / "astalabs"),
        )
        (exp_dir / "rerun_output.txt").write_text(
            result.stdout + "\n--- STDERR ---\n" + result.stderr,
            encoding="utf-8",
        )
        # List generated figures
        figs = list(fig_dir.glob("*.png"))
        if figs:
            print(f"    Generated {len(figs)} figure(s)")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        (exp_dir / "rerun_output.txt").write_text(
            "TIMEOUT: Code execution exceeded 120s", encoding="utf-8"
        )
        return False
    except Exception as e:
        (exp_dir / "rerun_output.txt").write_text(
            f"ERROR: {e}", encoding="utf-8"
        )
        return False
    finally:
        wrapper_path.unlink(missing_ok=True)


# ── Master summary generators ──────────────────────────────────────────────

def write_summary_csv(summaries: list[dict], out_dir: Path):
    """Write summary.csv with one row per experiment."""
    path = out_dir / "summary.csv"
    if not summaries:
        return
    keys = summaries[0].keys()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"  Summary CSV: {path}")


def write_summary_md(summaries: list[dict], meta: dict, out_dir: Path):
    """Write a master summary markdown."""
    path = out_dir / "summary.md"

    # Statistics
    succeeded = [s for s in summaries if s["status"] == "SUCCEEDED"]
    failed = [s for s in summaries if s["status"] == "FAILED"]
    surprising = [s for s in summaries if s.get("is_surprising")]

    confirmed = [s for s in succeeded if (s.get("surprise") or 0) > 0.05]
    rejected = [s for s in succeeded if (s.get("surprise") or 0) < -0.05]
    neutral = [s for s in succeeded
               if -0.05 <= (s.get("surprise") or 0) <= 0.05]

    md = f"""# AstaLabs AutoDiscovery – Experiment Archive

**Run ID:** `{meta.get('runid', 'unknown')}`
**Job Completed:** {meta.get('has_job_completed', False)}
**Archive Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Total Experiments:** {len(summaries)}

## Statistics

| Metric | Count |
|---|---|
| Succeeded | {len(succeeded)} |
| Failed | {len(failed)} |
| Surprising | {len(surprising)} |
| Hypotheses Supported (surprise > +0.05) | {len(confirmed)} |
| Hypotheses Contradicted (surprise < -0.05) | {len(rejected)} |
| Neutral (|surprise| ≤ 0.05) | {len(neutral)} |

## Experiment Index

| # | ID | Status | Prior | Post | Surprise | Hypothesis |
|---|---|---|---|---|---|---|
"""
    for s in sorted(summaries, key=lambda x: x.get("exp_num", 0)):
        hyp_short = (s.get("hypothesis", "") or "")[:80]
        sur = f"{s['surprise']:+.3f}" if s.get("surprise") is not None else "N/A"
        pri = f"{s['prior']:.3f}" if s.get("prior") is not None else "N/A"
        pos = f"{s['posterior']:.3f}" if s.get("posterior") is not None else "N/A"
        md += f"| {s['exp_num']} | [{s['experiment_id']}]({s['folder']}/report.md) "
        md += f"| {s['status']} | {pri} | {pos} | {sur} | {hyp_short}… |\n"

    # Highlight key findings
    if confirmed:
        md += "\n## Hypotheses Supported\n\n"
        for s in sorted(confirmed, key=lambda x: -(x.get("surprise") or 0)):
            md += f"- **{s['experiment_id']}** (surprise={s['surprise']:+.3f}): "
            md += f"{(s.get('hypothesis','') or '')[:120]}…\n"

    if rejected:
        md += "\n## Hypotheses Contradicted\n\n"
        for s in sorted(rejected, key=lambda x: (x.get("surprise") or 0)):
            md += f"- **{s['experiment_id']}** (surprise={s['surprise']:+.3f}): "
            md += f"{(s.get('hypothesis','') or '')[:120]}…\n"

    md += f"\n---\n\n*Generated by `07_archive_astalabs_experiments.py`*\n"

    path.write_text(md, encoding="utf-8")
    print(f"  Summary MD:  {path}")


def write_tree(experiments: list[dict], meta: dict, out_dir: Path):
    """Write hypothesis tree showing parent-child relationships."""
    tree = {
        "runid": meta.get("runid"),
        "has_job_completed": meta.get("has_job_completed"),
        "nodes": {},
    }
    for exp in experiments:
        eid = exp.get("experiment_id", "")
        tree["nodes"][eid] = {
            "id_in_run": exp.get("id_in_run"),
            "parent_id": exp.get("parent_id"),
            "child_ids": exp.get("child_ids", []),
            "status": exp.get("status"),
            "surprise": exp.get("surprise"),
            "prior": exp.get("prior"),
            "posterior": exp.get("posterior"),
            "hypothesis": (exp.get("hypothesis", "") or "")[:150],
        }
    path = out_dir / "tree.json"
    path.write_text(json.dumps(tree, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Tree JSON:   {path}")


# ── Fetch fresh data ───────────────────────────────────────────────────────

def fetch_fresh() -> Path:
    """Run the fetch script to get latest data, return path."""
    # Import and run
    fetch_script = ROOT / "scripts" / "06_fetch_astalabs_experiments.py"
    result = subprocess.run(
        [sys.executable, str(fetch_script), "--no-poll"],
        capture_output=True, text=True, timeout=60,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Warning: fetch returned {result.returncode}")
        print(result.stderr)
    return latest_export()


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Archive AstaLabs experiments into structured local files"
    )
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Path to experiments JSON. Default: latest in exports dir.",
    )
    parser.add_argument(
        "--fetch", action="store_true",
        help="Fetch fresh data from API before archiving.",
    )
    parser.add_argument(
        "--rerun", action="store_true",
        help="Re-execute experiment code locally to regenerate figures.",
    )
    parser.add_argument(
        "--out", type=Path, default=ARCHIVE_DIR,
        help=f"Output directory. Default: {ARCHIVE_DIR}",
    )
    args = parser.parse_args()

    # Find or fetch source
    if args.fetch:
        print("── Fetching fresh data from AstaLabs…")
        source = fetch_fresh()
    elif args.source:
        source = args.source
    else:
        source = latest_export()

    print(f"\n── Source: {source.name}")
    with open(source, encoding="utf-8") as f:
        data = json.load(f)

    experiments = data.get("experiments", [])
    meta = {k: v for k, v in data.items() if k != "experiments"}

    print(f"── Experiments: {len(experiments)}")
    print(f"── Job completed: {meta.get('has_job_completed', '?')}")
    print(f"── Output: {args.out}\n")

    args.out.mkdir(parents=True, exist_ok=True)

    # Archive each experiment
    summaries = []
    for i, exp in enumerate(experiments):
        eid = exp.get("experiment_id", f"unknown_{i}")
        exp_num = exp.get("id_in_run", i)
        status = exp.get("status", "?")
        print(f"  [{i+1:3d}/{len(experiments)}] {eid:20s} ({status})", end="")

        summary = archive_experiment(exp, i, args.out)
        summaries.append(summary)

        if args.rerun and status == "SUCCEEDED" and exp.get("code"):
            print(" → re-running…", end="")
            ok = rerun_experiment(args.out / summary["folder"])
            print(" ✓" if ok else " ✗", end="")

        print()

    # Write summaries
    print("\n── Writing summaries…")
    write_summary_csv(summaries, args.out)
    write_summary_md(summaries, meta, args.out)
    write_tree(experiments, meta, args.out)

    # Save metadata
    meta_path = args.out / "run_metadata.json"
    meta["archive_date"] = datetime.now(timezone.utc).isoformat()
    meta["source_file"] = source.name
    meta["n_experiments"] = len(experiments)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Stats
    ok = sum(1 for s in summaries if s["status"] == "SUCCEEDED")
    fail = sum(1 for s in summaries if s["status"] == "FAILED")
    sur_pos = sum(1 for s in summaries if (s.get("surprise") or 0) > 0.05)
    sur_neg = sum(1 for s in summaries if (s.get("surprise") or 0) < -0.05)

    print(f"\n{'='*50}")
    print(f"✅ Archived {len(summaries)} experiments")
    print(f"   Succeeded: {ok}  |  Failed: {fail}")
    print(f"   Supported: {sur_pos}  |  Contradicted: {sur_neg}")
    print(f"   Output: {args.out}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
