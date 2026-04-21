#!/usr/bin/env python3
"""
Script 14 – ATLAS Threat Characterization & EO Governance Bridge
================================================================
Addresses reviewer weakness K: foregrounds the adversarial-threat leg
(ATLAS) in the Threat → Gap → Safeguard logic.

Outputs:
  1. Resolved tactic frequency distribution across 52 case studies
  2. Attack-chain transition matrix (tactic → tactic)
  3. Mitigation demand analysis (which ATLAS mitigations are most needed)
  4. ATLAS mitigation → EO 13960 analog bridge table (Table 6)

Reproducibility: deterministic; no random seed needed.
"""

import pathlib, re, yaml, glob, json
from collections import Counter

import pandas as pd

BASE = pathlib.Path(__file__).resolve().parents[1]
ATLAS_DIR = BASE / "data" / "raw" / "atlas" / "data"
OUT = BASE / "data" / "processed"


# ─── 1. Build template-reference → ID mappings ────────────────────────────────

def _build_anchor_map(yaml_path: pathlib.Path, id_prefix: str) -> dict:
    """Parse YAML anchors (&name) and their IDs from raw text."""
    raw = yaml_path.read_text(encoding="utf-8")
    pattern = rf"- &(\w+)\s*\n\s*id:\s*({re.escape(id_prefix)}\S+)"
    return {m.group(1): m.group(2) for m in re.finditer(pattern, raw)}


def _resolve_ref(ref: str, anchor_map: dict) -> str | None:
    """Resolve '{{name.id}}' → actual ID using anchor_map."""
    m = re.match(r"\{\{(\w+)\.id\}\}", ref)
    if m:
        return anchor_map.get(m.group(1))
    return None


# ─── 2. Load ATLAS data ───────────────────────────────────────────────────────

tactic_anchor = _build_anchor_map(ATLAS_DIR / "tactics.yaml", "AML.TA")
tech_anchor = _build_anchor_map(ATLAS_DIR / "techniques.yaml", "AML.T")

with open(ATLAS_DIR / "tactics.yaml", encoding="utf-8") as f:
    tactics_list = yaml.safe_load(f)
tactic_names = {t["id"]: t["name"] for t in tactics_list}

with open(ATLAS_DIR / "techniques.yaml", encoding="utf-8") as f:
    techs_list = yaml.safe_load(f)
tech_names = {t["id"]: t["name"] for t in techs_list}

with open(ATLAS_DIR / "mitigations.yaml", encoding="utf-8") as f:
    mits_list = yaml.safe_load(f)
mit_names = {m["id"]: m["name"] for m in mits_list}

# ─── 3. Mitigation → technique coverage ───────────────────────────────────────

mit_covers_tech: dict[str, set[str]] = {}
for m in mits_list:
    mid = m["id"]
    covered = set()
    for t in m.get("techniques", []):
        ref = t.get("id", "") if isinstance(t, dict) else str(t)
        tid = _resolve_ref(ref, tech_anchor)
        if tid:
            covered.add(tid)
    mit_covers_tech[mid] = covered

# ─── 4. Process case studies ──────────────────────────────────────────────────

case_files = sorted(glob.glob(str(ATLAS_DIR / "case-studies" / "*.yaml")))
all_tactic_mentions: list[str] = []
all_tech_mentions: list[str] = []
chain_transitions: Counter = Counter()
case_stats: list[dict] = []

for cf in case_files:
    with open(cf, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    case_tactics = set()
    case_techs = set()
    prev_tactic = None
    for step in doc.get("procedure", []):
        tac_id = _resolve_ref(step.get("tactic", ""), tactic_anchor)
        tech_id = _resolve_ref(step.get("technique", ""), tech_anchor)
        if tac_id:
            case_tactics.add(tac_id)
            all_tactic_mentions.append(tac_id)
        if tech_id:
            case_techs.add(tech_id)
            all_tech_mentions.append(tech_id)
        if prev_tactic and tac_id and prev_tactic != tac_id:
            chain_transitions[(prev_tactic, tac_id)] += 1
        if tac_id:
            prev_tactic = tac_id

    case_stats.append({
        "case_id": doc.get("id"),
        "n_tactics": len(case_tactics),
        "n_techniques": len(case_techs),
    })

n_cases = len(case_files)
tactic_counts = Counter(all_tactic_mentions)
tech_counts = Counter(all_tech_mentions)

# ─── 5. Mitigation demand ────────────────────────────────────────────────────

tech_set_used = set(all_tech_mentions)
mit_demand: dict[str, int] = {}
for mid, covered in mit_covers_tech.items():
    hits = sum(tech_counts.get(tid, 0) for tid in covered)
    if hits > 0:
        mit_demand[mid] = hits

# ─── 6. Bridge table: top ATLAS mitigations → EO analogs ─────────────────────
#
# The mapping below was constructed by the authors based on functional
# equivalence between ATLAS mitigation descriptions and EO 13960 inventory
# fields.  EO prevalence figures come from the scored dataset (n = 1,757).

BRIDGE = [
    {
        "atlas_id": "AML.M0024",
        "atlas_name": "AI Telemetry Logging",
        "demand": mit_demand.get("AML.M0024", 0),
        "eo_analog": "Post-deployment monitoring",
        "eo_field": "56_monitor_postdeploy",
        "eo_prevalence": "8.5%",
        "gap_note": "Logging prerequisite for monitoring; 91.5% of federal AI lacks it",
    },
    {
        "atlas_id": "AML.M0005",
        "atlas_name": "Access Control (Models/Data at Rest)",
        "demand": mit_demand.get("AML.M0005", 0),
        "eo_analog": "Code access + Data catalog",
        "eo_field": "38_code_access, 31_data_catalog",
        "eo_prevalence": "34–83%",
        "gap_note": "Vendor opacity reduces code access (OR=0.56); data catalog varies by agency",
    },
    {
        "atlas_id": "AML.M0015",
        "atlas_name": "Adversarial Input Detection",
        "demand": mit_demand.get("AML.M0015", 0),
        "eo_analog": "Real-world testing",
        "eo_field": "53_real_world_testing",
        "eo_prevalence": "8.5%",
        "gap_note": "Only 8.5% test in operational environment; adversarial testing even rarer",
    },
    {
        "atlas_id": "AML.M0003",
        "atlas_name": "Model Hardening",
        "demand": mit_demand.get("AML.M0003", 0),
        "eo_analog": "Impact assessment + Independent eval",
        "eo_field": "52_impact_assessment, 55_independent_eval",
        "eo_prevalence": "5.2–6.8%",
        "gap_note": "Hardening requires model internals access; blocked by vendor opacity (§4.5)",
    },
    {
        "atlas_id": "AML.M0033",
        "atlas_name": "I/O Validation for AI Agents",
        "demand": mit_demand.get("AML.M0033", 0),
        "eo_analog": "No direct EO analog",
        "eo_field": "—",
        "eo_prevalence": "—",
        "gap_note": "Agentic AI not covered by EO 13960 taxonomy; emerging gap",
    },
]


# ─── 7. Print results ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import statistics

    nt = [c["n_tactics"] for c in case_stats]
    print("=" * 72)
    print("ATLAS Threat Characterization  (n = %d case studies)" % n_cases)
    print("=" * 72)

    print("\n── Tactic Frequency ──")
    for tid, cnt in tactic_counts.most_common():
        pct = cnt / n_cases * 100
        print(f"  {tid}  {tactic_names[tid]:35s}  {cnt:3d}  ({pct:.0f}%)")

    print(f"\n  Tactics/case: mean={statistics.mean(nt):.1f}, "
          f"median={statistics.median(nt):.1f}, max={max(nt)}, min={min(nt)}")

    print("\n── Top 10 Attack-Chain Transitions ──")
    for (t1, t2), cnt in chain_transitions.most_common(10):
        print(f"  {tactic_names[t1]:30s} → {tactic_names[t2]:30s}  {cnt}")

    print("\n── Top 10 Mitigation Demand ──")
    for mid, demand in sorted(mit_demand.items(), key=lambda x: -x[1])[:10]:
        n_tech = len(mit_covers_tech[mid] & tech_set_used)
        print(f"  {mid}  {mit_names[mid]:45s}  demand={demand:3d}  ({n_tech} techs)")

    print("\n── ATLAS → EO Bridge Table ──")
    print(f"  {'ATLAS Mitigation':40s} {'Demand':>6s}  {'EO Analog':30s}  {'EO Prev':>8s}")
    print("  " + "─" * 90)
    for row in BRIDGE:
        print(f"  {row['atlas_name']:40s} {row['demand']:6d}  "
              f"{row['eo_analog']:30s}  {row['eo_prevalence']:>8s}")

    print("\nDone.")
