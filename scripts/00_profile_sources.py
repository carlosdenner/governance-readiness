"""
Data Source Profiling — ATLAS, AIID, EO 13960
Generates a comprehensive profile of each secondary data source
to evaluate research angles for the AMCIS 2026 paper.
"""

import pandas as pd
import yaml
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw"
OUT  = ROOT / "data" / "processed"


# ─────────────────────────────────────────────
# 1. MITRE ATLAS
# ─────────────────────────────────────────────
def profile_atlas():
    print("=" * 70)
    print("SOURCE 1: MITRE ATLAS — Adversarial Threat Landscape for AI Systems")
    print("=" * 70)

    atlas_dir = DATA / "atlas" / "data"

    # Tactics
    with open(atlas_dir / "tactics.yaml", encoding="utf-8") as f:
        tactics = yaml.safe_load(f)
    print(f"\n  Tactics: {len(tactics)}")
    for t in tactics:
        print(f"    {t['id']:12s}  {t['name']}")

    # Techniques
    with open(atlas_dir / "techniques.yaml", encoding="utf-8") as f:
        techniques = yaml.safe_load(f)
    parent_techniques = [t for t in techniques if "." not in t["id"].split("AML.T")[-1]]
    sub_techniques = [t for t in techniques if "." in t["id"].split("AML.T")[-1]]
    print(f"\n  Techniques: {len(techniques)} total ({len(parent_techniques)} parent, {len(sub_techniques)} sub)")

    # Mitigations
    with open(atlas_dir / "mitigations.yaml", encoding="utf-8") as f:
        mitigations = yaml.safe_load(f)
    print(f"  Mitigations: {len(mitigations)}")

    # Mitigation categories
    cats = Counter()
    ml_phases = Counter()
    for m in mitigations:
        for c in m.get("category", []):
            cats[c] += 1
        for p in m.get("ml-lifecycle", []):
            ml_phases[p] += 1
    print(f"\n  Mitigation categories:")
    for c, n in cats.most_common():
        print(f"    {c:30s} {n}")
    print(f"\n  ML lifecycle phases covered:")
    for p, n in ml_phases.most_common():
        print(f"    {p:40s} {n}")

    # Case studies
    cs_dir = atlas_dir / "case-studies"
    case_studies = []
    for fp in sorted(cs_dir.glob("*.yaml")):
        with open(fp, encoding="utf-8") as f:
            cs = yaml.safe_load(f)
            case_studies.append(cs)

    print(f"\n  Case Studies: {len(case_studies)}")

    # Case study types
    cs_types = Counter(cs.get("case-study-type", "unknown") for cs in case_studies)
    print(f"  Case study types:")
    for t, n in cs_types.most_common():
        print(f"    {t:20s} {n}")

    # Procedures per case study — average tactics used
    tactic_usage = Counter()
    tech_usage = Counter()
    for cs in case_studies:
        for step in cs.get("procedure", []):
            tactic_ref = step.get("tactic", "")
            tech_ref = step.get("technique", "")
            tactic_usage[tactic_ref] += 1
            tech_usage[tech_ref] += 1

    avg_steps = sum(len(cs.get("procedure", [])) for cs in case_studies) / max(len(case_studies), 1)
    print(f"  Average procedure steps per case study: {avg_steps:.1f}")

    return {
        "tactics": len(tactics),
        "techniques": len(techniques),
        "mitigations": len(mitigations),
        "case_studies": len(case_studies),
        "mitigation_categories": dict(cats),
        "ml_phases": dict(ml_phases),
    }


# ─────────────────────────────────────────────
# 2. AI Incident Database (AIID)
# ─────────────────────────────────────────────
def profile_aiid():
    print("\n" + "=" * 70)
    print("SOURCE 2: AI Incident Database (AIID)")
    print("=" * 70)

    snap = DATA / "aiid" / "mongodump_full_snapshot"

    # Incidents
    inc = pd.read_csv(snap / "incidents.csv", encoding="utf-8", on_bad_lines="skip")
    print(f"\n  Incidents: {len(inc):,}")
    print(f"  Columns: {list(inc.columns)}")

    # Date range
    inc["date"] = pd.to_datetime(inc["date"], errors="coerce")
    print(f"  Date range: {inc['date'].min():%Y-%m-%d} → {inc['date'].max():%Y-%m-%d}")

    # Reports
    rpt = pd.read_csv(snap / "reports.csv", encoding="utf-8", on_bad_lines="skip")
    print(f"\n  Reports: {len(rpt):,}")
    print(f"  Report columns: {list(rpt.columns)}")

    # Reports per incident
    reports_col = inc["reports"].dropna()
    # Parse the list strings
    def count_items(s):
        try:
            return len(json.loads(s.replace("'", '"'))) if isinstance(s, str) else 0
        except Exception:
            return str(s).count(",") + 1

    inc["report_count"] = inc["reports"].apply(count_items)
    print(f"\n  Reports per incident: mean={inc['report_count'].mean():.1f}, median={inc['report_count'].median():.0f}, max={inc['report_count'].max()}")

    # Top alleged deployers
    def flatten_field(series):
        items = []
        for val in series.dropna():
            for item in str(val).split(","):
                item = item.strip().strip("[]'\"")
                if item:
                    items.append(item)
        return Counter(items)

    deployers = flatten_field(inc["Alleged deployer of AI system"])
    print(f"\n  Top 15 alleged deployers:")
    for d, n in deployers.most_common(15):
        print(f"    {d:45s} {n}")

    developers = flatten_field(inc["Alleged developer of AI system"])
    print(f"\n  Top 15 alleged developers:")
    for d, n in developers.most_common(15):
        print(f"    {d:45s} {n}")

    harmed = flatten_field(inc["Alleged harmed or nearly harmed parties"])
    print(f"\n  Top 15 harmed parties:")
    for h, n in harmed.most_common(15):
        print(f"    {h:45s} {n}")

    # Classifications — CSET taxonomy
    cset = pd.read_csv(snap / "classifications_CSETv1.csv", encoding="utf-8", on_bad_lines="skip")
    print(f"\n  CSET v1 Classifications: {len(cset):,}")

    # Key CSET fields
    for col in ["Harm Domain", "Tangible Harm", "AI System", "Sector of Deployment",
                 "Autonomy Level", "Level of Automation"]:
        if col in cset.columns:
            vals = flatten_field(cset[col])
            print(f"\n  CSET '{col}' distribution:")
            for v, n in vals.most_common(10):
                print(f"    {v:50s} {n}")

    # GMF classifications
    gmf = pd.read_csv(snap / "classifications_GMF.csv", encoding="utf-8", on_bad_lines="skip")
    print(f"\n  GMF Classifications: {len(gmf):,}")
    print(f"  GMF columns: {list(gmf.columns)}")

    return {
        "incidents": len(inc),
        "reports": len(rpt),
        "cset_classifications": len(cset),
        "date_range": f"{inc['date'].min():%Y-%m-%d} to {inc['date'].max():%Y-%m-%d}",
    }


# ─────────────────────────────────────────────
# 3. EO 13960 Federal AI Use Case Inventory
# ─────────────────────────────────────────────
def profile_eo13960():
    print("\n" + "=" * 70)
    print("SOURCE 3: EO 13960 Federal AI Use Case Inventory (2024)")
    print("=" * 70)

    eo = pd.read_csv(
        DATA / "eo13960" / "2024_consolidated_ai_inventory_raw.csv",
        encoding="utf-8", on_bad_lines="skip"
    )
    print(f"\n  Use cases: {len(eo):,}")
    print(f"  Columns ({len(eo.columns)}): {list(eo.columns)}")

    # Agency distribution
    agency_counts = eo["3_agency"].value_counts()
    print(f"\n  Agencies represented: {eo['3_agency'].nunique()}")
    print(f"  Top 15 agencies by use case count:")
    for agency, n in agency_counts.head(15).items():
        print(f"    {agency:55s} {n}")

    # Topic area
    if "8_topic_area" in eo.columns:
        topic = eo["8_topic_area"].value_counts()
        print(f"\n  Topic areas:")
        for t, n in topic.items():
            print(f"    {t:55s} {n}")

    # Development stage
    if "16_dev_stage" in eo.columns:
        stage = eo["16_dev_stage"].value_counts()
        print(f"\n  Development stage:")
        for s, n in stage.items():
            print(f"    {s:55s} {n}")

    # Impact type (rights/safety)
    if "17_impact_type" in eo.columns:
        impact = eo["17_impact_type"].value_counts()
        print(f"\n  Impact type (rights/safety):")
        for i, n in impact.items():
            print(f"    {i:30s} {n}")

    # Governance-related fields completeness
    governance_cols = [
        "40_has_ato", "50_internal_review", "52_impact_assessment",
        "53_real_world_testing", "54_key_risks", "55_independent_eval",
        "56_monitor_postdeploy", "57_autonomous_impact",
        "59_ai_notice", "61_adverse_impact", "62_disparity_mitigation",
        "63_stakeholder_consult", "65_appeal_process", "67_opt_out"
    ]
    print(f"\n  Governance field completeness (% non-null):")
    for col in governance_cols:
        if col in eo.columns:
            pct = eo[col].notna().mean() * 100
            print(f"    {col:35s} {pct:5.1f}%")

    # Development method
    if "22_dev_method" in eo.columns:
        dev = eo["22_dev_method"].value_counts()
        print(f"\n  Development method:")
        for d, n in dev.items():
            print(f"    {d:55s} {n}")

    # PII
    if "29_contains_pii" in eo.columns:
        pii = eo["29_contains_pii"].value_counts()
        print(f"\n  Contains PII:")
        for p, n in pii.items():
            print(f"    {p:30s} {n}")

    # Infrastructure provisioned
    if "43_infra_provisioned" in eo.columns:
        infra = eo["43_infra_provisioned"].value_counts()
        print(f"\n  Infrastructure provisioned:")
        for i, n in infra.head(10).items():
            print(f"    {str(i):55s} {n}")

    # Extension requests (risk management compliance)
    if "51_extension_request" in eo.columns:
        ext = eo["51_extension_request"].value_counts()
        print(f"\n  Extension request (compliance with M-24-10 safeguards):")
        for e, n in ext.items():
            print(f"    {str(e):55s} {n}")

    return {
        "use_cases": len(eo),
        "agencies": eo["3_agency"].nunique(),
        "columns": len(eo.columns),
    }


# ─────────────────────────────────────────────
# 4. Cross-source angle analysis
# ─────────────────────────────────────────────
def angle_analysis(atlas_stats, aiid_stats, eo_stats):
    print("\n" + "=" * 70)
    print("CROSS-SOURCE ANGLE ANALYSIS")
    print("=" * 70)

    print("""
┌─────────────────────────────────────────────────────────────────────┐
│ ANGLE A: Trust/Governance Readiness (Threat → Gap → Safeguard)     │
│                                                                     │
│ ATLAS threats → AIID real-world failures → EO 13960 safeguards      │
│                                                                     │
│ Logic: Map ATLAS tactics to AIID incident categories to show what   │
│ threats materialize as real harms, then compare against EO 13960    │
│ governance safeguards to identify where federal agencies are/aren't │
│ prepared. Yields CIO competency gaps.                               │
│                                                                     │
│ Data support:                                                       │
│   ATLAS:  {atlas_cs} case studies, {atlas_mit} mitigations          │
│   AIID:   {aiid_inc:,} incidents with harm taxonomy                 │
│   EO:     14 governance fields across {eo_uc:,} use cases           │
│                                                                     │
│ Strength: Triangulation across threat / incident / practice data    │
│ Risk: Mapping between different taxonomies requires careful coding  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ ANGLE B: Integration/Architecture Readiness (Maturity Stages)      │
│                                                                     │
│ EO 13960 dev stages + infra + dev method → maturity patterns        │
│                                                                     │
│ Logic: Use the 62-variable EO 13960 dataset to model AI integration │
│ maturity across federal agencies. Cluster agencies by governance +   │
│ architecture readiness patterns. Relate to AIID harm types to show  │
│ which maturity gaps lead to incidents.                               │
│                                                                     │
│ Data support:                                                       │
│   EO:     {eo_uc:,} use cases × 62 variables × {eo_ag} agencies    │
│   AIID:   sector-specific incident data                             │
│                                                                     │
│ Strength: Large structured dataset, quantitative analysis possible  │
│ Risk: Federal-only scope may limit generalizability                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ ANGLE C: Combined Framework — CIO AI Orientation Enablers          │
│                                                                     │
│ Synthesize all 3 sources into a unified competency framework:       │
│                                                                     │
│ 1. ATLAS + AIID → Identify threat/incident landscape (what can go   │
│    wrong and what has gone wrong)                                    │
│ 2. EO 13960 → Identify governance & integration practices in use    │
│    (what orgs are actually doing)                                    │
│ 3. Gap analysis → Where practice doesn't match threat reality       │
│ 4. Map gaps to CIO competency dimensions                            │
│                                                                     │
│ This is the full thesis but may be too ambitious for one paper.      │
└─────────────────────────────────────────────────────────────────────┘
""".format(
        atlas_cs=atlas_stats["case_studies"],
        atlas_mit=atlas_stats["mitigations"],
        aiid_inc=aiid_stats["incidents"],
        eo_uc=eo_stats["use_cases"],
        eo_ag=eo_stats["agencies"],
    ))


def main():
    atlas_stats = profile_atlas()
    aiid_stats  = profile_aiid()
    eo_stats    = profile_eo13960()
    angle_analysis(atlas_stats, aiid_stats, eo_stats)


if __name__ == "__main__":
    main()
