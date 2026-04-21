"""
prepare_datasets.py
===================
Creates analysis-ready, merged datasets from the three raw data sources:

  1. aiid_incidents_classified.csv  — AIID incidents joined with CSET + GMF
     classification taxonomies (1 row per incident, harm + failure fields)
  2. atlas_cases_enriched.csv       — ATLAS case studies with tactic / technique
     / mitigation lists exploded + McKinsey constraint tags
  3. eo13960_scored.csv             — EO 13960 use cases with a governance
     readiness score (Tier-1 + Tier-2) and McKinsey constraint coverage flags
  4. unified_evidence_base.csv      — Long-format table that stacks evidence
     rows from all three sources for cross-source analysis
"""

from __future__ import annotations
import pathlib
import re
import yaml
import pandas as pd
import numpy as np

BASE = pathlib.Path(__file__).resolve().parent.parent
AIID = BASE / "data" / "raw" / "aiid" / "mongodump_full_snapshot"
ATLAS = BASE / "data" / "raw" / "atlas" / "data"
EO = BASE / "data" / "raw" / "eo13960"
OUTPUT = BASE / "data" / "processed"
OUTPUT.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  1.  AIID — merge incidents + CSET + GMF classifications
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_aiid():
    print("─── AIID: loading & merging ───")
    inc = pd.read_csv(AIID / "incidents.csv")
    cset = pd.read_csv(AIID / "classifications_CSETv1.csv")
    gmf  = pd.read_csv(AIID / "classifications_GMF.csv")

    # Normalise incident ID column names
    inc.rename(columns={"incident_id": "incident_id"}, inplace=True)
    cset.rename(columns={"Incident Number": "incident_id"}, inplace=True)
    gmf.rename(columns={"Incident ID": "incident_id"}, inplace=True)

    # Select useful CSET columns
    cset_cols = [
        "incident_id",
        "Harm Domain", "Tangible Harm", "AI Harm Level",
        "Harm Distribution Basis", "Special Interest Intangible Harm",
        "Sector of Deployment", "Infrastructure Sectors",
        "Public Sector Deployment", "Autonomy Level",
        "Lives Lost", "Intentional Harm",
    ]
    cset_cols = [c for c in cset_cols if c in cset.columns]
    cset_sel = cset[cset_cols].copy()
    # Deduplicate — keep first annotation per incident
    cset_sel = cset_sel.drop_duplicates(subset="incident_id", keep="first")

    # Select useful GMF columns
    gmf_cols = [
        "incident_id",
        "Known AI Goal", "Known AI Technology",
        "Known AI Technical Failure",
        "Potential AI Technology", "Potential AI Technical Failure",
    ]
    gmf_cols = [c for c in gmf_cols if c in gmf.columns]
    gmf_sel = gmf[gmf_cols].copy()
    gmf_sel = gmf_sel.drop_duplicates(subset="incident_id", keep="first")

    # Merge
    merged = inc.merge(cset_sel, on="incident_id", how="left")
    merged = merged.merge(gmf_sel, on="incident_id", how="left")

    # Add helper columns
    merged["has_cset"] = merged["Harm Domain"].notna()
    merged["has_gmf"]  = merged["Known AI Technical Failure"].notna()
    merged["year"] = pd.to_datetime(merged["date"], errors="coerce").dt.year

    out = OUTPUT / "aiid_incidents_classified.csv"
    merged.to_csv(out, index=False)
    print(f"  Rows: {len(merged)}")
    print(f"  With CSET: {merged['has_cset'].sum()}")
    print(f"  With GMF:  {merged['has_gmf'].sum()}")
    print(f"  → {out.name}")
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
#  2.  ATLAS — enrich case studies with tactic/technique/mitigation details
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_atlas():
    print("\n─── ATLAS: loading case studies ───")
    cs_dir = ATLAS / "case-studies"
    cases = []
    for yf in sorted(cs_dir.glob("*.yaml")):
        with open(yf) as f:
            doc = yaml.safe_load(f)
        case_id = doc.get("id", yf.stem)
        name = doc.get("name", "")
        summary = doc.get("summary", "")
        case_type = doc.get("type", "")

        tactics_used, techniques_used, mitigations_used = [], [], []
        for step in doc.get("procedure", []):
            tactic = step.get("tactic", "")
            technique = step.get("technique", "")
            if tactic:
                tactics_used.append(tactic)
            if technique:
                techniques_used.append(technique)
        for mit in doc.get("mitigations", []):
            mid = mit if isinstance(mit, str) else mit.get("id", "")
            if mid:
                mitigations_used.append(mid)

        cases.append({
            "case_id": case_id,
            "name": name,
            "summary": summary[:300],
            "type": case_type,
            "n_tactics": len(set(tactics_used)),
            "n_techniques": len(set(techniques_used)),
            "n_mitigations": len(set(mitigations_used)),
            "tactics": "|".join(sorted(set(tactics_used))),
            "techniques": "|".join(sorted(set(techniques_used))),
            "mitigations": "|".join(sorted(set(mitigations_used))),
        })

    df = pd.DataFrame(cases)

    # Tag each case with the McKinsey constraints it touches
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from importlib import import_module
    mod = import_module("01_cross_taxonomy_mapping")
    ATLAS_TACTIC_TO_CONSTRAINT = mod.ATLAS_TACTIC_TO_CONSTRAINT
    def case_constraints(row):
        cids = set()
        for tid in row["tactics"].split("|"):
            cids.update(ATLAS_TACTIC_TO_CONSTRAINT.get(tid, []))
        return "|".join(sorted(cids))

    df["mckinsey_constraints_touched"] = df.apply(case_constraints, axis=1)

    out = OUTPUT / "atlas_cases_enriched.csv"
    df.to_csv(out, index=False)
    print(f"  Cases: {len(df)}")
    print(f"  → {out.name}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  3.  EO 13960 — governance readiness scoring
# ═══════════════════════════════════════════════════════════════════════════════

TIER1_COLS = ["40_has_ato", "50_internal_review"]
TIER2_COLS = [
    "52_impact_assessment", "56_monitor_postdeploy",
    "61_adverse_impact", "53_real_world_testing",
    "57_autonomous_impact", "63_stakeholder_consult",
    "67_opt_out", "65_appeal_process", "59_ai_notice",
    "55_independent_eval", "54_key_risks",
    "62_disparity_mitigation",
]

def prepare_eo13960():
    print("\n─── EO 13960: scoring governance readiness ───")
    df = pd.read_csv(EO / "2024_consolidated_ai_inventory_raw.csv",
                     low_memory=False)

    # Normalise column names — lowercase, underscores
    df.columns = [
        re.sub(r'\s+', '_', c.strip().lower().replace("(", "").replace(")", ""))
        for c in df.columns
    ]

    # Identify the actual column names that match our tier definitions
    # (some columns may have slightly different names)
    all_cols_lower = {c.lower(): c for c in df.columns}

    def find_col(pattern):
        """Find actual column name matching a pattern."""
        for c in df.columns:
            if pattern in c:
                return c
        return None

    # Map tier columns to actual names
    tier1_actual = []
    tier2_actual = []
    for c in TIER1_COLS:
        actual = find_col(c)
        if actual:
            tier1_actual.append(actual)
    for c in TIER2_COLS:
        actual = find_col(c)
        if actual:
            tier2_actual.append(actual)

    print(f"  Total rows: {len(df)}")
    print(f"  Tier-1 cols found: {len(tier1_actual)} / {len(TIER1_COLS)}")
    print(f"  Tier-2 cols found: {len(tier2_actual)} / {len(TIER2_COLS)}")

    # Score: count non-empty, non-NaN, non-"No" answers as "present"
    # Many columns use free-text (not just Yes/No), so presence of substantive
    # content = safeguard is in place.
    def safeguard_present(val):
        """Return True if the value indicates the safeguard is present."""
        if pd.isna(val):
            return False
        s = str(val).strip().lower()
        if s in ("", "no", "n/a", "nan", "none", "n/a.", "na"):
            return False
        return True

    df["tier1_score"] = df[tier1_actual].apply(
        lambda row: sum(safeguard_present(v) for v in row), axis=1
    )
    df["tier2_score"] = df[tier2_actual].apply(
        lambda row: sum(safeguard_present(v) for v in row), axis=1
    )
    df["tier1_max"]   = len(tier1_actual)
    df["tier2_max"]   = len(tier2_actual)
    df["total_gov_score"] = df["tier1_score"] + df["tier2_score"]
    df["total_gov_max"]   = df["tier1_max"] + df["tier2_max"]
    df["gov_readiness_pct"] = (df["total_gov_score"] / df["total_gov_max"] * 100).round(1)

    # McKinsey constraint coverage flags
    # Does this use case have safeguards addressing each constraint?
    from importlib import import_module
    mod = import_module("01_cross_taxonomy_mapping")
    EO13960_SAFEGUARD_TO_CONSTRAINT = mod.EO13960_SAFEGUARD_TO_CONSTRAINT
    for cid_label in ["C3", "C4", "C5", "C7", "C8"]:
        relevant_safeguards = [
            s for s, cids in EO13960_SAFEGUARD_TO_CONSTRAINT.items()
            if cid_label in cids
        ]
        relevant_actual = [find_col(s) for s in relevant_safeguards]
        relevant_actual = [c for c in relevant_actual if c]
        df[f"covers_{cid_label}"] = df.apply(
            lambda r, cols=relevant_actual: any(
                safeguard_present(r.get(c))
                for c in cols
            ),
            axis=1,
        )

    out = OUTPUT / "eo13960_scored.csv"
    df.to_csv(out, index=False)

    # Summary
    print(f"  Mean tier-1 score: {df['tier1_score'].mean():.2f} / {df['tier1_max'].iloc[0]}")
    print(f"  Mean tier-2 score: {df['tier2_score'].mean():.2f} / {df['tier2_max'].iloc[0]}")
    print(f"  Mean total readiness: {df['gov_readiness_pct'].mean():.1f}%")
    print(f"  → {out.name}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  4.  Unified evidence base (long format)
# ═══════════════════════════════════════════════════════════════════════════════

def build_unified(aiid_df, atlas_df, eo_df):
    print("\n─── Unified evidence base ───")
    rows = []

    # AIID incidents → rows
    for _, r in aiid_df.iterrows():
        rows.append({
            "source": "AIID",
            "record_id": f"AIID-{r['incident_id']}",
            "title": str(r.get("title", ""))[:200],
            "year": r.get("year"),
            "harm_type": r.get("AI Harm Level", ""),
            "failure_type": r.get("Known AI Technical Failure", ""),
            "sector": r.get("Sector of Deployment", ""),
            "autonomy": r.get("Autonomy Level", ""),
        })

    # ATLAS cases → rows
    for _, r in atlas_df.iterrows():
        rows.append({
            "source": "ATLAS",
            "record_id": r["case_id"],
            "title": r["name"][:200],
            "year": None,
            "harm_type": "",
            "failure_type": r.get("tactics", ""),
            "sector": "",
            "autonomy": "",
        })

    # EO 13960 → rows (sample summary)
    for _, r in eo_df.iterrows():
        agency = r.get("agency", r.get("agency_name", ""))
        name = r.get("ai_use_case_name", r.get("use_case_name", ""))
        rows.append({
            "source": "EO13960",
            "record_id": f"EO-{r.name}",
            "title": f"{agency}: {name}"[:200] if pd.notna(name) else str(agency)[:200],
            "year": None,
            "harm_type": "",
            "failure_type": "",
            "sector": str(agency)[:100] if pd.notna(agency) else "",
            "autonomy": "",
        })

    udf = pd.DataFrame(rows)
    out = OUTPUT / "unified_evidence_base.csv"
    udf.to_csv(out, index=False)
    print(f"  Total rows: {len(udf)}")
    for src, grp in udf.groupby("source"):
        print(f"    {src}: {len(grp)}")
    print(f"  → {out.name}")
    return udf


# ═══════════════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("PREPARE ANALYSIS-READY DATASETS")
    print("=" * 72)

    aiid_df  = prepare_aiid()
    atlas_df = prepare_atlas()
    eo_df    = prepare_eo13960()
    _        = build_unified(aiid_df, atlas_df, eo_df)

    print("\n" + "=" * 72)
    print("All datasets written to analysis/output/")
    print("=" * 72)


if __name__ == "__main__":
    main()
