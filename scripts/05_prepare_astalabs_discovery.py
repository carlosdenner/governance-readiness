"""
05_prepare_astalabs_discovery.py
================================
Consolidate ALL analytical data into a single CSV for AstaLabs AutoDiscovery,
plus copy the best supplementary MD/JSON files for upload.

Outputs (in data/astalabs/):
  - astalabs_discovery_all_data.csv          (mega-CSV, all tables stacked)
  - context_construct_definitions.md         (research constructs)
  - context_propositions.md                  (5 propositions narrative)
  - context_validation_report.md             (step-5 validation)
  - context_crosswalk_evidence.json          (crosswalk evidence)
  - context_propositions.json                (propositions evidence)
  - SESSION_README.md                        (fill-in guide for AstaLabs form)
"""

import pandas as pd
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
ENRICHED = PROC / "enriched"
OUT = ROOT / "data" / "astalabs"
OUT.mkdir(parents=True, exist_ok=True)

# ── 1. Load every processed CSV ────────────────────────────────────────────
tables = {
    "eo13960_scored":           PROC / "eo13960_scored.csv",
    "aiid_incidents":           PROC / "aiid_incidents_classified.csv",
    "atlas_cases":              PROC / "atlas_cases_enriched.csv",
    "unified_evidence_base":    PROC / "unified_evidence_base.csv",
    "cross_taxonomy_map":       PROC / "cross_taxonomy_map.csv",
    "step1_sub_competencies":   PROC / "step1_sub_competencies.csv",
    "step2_crosswalk_matrix":   PROC / "step2_crosswalk_matrix.csv",
    "step2_competency_stmts":   PROC / "step2_competency_statements.csv",
    "step3_incident_coding":    PROC / "step3_incident_coding.csv",
    "step3_mitigation_gaps":    PROC / "step3_mitigation_gaps.csv",
    "step3_tactic_frequency":   PROC / "step3_tactic_frequency.csv",
    "step4_propositions":       PROC / "step4_propositions.csv",
}

# Also grab enriched-only tables
enriched_only = {
    "step3_coverage_map":        ENRICHED / "step3_coverage_map.csv",
    "step5_validation_issues":   ENRICHED / "step5_validation_issues.csv",
}

all_tables = {**tables, **enriched_only}

frames = []
for name, path in all_tables.items():
    if not path.exists():
        print(f"  [SKIP] {name}: {path} not found")
        continue
    df = pd.read_csv(path, low_memory=False)
    df.insert(0, "source_table", name)
    df.insert(1, "source_row_num", range(1, len(df) + 1))
    frames.append(df)
    print(f"  [OK]  {name:30s}  {len(df):>6,} rows × {len(df.columns):>3} cols")

mega = pd.concat(frames, ignore_index=True, sort=False)

# Move source_table and source_row_num to the front, keep the rest as-is
cols = mega.columns.tolist()
cols.remove("source_table")
cols.remove("source_row_num")
mega = mega[["source_table", "source_row_num"] + cols]

csv_path = OUT / "astalabs_discovery_all_data.csv"
mega.to_csv(csv_path, index=False)
print(f"\n✅  Mega CSV written: {csv_path}")
print(f"    Total rows : {len(mega):,}")
print(f"    Total cols : {len(mega.columns)}")
print(f"    File size  : {csv_path.stat().st_size / 1024 / 1024:.1f} MB")

# ── 2. Column sparsity report ──────────────────────────────────────────────
print("\n── Column fill-rate (top 30) ──")
fill = mega.notna().mean().sort_values(ascending=False)
for col, pct in fill.head(30).items():
    print(f"  {pct:6.1%}  {col}")

# ── 3. Copy supplementary MD / JSON files ──────────────────────────────────
supplementary = {
    "context_construct_definitions.md":  PROC / "step1_construct_definitions.md",
    "context_propositions.md":           PROC / "step4_propositions.md",
    "context_validation_report.md":      ENRICHED / "step5_validation_report.md",
    "context_crosswalk_evidence.json":   ENRICHED / "step2_crosswalk_evidence.json",
    "context_propositions.json":         ENRICHED / "step4_propositions.json",
    "context_step1_evidence.json":       ENRICHED / "step1_evidence.json",
}

print("\n── Supplementary files ──")
for dest_name, src_path in supplementary.items():
    if src_path.exists():
        shutil.copy2(src_path, OUT / dest_name)
        sz = (OUT / dest_name).stat().st_size / 1024
        print(f"  [OK]  {dest_name:45s} ({sz:.0f} KB)")
    else:
        print(f"  [SKIP] {dest_name}: source not found")

# ── 4. Write a session-filling guide ───────────────────────────────────────
guide = f"""\
# AstaLabs AutoDiscovery – Session Setup Guide

## Discovery session name
```
AMCIS2026 – AI Orientation Trust-Integration Readiness
```

## Dataset context
```
This dataset consolidates three public secondary-data sources used in a
design-science / framework-synthesis study targeting AMCIS 2026.

Sources:
1. AIID – AI Incident Database (1,366 real-world AI failure incidents
   classified by harm domain, tangible harm, sector, autonomy level,
   AI technology, and technical failure type).
2. MITRE ATLAS – Adversarial Threat Landscape for AI Systems (84
   adversarial case studies coded with tactics, techniques, and
   mitigations; cross-mapped to McKinsey AI-scaling constraints C1–C8).
3. EO 13960 – U.S. Federal AI Use-Case Inventory (3,658 government AI
   deployments scored for governance readiness across two tiers: basic
   controls and deep governance safeguards including impact assessments,
   bias mitigation, and independent evaluation).

Analytical layers built on top of these three sources include:
- A crosswalk matrix linking 42 governance requirements from NIST AI RMF,
  EU AI Act, ISO 42001, and OWASP Top-10 LLM to 18 architecture controls.
- 16 sub-competencies in two bundles: Trust Readiness (TR-1…TR-8) and
  Integration Readiness (IR-1…IR-8).
- Incident-coding of ATLAS cases mapping tactics to competency gaps.
- Five testable propositions (P1–P5) with falsifiability criteria.

The CSV concatenates ALL tables (sparse — use the `source_table` column
to segment by data source). Supplementary .md and .json files provide
construct definitions, proposition narratives, and validation reports.

Known limitation: EO 13960 data skews toward U.S. federal agencies;
AIID coverage varies by year; ATLAS cases are curated (not exhaustive).
```

## Domain of datasets
```
Information Systems / AI Governance / Enterprise Architecture
```

## Files to upload
| # | File | Type | Purpose |
|---|------|------|---------|
| 1 | `astalabs_discovery_all_data.csv` | CSV | All {len(mega):,} rows across {len(all_tables)} tables |
| 2 | `context_construct_definitions.md` | MD | Trust & Integration Readiness construct definitions |
| 3 | `context_propositions.md` | MD | Five propositions (P1–P5) with evidence |
| 4 | `context_validation_report.md` | MD | Step-5 validation findings |
| 5 | `context_crosswalk_evidence.json` | JSON | Crosswalk evidence details |
| 6 | `context_propositions.json` | JSON | Proposition evidence details |
| 7 | `context_step1_evidence.json` | JSON | Construct-building evidence |
"""

guide_path = OUT / "SESSION_README.md"
guide_path.write_text(guide, encoding="utf-8")
print(f"\n✅  Session guide written: {guide_path}")
print("\nDone! Upload the files in data/astalabs/ to AstaLabs AutoDiscovery.")
