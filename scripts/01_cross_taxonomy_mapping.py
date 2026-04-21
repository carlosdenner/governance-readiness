"""
cross_taxonomy_mapping.py
=========================
Builds a unified bridging taxonomy that connects:
  1. McKinsey Exhibit 5 "constraints to scale" (8 barriers)
  2. ATLAS tactics / mitigations (adversarial threat surface)
  3. AIID harm & failure categories (real-world AI failures)
  4. EO 13960 governance safeguard variables (U.S. federal practice)

The output is a single CSV ("cross_taxonomy_map.csv") where every row is a
*mapping link* between a source taxonomy concept and a target taxonomy concept,
tagged with the McKinsey constraint domain it addresses.

This table is angle-agnostic: whichever research angle the co-author picks,
the mapping feeds directly into the analysis.
"""

from __future__ import annotations
import csv
import os
import pathlib
import yaml
import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent.parent
OUTPUT = BASE / "data" / "processed"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── McKinsey Exhibit 5: constraints to scale ──────────────────────────────────
# Source: McKinsey, "The new CIO mandate", Feb 2026, Exhibit 5
MCKINSEY_CONSTRAINTS = {
    "C1": {"label": "Talent / capability gaps",                      "pct": 31},
    "C2": {"label": "Integration complexity with existing systems",  "pct": 29},
    "C3": {"label": "Security / reliability / hallucinations",       "pct": 26},
    "C4": {"label": "Regulatory / privacy / compliance",             "pct": 24},
    "C5": {"label": "Lack of modern data foundations",               "pct": 21},
    "C6": {"label": "Lack of clear business use cases",              "pct": 18},
    "C7": {"label": "Difficulty measuring ROI / value",              "pct": 17},
    "C8": {"label": "Internal resistance / change management",       "pct": 16},
}

# Competency Domains (from Planning document) mapped to constraints
COMPETENCY_DOMAINS = {
    "A": {"label": "Strategy Co-Creation & AI Value Architecture",          "constraints": ["C6", "C7"]},
    "B": {"label": "Agentic Operating Model & Delivery Velocity",           "constraints": ["C2", "C8"]},
    "C": {"label": "Enterprise Architecture & Integration",                 "constraints": ["C2"]},
    "D": {"label": "Data Foundations & Data as a Product",                   "constraints": ["C5", "C3"]},
    "E": {"label": "Trust, Risk & Compliance Governance",                   "constraints": ["C3", "C4"]},
    "F": {"label": "Talent Systems & Capability Building",                  "constraints": ["C1", "C2", "C8"]},
    "G": {"label": "Change Leadership & Human-AI Work Redesign",            "constraints": ["C8", "C1"]},
}


# ── 1. ATLAS → McKinsey Constraint Mapping ────────────────────────────────────
# Each ATLAS tactic is mapped to the McKinsey constraint(s) it most threatens.
# Rationale: if this threat is unaddressed, *which* scaling barrier does it
# worsen?

ATLAS_TACTIC_TO_CONSTRAINT = {
    # --- Security / reliability / hallucinations (C3) ---
    "AML.TA0000": ["C3"],           # AI Model Access
    "AML.TA0001": ["C3"],           # AI Attack Staging
    "AML.TA0004": ["C3"],           # Initial Access
    "AML.TA0005": ["C3"],           # Execution
    "AML.TA0006": ["C3"],           # Persistence
    "AML.TA0007": ["C3"],           # Defense Evasion
    "AML.TA0010": ["C3", "C4"],     # Exfiltration → security + privacy
    "AML.TA0011": ["C3"],           # Impact
    "AML.TA0012": ["C3"],           # Privilege Escalation
    "AML.TA0013": ["C3"],           # Credential Access
    "AML.TA0014": ["C3"],           # Command and Control
    "AML.TA0015": ["C3", "C2"],     # Lateral Movement → security + integration
    # --- Reconnaissance / Resource Dev → talent/process gaps ---
    "AML.TA0002": ["C3", "C1"],     # Reconnaissance → security + talent gap
    "AML.TA0003": ["C3", "C1"],     # Resource Development → security + talent
    # --- Discovery / Collection → data foundations ---
    "AML.TA0008": ["C5", "C3"],     # Discovery → data exposure + security
    "AML.TA0009": ["C5", "C3"],     # Collection → data + security
}

# ATLAS mitigations mapped to constraints they *address*
ATLAS_MITIGATION_TO_CONSTRAINT = {
    # Technical-ML mitigations → C3 (security/reliability)
    "AML.M0002": ["C3"],   # Passive AI Output Obfuscation
    "AML.M0003": ["C3"],   # Model Hardening
    "AML.M0004": ["C3"],   # Restrict Queries
    "AML.M0006": ["C3"],   # Ensemble Methods
    "AML.M0007": ["C5", "C3"],   # Sanitize Training Data → data + security
    "AML.M0008": ["C3"],   # Validate AI Model
    "AML.M0009": ["C3"],   # Multi-Modal Sensors
    "AML.M0010": ["C3"],   # Input Restoration
    "AML.M0015": ["C3"],   # Adversarial Input Detection
    "AML.M0020": ["C3", "C4"],   # GenAI Guardrails → security + compliance
    "AML.M0022": ["C3", "C4"],   # GenAI Model Alignment → security + compliance
    "AML.M0034": ["C3"],   # Deepfake Detection
    # Technical-Cyber mitigations → C3 + C2
    "AML.M0005": ["C3", "C5"],   # Access Control at Rest → security + data
    "AML.M0011": ["C3"],   # Restrict Library Loading
    "AML.M0012": ["C3", "C4"],   # Encrypt Sensitive Info → security + privacy
    "AML.M0013": ["C3"],   # Code Signing
    "AML.M0014": ["C3"],   # Verify AI Artifacts
    "AML.M0016": ["C3"],   # Vulnerability Scanning
    "AML.M0019": ["C3", "C2"],   # Access Control in Production → security + integration
    "AML.M0031": ["C3"],   # Memory Hardening
    "AML.M0033": ["C3", "C2"],   # I/O Validation for Agents → security + integration
    # Policy/process mitigations → C4, C1, C5
    "AML.M0000": ["C4"],   # Limit Public Release → compliance
    "AML.M0001": ["C4"],   # Limit Model Artifact Release → compliance
    "AML.M0017": ["C2", "C3"],   # AI Model Distribution → integration + security
    "AML.M0018": ["C1"],   # User Training → talent
    "AML.M0021": ["C4", "C1"],   # GenAI Guidelines → compliance + talent
    "AML.M0023": ["C5", "C4"],   # AI Bill of Materials → data + compliance
    "AML.M0024": ["C3", "C5"],   # AI Telemetry Logging → security + data
    "AML.M0025": ["C5"],   # Dataset Provenance → data
    # Agentic-specific mitigations → C3 + C2
    "AML.M0026": ["C3", "C2"],   # Privileged Agent Permissions
    "AML.M0027": ["C3", "C2"],   # Single-User Agent Permissions
    "AML.M0028": ["C3", "C2"],   # Agent Tools Permissions
    "AML.M0029": ["C8", "C3"],   # Human-in-the-Loop → change mgmt + security
    "AML.M0030": ["C3", "C5"],   # Restrict Agent Tool on Untrusted Data
    "AML.M0032": ["C2", "C3"],   # Segmentation of Agent Components → integration
}


# ── 2. AIID Failure/Harm → McKinsey Constraint Mapping ───────────────────────
# GMF "Known AI Technical Failure" categories → which scaling barrier each
# failure type evidences.

AIID_FAILURE_TO_CONSTRAINT = {
    "Generalization Failure":               ["C3"],        # reliability
    "Misinformation Generation Hazard":     ["C3", "C4"],  # hallucinations + compliance
    "Distributional Bias":                  ["C4", "C3"],  # compliance/fairness + reliability
    "Context Misidentification":            ["C3"],        # reliability
    "Lack of Transparency":                 ["C4"],        # compliance/trustworthiness
    "Unsafe Exposure or Access":            ["C3", "C4"],  # security + privacy
    "Harmful Application":                  ["C4"],        # compliance
    "Algorithmic Bias":                     ["C4", "C3"],  # compliance + reliability
    "Latency Issues":                       ["C2"],        # integration/performance
    "Adversarial Data":                     ["C3"],        # security
    "Gaming Vulnerability":                 ["C3"],        # security
    "Privacy Concerns":                     ["C4"],        # privacy/compliance
    "Robustness Failure":                   ["C3"],        # reliability
    "Data or Labelling Noise":              ["C5"],        # data foundations
    "Inadequate Data":                      ["C5"],        # data foundations
    "Limited Dataset":                      ["C5"],        # data foundations
    "Concept Drift":                        ["C5", "C3"],  # data + reliability
    "Misuse":                               ["C1", "C8"],  # talent + change mgmt
    "Lack of Capability Control":           ["C3", "C2"],  # security + integration
    "Underspecification":                   ["C3"],        # reliability
}

# CSET v1 harm levels → constraint
AIID_HARM_LEVEL_TO_CONSTRAINT = {
    "AI tangible harm event":     ["C3", "C4"],   # security/reliability + compliance
    "AI tangible harm issue":     ["C4"],          # compliance concern
    "AI tangible harm near-miss": ["C3"],          # security/reliability
}

# AIID sector → constraint
AIID_SECTOR_TO_CONSTRAINT = {
    "information and communication":    ["C2", "C3"],
    "transportation and storage":       ["C3"],
    "healthcare":                       ["C3", "C4"],
    "financial services":               ["C4", "C3"],
    "law enforcement":                  ["C4"],
    "Education":                        ["C4", "C1"],
}


# ── 3. EO 13960 Safeguards → McKinsey Constraint Mapping ─────────────────────
# Each EO 13960 governance variable addresses (or fails to address) specific
# scaling barriers.

EO13960_SAFEGUARD_TO_CONSTRAINT = {
    # Tier 1 — Basic controls
    "40_has_ato":                    ["C4"],          # Authorization → compliance
    "50_internal_review":            ["C4"],          # Internal review → compliance
    # Tier 2 — Deep governance
    "52_impact_assessment":          ["C4", "C3"],    # Impact assess → compliance + reliability
    "56_monitor_postdeploy":         ["C3", "C7"],    # Monitoring → reliability + ROI
    "61_adverse_impact":             ["C4"],          # Adverse impact → compliance
    "53_real_world_testing":         ["C3"],          # Testing → reliability
    "57_autonomous_impact":          ["C3", "C8"],    # Autonomy → reliability + change mgmt
    "63_stakeholder_consult":        ["C8", "C4"],    # Stakeholders → change mgmt + compliance
    "67_opt_out":                    ["C4", "C8"],    # Opt-out → compliance + change mgmt
    "65_appeal_process":             ["C4"],          # Appeal → compliance
    "59_ai_notice":                  ["C4", "C8"],    # Notice → compliance + change mgmt
    "55_independent_eval":           ["C3", "C4"],    # Evaluation → reliability + compliance
    "54_key_risks":                  ["C3"],          # Risk ID → security/reliability
    "62_disparity_mitigation":       ["C4"],          # Bias mitigation → compliance
}


# ── Build the cross-taxonomy CSV ──────────────────────────────────────────────

def load_atlas_names():
    """Load ATLAS tactic and mitigation names from YAML."""
    t_path = BASE / "data" / "raw" / "atlas" / "data" / "tactics.yaml"
    m_path = BASE / "data" / "raw" / "atlas" / "data" / "mitigations.yaml"
    names = {}
    with open(t_path) as f:
        for t in yaml.safe_load(f):
            names[t["id"]] = t["name"]
    with open(m_path) as f:
        for m in yaml.safe_load(f):
            names[m["id"]] = m["name"]
    return names

def build_rows():
    atlas_names = load_atlas_names()
    rows = []

    # ATLAS tactics → constraints
    for tid, cids in ATLAS_TACTIC_TO_CONSTRAINT.items():
        for cid in cids:
            rows.append({
                "source_taxonomy": "ATLAS",
                "source_type": "tactic",
                "source_id": tid,
                "source_label": atlas_names.get(tid, tid),
                "target_taxonomy": "McKinsey",
                "target_id": cid,
                "target_label": MCKINSEY_CONSTRAINTS[cid]["label"],
                "target_pct": MCKINSEY_CONSTRAINTS[cid]["pct"],
                "relationship": "threatens",
                "competency_domain": "|".join(
                    d for d, v in COMPETENCY_DOMAINS.items()
                    if cid in v["constraints"]
                ),
            })

    # ATLAS mitigations → constraints
    for mid, cids in ATLAS_MITIGATION_TO_CONSTRAINT.items():
        for cid in cids:
            rows.append({
                "source_taxonomy": "ATLAS",
                "source_type": "mitigation",
                "source_id": mid,
                "source_label": atlas_names.get(mid, mid),
                "target_taxonomy": "McKinsey",
                "target_id": cid,
                "target_label": MCKINSEY_CONSTRAINTS[cid]["label"],
                "target_pct": MCKINSEY_CONSTRAINTS[cid]["pct"],
                "relationship": "addresses",
                "competency_domain": "|".join(
                    d for d, v in COMPETENCY_DOMAINS.items()
                    if cid in v["constraints"]
                ),
            })

    # AIID failures → constraints
    for fail, cids in AIID_FAILURE_TO_CONSTRAINT.items():
        for cid in cids:
            rows.append({
                "source_taxonomy": "AIID-GMF",
                "source_type": "technical_failure",
                "source_id": "",
                "source_label": fail,
                "target_taxonomy": "McKinsey",
                "target_id": cid,
                "target_label": MCKINSEY_CONSTRAINTS[cid]["label"],
                "target_pct": MCKINSEY_CONSTRAINTS[cid]["pct"],
                "relationship": "evidences",
                "competency_domain": "|".join(
                    d for d, v in COMPETENCY_DOMAINS.items()
                    if cid in v["constraints"]
                ),
            })

    # AIID harm levels → constraints
    for hlevel, cids in AIID_HARM_LEVEL_TO_CONSTRAINT.items():
        for cid in cids:
            rows.append({
                "source_taxonomy": "AIID-CSET",
                "source_type": "harm_level",
                "source_id": "",
                "source_label": hlevel,
                "target_taxonomy": "McKinsey",
                "target_id": cid,
                "target_label": MCKINSEY_CONSTRAINTS[cid]["label"],
                "target_pct": MCKINSEY_CONSTRAINTS[cid]["pct"],
                "relationship": "evidences",
                "competency_domain": "|".join(
                    d for d, v in COMPETENCY_DOMAINS.items()
                    if cid in v["constraints"]
                ),
            })

    # EO 13960 safeguards → constraints
    for saf, cids in EO13960_SAFEGUARD_TO_CONSTRAINT.items():
        for cid in cids:
            rows.append({
                "source_taxonomy": "EO13960",
                "source_type": "governance_safeguard",
                "source_id": saf,
                "source_label": saf.replace("_", " ").title(),
                "target_taxonomy": "McKinsey",
                "target_id": cid,
                "target_label": MCKINSEY_CONSTRAINTS[cid]["label"],
                "target_pct": MCKINSEY_CONSTRAINTS[cid]["pct"],
                "relationship": "measures_readiness_for",
                "competency_domain": "|".join(
                    d for d, v in COMPETENCY_DOMAINS.items()
                    if cid in v["constraints"]
                ),
            })

    return rows


def main():
    rows = build_rows()
    df = pd.DataFrame(rows)
    out_path = OUTPUT / "cross_taxonomy_map.csv"
    df.to_csv(out_path, index=False)

    # ── Summary statistics ────────────────────────────────────────────────
    print("=" * 72)
    print("CROSS-TAXONOMY MAPPING — Summary")
    print("=" * 72)
    print(f"\nTotal mapping links:  {len(df)}")
    print(f"Source taxonomies:    {df['source_taxonomy'].nunique()}")
    print(f"Unique source items:  {df.groupby(['source_taxonomy','source_label']).ngroups}")
    print()

    # Links per McKinsey constraint
    print("── Links per McKinsey Constraint ──")
    constraint_counts = (
        df.groupby("target_id")
        .agg(links=("source_label", "count"),
             label=("target_label", "first"),
             pct=("target_pct", "first"))
        .sort_values("target_id")
    )
    for cid, row in constraint_counts.iterrows():
        bar = "#" * (row["links"] // 2)
        print(f"  {cid} ({row['pct']:2d}%) {row['label']:<50s} {row['links']:3d} links  {bar}")

    print()

    # Links by source taxonomy
    print("── Links by Source Taxonomy ──")
    for src, grp in df.groupby("source_taxonomy"):
        print(f"  {src:<12s}  {len(grp):3d} links  "
              f"({grp['source_type'].nunique()} types: "
              f"{', '.join(grp['source_type'].unique())})")

    print()

    # Coverage: which constraints have evidence from ALL three data sources?
    print("── Triangulation Coverage (constraints with links from all 3 data repos) ──")
    src_per_constraint = (
        df.groupby("target_id")["source_taxonomy"]
        .apply(lambda s: set(s))
    )
    full_tri = {"ATLAS", "AIID-GMF", "AIID-CSET", "EO13960"}
    for cid, sources in src_per_constraint.items():
        status = "FULL" if sources >= {"ATLAS", "EO13960"} and sources & {"AIID-GMF", "AIID-CSET"} else "PARTIAL"
        print(f"  {cid}: {status:<8s}  sources = {', '.join(sorted(sources))}")

    print()

    # Competency domain coverage
    print("── Competency Domain Coverage ──")
    all_domains = set()
    for _, r in df.iterrows():
        all_domains.update(r["competency_domain"].split("|"))
    for dom in sorted(all_domains):
        dom_label = COMPETENCY_DOMAINS[dom]["label"]
        dom_constraints = COMPETENCY_DOMAINS[dom]["constraints"]
        dom_links = df[df["competency_domain"].str.contains(dom, na=False)]
        print(f"  Domain {dom}: {dom_label}")
        print(f"    Constraints: {', '.join(dom_constraints)}")
        print(f"    Total links: {len(dom_links)}  "
              f"(threatens: {len(dom_links[dom_links['relationship']=='threatens'])}, "
              f"addresses: {len(dom_links[dom_links['relationship']=='addresses'])}, "
              f"evidences: {len(dom_links[dom_links['relationship']=='evidences'])}, "
              f"measures: {len(dom_links[dom_links['relationship']=='measures_readiness_for'])})")
        print()

    print(f"\nOutput: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
