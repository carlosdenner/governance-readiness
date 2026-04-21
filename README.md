# Governance Readiness Gaps in Organizational AI Deployment
### A Triangulated Analysis of Threats, Incidents, and Practice

**AMCIS 2026 Full Paper** · Carlos Santos · Université de Sherbrooke

> **Replication package** — see [REPLICATION.md](REPLICATION.md) for the full analysis pipeline, construct operationalization, and Bayesian exploratory protocol documentation.

---

Research project for the **Americas Conference on Information Systems (AMCIS) 2026**.

## Thesis

CIOs shape a firm's **AI orientation** (strategic intent / direction for AI), but the ability to realise that orientation depends on two enabling capability bundles:

1. **Trust / Governance Readiness** — grounded in NIST AI RMF, EU AI Act, OWASP Top-10 for LLMs, and MITRE ATLAS.
2. **Integration / Architecture Readiness** — grounded in agentic-AI design patterns, Azure Well-Architected AI guidance, and GenAIOps practices.

## Repository Structure

```
├── README.md
├── requirements.txt              # Python dependencies
│
├── data/
│   ├── raw/                      # Immutable source data (do NOT edit)
│   │   ├── atlas/                # MITRE ATLAS adversarial-threat KB
│   │   ├── aiid/                 # AI Incident Database snapshot
│   │   └── eo13960/              # EO 13960 federal AI inventory
│   └── processed/                # Analysis-ready artefacts (regenerable)
│       ├── cross_taxonomy_map.csv
│       ├── aiid_incidents_classified.csv
│       ├── atlas_cases_enriched.csv
│       ├── eo13960_scored.csv
│       ├── unified_evidence_base.csv
│       ├── step1–4 legacy outputs
│       └── enriched/             # LLM-enriched artefacts
│
├── scripts/                      # Reproducible analysis pipeline
│   ├── 00_profile_sources.py     # Profile all 3 raw data sources
│   ├── 01_cross_taxonomy_mapping.py  # Cross-taxonomy bridging table
│   ├── 02_prepare_datasets.py    # Merge & score → analysis CSVs
│   ├── 03_generate_figures.py    # Publication-ready figures
│   ├── 04_profile_eo13960.py     # Deep EO 13960 governance profiling
│   ├── agents/                   # LLM-powered enrichment pipeline
│   └── legacy/                   # Original 4-step pipeline (archived)
│
├── paper/
│   ├── paper.md                  # Manuscript (Markdown)
│   └── figures/                  # Publication-ready PNGs
│
├── literature/                   # Reference documents (HTML, MD, PDF)
│
└── docs/                         # Project planning & design documents
    ├── Goal.txt
    ├── Planning - Concepts, Framework and Methods.*
    └── research_angles_summary.md
```

## Data Sources (3 secondary data repositories)

| Source | Records | Type | Licence |
|--------|--------:|------|---------|
| MITRE ATLAS | 52 case studies, 16 tactics, 155 techniques, 35 mitigations | Adversarial threat knowledge base | Apache 2.0 |
| AI Incident Database (AIID) | 1,362 incidents, 6,681 reports | Real-world AI failure repository | CC BY-SA 4.0 |
| EO 13960 Federal AI Inventory | 1,757 use cases × 62 variables × 38 agencies | Government AI practice data | Public domain |

## McKinsey "Constraints to Scale" Alignment

The cross-taxonomy mapping anchors all three data sources to McKinsey's
Exhibit 5 barriers (Feb 2026, "The new CIO mandate"):

| ID | Constraint | % | Primary Data Evidence |
|----|-----------|---:|----------------------|
| C1 | Talent / capability gaps | 31% | ATLAS (reconnaissance), AIID (misuse) |
| C2 | Integration complexity | 29% | ATLAS (lateral movement, agent segmentation), AIID (latency) |
| C3 | Security / reliability / hallucinations | 26% | ATLAS (65 links), AIID (generalization, bias), EO 13960 (Tier-2 gap) |
| C4 | Regulatory / privacy / compliance | 24% | ATLAS (exfiltration), AIID (transparency, bias), EO 13960 (safeguards) |
| C5 | Lack of modern data foundations | 21% | ATLAS (collection/discovery), AIID (data noise) |
| C7 | Difficulty measuring ROI / value | 17% | EO 13960 (post-deploy monitoring) |
| C8 | Internal resistance / change management | 16% | ATLAS (HITL), AIID (misuse), EO 13960 (stakeholders) |

## Reproducibility

### Prerequisites

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Running the Pipeline

From the project root:

```bash
# Step 0 — Profile raw sources (optional, exploratory)
python scripts/00_profile_sources.py

# Step 1 — Build cross-taxonomy bridging table → data/processed/cross_taxonomy_map.csv
python scripts/01_cross_taxonomy_mapping.py

# Step 2 — Prepare analysis-ready datasets → data/processed/*.csv
python scripts/02_prepare_datasets.py

# Step 3 — Generate publication figures → paper/figures/*.png
python scripts/03_generate_figures.py

# Step 4 — Deep EO 13960 governance profiling (optional)
python scripts/04_profile_eo13960.py
```

### Agentic Enrichment Pipeline (optional)

The `scripts/agents/` directory contains an LLM-powered enrichment pipeline that
reads literature sources, extracts evidence passages, and generates grounded
artefacts with auditable citation trails.

```bash
# Requires OPENAI_API_KEY environment variable
python -m scripts.agents.run_agents          # full pipeline
python -m scripts.agents.run_agents --step 1 # single step
```

| Step | Agent | Purpose |
|------|-------|---------|
| 1 | `step1_extract.py` | Extract evidence from literature → enriched construct definitions |
| 2 | `step2_crosswalk.py` | Generate crosswalk with source-passage citations |
| 3 | `step3_enrich.py` | Fix mitigation mapping + LLM competency gap descriptions |
| 4 | `step4_synthesize.py` | Compute cross-step statistics + grounded propositions |

Uses `gpt-4.1-mini` for extraction and `gpt-4.1` for synthesis.

## License

Private academic research — all rights reserved.
