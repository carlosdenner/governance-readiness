# Replication Package

## Governance Readiness Gaps in Organizational AI Deployment: A Triangulated Analysis of Threats, Incidents, and Practice

**AMCIS 2026 | Full Paper**  
Carlos Santos · Université de Sherbrooke

---

## Overview

This repository contains all analysis code, processed data artefacts, and exploratory-experiment records for the above paper. The study triangulates three public datasets to characterize the gap between formal AI governance adoption and substantive implementation across 1,757 U.S. federal AI deployments.

| Dataset | Source | N |
|---|---|---|
| MITRE ATLAS v4.x | [atlas.mitre.org](https://atlas.mitre.org) | 52 adversarial case studies |
| AI Incident Database (AIID) | [incidentdatabase.ai](https://incidentdatabase.ai) | 1,362 incidents |
| EO 13960 Federal AI Use Case Inventory | [ai.gov](https://ai.gov/ai-use-cases/) | 1,757 deployments |

---

## Quick Start

```bash
git clone https://github.com/[yourusername]/governance-readiness
cd governance-readiness
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

Download the EO 13960 raw inventory (`2024_consolidated_ai_inventory_raw.csv`) from [ai.gov](https://ai.gov/ai-use-cases/) and place it in `data/raw/eo13960/`.

---

## Analysis Pipeline

Run scripts in numbered order from the `scripts/` directory:

| Script | Purpose |
|---|---|
| `00_profile_sources.py` | Characterize the three raw data sources |
| `01_cross_taxonomy_mapping.py` | Map ATLAS/AIID/EO 13960 onto TR/IR taxonomy |
| `02_prepare_datasets.py` | Build analysis-ready artefacts |
| `03_generate_figures.py` | Reproduce figures 2–4 |
| `04_profile_eo13960.py` | EO 13960 descriptive statistics |
| `09_pathway_model.py` | **Main result** — nested logistic regressions (Table 2) |
| `10_ir_analysis.py` | Integration Readiness sub-analyses |
| `11_mapping_sensitivity.py` | TR/IR robustness checks |
| `12_procurement_confounding.py` | Vendor/procurement analyses (H4, P3) |
| `13_aiid_coverage_robustness.py` | AIID sector harm fingerprints |
| `14_atlas_threat_characterization.py` | ATLAS threat characterization |
| `17_table2_split_tr.py` | Split-TR suppression-effect model |

---

## Bayesian Exploratory Protocol

The paper uses an inductive-then-deductive research design. The inductive phase used a **Bayesian-surprise-based exploratory analysis engine** to surface candidate hypotheses from the data. All quantitative findings in the paper were subsequently confirmed via standard statistical tests (logistic regression, chi-square). This section documents the exploratory protocol in full.

### Design Rationale

Secondary data analysis at the intersection of three heterogeneous public datasets creates a large, underspecified hypothesis space. Rather than pre-registering a fixed set of hypotheses from prior theory alone (which would risk anchoring on existing IS governance literature that does not yet have empirical grounding in AI deployment data), we adopted an inductive-first approach: use the data itself to surface surprising patterns, then test those patterns deductively.

### Engine Architecture

Each experiment proceeds as follows:

1. **Hypothesis generation** — The engine draws a hypothesis from one of three sources:
   - *Theory-driven*: derived from the TR/IR/governance-theater framework (e.g., "vendor procurement negatively predicts impact assessment more than independent evaluation")
   - *Data-driven*: auto-generated from the data schema by pairing variable families (e.g., governance indicators × deployment stage)
   - *Literature-derived*: operationalizations of institutional decoupling or dynamic capabilities propositions
   
2. **Prior elicitation** — A prior belief is assigned (0–1 scale) based on theoretical expectation. Novel cross-dataset hypotheses receive a flat prior of 0.50; theory-consistent hypotheses receive elevated priors (0.70–0.90).

3. **Evidence collection** — A Python script is auto-generated and executed against the processed datasets. Output includes test statistics, effect sizes, and confidence intervals.

4. **Posterior update** — Belief is updated via a Bayes factor computed from the evidence. The resulting posterior reflects the degree to which the data supports or contradicts the hypothesis.

5. **Surprise scoring & filtering** — An experiment is flagged as *surprising* when the posterior odds exceed **10:1** in either direction relative to the prior odds (i.e., a Bayes factor ≥ 10). Hypotheses that fall below this threshold are discarded as uninformative.

### Experiment Record Structure

Each experiment is stored as a directory (`EXP_NNN_*/`) containing:

```
EXP_042_node_3_1/
├── experiment.json    # full record (hypothesis, prior, posterior, analysis)
├── code.py            # auto-generated analysis script
├── code_output.txt    # stdout from running code.py
└── report.md          # narrative interpretation
```

Key fields in `experiment.json`:

| Field | Type | Description |
|---|---|---|
| `hypothesis` | string | The tested proposition |
| `prior` | float | Prior belief (0–1) |
| `posterior` | float | Updated belief after evidence |
| `prior_belief` | object | Full distribution (definitely_true/false, maybe, uncertain) |
| `posterior_belief` | object | Updated full distribution |
| `is_surprising` | bool | True if posterior odds ≥ 10:1 vs prior odds |
| `status` | string | `SUCCEEDED` / `FAILED` |
| `analysis` | string | LLM narrative interpretation |
| `parent_id` / `child_ids` | string/list | Tree links for multi-step experiments |

### Session Summary

| | Session 1 | Session 2 | Total |
|---|---|---|---|
| Experiments run | 100 | 300 | **400** |
| Hypotheses confirmed (posterior > prior, posterior > 0.70) | ~35 | ~58 | **~93** |
| Hypotheses contradicted (posterior < prior, posterior < 0.30) | ~60 | ~96 | **~156** |
| Contradicted rate | — | — | **67.7%** |
| Hypotheses reported in paper | — | — | **4 (H1–H4 + P1–P3 + E1)** |

### Sample Hypotheses: Confirmed

The following were confirmed at posterior ≥ 0.70 and formed the basis for propositions in the paper:

| # | Hypothesis (abbreviated) | Prior → Posterior |
|---|---|---|
| C1 | Governance requirements (EU AI Act, NIST RMF) map disproportionately to Trust Readiness; technical frameworks (OWASP, NIST GenAI) map disproportionately to Integration Readiness | 0.74 → 0.75 |
| C2 | AI incident complexity shows a positive temporal trend (later incidents involve more attack techniques) | 0.72 → 0.93 |
| C3 | Governance requirements follow a Pareto-like distribution: a small set of keystone controls accounts for a disproportionate share of framework coverage | 0.89 → 0.97 |
| C4 | Attack complexity is positively correlated with organizational competency gap breadth | 0.61 → 0.90 |

### Sample Hypotheses: Discarded (Contradicted)

The following were contradicted at posterior ≤ 0.30 and were excluded from the final analysis:

| # | Hypothesis (abbreviated) | Prior → Posterior | Reason for discard |
|---|---|---|---|
| D1 | Security-related harm types are *exclusively* associated with Integration Readiness gaps | 0.50 → 0.13 | Harm types cross both capability bundles |
| D2 | Integration Readiness competencies have significantly higher empirical confidence scores than Trust Readiness | 0.76 → 0.19 | Confidence scores are similar across bundles |
| D3 | Uncovered sub-competencies are concentrated in Trust Readiness | 0.74 → 0.19 | Coverage gaps are distributed across both bundles |
| D4 | Trust Readiness competencies require more distinct architecture controls than Integration Readiness | 0.74 → 0.19 | Control requirements are roughly symmetric |

### From Inductive to Deductive

Hypotheses that survived the surprise filter were translated into the four testable propositions (H1–H4) reported in the paper, and confirmed via:
- Nested logistic regression (Table 2, `scripts/09_pathway_model.py` and `scripts/17_table2_split_tr.py`)
- Chi-square tests for sector-harm associations (`scripts/13_aiid_coverage_robustness.py`)
- Vendor-stratified sub-analyses (`scripts/12_procurement_confounding.py`)

The full experiment archives (`data/astalabs_experiments_session1/`, `data/astalabs_experiments_session2/`) are included in this repository, enabling complete audit of the exploratory phase.

---

## Construct Operationalization

### Trust Readiness (TR)

TR is operationalized as two additive indices from the EO 13960 governance fields. Equal weighting reflects the treatment of each indicator as a necessary constituent of the governance capability bundle, in the absence of empirical evidence favoring differential weights at this stage of the literature.

| Sub-index | Items | EO 13960 field |
|---|---|---|
| TR-surface (0–2) | Internal review approval | `limited_access_restrictions_applied` = yes |
| TR-surface (0–2) | Authorization to operate (ATO) | `ato` = yes |
| TR-substantive (0–7) | Impact assessment | `impact_assessment` = yes |
| TR-substantive (0–7) | Independent evaluation | `independent_evaluation` = yes |
| TR-substantive (0–7) | Bias mitigation | `bias_mitigation` = yes |
| TR-substantive (0–7) | Ongoing monitoring | `ongoing_monitoring` = yes |
| TR-substantive (0–7) | Data quality | `data_quality` = yes |
| TR-substantive (0–7) | Explainability | `explainability` = yes |
| TR-substantive (0–7) | Appeal/redress | `appeal` = yes |

### Integration Readiness (IR)

IR counts seven binary architecture indicators from the EO 13960 inventory:
data pipeline governance, evaluation infrastructure, deployment controls, code access, model access, testing/validation, and documentation.

### Governance Theater Operationalization

Governance theater is operationalized as TR-surface > 0 AND TR-substantive = 0: the system holds formal authorization artifacts without the operational safeguards needed to make authorization meaningful.

---

## Robustness Checks

| Check | Script | Finding |
|---|---|---|
| Vendor analysis with 650 excluded use cases included | `scripts/16_vendor_divergence_diagnostic.py` | Bivariate vendor direction reverses — confirms sensitivity to sample composition; addressed in Limitations |
| Agency fixed effects for vendor finding | `scripts/12_procurement_confounding.py` | Vendor OR for impact assessment robust: 0.47 (p=0.046) |
| TR composite vs split-TR model comparison | `scripts/17_table2_split_tr.py` | Split model improves AIC (2,080 vs 2,089); suppression effect confirmed |
| Sector-harm association sensitivity | `scripts/13_aiid_coverage_robustness.py` | χ²=12.97, p=0.0003, robust to taxonomy subsets |

---

## Citation

If you use this code or data, please cite:

```bibtex
@inproceedings{santos2026governance,
  author    = {Santos, Carlos},
  title     = {Governance Readiness Gaps in Organizational {AI} Deployment:
               A Triangulated Analysis of Threats, Incidents, and Practice},
  booktitle = {Proceedings of the Americas Conference on Information Systems (AMCIS 2026)},
  year      = {2026}
}
```

---

## License

Code: MIT License. See `LICENSE`.  
Data: Subject to terms of the original sources (MITRE ATLAS, AIID, ai.gov). See `data/README.md`.
