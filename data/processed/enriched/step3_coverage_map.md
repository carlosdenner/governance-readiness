# Step 3 -- Incident-Mapped Validation Appendix (Coverage Map)

**Source**: MITRE ATLAS adversarial ML case studies
**Purpose**: Validate that the Trust & Integration Readiness framework
covers real-world AI failure modes (negative-case evidence)

## Coverage Summary

- **Validated** (3+ incidents): 8 sub-competencies
- **Partial** (1-2 incidents): 4 sub-competencies
- **Uncovered** (0 incidents): 4 sub-competencies
- **Total incidents coded**: 52

## Trust Readiness Coverage

| Sub-Competency | Incidents | Status | Primary Harm Types |
|----------------|-----------|--------|-------------------|
| TR-1: Risk Policy & Accountability | 19 | validated | security(16); intellectual_property(2); autonomy_misuse(1) |
| TR-2: Threat Mapping & Reconnaissance Defense | 1 | partial | privacy(1) |
| TR-3: Monitoring & Detection | 9 | validated | security(5); reliability(2); bias_discrimination(1) |
| TR-4: Data Governance & Exfiltration Prevention | 2 | partial | reliability(2) |
| TR-5: Regulatory Compliance | 0 | uncovered |  |
| TR-6: Incident Response & Recovery | 9 | validated | security(7); privacy(2) |
| TR-7: Human Override & Control | 0 | uncovered |  |
| TR-8: Supply Chain & Third-Party Risk | 5 | validated | supply_chain(4); security(1) |

## Integration Readiness Coverage

| Sub-Competency | Incidents | Status | Primary Harm Types |
|----------------|-----------|--------|-------------------|
| IR-1: Orchestration & Execution Controls | 8 | validated | security(5); reliability(2); supply_chain(1) |
| IR-2: Tool-Use Boundaries & Access Control | 8 | validated | security(5); privacy(2); supply_chain(1) |
| IR-3: Nondeterminism Management | 18 | validated | security(14); reliability(2); intellectual_property(1) |
| IR-4: RAG Architecture & Data Grounding | 2 | partial | privacy(1); supply_chain(1) |
| IR-5: GenAIOps / MLOps Lifecycle | 0 | uncovered |  |
| IR-6: Modular Architecture & Resource Controls | 7 | validated | security(5); supply_chain(1); autonomy_misuse(1) |
| IR-7: HITL Architecture Patterns | 1 | partial | privacy(1) |
| IR-8: Evaluation & Monitoring Infrastructure | 0 | uncovered |  |

## Harm Type Distribution (All Incidents)

| Harm Type | Count | Percentage |
|-----------|-------|------------|
| security | 36 | 69.2% |
| reliability | 4 | 7.7% |
| supply_chain | 4 | 7.7% |
| privacy | 4 | 7.7% |
| intellectual_property | 2 | 3.8% |
| bias_discrimination | 1 | 1.9% |
| autonomy_misuse | 1 | 1.9% |
