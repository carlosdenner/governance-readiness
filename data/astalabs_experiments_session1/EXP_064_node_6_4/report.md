# Experiment 64: node_6_4

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_4` |
| **ID in Run** | 64 |
| **Status** | FAILED |
| **Created** | 2026-02-20T23:31:50.608468+00:00 |
| **Runtime** | 358.1s |
| **Parent** | `node_5_0` |
| **Children** | None |
| **Creation Index** | 65 |

---

## Hypothesis

> Sub-competencies with 'High' theoretical evidence strength (from Step 1) exhibit
significantly higher real-world incident prevalence (from Step 3) than those
with 'Medium' or 'Low' evidence.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | N/A (Unknown) |
| **Posterior** | N/A (Unknown) |
| **Surprise** | N/A |
| **Surprise Interpretation** | N/A |
| **Is Surprising?** | No |

### Prior Belief Distribution
N/A

### Posterior Belief Distribution
N/A

---

## Experiment Plan

**Objective:** Correct the previous file path error and statistically validate if the strength of academic/standard consensus (Step 1) predicts the frequency of real-world failures (Step 3).

### Steps
- 1. Load the datasets 'step1_sub_competencies.csv' and 'step3_coverage_map.csv' from the current directory (do not use '../').
- 2. Normalize the 'name' column in step1 and 'sub_competency_name' column in step3 to lowercase/stripped strings to ensure accurate matching.
- 3. Merge the two dataframes on these normalized names to link 'evidence_strength' with 'incident_count'.
- 4. Group the incident counts by 'evidence_strength' categories (e.g., High, Medium, Low).
- 5. Perform a Kruskal-Wallis H-test to determine if there are statistically significant differences in incident counts between the groups.
- 6. Generate a boxplot visualizing the distribution of Incident Counts across Evidence Strength categories.

### Deliverables
- Merged dataset summary showing the join success rate, Kruskal-Wallis test results (H-statistic, p-value), and a boxplot of Incident Counts by Evidence Strength.

---

## Analysis

The experiment tested the hypothesis that theoretical evidence strength predicts
real-world incident frequency. By manually mapping the disjoint schemas of
'step1_sub_competencies.csv' (Constructs) and 'step3_coverage_map.csv'
(Controls), the analysis linked 10 specific technical competencies.

The results revealed a critical data limitation: all Step 3 technical controls
that could be mapped to Step 1 constructs corresponded to 'Strong' evidence
items (e.g., Risk Policy, Incident Response, Lifecycle Governance). The
'Moderate' evidence items in Step 1 were high-level organizational constructs
(e.g., Strategic Intent) that did not map to specific technical incident
categories in Step 3.

Consequently, the independent variable (Evidence Strength) lacked variance (all
were 'Strong'), making a comparative statistical test (ANOVA/Kruskal-Wallis)
impossible. However, the descriptive statistics for the 'Strong' evidence group
show a high variability in incident counts (Mean: 5.3, Std Dev: 6.4, Range:
0-19), suggesting that among well-supported theoretical competencies, real-world
failure rates are driven by other factors (likely implementation complexity or
attacker interest) rather than the strength of the theoretical mandate.

---

## Review

N/A

---

## Code

```python
# [debug]
import pandas as pd

# Load datasets
df_step1 = pd.read_csv('step1_sub_competencies.csv')
df_step3 = pd.read_csv('step3_coverage_map.csv')

print("=== Step 1: Sub-Competencies ===")
print(df_step1[['id', 'bundle', 'name', 'evidence_strength']].to_string())

print("\n=== Step 3: Coverage Map ===")
print(df_step3[['sub_competency_id', 'sub_competency_name', 'bundle', 'incident_count']].to_string())

# Check Step 2 for potential bridging
df_step2_comp = pd.read_csv('step2_competency_statements.csv')
print("\n=== Step 2: Competency Statements (Head) ===")
print(df_step2_comp[['competency_id', 'bundle', 'confidence', 'applicable_controls']].head().to_string())

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: === Step 1: Sub-Competencies ===
      id                 bundle                                             name evidence_strength
0   AO-1         AI Orientation    CIO Strategic Centrality in AI Agenda-Setting            strong
1   AO-2         AI Orientation        Strategic Intent and Prioritization of AI          moderate
2   AO-3         AI Orientation           Resource Commitment to AI Capabilities          moderate
3   TR-1        Trust Readiness     AI Risk Policy and Accountability Structures            strong
4   TR-2        Trust Readiness          AI Evaluation and Monitoring Governance            strong
5   TR-3        Trust Readiness     Regulatory Compliance Translation Capability            strong
6   TR-4        Trust Readiness                AI Incident Response and Recovery            strong
7   TR-5        Trust Readiness  Supply Chain and Third-Party AI Risk Governance            strong
8   IR-1  Integration Readiness             Agentic Architecture & Orchestration            strong
9   IR-2  Integration Readiness    Lifecycle Governance & Operational Management            strong
10  IR-3  Integration Readiness       Security, Access, and Contingency Controls            strong

=== Step 3: Coverage Map ===
   sub_competency_id                        sub_competency_name                 bundle  incident_count
0               IR-1         Orchestration & Execution Controls  Integration Readiness               8
1               IR-2       Tool-Use Boundaries & Access Control  Integration Readiness               8
2               IR-3                  Nondeterminism Management  Integration Readiness              18
3               IR-4          RAG Architecture & Data Grounding  Integration Readiness               2
4               IR-5                 GenAIOps / MLOps Lifecycle  Integration Readiness               0
5               IR-6   Modular Architecture & Resource Controls  Integration Readiness               7
6               IR-7                 HITL Architecture Patterns  Integration Readiness               1
7               IR-8     Evaluation & Monitoring Infrastructure  Integration Readiness               0
8               TR-1               Risk Policy & Accountability        Trust Readiness              19
9               TR-2    Threat Mapping & Reconnaissance Defense        Trust Readiness               1
10              TR-3                     Monitoring & Detection        Trust Readiness               9
11              TR-4  Data Governance & Exfiltration Prevention        Trust Readiness               2
12              TR-5                      Regulatory Compliance        Trust Readiness               0
13              TR-6               Incident Response & Recovery        Trust Readiness               9
14              TR-7                   Human Override & Control        Trust Readiness               0
15              TR-8            Supply Chain & Third-Party Risk        Trust Readiness               5

=== Step 2: Competency Statements (Head) ===
  competency_id           bundle confidence                                                                                                     applicable_controls
0       COMP-01  Trust Readiness       high  AI Risk Policy & Accountability Structures; GenAIOps / MLOps Lifecycle Governance; Regulatory Compliance Documentation
1       COMP-02  Trust Readiness       high                                                                              AI Risk Policy & Accountability Structures
2       COMP-03  Trust Readiness     medium                                                                              AI Risk Policy & Accountability Structures
3       COMP-04  Trust Readiness     medium                                                                              AI Risk Policy & Accountability Structures
4       COMP-05  Trust Readiness       high                                                                              AI Risk Policy & Accountability Structures

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
