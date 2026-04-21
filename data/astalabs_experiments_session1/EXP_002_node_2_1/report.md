# Experiment 2: node_2_1

| Property | Value |
|---|---|
| **Experiment ID** | `node_2_1` |
| **ID in Run** | 2 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T21:31:16.722208+00:00 |
| **Runtime** | 169.3s |
| **Parent** | `node_1_0` |
| **Children** | `node_3_1`, `node_3_5`, `node_3_17` |
| **Creation Index** | 3 |

---

## Hypothesis

> Competencies classified as 'Integration Readiness' have significantly higher
empirical 'Confidence' scores in the literature than 'Trust Readiness'
competencies, reflecting a mature engineering body of knowledge versus a nascent
governance frameworks.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7581 (Likely True) |
| **Posterior** | 0.1942 (Likely False) |
| **Surprise** | -0.6544 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 2.0 |
| Maybe True | 28.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 90.0 |

---

## Experiment Plan

**Objective:** Compare the evidence confidence levels between bundles.

### Steps
- 1. Load 'step2_competency_statements.csv'.
- 2. Map the categorical 'confidence' column to numerical values (High=3, Medium=2, Low=1).
- 3. Group by 'bundle'.
- 4. Calculate mean and standard deviation of confidence scores for Trust vs. Integration.
- 5. Perform a Mann-Whitney U test (ordinal data) to determine if one bundle has statistically higher confidence rankings than the other.

### Deliverables
- Mean confidence scores, Mann-Whitney U test statistic, and p-value.

---

## Analysis

The experiment successfully loaded 'step2_competency_statements.csv' and mapped
the categorical 'confidence' levels to numerical scores (High=3, Medium=2,
Low=1).

Descriptive statistics show nearly identical high levels of confidence for both
competency bundles:
- **Integration Readiness:** Mean = 2.87 (SD = 0.34, n = 23)
- **Trust Readiness:** Mean = 2.84 (SD = 0.37, n = 19)

The Mann-Whitney U test resulted in a U-statistic of 224.5 and a p-value of
0.81867. Since the p-value is significantly greater than 0.05, the difference is
not statistically significant.

**Conclusion:** The hypothesis that 'Integration Readiness' competencies have
higher empirical confidence than 'Trust Readiness' is **not supported**. The
analysis indicates that both the engineering-focused (Integration) and
governance-focused (Trust) competencies in this dataset are supported by
similarly high-confidence evidence from the literature sources.

---

## Review

The experiment was faithfully implemented according to the plan. The code
successfully loaded the target dataset ('step2_competency_statements.csv'),
correctly handled file path variations, mapped categorical confidence ratings to
numerical values, and executed the specified statistical test (Mann-Whitney U).

**Summary of Findings:**
- **Hypothesis:** The experiment tested whether 'Integration Readiness'
competencies possess significantly higher evidence confidence than 'Trust
Readiness' competencies.
- **Results:** The analysis showed nearly identical mean confidence scores for
Integration Readiness (Mean=2.87, SD=0.34) and Trust Readiness (Mean=2.84,
SD=0.37). The Mann-Whitney U test yielded a p-value of 0.819, which is far above
the significance threshold (alpha=0.05).
- **Conclusion:** The hypothesis is rejected. There is no statistically
significant difference in the strength of literature evidence between the two
competency bundles; both are supported by high-confidence sources.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import os
import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

try:
    from scipy.stats import mannwhitneyu
except ImportError:
    install('scipy')
    from scipy.stats import mannwhitneyu

# Handle file loading based on location note
filename = 'step2_competency_statements.csv'
possible_paths = [f"../{filename}", filename]
file_path = None

for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        break

if not file_path:
    print(f"Error: Could not find {filename} in {possible_paths}")
else:
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)

    # Map categorical confidence to numerical scores
    confidence_map = {'high': 3, 'medium': 2, 'low': 1}
    
    # Normalize string to handle case/whitespace
    df['confidence_mapped'] = df['confidence'].astype(str).str.lower().str.strip().map(confidence_map)

    # Validation
    if df['confidence_mapped'].isnull().any():
        print("Warning: Some confidence values could not be mapped. Dropping NaNs.")
        print("Unique raw values:", df['confidence'].unique())
        df = df.dropna(subset=['confidence_mapped'])

    # Group by bundle
    trust_bundle = 'Trust Readiness'
    integration_bundle = 'Integration Readiness'
    
    group_stats = df.groupby('bundle')['confidence_mapped'].agg(['count', 'mean', 'std'])
    print("\n=== Descriptive Statistics (Confidence Score: High=3, Medium=2, Low=1) ===")
    print(group_stats)

    # Prepare vectors for statistical test
    trust_scores = df[df['bundle'] == trust_bundle]['confidence_mapped']
    integration_scores = df[df['bundle'] == integration_bundle]['confidence_mapped']

    if len(trust_scores) == 0 or len(integration_scores) == 0:
        print("\nError: One of the bundles has no data. Cannot perform statistical test.")
    else:
        # Mann-Whitney U Test (Two-sided)
        # Using two-sided to detect any difference, though hypothesis suggests Integration > Trust
        u_stat, p_val = mannwhitneyu(integration_scores, trust_scores, alternative='two-sided')

        print("\n=== Mann-Whitney U Test Results ===")
        print(f"Comparison: {integration_bundle} vs {trust_bundle}")
        print(f"U-statistic: {u_stat}")
        print(f"P-value: {p_val:.5f}")
        
        alpha = 0.05
        if p_val < alpha:
            direction = integration_bundle if integration_scores.mean() > trust_scores.mean() else trust_bundle
            print(f"Conclusion: Statistically significant difference detected (p < {alpha}).")
            print(f"Direction: {direction} has higher confidence scores.")
        else:
            print(f"Conclusion: No statistically significant difference detected (p >= {alpha}).")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step2_competency_statements.csv

=== Descriptive Statistics (Confidence Score: High=3, Medium=2, Low=1) ===
                       count      mean       std
bundle                                          
Integration Readiness     23  2.869565  0.344350
Trust Readiness           19  2.842105  0.374634

=== Mann-Whitney U Test Results ===
Comparison: Integration Readiness vs Trust Readiness
U-statistic: 224.5
P-value: 0.81867
Conclusion: No statistically significant difference detected (p >= 0.05).

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
