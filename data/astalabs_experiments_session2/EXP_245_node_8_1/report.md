# Experiment 245: node_8_1

| Property | Value |
|---|---|
| **Experiment ID** | `node_8_1` |
| **ID in Run** | 245 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T12:38:00.219903+00:00 |
| **Runtime** | 291.2s |
| **Parent** | `node_7_8` |
| **Children** | None |
| **Creation Index** | 246 |

---

## Hypothesis

> Commercial Opacity: Commercially acquired AI systems are significantly less
likely to provide 'Code Access' for governance review compared to government-
developed systems, creating a transparency barrier due to intellectual property
constraints.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9839 (Definitely True) |
| **Posterior** | 0.9945 (Definitely True) |
| **Surprise** | +0.0128 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 30.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 60.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Compare code accessibility rates between commercial and government-developed AI systems.

### Steps
- 1. Load 'eo13960_scored' from the dataset.
- 2. Filter for rows where '10_commercial_ai' and '38_code_access' are not null.
- 3. Create two groups based on '10_commercial_ai': 'Commercial' (Yes) and 'Custom/Government' (No).
- 4. Convert '38_code_access' to binary (Yes=1, No=0).
- 5. Calculate the percentage of systems with code access in each group.
- 6. Perform a Chi-square test of independence to assess statistical significance.

### Deliverables
- Compliance rates for both groups; Chi-square test statistics and p-value.

---

## Analysis

The experiment successfully tested the 'Commercial Opacity' hypothesis using the
EO 13960 dataset. By correctly identifying '22_dev_method' as the proxy for
system origin (Contracted vs. In-house) and applying fuzzy matching to the
'38_code_access' column, the code generated a valid dataset of 781 systems.

The results provide strong statistical support for the hypothesis:
- **Government (In-house) Systems**: High transparency, with 90.5% (354/391)
providing source code access.
- **Commercial (Contracted) Systems**: Low transparency, with only 33.8%
(132/390) providing source code access.

The Chi-square test yielded a massive statistic (264.56) and a p-value near zero
(1.74e-59), confirming that intellectual property or vendor constraints in
contracted AI systems create a significant barrier to governance transparency
compared to government-developed tools.

---

## Review

The experiment was successfully executed and the analysis is methodologically
sound. The programmer correctly pivoted to using '22_dev_method' as a proxy for
the 'Commercial' vs. 'Government' distinction after the initial attempt revealed
that '10_commercial_ai' contained use-case descriptions rather than origin data.
The logic to parse '38_code_access' using fuzzy matching (startswith)
effectively handled the verbose response strings in the dataset.

The results provide robust statistical support for the 'Commercial Opacity'
hypothesis. With a sample size of n=781, the analysis revealed a stark contrast
in transparency: 90.5% of in-house government systems provide source code access
for governance review, compared to only 33.8% of commercially contracted
systems. The Chi-square test (p < 1e-50) confirms that this is not a random
fluctuation but a systemic barrier, likely driven by intellectual property
constraints in vendor agreements.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import sys
import os

# Load dataset
paths = ['../astalabs_discovery_all_data.csv', 'astalabs_discovery_all_data.csv']
df = None
for p in paths:
    if os.path.exists(p):
        df = pd.read_csv(p, low_memory=False)
        break

if df is None:
    print("Dataset not found.")
    sys.exit(1)

# Filter for EO 13960 Scored table
subset = df[df['source_table'] == 'eo13960_scored'].copy()

# define columns
col_dev_method = '22_dev_method'
col_code_access = '38_code_access'

# Drop rows with NaN in critical columns
subset = subset.dropna(subset=[col_dev_method, col_code_access])

# Define Groups based on Development Method
# Hypothesis: Commercial (Contracted) vs Government (In-house)
def classify_source(val):
    val = str(val).strip()
    if 'contracting resources' in val and 'in-house' not in val:
        return 'Commercial'
    elif 'in-house' in val and 'contracting' not in val:
        return 'Government'
    else:
        return None

subset['group'] = subset[col_dev_method].apply(classify_source)

# Filter only for the two groups of interest
subset = subset[subset['group'].notna()]

# Define Code Access (Yes/No)
def classify_access(val):
    val = str(val).strip().upper()
    if val.startswith('YES'):
        return 'Yes'
    elif val.startswith('NO'):
        return 'No'
    else:
        return None

subset['access_binary'] = subset[col_code_access].apply(classify_access)
subset = subset[subset['access_binary'].notna()]

# Summary stats
group_counts = subset.groupby(['group', 'access_binary']).size().unstack(fill_value=0)
print("Contingency Table (Group vs Code Access):")
print(group_counts)

# Calculate percentages
comm_stats = group_counts.loc['Commercial']
gov_stats = group_counts.loc['Government']

comm_total = comm_stats.sum()
comm_yes = comm_stats.get('Yes', 0)
comm_rate = (comm_yes / comm_total) * 100 if comm_total > 0 else 0

gov_total = gov_stats.sum()
gov_yes = gov_stats.get('Yes', 0)
gov_rate = (gov_yes / gov_total) * 100 if gov_total > 0 else 0

print(f"\nCommercial (Contracted) Code Access Rate: {comm_rate:.1f}% ({comm_yes}/{comm_total})")
print(f"Government (In-house) Code Access Rate:   {gov_rate:.1f}% ({gov_yes}/{gov_total})")

# Statistical Test
if comm_total > 0 and gov_total > 0:
    contingency = group_counts.values
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test Results:")
    print(f"Statistic: {chi2:.4f}")
    print(f"P-value: {p:.6e}")
    
    if p < 0.05:
        print("Result: Significant difference found.")
        if comm_rate < gov_rate:
            print("Supports Hypothesis: Commercial systems have significantly lower code access.")
        else:
            print("Contradicts Hypothesis: Commercial systems have higher code access.")
    else:
        print("Result: No significant difference found.")
else:
    print("Insufficient data for statistical test.")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Contingency Table (Group vs Code Access):
access_binary   No  Yes
group                  
Commercial     258  132
Government      37  354

Commercial (Contracted) Code Access Rate: 33.8% (132/390)
Government (In-house) Code Access Rate:   90.5% (354/391)

Chi-Square Test Results:
Statistic: 264.5628
P-value: 1.737640e-59
Result: Significant difference found.
Supports Hypothesis: Commercial systems have significantly lower code access.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
