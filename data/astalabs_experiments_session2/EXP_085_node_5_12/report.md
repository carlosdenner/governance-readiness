# Experiment 85: node_5_12

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_12` |
| **ID in Run** | 85 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T05:09:36.353863+00:00 |
| **Runtime** | 217.1s |
| **Parent** | `node_4_21` |
| **Children** | `node_6_14` |
| **Creation Index** | 86 |

---

## Hypothesis

> The Public Scrutiny Effect: AI systems designated as 'Publicly Accessible'
(providing public information or services) are significantly more likely to
comply with 'AI Notice' requirements compared to internal-facing systems, driven
by external transparency pressures.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.9121 (Definitely True) |
| **Surprise** | +0.2042 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 30.0 |
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

**Objective:** Compare the rate of 'AI Notice' compliance between public-facing and internal-only AI systems in the federal inventory.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `eo13960_scored`.
- 2. Create a binary variable `is_public` based on columns `26_public_service` and `27_public_info`. If either is 'Yes'/'True', set to 1, else 0.
- 3. Create a binary variable `has_notice` based on column `59_ai_notice` (convert 'Yes' to 1, others to 0).
- 4. Generate a contingency table of `is_public` vs. `has_notice`.
- 5. Perform a Chi-square test of independence.
- 6. Calculate and compare the percentage of notice compliance for both groups.

### Deliverables
- Contingency table, Chi-square statistics (p-value), and compliance percentages for public vs. internal systems.

---

## Analysis

The experiment successfully tested the 'Public Scrutiny Effect' hypothesis.
After correcting for text-based data representation in the 'AI Notice' and
'Public Service' columns (which previously caused debug failures), the analysis
classified 1,757 federal AI systems into Public-Facing (n=255) and Internal-Only
(n=1,502) groups.

**Findings:**
1. **Compliance Rates:** Public-facing systems demonstrated a significantly
higher rate of AI Notice compliance (10.20%, 26/255) compared to internal
systems (3.00%, 45/1,502).
2. **Statistical Significance:** A Chi-Square test confirmed this difference is
highly significant (Chi2 = 27.32, p < 0.0001).

**Conclusion:**
The hypothesis is **supported**. Systems interacting with the public are over 3
times more likely to provide AI transparency notices than internal
administrative systems, suggesting that external visibility acts as a strong
driver for transparency compliance.

---

## Review

The experiment was successfully executed and the code faithfully implemented the
analysis plan. After correcting for text-based data representation in the 'AI
Notice' and 'Public Service' columns (which previously caused debug failures),
the analysis successfully classified 1,757 federal AI systems into Public-Facing
(n=255) and Internal-Only (n=1,502) groups.

**Hypothesis:** The Public Scrutiny Effect (Public-facing systems are more
likely to provide AI Notice).

**Findings:**
1.  **Compliance Rates:** Public-facing systems demonstrated a significantly
higher rate of AI Notice compliance (10.20%, 26/255) compared to internal
systems (3.00%, 45/1,502).
2.  **Statistical Test:** A Chi-Square test confirmed this difference is highly
significant (Chi2 = 27.32, p < 0.0001).

**Conclusion:**
The hypothesis is **supported**. Systems interacting with the public are over 3
times more likely to provide AI transparency notices than internal
administrative systems, suggesting that external visibility acts as a strong
driver for transparency compliance.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

# Load dataset
file_path = "astalabs_discovery_all_data.csv"
if not os.path.exists(file_path):
    file_path = "../astalabs_discovery_all_data.csv"

df = pd.read_csv(file_path, low_memory=False)

# Filter for relevant source table
subset = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Subset size: {len(subset)}")

# --- Logic for 'is_public' ---
# Column 26: Public Service (Descriptive text = Yes, 'No'/Empty = No)
def map_public_service(val):
    if pd.isna(val):
        return 0
    s = str(val).strip()
    if len(s) < 2:  # Filters out empty strings, single chars like ' '
        return 0
    if s.lower() in ['no', 'none', 'n/a']:
        return 0
    return 1  # Contains description of service

# Column 27: Public Info (Explicit 'Yes')
def map_public_info(val):
    if pd.isna(val):
        return 0
    if str(val).strip().lower() == 'yes':
        return 1
    return 0

subset['public_service_flag'] = subset['26_public_service'].apply(map_public_service)
subset['public_info_flag'] = subset['27_public_info'].apply(map_public_info)
subset['is_public'] = ((subset['public_service_flag'] == 1) | (subset['public_info_flag'] == 1)).astype(int)

# --- Logic for 'has_notice' ---
# Column 59: AI Notice
# Positives: 'Online', 'In-person', 'Email', 'Other', 'Telephone'
# Negatives: NaN, 'None of the above', 'N/A', 'Waived', 'Not safety'
def map_notice(val):
    if pd.isna(val):
        return 0
    s = str(val).strip().lower()
    # Negative keywords
    negatives = ['none of the above', 'n/a', 'waived', 'not safety', 'nan']
    if any(neg in s for neg in negatives):
        return 0
    # If it's not negative and has content, assume positive notice info
    if len(s) > 2:
        return 1
    return 0

subset['has_notice'] = subset['59_ai_notice'].apply(map_notice)

# --- Analysis ---
print("\n--- Value Counts ---")
print(f"Public Systems (is_public=1): {subset['is_public'].sum()}")
print(f"Internal Systems (is_public=0): {len(subset) - subset['is_public'].sum()}")
print(f"Systems with Notice (has_notice=1): {subset['has_notice'].sum()}")

# Generate Contingency Table with reindexing to ensure 2x2
contingency_table = pd.crosstab(subset['is_public'], subset['has_notice'])
# Reindex to ensure all categories exist (0 and 1)
contingency_table = contingency_table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)

contingency_table.index = ['Internal', 'Public']
contingency_table.columns = ['No Notice', 'Has Notice']

print("\n--- Contingency Table ---")
print(contingency_table)

# Percentages
internal_total = contingency_table.loc['Internal'].sum()
public_total = contingency_table.loc['Public'].sum()

internal_compliance = (contingency_table.loc['Internal', 'Has Notice'] / internal_total * 100) if internal_total > 0 else 0
public_compliance = (contingency_table.loc['Public', 'Has Notice'] / public_total * 100) if public_total > 0 else 0

print(f"\nInternal Compliance: {internal_compliance:.2f}% (n={internal_total})")
print(f"Public Compliance:   {public_compliance:.2f}% (n={public_total})")

# Statistical Test
if internal_total > 0 and public_total > 0:
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print("\n--- Chi-Square Test Results ---")
    print(f"Chi2: {chi2:.4f}")
    print(f"p-value: {p:.4e}")
    
    if p < 0.05:
        print("Result: Significant difference found.")
        if public_compliance > internal_compliance:
            print("Hypothesis Supported: Public systems are more likely to have notice.")
        else:
            print("Hypothesis Refuted: Public systems are LESS likely to have notice.")
    else:
        print("Result: No significant difference.")
else:
    print("Insufficient data for test.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Subset size: 1757

--- Value Counts ---
Public Systems (is_public=1): 255
Internal Systems (is_public=0): 1502
Systems with Notice (has_notice=1): 71

--- Contingency Table ---
          No Notice  Has Notice
Internal       1457          45
Public          229          26

Internal Compliance: 3.00% (n=1502)
Public Compliance:   10.20% (n=255)

--- Chi-Square Test Results ---
Chi2: 27.3161
p-value: 1.7276e-07
Result: Significant difference found.
Hypothesis Supported: Public systems are more likely to have notice.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
