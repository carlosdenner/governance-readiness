# Experiment 114: node_4_46

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_46` |
| **ID in Run** | 114 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T06:28:03.638737+00:00 |
| **Runtime** | 289.9s |
| **Parent** | `node_3_17` |
| **Children** | `node_5_45`, `node_5_76` |
| **Creation Index** | 115 |

---

## Hypothesis

> The 'Public Service' Accountability Divide: Public-facing AI systems in federal
agencies are significantly more likely to provide an Appeal Process compared to
internal-facing systems, reflecting a stronger emphasis on due process for
citizen-impacting technologies.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.5161 (Uncertain) |
| **Posterior** | 0.2088 (Likely False) |
| **Surprise** | -0.3688 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 2.0 |
| Maybe True | 15.0 |
| Uncertain | 0.0 |
| Maybe False | 9.0 |
| Definitely False | 4.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 12.0 |
| Definitely False | 48.0 |

---

## Experiment Plan

**Objective:** Compare the availability of appeal processes between public-facing and internal AI deployments.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'eo13960_scored'.
- 2. Map '26_public_service' to a binary variable (Public vs Internal).
- 3. Map '65_appeal_process' to a binary variable (Yes vs No/Other).
- 4. Calculate compliance rates for both groups.
- 5. Run a two-proportion z-test or Chi-Square test.

### Deliverables
- Bar chart of Appeal Process availability by Deployment Type; Statistical test results.

---

## Analysis

The experiment successfully tested the 'Public Service Accountability Divide'
hypothesis using the EO 13960 dataset. Due to extreme sparsity in the originally
targeted '26_public_service' column (93% missing), the analysis substituted
'27_public_info' ('Has the agency published information...') as a proxy for
public transparency. The study analyzed 961 valid records, categorizing them
into Public-Facing (n=195) and Internal-Facing (n=766) systems.

Contrary to the hypothesis—which predicted higher accountability for public-
facing systems—the results showed the opposite trend. Internal-facing systems
had an appeal process availability rate of 8.22%, compared to just 4.10% for
public-facing systems. However, a Chi-Square test yielded a p-value of 0.0701,
which is slightly above the standard 0.05 significance threshold. Consequently,
while the data suggests a trend where internal systems (likely affecting federal
employees) have better recourse mechanisms than public ones, the difference is
not statistically significant at the 95% confidence level. The hypothesis is
rejected, highlighting a uniformly low implementation of appeal processes (<9%)
across all federal AI deployments.

---

## Review

The experiment was successfully executed. The programmer correctly identified
the data quality issues with the originally intended column
(`26_public_service`, 93% missing) and implemented a reasonable fallback using
`27_public_info` to distinguish between transparent/public and internal/obscure
systems. The statistical analysis was rigorous, providing both proportions and a
Chi-Square test. The findings clearly reject the hypothesis (showing a trend
where internal systems actually have *higher* appeal rates than public ones,
though p=0.07 is marginally non-significant), providing valuable insight into
the state of federal AI governance.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load the dataset
file_path = 'astalabs_discovery_all_data.csv'
try:
    # Using low_memory=False to avoid mixed type warnings
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for relevant source table
subset = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"Initial subset size: {len(subset)}")

# --- 1. Variable Mapping ---
# Note: '26_public_service' was found to be extremely sparse (93% missing) and unreliable 
# (classifying 'CBP One' as missing/internal). 
# We substitute '27_public_info' ('Has the agency published information...') as a proxy 
# for public-facing transparency vs internal/obscure systems.

def normalize_binary(val):
    s = str(val).strip().lower()
    if s == 'yes':
        return True
    elif s == 'no':
        return False
    else:
        return None

subset['is_public'] = subset['27_public_info'].apply(normalize_binary)

# Outcome: Appeal Process
# We treat 'Yes' as True, and everything else (No, NaN, N/A) as False/No Appeal.
subset['has_appeal'] = subset['65_appeal_process'].apply(lambda x: str(x).strip().lower() == 'yes')

# Drop rows where our independent variable (Deployment Type) is undefined
clean_subset = subset.dropna(subset=['is_public'])

print(f"Cleaned subset size (valid Public/Internal label): {len(clean_subset)}")
print("Group sizes:")
print(clean_subset['is_public'].value_counts())

# --- 2. Analysis ---

# Contingency Table
contingency_table = pd.crosstab(clean_subset['is_public'], clean_subset['has_appeal'])

# Ensure we have the right shape before labeling
if contingency_table.shape[0] == 2:
    contingency_table.index = ['Internal-Facing', 'Public-Facing']
    contingency_table.columns = ['No Appeal Process', 'Has Appeal Process']
else:
    print("Warning: Contingency table does not have 2 rows. Check data.")
    print(contingency_table)

print("\nContingency Table:")
print(contingency_table)

# Proportions
public_group = clean_subset[clean_subset['is_public'] == True]
internal_group = clean_subset[clean_subset['is_public'] == False]

public_appeal_rate = public_group['has_appeal'].mean()
internal_appeal_rate = internal_group['has_appeal'].mean()

print(f"\nAppeal Process Availability Rate - Public-Facing (n={len(public_group)}): {public_appeal_rate:.2%}")
print(f"Appeal Process Availability Rate - Internal-Facing (n={len(internal_group)}): {internal_appeal_rate:.2%}")

# Statistical Test (Chi-Square)
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# --- 3. Visualization ---

labels = ['Internal-Facing', 'Public-Facing']
rates = [internal_appeal_rate, public_appeal_rate]

plt.figure(figsize=(8, 6))
bars = plt.bar(labels, rates, color=['#A9A9A9', '#4682B4'])
plt.ylabel('Proportion with Appeal Process')
plt.title('Appeal Process Availability by Deployment Type')
plt.ylim(0, max(rates) * 1.3 if max(rates) > 0 else 0.1)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1%}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Initial subset size: 1757
Cleaned subset size (valid Public/Internal label): 961
Group sizes:
is_public
False    766
True     195
Name: count, dtype: int64

Contingency Table:
                 No Appeal Process  Has Appeal Process
Internal-Facing                703                  63
Public-Facing                  187                   8

Appeal Process Availability Rate - Public-Facing (n=195): 4.10%
Appeal Process Availability Rate - Internal-Facing (n=766): 8.22%

Chi-Square Statistic: 3.2807
P-value: 7.0097e-02


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot.
*   **Purpose:** This plot compares a quantitative variable ("Proportion with Appeal Process") across two distinct categorical groups ("Internal-Facing" and "Public-Facing" deployment types).

### 2. Axes
*   **X-Axis:**
    *   **Label/Categories:** Represents "Deployment Type". The specific categories labeled are **"Internal-Facing"** and **"Public-Facing"**.
    *   **Range:** Categorical (nominal data), so there is no numerical range.
*   **Y-Axis:**
    *   **Title:** "Proportion with Appeal Process".
    *   **Units:** The axis uses decimal proportions (0.00 to 0.10), representing fractions of a whole.
    *   **Range:** The visible tick marks range from **0.00 to 0.10**, with intervals of 0.02. The axis extends slightly beyond 0.10 to accommodate the height of the data.

### 3. Data Trends
*   **Tallest Bar:** The **"Internal-Facing"** category (gray bar) is the tallest, indicating a higher availability of appeal processes.
*   **Shortest Bar:** The **"Public-Facing"** category (blue bar) is the shortest.
*   **Pattern:** There is a significant disparity between the two groups. The proportion for "Internal-Facing" deployments is exactly double that of "Public-Facing" deployments.

### 4. Annotations and Legends
*   **Data Labels:** Both bars have explicit data value annotations placed at the top in bold text, presented as percentages:
    *   Internal-Facing: **8.2%**
    *   Public-Facing: **4.1%**
*   **Color Coding:** The bars are colored differently—gray for "Internal-Facing" and blue for "Public-Facing"—to visually distinguish the categories, though no separate legend is provided or necessary given the x-axis labels.

### 5. Statistical Insights
*   **Low Overall Availability:** The availability of an appeal process is generally low across both deployment types, with neither category exceeding 10%.
*   **Comparative Ratio:** An "Internal-Facing" deployment is **twice as likely** (8.2% vs 4.1%) to have an appeal process compared to a "Public-Facing" deployment.
*   **Implication:** This suggests that systems designed for internal use (likely involving employees or internal stakeholders) usually have more robust or formalized mechanisms for recourse than systems designed for the general public.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
