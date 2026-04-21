# Experiment 211: node_6_37

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_37` |
| **ID in Run** | 211 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T10:53:02.706669+00:00 |
| **Runtime** | 263.0s |
| **Parent** | `node_5_78` |
| **Children** | None |
| **Creation Index** | 212 |

---

## Hypothesis

> The 'Financial Malice' Profile: The 'Financial and Insurance' sector has a
significantly higher proportion of incidents labeled as 'Intentional Harm'
(Adversarial) compared to the 'Healthcare' sector, which is dominated by
accidental/negligent failures.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7258 (Likely True) |
| **Posterior** | 0.8077 (Likely True) |
| **Surprise** | +0.0983 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 29.0 |
| Uncertain | 0.0 |
| Maybe False | 1.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 24.0 |
| Maybe True | 36.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Compare the adversarial nature of incidents between Finance and Healthcare.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `aiid_incidents`.
- 2. Filter for `Sector of Deployment` containing 'Financial' or 'Healthcare'.
- 3. Create `Is_Intentional`: True if `Intentional Harm` is 'Yes' or 'True'.
- 4. Compare the rate of Intentional incidents between the two sectors using a Z-test.

### Deliverables
- 1. Intentional Harm rates for Finance vs. Healthcare.
- 2. Z-test results.
- 3. Interpretation of the sector risk profile.

---

## Analysis

The experiment successfully tested the 'Financial Malice' hypothesis using the
AIID dataset. The code correctly loaded the data, filtered for the relevant
sectors, and performed a two-proportion Z-test.

1. **Data Segmentation**: The analysis identified a small subset of relevant
incidents: 7 in 'Finance & Insurance' and 15 in 'Healthcare'.
2. **Intentionality Rates**: The hypothesis was strongly supported by the raw
rates. The Finance sector showed a 28.6% rate of intentional harm (2/7
incidents), whereas the Healthcare sector showed a 0.0% rate (0/15 incidents).
3. **Statistical Outcome**: Despite the small sample sizes, the difference was
found to be statistically significant (Z-score: 2.17, p-value: 0.0299), which is
below the standard alpha of 0.05.
4. **Conclusion**: The hypothesis is **SUPPORTED**. The analysis indicates that
AI incidents in the Financial sector are significantly more likely to be
adversarial/intentional compared to the Healthcare sector, which appears
dominated by accidental failures in this dataset.

---

## Review

The experiment successfully tested the 'Financial Malice' hypothesis using the
AIID dataset.

**Hypothesis**: The 'Financial Malice' Profile (that Financial/Insurance sectors
have a significantly higher proportion of 'Intentional Harm' incidents compared
to Healthcare) was **SUPPORTED** by the analysis.

**Findings**:
1. **Data Segmentation**: The code correctly identified relevant incidents,
though the sample sizes were small: 7 incidents in the 'Financial and Insurance'
sector and 15 in 'Healthcare'.
2. **Intentionality Rates**: The Financial sector displayed a significantly
higher rate of intentional/adversarial harm (28.6%, 2/7 incidents) compared to
the Healthcare sector, which had zero intentional incidents (0.0%, 0/15).
3. **Statistical Outcome**: The two-proportion Z-test yielded a Z-score of
2.1712 and a p-value of 0.0299. Since p < 0.05, the difference is statistically
significant.
4. **Interpretation**: The results confirm the hypothesis for this dataset,
suggesting that while Healthcare AI failures are predominantly accidental
(negligence/safety issues), Financial AI failures have a distinct profile
involving a significant component of malicious or adversarial intent (e.g.,
fraud, cyberattacks).

---

## Code

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import sys

# [debug] Print standard start message
print("Starting Financial Malice Hypothesis Experiment...\n")

# 1. Load the dataset with robust path checking
file_name = 'astalabs_discovery_all_data.csv'
possible_paths = [file_name, f'../{file_name}']
data_path = None

for path in possible_paths:
    if os.path.exists(path):
        data_path = path
        break

if data_path is None:
    print(f"Error: Could not find {file_name} in current or parent directory.")
    # Debugging info to help locate file if this fails again
    print("Current working directory:", os.getcwd())
    print("Files in current directory:", os.listdir('.'))
    try:
        print("Files in parent directory:", os.listdir('..'))
    except Exception as e:
        print("Could not list parent directory:", e)
    sys.exit(1)

print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path, low_memory=False)

# 2. Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents subset shape: {aiid_df.shape}")

# 3. Identify Column Names
# Based on metadata, columns of interest are 'Sector of Deployment' and 'Intentional Harm'
sector_col = 'Sector of Deployment'
intent_col = 'Intentional Harm'

# Verify columns exist
missing_cols = [c for c in [sector_col, intent_col] if c not in aiid_df.columns]
if missing_cols:
    print(f"Error: Missing columns {missing_cols}")
    print("Available columns:", aiid_df.columns.tolist())
    sys.exit(1)

# 4. Standardize Sector and Intentionality
aiid_df['sector_norm'] = aiid_df[sector_col].astype(str).str.lower()
aiid_df['intent_norm'] = aiid_df[intent_col].astype(str).str.lower()

# Define Filters
# Finance: 'financial', 'insurance', 'finance'
# Healthcare: 'healthcare', 'medical', 'health'
finance_mask = aiid_df['sector_norm'].str.contains('financ|insur', na=False)
health_mask = aiid_df['sector_norm'].str.contains('health|medic', na=False)

finance_df = aiid_df[finance_mask]
health_df = aiid_df[health_mask]

print(f"Finance/Insurance incidents found: {len(finance_df)}")
print(f"Healthcare incidents found: {len(health_df)}")

# Define Intentionality Logic
# We consider 'true' or 'yes' as intentional. Note: AIID data often has 'true'/'false' strings or booleans.
def is_intentional(val):
    v = str(val).lower()
    return 'true' in v or 'yes' in v

finance_intent_count = finance_df['intent_norm'].apply(is_intentional).sum()
health_intent_count = health_df['intent_norm'].apply(is_intentional).sum()

n_finance = len(finance_df)
n_health = len(health_df)

if n_finance == 0 or n_health == 0:
    print("Error: One of the sectors has 0 records. Cannot perform statistical test.")
    sys.exit(0)

prop_finance = finance_intent_count / n_finance
prop_health = health_intent_count / n_health

print(f"\n--- Results ---")
print(f"Finance: {finance_intent_count}/{n_finance} ({prop_finance:.2%}) intentional")
print(f"Healthcare: {health_intent_count}/{n_health} ({prop_health:.2%}) intentional")

# 5. Statistical Test (Two-proportion Z-test)
# Pooled proportion
p_pooled = (finance_intent_count + health_intent_count) / (n_finance + n_health)
se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_finance + 1/n_health))

if se == 0:
    print("Standard Error is 0, cannot calculate Z-score (proportions might be identical or zero).")
    z_score = 0
    p_value = 1.0
else:
    z_score = (prop_finance - prop_health) / se
    p_value = stats.norm.sf(abs(z_score)) * 2  # Two-tailed test

print(f"\nZ-score: {z_score:.4f}")
print(f"P-value: {p_value:.4e}")

if p_value < 0.05:
    print("Conclusion: Statistically Significant Difference.")
else:
    print("Conclusion: No Statistically Significant Difference.")

# 6. Visualization
labels = ['Finance & Insurance', 'Healthcare']
intent_rates = [prop_finance, prop_health]
accidental_rates = [1-prop_finance, 1-prop_health]

fig, ax = plt.subplots(figsize=(8, 6))

bar_width = 0.5
x_pos = np.arange(len(labels))

# Stacked bar chart
p1 = ax.bar(x_pos, intent_rates, bar_width, label='Intentional Harm', color='#d62728', alpha=0.8)
p2 = ax.bar(x_pos, accidental_rates, bar_width, bottom=intent_rates, label='Accidental/Other', color='#1f77b4', alpha=0.6)

ax.set_ylabel('Proportion of Incidents')
ax.set_title('Intentional Harm Rate: Finance vs Healthcare')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.legend(loc='lower right')

# Add percentage labels
for i, v in enumerate(intent_rates):
    if v > 0.05: # Only label if bar is visible enough
        ax.text(i, v/2, f"{v:.1%}", ha='center', va='center', color='white', fontweight='bold')

for i, v in enumerate(accidental_rates):
    if v > 0.05:
        ax.text(i, intent_rates[i] + v/2, f"{v:.1%}", ha='center', va='center', color='white')

plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting Financial Malice Hypothesis Experiment...

Loading dataset from: astalabs_discovery_all_data.csv
AIID Incidents subset shape: (1362, 196)
Finance/Insurance incidents found: 7
Healthcare incidents found: 15

--- Results ---
Finance: 2/7 (28.57%) intentional
Healthcare: 0/15 (0.00%) intentional

Z-score: 2.1712
P-value: 2.9913e-02
Conclusion: Statistically Significant Difference.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Stacked Bar Chart (specifically a Normalized or 100% Stacked Bar Chart).
*   **Purpose:** This chart is designed to compare the relative proportions (composition) of incident types ("Intentional Harm" vs. "Accidental/Other") across two distinct industry sectors ("Finance & Insurance" and "Healthcare"). By normalizing the height to 1.0 (100%), it focuses on the ratio of incidents rather than the total volume of incidents.

### 2. Axes
*   **X-Axis:**
    *   **Label/Title:** Categories representing industry sectors.
    *   **Values:** "Finance & Insurance" and "Healthcare".
*   **Y-Axis:**
    *   **Label/Title:** "Proportion of Incidents".
    *   **Units:** The axis uses decimal proportions representing percentages.
    *   **Range:** 0.0 to 1.0 (equivalent to 0% to 100%). Intervals are marked every 0.2 units.

### 3. Data Trends
*   **Finance & Insurance:**
    *   The column is divided into two distinct segments.
    *   The bottom segment (Red) indicates that a significant portion of incidents are classified as **Intentional Harm**.
    *   The top segment (Blue) indicates the majority of incidents are **Accidental/Other**.
*   **Healthcare:**
    *   The column represents a single, uniform block of color (Blue).
    *   There is no visible red segment, indicating that **Intentional Harm** is absent or statistically negligible in this specific dataset for the Healthcare sector.
    *   All recorded incidents fall under the **Accidental/Other** category.

### 4. Annotations and Legends
*   **Legend:** Located at the bottom right of the chart.
    *   **Red Square:** Represents "Intentional Harm".
    *   **Blue Square:** Represents "Accidental/Other".
*   **Annotations (Data Labels):** White text overlaid on the bars provides precise percentages.
    *   **Finance & Insurance:**
        *   **28.6%** for Intentional Harm (Red section).
        *   **71.4%** for Accidental/Other (Blue section).
    *   **Healthcare:**
        *   **100.0%** for Accidental/Other (Blue section).

### 5. Statistical Insights
*   **Disparity in Intent:** There is a stark contrast in the nature of incidents between the two sectors. While the Healthcare sector (in this dataset) deals exclusively with accidental or non-malicious issues, the Finance & Insurance sector faces a substantial threat from intentional actors.
*   **Risk Profile:** Nearly 3 out of every 10 incidents (28.6%) in Finance & Insurance are intentional. This suggests that security protocols in Finance need to be heavily weighted toward preventing fraud, insider threats, or cyberattacks, whereas Healthcare risk mitigation (based on this chart) should focus on error reduction, process safety, and accident prevention.
*   **Absence of Malice in Healthcare Data:** The 100.0% "Accidental/Other" rate in Healthcare is notable. It implies that in the context of this specific study or data collection period, malicious intent was not a contributing factor to any recorded operational incidents.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
