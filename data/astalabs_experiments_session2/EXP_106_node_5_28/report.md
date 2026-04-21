# Experiment 106: node_5_28

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_28` |
| **ID in Run** | 106 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T06:05:33.446562+00:00 |
| **Runtime** | 180.4s |
| **Parent** | `node_4_2` |
| **Children** | `node_6_60` |
| **Creation Index** | 107 |

---

## Hypothesis

> The 'Shadow AI' Security Gap: Commercial AI procurements ('COTS') in federal
agencies are significantly less likely to report a formal 'Authority to Operate'
(ATO) compared to Custom/Government-developed systems, suggesting a reliance on
vendor assurances over standard security accreditation.

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

**Objective:** Compare security accreditation rates (ATO) between Commercial and Custom AI systems.

### Steps
- 1. Load 'eo13960_scored' subset.
- 2. Define 'is_commercial' based on '10_commercial_ai' (as done in previous explorations).
- 3. Define 'has_ato': Parse column '40_has_ato'. Look for 'Yes' or explicit confirmation strings; treat 'No', 'Pending', or blank as 0.
- 4. Construct a contingency table: Commercial Status vs ATO Presence.
- 5. Perform Chi-square test of independence and calculate Odds Ratio.

### Deliverables
- Contingency table showing ATO rates, Chi-square statistic, and Odds Ratio indicating the likelihood of ATO compliance for commercial systems.

---

## Analysis

The experiment successfully tested the 'Shadow AI' Security Gap hypothesis using
the EO 13960 dataset. The analysis processed 1,757 AI systems, classifying 1,404
as Custom/Government-developed and 353 as Commercial. The results reveal a
significant disparity in security accreditation: only 22.1% (78/353) of
Commercial systems have a confirmed 'Authority to Operate' (ATO), compared to
39.7% (558/1,404) of Custom/Gov systems. The Chi-square test confirms this
difference is statistically significant (Chi2 = 37.28, p < 0.001). The Odds
Ratio analysis indicates that Custom systems are approximately 2.33 times more
likely to hold a formal ATO than Commercial systems. This strongly supports the
hypothesis that commercial AI procurements in the federal sector frequently
bypass standard security accreditation processes, creating a potential 'Shadow
AI' governance gap where vendor solutions operate with less oversight than
internal developments.

---

## Review

The experiment was executed successfully and faithfully followed the plan. The
analysis of the EO 13960 dataset (n=1,757) strongly supports the 'Shadow AI'
Security Gap hypothesis. The results demonstrate a statistically significant
disparity in security accreditation: Custom/Government-developed AI systems are
more likely to have a formal 'Authority to Operate' (ATO) (39.7%) compared to
Commercial 'COTS' systems (22.1%). The Chi-square test (X2=37.28, p<0.001) and
Odds Ratio (Custom systems are ~2.33 times more likely to hold an ATO) confirm
that commercial AI procurements often bypass standard federal security
accreditation processes, validating the concern regarding 'Shadow AI'
governance.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored subset
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"Loaded EO 13960 dataset with {len(eo_data)} records.")

# --- Variable Construction ---

# 1. Define 'is_commercial'
# Logic: If '10_commercial_ai' is NaN or contains 'None of the above', it's Custom (0). Else Commercial (1).
def classify_commercial(val):
    if pd.isna(val):
        return 0
    s = str(val).lower()
    if 'none of the above' in s:
        return 0
    return 1

eo_data['is_commercial'] = eo_data['10_commercial_ai'].apply(classify_commercial)

# 2. Define 'has_ato'
# Logic: Parse column '40_has_ato'. Check if starts with 'yes'.
print("\nUnique values in '40_has_ato' (top 10):")
print(eo_data['40_has_ato'].value_counts().head(10))

def classify_ato(val):
    if pd.isna(val):
        return 0
    s = str(val).lower().strip()
    if s.startswith('yes'):
        return 1
    return 0

eo_data['has_ato'] = eo_data['40_has_ato'].apply(classify_ato)

# --- Analysis ---

# Contingency Table
contingency_table = pd.crosstab(eo_data['is_commercial'], eo_data['has_ato'])
contingency_table.index = ['Custom/Gov', 'Commercial']
contingency_table.columns = ['No ATO', 'Has ATO']

print("\n--- Contingency Table: Commercial Status vs. ATO ---")
print(contingency_table)

# Chi-square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Odds Ratio Calculation
# OR = (ad)/(bc)
# a = Custom, Has ATO
# b = Custom, No ATO
# c = Commercial, Has ATO
# d = Commercial, No ATO
# Note: crosstab structure is:
#             No ATO (0)   Has ATO (1)
# Custom (0)      A            B
# Comm   (1)      C            D
# So OR (Commercial having ATO vs Custom having ATO) = (D/C) / (B/A) = (D*A) / (C*B)

# Extract values
custom_no_ato = contingency_table.loc['Custom/Gov', 'No ATO']
custom_has_ato = contingency_table.loc['Custom/Gov', 'Has ATO']
comm_no_ato = contingency_table.loc['Commercial', 'No ATO']
comm_has_ato = contingency_table.loc['Commercial', 'Has ATO']

# Calculate rates
custom_rate = custom_has_ato / (custom_has_ato + custom_no_ato)
comm_rate = comm_has_ato / (comm_has_ato + comm_no_ato)

print(f"\nATO Rate (Custom/Gov): {custom_rate:.1%} ({custom_has_ato}/{custom_has_ato + custom_no_ato})")
print(f"ATO Rate (Commercial): {comm_rate:.1%} ({comm_has_ato}/{comm_has_ato + comm_no_ato})")

try:
    odds_ratio = (comm_has_ato * custom_no_ato) / (comm_no_ato * custom_has_ato)
    print(f"\nOdds Ratio (Commercial vs Custom for having ATO): {odds_ratio:.4f}")
    
    # Inverse OR for interpretation if Custom is higher
    if odds_ratio < 1:
        inv_or = 1 / odds_ratio
        print(f"Interpretation: Custom systems are {inv_or:.2f} times more likely to have an ATO than Commercial systems.")
    else:
        print(f"Interpretation: Commercial systems are {odds_ratio:.2f} times more likely to have an ATO than Custom systems.")
except ZeroDivisionError:
    print("\nCannot calculate Odds Ratio due to zero division.")

# Visualization
plt.figure(figsize=(8, 6))
rates = [custom_rate, comm_rate]
labels = ['Custom/Gov', 'Commercial']
colors = ['#1f77b4', '#ff7f0e']

bars = plt.bar(labels, rates, color=colors)
plt.ylabel('ATO Compliance Rate')
plt.title('ATO Compliance: Custom vs. Commercial AI')
plt.ylim(0, 1.0)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.1%}", ha='center', va='bottom')

plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded EO 13960 dataset with 1757 records.

Unique values in '40_has_ato' (top 10):
40_has_ato
Yes                                             636
No                                              448
                                                 15
No.  Engineering Review and Resease instead.     15
                                                  2
Operated in an approved enclave                   2
Data.State-SBU                                    1
Name: count, dtype: int64

--- Contingency Table: Commercial Status vs. ATO ---
            No ATO  Has ATO
Custom/Gov     846      558
Commercial     275       78

Chi-Square Statistic: 37.2767
P-value: 1.0250e-09

ATO Rate (Custom/Gov): 39.7% (558/1404)
ATO Rate (Commercial): 22.1% (78/353)

Odds Ratio (Commercial vs Custom for having ATO): 0.4300
Interpretation: Custom systems are 2.33 times more likely to have an ATO than Commercial systems.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Chart.
*   **Purpose:** The plot compares the ATO (Authority to Operate) compliance rates between two distinct categories of AI solutions: "Custom/Gov" and "Commercial."

### 2. Axes
*   **Y-Axis (Vertical):**
    *   **Label:** "ATO Compliance Rate".
    *   **Range:** The scale ranges from **0.0 to 1.0** (representing 0% to 100%).
    *   **Increments:** The axis is marked in increments of 0.2.
*   **X-Axis (Horizontal):**
    *   **Labels:** The categories represent the source/type of AI: **"Custom/Gov"** and **"Commercial"**.

### 3. Data Trends
*   **Tallest Bar:** The blue bar representing "Custom/Gov" is the tallest, indicating a higher compliance rate.
*   **Shortest Bar:** The orange bar representing "Commercial" is the shortest, indicating a lower compliance rate.
*   **Comparison:** There is a visibly significant gap between the two categories, with the Custom/Gov bar appearing nearly twice as tall as the Commercial bar.

### 4. Annotations and Legends
*   **Title:** "ATO Compliance: Custom vs. Commercial AI".
*   **Value Labels:** Specific percentage values are annotated directly above each bar to provide precise data points:
    *   **Custom/Gov:** Annotated with **39.7%**.
    *   **Commercial:** Annotated with **22.1%**.
*   **Color Coding:** While there is no separate legend box, the bars are distinct colors (Blue for Custom/Gov, Orange for Commercial) to aid visual differentiation.

### 5. Statistical Insights
*   **Significant Performance Gap:** Custom or Government-developed AI solutions have a significantly higher success rate in achieving ATO compliance compared to Commercial solutions. The Custom/Gov sector outperforms the Commercial sector by **17.6 percentage points** ($39.7\% - 22.1\%$).
*   **Relative Ratio:** The compliance rate for Custom/Gov AI is approximately **1.8 times higher** than that of Commercial AI ($39.7 / 22.1 \approx 1.79$).
*   **Overall Low Compliance:** Despite the Custom/Gov sector performing better, both categories show relatively low compliance rates (both are below 40%). This suggests that obtaining ATO compliance is a difficult hurdle for AI systems in general, regardless of their origin.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
