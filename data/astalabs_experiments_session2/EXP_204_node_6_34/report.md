# Experiment 204: node_6_34

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_34` |
| **ID in Run** | 204 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T10:32:15.335150+00:00 |
| **Runtime** | 312.4s |
| **Parent** | `node_5_6` |
| **Children** | None |
| **Creation Index** | 205 |

---

## Hypothesis

> AI systems procured commercially ('COTS') exhibit significantly lower compliance
with 'Code Access' requirements compared to systems developed 'In-House',
creating a 'Transparency Black Box'.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9800 (Definitely True) |
| **Posterior** | 0.9937 (Definitely True) |
| **Surprise** | +0.0164 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 24.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 54.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Compare transparency controls between Commercial and In-House AI systems.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' (EO13960 subset).
- 2. Parse '10_commercial_ai' to classify systems as 'Commercial' vs. 'Custom/In-House'.
- 3. Parse '38_code_access' into binary (Access vs No Access).
- 4. Perform a Chi-Square test of independence.

### Deliverables
- Comparison plot of Code Access rates; Chi-Square test summary.

---

## Analysis

The experiment successfully tested the hypothesis that commercial AI systems
create a 'Transparency Black Box' compared to in-house development. After
correcting the data selection logic to use the `22_dev_method` column, the
analysis categorized 781 AI systems from the EO 13960 inventory.

The results provide overwhelming support for the hypothesis:

1.  **Stark Contrast in Transparency**: In-house developed systems demonstrated
a 90.5% compliance rate for code access. In sharp contrast, systems developed by
commercial contractors showed only a 33.8% compliance rate.
2.  **Statistical Significance**: The Chi-Square test yielded a statistic of
264.56 and a p-value of 1.74e-59, indicating that the association between
development method and code accessibility is highly statistically significant.
3.  **Operational Implication**: This 56.7 percentage point gap confirms that
federal agencies frequently deploy commercial AI tools without access to the
underlying source code, validating the concern that procurement mechanisms often
result in 'black box' governance risks where external vendors retain
intellectual property rights at the expense of government transparency.

---

## Review

The experiment successfully tested the hypothesis regarding the 'Transparency
Black Box' in commercial AI procurement. Although the initial attempt failed due
to an incorrect column assumption (`10_commercial_ai`), the programmer correctly
identified the issue via debugging and successfully pivoted to the correct
column (`22_dev_method`). The analysis of 781 AI systems revealed a massive,
statistically significant disparity: in-house systems have a 90.5% code access
rate compared to just 33.8% for commercial/contractor systems. The Chi-Square
test (p < 1.74e-59) overwhelmingly supports the hypothesis.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO13960 data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# --- Preprocessing System Origin using '22_dev_method' ---
# Previous attempt with '10_commercial_ai' failed as it contained use-case descriptions.
# '22_dev_method' contains 'Developed in-house.' and 'Developed with contracting resources.'

def classify_origin(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower()
    if 'in-house' in val_str and 'contracting' not in val_str:
        return 'In-House'
    elif 'contracting' in val_str and 'in-house' not in val_str:
        return 'Commercial/Contractor'
    return np.nan # Exclude 'Both' or others for clear comparison

eo_df['system_origin'] = eo_df['22_dev_method'].apply(classify_origin)

# --- Preprocessing '38_code_access' ---
def check_access(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower()
    # 'Yes - ...' counts as Yes.
    if val_str.startswith('yes'):
        return 1
    elif val_str.startswith('no'):
        return 0
    return np.nan

eo_df['has_code_access'] = eo_df['38_code_access'].apply(check_access)

# Filter analysis data
analysis_df = eo_df.dropna(subset=['system_origin', 'has_code_access'])

print(f"Analysis Data Shape: {analysis_df.shape}")
print("Group Counts:")
print(analysis_df['system_origin'].value_counts())

# --- Statistical Analysis ---
contingency_table = pd.crosstab(analysis_df['system_origin'], analysis_df['has_code_access'])
print("\nContingency Table (0=No Access, 1=Access):")
print(contingency_table)

if not contingency_table.empty:
    chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
    print(f"\nChi-Square Test Results:\nChi2 Statistic: {chi2:.4f}\nP-value: {p:.4e}")
else:
    print("\nError: Contingency table is empty.")

# Calculate rates
rates = analysis_df.groupby('system_origin')['has_code_access'].mean()
print("\nCode Access Rates:")
print(rates)

# --- Visualization ---
plt.figure(figsize=(10, 6))
bars = plt.bar(rates.index, rates.values * 100, color=['#d62728', '#1f77b4'], alpha=0.8)

plt.title('Code Access Compliance: Commercial/Contractor vs. In-House AI', fontsize=14)
plt.ylabel('Percentage with Code Access (%)', fontsize=12)
plt.xlabel('System Origin', fontsize=12)
plt.ylim(0, 110)

# Add labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Analysis Data Shape: (781, 198)
Group Counts:
system_origin
In-House                 391
Commercial/Contractor    390
Name: count, dtype: int64

Contingency Table (0=No Access, 1=Access):
has_code_access        0.0  1.0
system_origin                  
Commercial/Contractor  258  132
In-House                37  354

Chi-Square Test Results:
Chi2 Statistic: 264.5628
P-value: 1.7376e-59

Code Access Rates:
system_origin
Commercial/Contractor    0.338462
In-House                 0.905371
Name: has_code_access, dtype: float64


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Chart.
*   **Purpose:** The plot is designed to compare a quantitative metric (percentage of code access compliance) across two distinct categorical groups (Commercial/Contractor vs. In-House AI systems).

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Label:** "System Origin"
    *   **Categories:** The axis displays two distinct categories: "Commercial/Contractor" and "In-House".
*   **Y-Axis (Vertical):**
    *   **Label:** "Percentage with Code Access (%)"
    *   **Range:** The numerical tick marks range from 0 to 100, with intervals of 20 (0, 20, 40, 60, 80, 100). The axis limits extend slightly beyond 100 (likely 110) to accommodate the height of the bars and labels.
    *   **Grid:** Horizontal dashed grid lines appear at every major interval (20%) to aid in readability.

### 3. Data Trends
*   **Pattern:** There is a stark contrast between the two categories.
*   **Tallest Bar:** The "In-House" category (represented by a blue bar) is the tallest, indicating a very high level of code access.
*   **Shortest Bar:** The "Commercial/Contractor" category (represented by a red bar) is significantly shorter, indicating a much lower level of code access.
*   **Visual Comparison:** The In-House bar is nearly three times the height of the Commercial/Contractor bar.

### 4. Annotations and Legends
*   **Title:** The chart is titled "Code Access Compliance: Commercial/Contractor vs. In-House AI".
*   **Data Labels:** Specific values are annotated directly on top of each bar in bold black text:
    *   Commercial/Contractor: **33.8%**
    *   In-House: **90.5%**
*   **Color Coding:** While there is no separate legend box, the bars are color-coded for visual distinction: Red for Commercial/Contractor and Blue for In-House.

### 5. Statistical Insights
*   **Significant Disparity:** There is a massive gap in code transparency/access depending on the origin of the system. In-House developed AI systems have a 90.5% compliance rate regarding code access, whereas Commercial or Contractor-supplied systems only have a 33.8% compliance rate.
*   **Magnitude of Difference:** The difference between the two groups is 56.7 percentage points.
*   **Implication:** An organization is approximately **2.7 times more likely** to have access to the code if the AI system is developed in-house compared to procuring it from a commercial vendor or contractor. This suggests that proprietary restrictions or "black box" delivery methods are prevalent in commercial AI solutions, while internal development prioritizes or inherently possesses code accessibility.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
