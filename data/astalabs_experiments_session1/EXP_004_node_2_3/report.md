# Experiment 4: node_2_3

| Property | Value |
|---|---|
| **Experiment ID** | `node_2_3` |
| **ID in Run** | 4 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T21:34:23.975228+00:00 |
| **Runtime** | 216.2s |
| **Parent** | `node_1_0` |
| **Children** | `node_3_9`, `node_3_11`, `node_3_20` |
| **Creation Index** | 5 |

---

## Hypothesis

> There has been a significant temporal shift in AI incident characteristics,
where the proportion of 'Integration Readiness' gaps has increased relative to
'Trust Readiness' gaps in incidents occurring post-2023 compared to pre-2023.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.5417 (Uncertain) |
| **Posterior** | 0.6667 (Maybe True) |
| **Surprise** | +0.1451 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 5.0 |
| Uncertain | 24.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 78.0 |
| Uncertain | 9.0 |
| Maybe False | 3.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Evaluate the time-based evolution of competency gap types.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Convert 'incident_date' to datetime objects.
- 3. Create a binary 'Period' variable: 'Post-2023' (date >= 2023-01-01) and 'Pre-2023'.
- 4. For each incident, determine if it involves 'Integration' gaps (based on 'trust_integration_split' or parsing 'sub_competency_ids').
- 5. Generate a contingency table of Period vs. Gap Type.
- 6. Run a Chi-Square test to check for a shift in distribution over time.

### Deliverables
- Temporal bar chart of gap types, contingency table, and Chi-Square test results.

---

## Analysis

The experiment successfully analyzed the temporal evolution of competency gaps
across 52 incidents. The data was split into Pre-2023 (n=19) and Post-2023
(n=33) periods. The analysis reveals a directional shift consistent with the
hypothesis: 'Integration-Dominant' gaps emerged only in the Post-2023 period,
rising from 0.00% to 12.12%, while 'Trust-Dominant' gaps decreased slightly from
5.26% to 3.03%. However, the 'Both' category remained overwhelmingly dominant in
both periods (94.74% vs 84.85%). Consequently, the Chi-Square test yielded a
p-value of 0.2735, meaning the observed shift is not statistically significant
at the 0.05 level, likely due to the small sample size. While the statistical
test failed to reject the null hypothesis, the descriptive statistics and
visualization highlight a qualitative emergence of pure integration failures in
recent years.

---

## Review

The experiment was successfully executed. The analysis of 52 incidents (19
Pre-2023, 33 Post-2023) tested the hypothesis that 'Integration Readiness' gaps
have increased relative to 'Trust Readiness' gaps in recent years. Descriptive
statistics support a qualitative shift: 'Integration-Dominant' gaps emerged from
0% (0/19) in the Pre-2023 period to 12.1% (4/33) in the Post-2023 period, while
'Trust-Dominant' gaps remained negligible (5.3% to 3.0%). However, the 'Both'
category remained the overwhelming majority in both periods (94.7% vs 84.8%).
Consequently, the Chi-Square test yielded a p-value of 0.2735, failing to reject
the null hypothesis at the alpha=0.05 level. While statistically insignificant
due to the small sample size, the emergence of purely integration-related
failures in the later period provides directional support for the proposition
that engineering challenges are becoming more distinct as AI systems scale.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('step3_incident_coding.csv')
except FileNotFoundError:
    # Fallback to parent directory just in case
    try:
        df = pd.read_csv('../step3_incident_coding.csv')
    except FileNotFoundError:
        raise FileNotFoundError("Dataset 'step3_incident_coding.csv' not found in current or parent directory.")

# Convert incident_date to datetime
df['incident_date_dt'] = pd.to_datetime(df['incident_date'], errors='coerce')

# Drop rows with invalid dates
df = df.dropna(subset=['incident_date_dt'])

# Define Period
# Post-2023 includes 2023-01-01 onwards
cutoff_date = pd.Timestamp('2023-01-01')
df['Period'] = df['incident_date_dt'].apply(lambda x: 'Post-2023' if x >= cutoff_date else 'Pre-2023')

# Clean and categorize Gap Type
# Metadata says values are: trust-dominant, integration-dominant, both
df['Gap_Type_Raw'] = df['trust_integration_split'].astype(str).str.strip().str.lower()

mapping = {
    'trust-dominant': 'Trust-Dominant',
    'integration-dominant': 'Integration-Dominant',
    'both': 'Both'
}
df['Gap_Type'] = df['Gap_Type_Raw'].map(mapping)

# Drop rows where Gap_Type mapping failed (if any)
df = df.dropna(subset=['Gap_Type'])

# Generate Contingency Table
contingency_table = pd.crosstab(df['Gap_Type'], df['Period'])

# Ensure column order for logical flow
expected_periods = ['Pre-2023', 'Post-2023']
# Filter to only existing columns in case data is missing for one period
existing_periods = [p for p in expected_periods if p in contingency_table.columns]
contingency_table = contingency_table[existing_periods]

print("=== Contingency Table (Gap Type vs Period) ===")
print(contingency_table)

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("\n=== Chi-Square Test Results ===")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")

alpha = 0.05
if p < alpha:
    print("Result: Significant shift in distribution (Reject H0)")
else:
    print("Result: No significant shift detected (Fail to reject H0)")

# Calculate Column Percentages (Distribution within each Period)
col_percentages = contingency_table.div(contingency_table.sum(axis=0), axis=1) * 100
print("\n=== Column Percentages (Distribution per Period) ===")
print(col_percentages.round(2))

# Visualization: Stacked 100% Bar Chart to visualize proportions
# Transpose so X-axis is Period
plot_data = col_percentages.T
ax = plot_data.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')

plt.title('Proportional Distribution of Competency Gap Types (Pre vs Post 2023)')
plt.xlabel('Period')
plt.ylabel('Percentage of Incidents (%)')
plt.legend(title='Gap Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: === Contingency Table (Gap Type vs Period) ===
Period                Pre-2023  Post-2023
Gap_Type                                 
Both                        18         28
Integration-Dominant         0          4
Trust-Dominant               1          1

=== Chi-Square Test Results ===
Chi2 Statistic: 2.5926
P-value: 0.2735
Degrees of Freedom: 2
Result: No significant shift detected (Fail to reject H0)

=== Column Percentages (Distribution per Period) ===
Period                Pre-2023  Post-2023
Gap_Type                                 
Both                     94.74      84.85
Integration-Dominant      0.00      12.12
Trust-Dominant            5.26       3.03


=== Plot Analysis (figure 1) ===
Based on the visual analysis of the provided plot, here is the detailed breakdown:

### 1. Plot Type
*   **Type:** Stacked Bar Chart (specifically a 100% Stacked Bar Chart).
*   **Purpose:** To compare the proportional distribution of three different "Gap Types" across two distinct time periods (Pre-2023 and Post-2023), showing how the composition of incidents has shifted over time while normalizing the total volume to 100%.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Period"
    *   **Labels:** Two categorical time bins: "Pre-2023" and "Post-2023".
*   **Y-Axis:**
    *   **Title:** "Percentage of Incidents (%)"
    *   **Range:** 0 to 100.
    *   **Increments:** Ticks are marked every 20 units (0, 20, 40, 60, 80, 100).

### 3. Data Trends
*   **"Both" Category (Dark Purple):**
    *   This is the dominant category in both periods.
    *   **Pre-2023:** Occupies approximately 95% of the incidents.
    *   **Post-2023:** Decreases noticeably to approximately 85% of the incidents.
*   **"Integration-Dominant" Category (Teal):**
    *   **Pre-2023:** This category is virtually non-existent or 0% (no teal segment is visible).
    *   **Post-2023:** Shows a significant emergence, occupying the segment roughly between 85% and 97% on the Y-axis (approximately 12% of the total).
*   **"Trust-Dominant" Category (Yellow):**
    *   **Pre-2023:** Occupies the top segment, roughly 5% (from ~95% to 100%).
    *   **Post-2023:** Occupies a slightly smaller top segment, roughly 3% (from ~97% to 100%).

### 4. Annotations and Legends
*   **Plot Title:** "Proportional Distribution of Competency Gap Types (Pre vs Post 2023)" - Located at the top center.
*   **Legend:** A box located to the right of the plot titled "**Gap Type**" maps colors to categories:
    *   **Purple:** "Both"
    *   **Teal:** "Integration-Dominant"
    *   **Yellow:** "Trust-Dominant"

### 5. Statistical Insights
*   **Shift in Composition:** There is a distinct shift in the nature of competency gaps after 2023. While incidents involving "Both" types of gaps remain the majority, their dominance has waned to allow for the rise of specific "Integration-Dominant" issues.
*   **Emergence of a New Trend:** The most statistically significant finding is the appearance of "Integration-Dominant" gaps in the Post-2023 era. This category went from being unobservable in the Pre-2023 data to representing a substantial minority share (approx. 1/8th of incidents) in the Post-2023 data.
*   **Stability of Trust Issues:** Purely "Trust-Dominant" issues have remained a small minority throughout both periods, potentially shrinking slightly in the more recent period.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
