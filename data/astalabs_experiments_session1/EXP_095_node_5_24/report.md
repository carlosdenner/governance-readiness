# Experiment 95: node_5_24

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_24` |
| **ID in Run** | 95 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-21T00:39:29.517948+00:00 |
| **Runtime** | 224.1s |
| **Parent** | `node_4_2` |
| **Children** | None |
| **Creation Index** | 96 |

---

## Hypothesis

> The proportion of 'Societal' harms (Bias, Privacy) has significantly increased
in the post-2023 era (Generative AI boom) compared to the pre-2023 era, while
'Security' harms have proportionally decreased.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.3760 (Maybe False) |
| **Surprise** | -0.4247 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
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
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 90.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Assess the temporal shift in AI risk profiles.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Convert 'incident_date' to datetime and create a binary 'era' column: 'Pre-2023' (date < 2023-01-01) and 'Post-2023' (date >= 2023-01-01).
- 3. Group 'harm_type' into 'Societal' (bias_discrimination, privacy) and 'Security/Reliability' (security, reliability, supply_chain).
- 4. Create a contingency table of Era vs. Harm Category.
- 5. Perform a Chi-Square test of independence to see if the distribution of harms has shifted over time.

### Deliverables
- Stacked bar chart of harm categories by era and Chi-Square test results.

---

## Analysis

The experiment successfully analyzed the temporal shift in AI incident harm
types using 'step3_incident_coding.csv'.

**Data Processing:**
The code successfully parsed dates and categorized 49 relevant incidents into
two eras ('Pre-2023' n=17, 'Post-2023' n=32) and two harm categories
('Security/Reliability' n=44, 'Societal' n=5).

**Results:**
- **Pre-2023:** 94.1% Security (16/17), 5.9% Societal (1/17).
- **Post-2023:** 87.5% Security (28/32), 12.5% Societal (4/32).

**Statistical Significance:**
While the proportion of Societal harms more than doubled (from 5.9% to 12.5%) in
the Generative AI era, the sample size for societal harms is very small (n=5
total). Consequently, the statistical tests failed to find a significant
difference:
- **Chi-Square P-value:** 0.8160
- **Fisher's Exact Test P-value:** 0.6463

**Conclusion:**
The hypothesis that Societal harms have *significantly* increased is **not
supported**. Although there is a directional trend consistent with the
hypothesis (a shift from 6% to 12.5%), the scarcity of recorded societal harm
incidents in the dataset prevents this from being statistically distinguishable
from chance. The dataset remains overwhelmingly dominated by
Security/Reliability incidents in both eras.

---

## Review

The experiment successfully analyzed the temporal evolution of AI incident harm
types using 'step3_incident_coding.csv'.

**Methodology:**
The code correctly parsed incident dates and categorized 49 relevant incidents
into two eras: 'Pre-2023' (n=17) and 'Post-2023' (n=32). Harm types were grouped
into 'Security/Reliability' (security, reliability, supply_chain) and 'Societal'
(bias, privacy).

**Findings:**
- **Pre-2023:** 94.1% Security/Reliability (16/17), 5.9% Societal (1/17).
- **Post-2023:** 87.5% Security/Reliability (28/32), 12.5% Societal (4/32).

**Hypothesis Verification:**
While the data shows a directional trend where the proportion of Societal harms
more than doubled (from 5.9% to 12.5%) in the Generative AI era, the absolute
counts are too low to achieve statistical significance. The dataset remains
overwhelmingly dominated by Security incidents.
- **Chi-Square P-value:** 0.8160
- **Fisher's Exact Test P-value:** 0.6463

**Conclusion:**
The hypothesis that Societal harms have *significantly* increased is **not
supported** statistically. The observed increase is indistinguishable from
chance given the current sample size.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load dataset
# Attempting to load from current directory as '../' failed in previous step
try:
    df = pd.read_csv('step3_incident_coding.csv')
except FileNotFoundError:
    # Fallback to absolute path or check if it's in a subdirectory if needed, 
    # but strictly following previous success patterns implies current dir.
    print("File not found in current directory. Creating dummy data for structure verification if needed, or exiting.")
    raise

# 1. Date Parsing
df['incident_date_dt'] = pd.to_datetime(df['incident_date'], errors='coerce')

# Filter out rows with invalid dates if any
df = df.dropna(subset=['incident_date_dt'])

# 2. Define Eras
cutoff_date = pd.Timestamp('2023-01-01')
df['era'] = df['incident_date_dt'].apply(lambda d: 'Post-2023' if d >= cutoff_date else 'Pre-2023')

# 3. Categorize Harm Types
# Prompt: 'Societal' (bias_discrimination, privacy)
# Prompt: 'Security/Reliability' (security, reliability, supply_chain)
societal_harms = ['bias_discrimination', 'privacy']
security_harms = ['security', 'reliability', 'supply_chain']

def categorize_harm(h_type):
    h_type = str(h_type).strip()
    if h_type in societal_harms:
        return 'Societal'
    elif h_type in security_harms:
        return 'Security/Reliability'
    else:
        return 'Other'

df['harm_category'] = df['harm_type'].apply(categorize_harm)

# Filter out 'Other' to test the specific hypothesis strictly
df_filtered = df[df['harm_category'] != 'Other'].copy()

print("=== Data Summary ===")
print(f"Total incidents processed: {len(df)}")
print(f"Incidents in hypothesis categories: {len(df_filtered)}")
print("\nHarm Category Counts:")
print(df_filtered['harm_category'].value_counts())
print("\nEra Counts:")
print(df_filtered['era'].value_counts())

# 4. Contingency Table
contingency_table = pd.crosstab(df_filtered['era'], df_filtered['harm_category'])
print("\n=== Contingency Table ===")
print(contingency_table)

# 5. Statistical Test (Chi-Square)
# Note: If counts are low (<5 in cells), Fisher's Exact Test is preferred, but for 2x2 Chi2 is standard start.
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("\n=== Statistical Test Results ===")
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4f}")

# Fisher's Exact Test (since it's 2x2 and likely small sample size)
if contingency_table.shape == (2, 2):
    # fisher_exact returns (odds_ratio, p_value)
    odds_ratio, fisher_p = stats.fisher_exact(contingency_table)
    print(f"Fisher's Exact Test P-value: {fisher_p:.4f}")
    print(f"Odds Ratio: {odds_ratio:.4f}")

# 6. Visualization
# Calculate proportions for plotting
props = contingency_table.div(contingency_table.sum(axis=1), axis=0)

ax = props.plot(kind='bar', stacked=True, color=['#d9534f', '#5bc0de'], figsize=(10, 6))
plt.title('Proportion of Harm Categories by Era (Pre vs Post 2023)')
plt.ylabel('Proportion')
plt.xlabel('Era')
plt.legend(title='Harm Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Add count annotations
for n, x in enumerate([*contingency_table.index.values]):
    for (proportion, count, y_loc) in zip(props.loc[x], contingency_table.loc[x], props.loc[x].cumsum()):                
        plt.text(x=n, y=(y_loc - proportion) + (proportion / 2), s=f'{count} ({proportion:.1%})', 
                 color="white", fontsize=10, fontweight="bold", ha="center", va="center")

plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: === Data Summary ===
Total incidents processed: 52
Incidents in hypothesis categories: 49

Harm Category Counts:
harm_category
Security/Reliability    44
Societal                 5
Name: count, dtype: int64

Era Counts:
era
Post-2023    32
Pre-2023     17
Name: count, dtype: int64

=== Contingency Table ===
harm_category  Security/Reliability  Societal
era                                          
Post-2023                        28         4
Pre-2023                         16         1

=== Statistical Test Results ===
Chi-Square Statistic: 0.0541
P-value: 0.8160
Fisher's Exact Test P-value: 0.6463
Odds Ratio: 0.4375


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Stacked Bar Plot (specifically, a 100% stacked bar chart).
*   **Purpose:** The plot compares the relative proportions of two specific categories of harm ("Security/Reliability" vs. "Societal") across two different time periods ("Post-2023" and "Pre-2023").

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Era"
    *   **Labels:** Two categorical time periods: "Post-2023" and "Pre-2023".
*   **Y-Axis:**
    *   **Title:** "Proportion"
    *   **Range:** 0.0 to 1.0 (representing 0% to 100%).
    *   **Units:** Decimal proportion.

### 3. Data Trends
*   **Dominant Category:** In both eras, the "Security/Reliability" category (red) makes up the vast majority of the data, exceeding 85% in both columns.
*   **Post-2023 Era:**
    *   **Tallest Segment:** Security/Reliability at 87.5% (28 counts).
    *   **Shortest Segment:** Societal at 12.5% (4 counts).
*   **Pre-2023 Era:**
    *   **Tallest Segment:** Security/Reliability at 94.1% (16 counts).
    *   **Shortest Segment:** Societal at 5.9% (1 count).
*   **Comparison:** The proportion of "Societal" harms is visibly larger in the "Post-2023" era compared to the "Pre-2023" era.

### 4. Annotations and Legends
*   **Legend:** Located in the top right corner with the title "Harm Category". It distinguishes the data series by color:
    *   **Red:** Security/Reliability
    *   **Light Blue:** Societal
*   **Bar Annotations:** Inside each segment of the bars, white text provides the raw count followed by the percentage in parentheses.
    *   **Post-2023:** Top: "4 (12.5%)", Bottom: "28 (87.5%)"
    *   **Pre-2023:** Top: "1 (5.9%)", Bottom: "16 (94.1%)"
*   **Title:** The chart is titled "Proportion of Harm Categories by Era (Pre vs Post 2023)".

### 5. Statistical Insights
*   **Shift in Harm Composition:** There has been a relative increase in "Societal" harms in the more recent era (Post-2023). The percentage of societal harms more than doubled from 5.9% in the Pre-2023 era to 12.5% in the Post-2023 era.
*   **Volume Increase:** Beyond proportions, the raw counts indicate an increase in the total number of incidents recorded. The Pre-2023 era shows a total of 17 incidents (16 + 1), while the Post-2023 era shows nearly double that amount with 32 incidents (28 + 4).
*   **Consistency:** Despite the increase in societal harms, "Security/Reliability" remains the primary concern in this dataset, consistently representing the overwhelming majority of cases across both timeframes.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
