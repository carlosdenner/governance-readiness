# Experiment 68: node_4_21

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_21` |
| **ID in Run** | 68 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:42:32.152442+00:00 |
| **Runtime** | 181.5s |
| **Parent** | `node_3_16` |
| **Children** | `node_5_25` |
| **Creation Index** | 69 |

---

## Hypothesis

> Incidents resulting in 'Security' harms are significantly more likely to be
characterized as 'Prevention Failures', whereas non-security harms (Reliability,
Safety) correlate with 'Detection' or 'Response' failures.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7339 (Likely True) |
| **Posterior** | 0.1880 (Likely False) |
| **Surprise** | -0.6335 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 1.0 |
| Maybe True | 28.0 |
| Uncertain | 0.0 |
| Maybe False | 1.0 |
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

**Objective:** Assess the relationship between the type of harm and the mode of control failure.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Create a binary 'Harm Category' variable: 'Security' (if harm_type == 'security') vs. 'Other'.
- 3. Create a contingency table of 'Harm Category' vs. 'failure_mode'.
- 4. Perform a Chi-Square test of independence.
- 5. Generate a Stacked Bar Chart showing Failure Mode proportions for Security vs. Other harms.

### Deliverables
- Chi-Square statistics and a Stacked Bar Chart.

---

## Analysis

The experiment successfully tested the relationship between harm types and
failure modes using the `step3_incident_coding.csv` dataset. The data revealed a
near-total lack of variance in the dependent variable: 51 out of 52 incidents
were classified as 'prevention_failure', with only a single 'detection_failure'
(occurring in the Security group) and zero 'response_failure' cases.
Consequently, the Chi-Square test yielded a p-value of 1.0, indicating no
statistical difference between 'Security' (n=36) and 'Other' (n=16) harm
categories. The hypothesis that non-security harms correlate with detection or
response failures was strictly rejected; in fact, the 'Other' category showed
100% prevention failures. The stacked bar chart visualizes this uniformity,
highlighting that the dataset is overwhelmingly dominated by prevention failures
regardless of the harm type.

---

## Review

The experiment successfully tested the relationship between harm types and
failure modes using the `step3_incident_coding.csv` dataset. The data revealed a
near-total lack of variance in the dependent variable: 51 out of 52 incidents
were classified as 'prevention_failure', with only a single 'detection_failure'
(occurring in the Security group) and zero 'response_failure' cases.
Consequently, the Chi-Square test yielded a p-value of 1.0, indicating no
statistical difference between 'Security' (n=36) and 'Other' (n=16) harm
categories. The hypothesis that non-security harms correlate with detection or
response failures was strictly rejected; in fact, the 'Other' category showed
100% prevention failures. The stacked bar chart visualizes this uniformity,
highlighting that the dataset is overwhelmingly dominated by prevention failures
regardless of the harm type.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# [debug] Check file existence
file_name = 'step3_incident_coding.csv'
possible_paths = [f'../{file_name}', file_name]
file_path = None
for p in possible_paths:
    if os.path.exists(p):
        file_path = p
        break

if not file_path:
    print(f"File {file_name} not found in checked paths: {possible_paths}")
    # Attempt to use step3_enrichments.json as fallback if csv is missing, as they share structure in metadata descriptions
    json_file = 'step3_enrichments.json'
    possible_json_paths = [f'../{json_file}', json_file]
    for p in possible_json_paths:
        if os.path.exists(p):
            print(f"Falling back to {p}")
            file_path = p
            break

if not file_path:
    raise FileNotFoundError("Neither incident coding CSV nor enrichments JSON found.")

# Load Data
print(f"Loading dataset from {file_path}...")
if file_path.endswith('.csv'):
    df = pd.read_csv(file_path)
else:
    df = pd.read_json(file_path)

# Verify columns
required_cols = ['harm_type', 'failure_mode']
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    print(f"Error: Missing columns {missing_cols}. Available: {df.columns.tolist()}")
    exit(1)

# Preprocessing
# Clean whitespace and standardize case
df['harm_type'] = df['harm_type'].fillna('').astype(str).str.strip().str.lower()
df['failure_mode'] = df['failure_mode'].fillna('').astype(str).str.strip().str.lower()

# Create Binary Harm Category
df['harm_category'] = df['harm_type'].apply(lambda x: 'Security' if 'security' in x else 'Other')

# Generate Contingency Table
contingency = pd.crosstab(df['harm_category'], df['failure_mode'])
print("\n--- Contingency Table (Observed) ---")
print(contingency)

# Check for empty columns or rows which might break Chi2
if contingency.empty or contingency.shape[0] < 2 or contingency.shape[1] < 2:
    print("\nWarning: Contingency table too small for Chi-Square test (need at least 2x2).")
else:
    # Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\n--- Chi-Square Test Results ---")
    print(f"Chi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    print(f"Degrees of Freedom: {dof}")
    print("\nExpected Frequencies:")
    print(expected)

# Visualization
# Calculate proportions for stacked bar chart
contingency_prop = contingency.div(contingency.sum(axis=1), axis=0)

# Plot
plt.figure(figsize=(10, 6))
# contingency_prop.plot(kind='bar', stacked=True) is cleaner, but let's use explicit ax for control
ax = contingency_prop.plot(kind='bar', stacked=True, colormap='viridis', figsize=(10, 6))

plt.title('Proportion of Failure Modes by Harm Category')
plt.xlabel('Harm Category')
plt.ylabel('Proportion')
plt.legend(title='Failure Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()

print("\nDisplaying plot...")
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from step3_incident_coding.csv...

--- Contingency Table (Observed) ---
failure_mode   detection_failure  prevention_failure
harm_category                                       
Other                          0                  16
Security                       1                  35

--- Chi-Square Test Results ---
Chi-Square Statistic: 0.0000
P-value: 1.0000
Degrees of Freedom: 1

Expected Frequencies:
[[ 0.30769231 15.69230769]
 [ 0.69230769 35.30769231]]

Displaying plot...


=== Plot Analysis (figure 2) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Normalized Stacked Bar Chart.
*   **Purpose:** This plot visualizes the relative composition of "Failure Modes" within two distinct "Harm Categories." By normalizing the height of the bars to 1.0 (or 100%), it allows for a direct comparison of proportions rather than absolute counts.

### 2. Axes
*   **X-axis:**
    *   **Label:** "Harm Category"
    *   **Categories:** Two discrete categories are displayed: "**Other**" and "**Security**."
*   **Y-axis:**
    *   **Label:** "Proportion"
    *   **Range:** The axis ranges from **0.0 to 1.0**, representing percentages from 0% to 100%.
    *   **Ticks:** Marks are placed at intervals of 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).

### 3. Data Trends
*   **Dominant Trend:** The failure mode **"prevention_failure"** (represented in yellow) is overwhelmingly dominant in both harm categories.
*   **Category "Other":**
    *   The bar appears to be entirely yellow. This indicates that for the "Other" harm category, nearly **100%** of the recorded failures are classified as prevention failures. There is no visible component for detection failures.
*   **Category "Security":**
    *   While still dominated by the yellow "prevention_failure," there is a distinct, albeit small, section of dark purple at the bottom of the bar.
    *   This indicates that a small proportion (likely less than 5%) of failures in the "Security" category are classified as **"detection_failure."**

### 4. Annotations and Legends
*   **Chart Title:** "Proportion of Failure Modes by Harm Category" is displayed at the top.
*   **Legend:** Located on the right side of the plot with the title **"Failure Mode."**
    *   **Dark Purple Square:** Corresponds to **"detection_failure."**
    *   **Yellow Square:** Corresponds to **"prevention_failure."**

### 5. Statistical Insights
*   **Prevalence of Prevention Failures:** The data suggests that regardless of whether the harm category is "Security" or "Other," the system is significantly more prone to (or classified as having) prevention failures rather than detection failures.
*   **Category Differentiation:** There is a slight statistical variance between the two categories. While the "Other" category shows an absolute hegemony of prevention failures, the "Security" category introduces a non-zero probability of detection failures. This suggests that detection mechanisms might be slightly more relevant or active (even if failing) within security contexts compared to other general contexts.
*   **Operational Implication:** If the goal is to improve system reliability, efforts should likely be focused on addressing "prevention_failures," as this is the primary bottleneck across the board.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
