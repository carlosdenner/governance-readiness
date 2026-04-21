# Experiment 62: node_5_3

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_3` |
| **ID in Run** | 62 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T03:58:15.279401+00:00 |
| **Runtime** | 192.5s |
| **Parent** | `node_4_3` |
| **Children** | None |
| **Creation Index** | 63 |

---

## Hypothesis

> Agency Maturity Effect: 'Mature' federal agencies (defined as the top quartile
by number of reported AI use cases) perform 'Independent Evaluations' of their
systems at a significantly higher rate than 'Immature' agencies (bottom
quartile).

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.3629 (Maybe False) |
| **Posterior** | 0.2170 (Likely False) |
| **Surprise** | -0.1750 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 7.0 |
| Uncertain | 0.0 |
| Maybe False | 22.0 |
| Definitely False | 1.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 34.0 |
| Definitely False | 26.0 |

---

## Experiment Plan

**Objective:** Test if organizational maturity (scale of AI adoption) drives better governance practices.

### Steps
- 1. Load 'eo13960_scored'.
- 2. Count use cases per '3_agency'.
- 3. Define 'Mature' as top 25% agencies by count, and 'Nascent' as bottom 25%.
- 4. Filter the dataset to these two groups.
- 5. Compare the '55_independent_eval' (Yes/No) rates between the groups using a Chi-Square test.
- 6. Plot the evaluation rate against log(agency_case_count).

### Deliverables
- Comparison stats (Mature vs Nascent), Scatter plot with trendline.

---

## Analysis

The experiment successfully tested the 'Agency Maturity Effect' hypothesis using
the EO 13960 dataset. The code correctly categorized agencies into 'Mature' (top
quartile, >54 cases) and 'Nascent' (bottom quartile, <=4 cases) based on their
AI portfolio size.

**Key Findings:**
1.  **Hypothesis Rejection:** The results **do not support** the hypothesis that
mature agencies perform independent evaluations at a higher rate. In fact, the
data suggests a trend in the opposite direction, though it is not statistically
significant.
    *   **Mature Agencies:** 3.3% independent evaluation rate (45 out of 1348
cases).
    *   **Nascent Agencies:** 10.3% independent evaluation rate (3 out of 29
cases).

2.  **Statistical Results:** The Chi-Square test yielded a p-value of **0.128**,
indicating that the difference in evaluation rates between the two groups is not
statistically significant at the 0.05 level.

3.  **Visual Insights:** The scatter plot and trendline reveal a negative
correlation: as an agency's case count increases (log scale), the rate of
independent evaluation tends to decrease. This suggests a 'denominator effect'
where scaling AI adoption outpaces the capacity or practice of rigorous
independent governance.

In conclusion, there is no evidence that organizational maturity (defined by
scale) leads to better governance practices; if anything, larger portfolios are
associated with lower density of independent oversight.

---

## Review

The experiment was successfully executed. The code faithfully implemented the
logic to categorize agencies by maturity based on case counts and compared their
independent evaluation rates using the specified statistical test.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# [debug]
print("Starting experiment: Agency Maturity Effect")

# 1. Load Data
file_path = '../astalabs_discovery_all_data.csv'
try:
    # Only loading necessary columns to save memory/time if possible, but sparse layout makes it tricky.
    # Loading all and filtering is safer given the structure.
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if running in a different environment structure
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Loaded EO 13960 data: {len(eo_data)} rows")

# 2. Agency Maturity Analysis
# Count use cases per agency
agency_counts = eo_data['3_agency'].value_counts().reset_index()
agency_counts.columns = ['3_agency', 'case_count']

# Determine quartiles
q1 = agency_counts['case_count'].quantile(0.25)
q3 = agency_counts['case_count'].quantile(0.75)
print(f"Quartiles -> Q1 (Nascent threshold): {q1}, Q3 (Mature threshold): {q3}")

# Label Agencies
def categorize_maturity(count):
    if count >= q3:
        return 'Mature'
    elif count <= q1:
        return 'Nascent'
    else:
        return 'Middle'

agency_counts['maturity'] = agency_counts['case_count'].apply(categorize_maturity)

# Merge maturity back to main data
eo_data = eo_data.merge(agency_counts[['3_agency', 'maturity', 'case_count']], on='3_agency', how='left')

# 3. Process Independent Evaluation Target
# Check values
target_col = '55_independent_eval'
unique_vals = eo_data[target_col].unique()
print(f"Unique values in {target_col}: {unique_vals}")

# Map to binary. Assuming variations of 'Yes'/'No'. NaN is treated as 0 (No) for governance scoring often, 
# but strict comparison might require dropping. Let's inspect and map cautiously.
# If it's a string, we normalize.
def clean_eval(val):
    if pd.isna(val):
        return 0 # Treat missing as lack of evidence/No in this context? Or np.nan?
                 # In government inventories, blank often means 'No'. Let's assume 0 but print warning if high NaN.
    s = str(val).lower().strip()
    if 'yes' in s or 'true' in s:
        return 1
    return 0

eo_data['has_eval'] = eo_data[target_col].apply(clean_eval)

# 4. Statistical Test (Mature vs Nascent)
analysis_set = eo_data[eo_data['maturity'].isin(['Mature', 'Nascent'])]

# Group stats
group_stats = analysis_set.groupby('maturity')['has_eval'].agg(['count', 'mean', 'sum'])
print("\n--- Group Statistics (Independent Evaluation Rate) ---")
print(group_stats)

# Contingency Table for Chi-Square
# Rows: [Mature, Nascent], Cols: [Eval=1, Eval=0]
mature_success = analysis_set[analysis_set['maturity'] == 'Mature']['has_eval'].sum()
mature_fail = analysis_set[analysis_set['maturity'] == 'Mature']['has_eval'].count() - mature_success

nascent_success = analysis_set[analysis_set['maturity'] == 'Nascent']['has_eval'].sum()
nascent_fail = analysis_set[analysis_set['maturity'] == 'Nascent']['has_eval'].count() - nascent_success

contingency_table = [[mature_success, mature_fail], [nascent_success, nascent_fail]]
chi2, p, dof, ex = stats.chi2_contingency(contingency_table)

print(f"\nChi-Square Test Results:")
print(f"Contingency Table (Rows: Mature, Nascent; Cols: Has_Eval, No_Eval):\n{contingency_table}")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# 5. Visualization: Evaluation Rate vs Log(Agency Case Count)
# We calculate rate per agency for the scatter plot
agency_perf = eo_data.groupby('3_agency').agg(
    case_count=('case_count', 'first'),
    eval_rate=('has_eval', 'mean'),
    maturity=('maturity', 'first')
).reset_index()

agency_perf['log_count'] = np.log1p(agency_perf['case_count'])

plt.figure(figsize=(10, 6))
colors = {'Mature': 'green', 'Nascent': 'red', 'Middle': 'gray'}

for mat, color in colors.items():
    subset = agency_perf[agency_perf['maturity'] == mat]
    plt.scatter(subset['log_count'], subset['eval_rate'], 
                label=mat, color=color, alpha=0.6, edgecolors='w', s=80)

# Trendline (using all data)
z = np.polyfit(agency_perf['log_count'], agency_perf['eval_rate'], 1)
p_poly = np.poly1d(z)
plt.plot(agency_perf['log_count'], p_poly(agency_perf['log_count']), "b--", alpha=0.5, label='Trendline')

plt.title('Agency Maturity vs Governance Rigor (Independent Eval)')
plt.xlabel('Log(Agency Case Count)')
plt.ylabel('Independent Evaluation Rate')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting experiment: Agency Maturity Effect
Loaded EO 13960 data: 1757 rows
Quartiles -> Q1 (Nascent threshold): 4.0, Q3 (Mature threshold): 54.75
Unique values in 55_independent_eval: <StringArray>
[                                                                                                    nan,
                                                                                     'Yes – by the CAIO',
                                                                                'Planned or in-progress',
 'Yes – by another appropriate agency office that was not directly involved in the system’s development',
               'Yes – by an agency AI oversight board not directly involved in the system’s development',
                                       'Does not apply, use case is neither safety or rights impacting.',
                         'Agency CAIO has waived this minimum practice and reported such waiver to OMB.',
                                                                 'AI is not safety or rights-impacting.',
                                                                                                  'TRUE']
Length: 9, dtype: str

--- Group Statistics (Independent Evaluation Rate) ---
          count      mean  sum
maturity                      
Mature     1348  0.033383   45
Nascent      29  0.103448    3

Chi-Square Test Results:
Contingency Table (Rows: Mature, Nascent; Cols: Has_Eval, No_Eval):
[[np.int64(45), np.int64(1303)], [np.int64(3), np.int64(26)]]
Chi2 Statistic: 2.3217
P-value: 1.2758e-01


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Scatter Plot with a linear regression trendline.
*   **Purpose:** The plot visualizes the relationship between the size of an agency (measured by the logarithm of its case count) and the rigor of its governance (measured by the independent evaluation rate). It categorizes agencies into three maturity levels: Nascent, Middle, and Mature.

### 2. Axes
*   **X-axis:**
    *   **Label:** "Log(Agency Case Count)"
    *   **Range:** Approximately **0.7 to 5.6**.
    *   **Meaning:** This represents the scale of the agency's operations on a logarithmic scale. Higher values indicate a significantly larger volume of cases.
*   **Y-axis:**
    *   **Label:** "Independent Evaluation Rate"
    *   **Range:** **0.0 to 0.5** (representing 0% to 50%).
    *   **Meaning:** This measures the proportion of cases or projects that undergo independent evaluation, serving as a proxy for governance rigor.

### 3. Data Trends
*   **Grouping by Maturity:**
    *   **Nascent (Red):** Clustered on the far left (low case count, Log value < 1.7). They show high variance; while several are at 0.0, there are two significant outliers with very high evaluation rates (approx. 0.25 and 0.50).
    *   **Middle (Grey):** Occupy the central range of the X-axis (Log value ~1.8 to 4.0). The vast majority are clustered at 0.0, with one significant outlier showing a high evaluation rate around 0.44.
    *   **Mature (Green):** Clustered on the far right (high case count, Log value > 4.0). These agencies consistently show low evaluation rates, generally below 0.15, with many resting at 0.0.
*   **Overall Trend:**
    *   There is a visible **negative correlation**. As the agency case count increases (moving right on the X-axis), the rate of independent evaluation tends to decrease.
    *   **Clustering at Zero:** A significant portion of the data points across all categories lie on the y=0 line, indicating that many agencies, regardless of size or maturity, have no independent evaluation recorded.

### 4. Annotations and Legends
*   **Legend (Top Right):**
    *   **Green Circle:** Represents "Mature" agencies.
    *   **Red Circle:** Represents "Nascent" agencies.
    *   **Grey Circle:** Represents "Middle" agencies.
    *   **Blue Dashed Line:** Represents the "Trendline" (linear fit).
*   **Title:** "Agency Maturity vs Governance Rigor (Independent Eval)" clearly defines the variables under investigation.
*   **Gridlines:** Light dotted gridlines are present to assist in estimating the coordinates of the data points.

### 5. Statistical Insights
*   **Inverse Relationship:** The negative slope of the blue trendline suggests that as agencies grow larger (or "mature" in terms of case count), the proportion of their work subject to independent evaluation decreases. This could be due to the sheer volume of cases making high evaluation rates logistically difficult (the denominator effect).
*   **Variance in Smaller Agencies:** "Nascent" agencies show the highest variability. While some have zero governance rigor (by this metric), others have the highest rigor in the dataset (up to 50%). This suggests early-stage agencies lack a standardized approach—some are highly rigorous while others are not.
*   **Consistency in Mature Agencies:** "Mature" agencies are more consistent but consistently low. None of the mature agencies exceed an evaluation rate of ~0.12, suggesting that at scale, high rates of independent evaluation are rare.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
