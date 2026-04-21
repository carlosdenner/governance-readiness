# Experiment 151: node_5_54

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_54` |
| **ID in Run** | 151 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T08:12:06.318393+00:00 |
| **Runtime** | 183.4s |
| **Parent** | `node_4_31` |
| **Children** | `node_6_56` |
| **Creation Index** | 152 |

---

## Hypothesis

> Sector-Demographic Harm: Incidents in the 'Financial' sector are significantly
more likely to have a 'Demographic' harm distribution basis (discrimination)
compared to the 'Transportation' sector, which skews towards general safety
harms.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9597 (Definitely True) |
| **Posterior** | 0.9093 (Definitely True) |
| **Surprise** | -0.0604 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 27.0 |
| Maybe True | 3.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 32.0 |
| Maybe True | 28.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Compare the basis of harm distribution across Financial and Transportation sectors.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` (AIID subset).
- 2. Filter for rows where `Sector of Deployment` is 'Financial' or 'Transportation' (or similar keywords).
- 3. Create binary target `is_demographic_harm` based on `Harm Distribution Basis` (True if contains 'Demographic' or 'Race'/'Gender', False otherwise).
- 4. Perform a Chi-Square test or Fisher's Exact test comparing the two sectors.

### Deliverables
- Stacked bar chart of Harm Basis by Sector; Statistical test results.

---

## Analysis

The experiment successfully tested the hypothesis regarding the nature of harm
in Financial versus Transportation AI incidents.

**Data Processing:**
- The programmer successfully filtered the AIID dataset and categorized
incidents into 'Financial' (n=7) and 'Transportation' (n=28) sectors using
keyword matching.
- A binary 'Demographic Harm' flag was created based on the 'Harm Distribution
Basis' column.

**Findings:**
- **Financial Sector:** 28.6% of incidents (2/7) involved demographic harm
(e.g., discrimination, bias).
- **Transportation Sector:** 0% of incidents (0/28) involved demographic harm.
- **Statistical Significance:** Fisher's Exact Test yielded a p-value of
**0.035**, which is below the standard alpha of 0.05.

**Conclusion:**
The results **support** the hypothesis. There is a statistically significant
association between the Financial sector and demographic harm compared to the
Transportation sector, which showed no instances of demographic harm in this
sample. This aligns with the expectation that financial AI risks often involve
allocative harms (bias in lending/credit), whereas transportation risks are
likely distinct (e.g., physical safety), though the small sample size for the
Financial sector (n=7) warrants caution in generalization.

---

## Review

The experiment successfully tested and supported the 'Sector-Demographic Harm'
hypothesis. The execution was faithful to the plan, correctly identifying the
small sample size issue (Financial sector n=7) and employing Fisher's Exact Test
appropriately instead of Chi-Square.

**Findings:**
1.  **Significant Association:** The analysis found a statistically significant
association (p=0.035) between the sector and the nature of harm.
2.  **Sector Differences:** Incidents in the **Financial** sector demonstrated a
much higher propensity for 'Demographic' harm (28.6%, n=2/7) compared to the
**Transportation** sector, which had zero recorded instances of demographic harm
(0%, n=0/28).
3.  **Conclusion:** The results confirm that AI failures in financial contexts
are distinctively prone to allocative or discriminatory harms, distinguishing
them from transportation-sector failures which likely skew towards physical
safety or reliability issues (captured here as 'Other Harm').

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# [debug]
# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# Normalize column names just in case, though they seem consistent based on previous output
# The relevant columns are 'Sector of Deployment' and 'Harm Distribution Basis'

# Clean and categorize Sector
def categorize_sector(val):
    if pd.isna(val):
        return None
    val_lower = str(val).lower()
    if 'financial' in val_lower or 'finance' in val_lower or 'banking' in val_lower:
        return 'Financial'
    if 'transport' in val_lower or 'automotive' in val_lower:
        return 'Transportation'
    return None

aiid_df['target_sector'] = aiid_df['Sector of Deployment'].apply(categorize_sector)

# Filter for only the two sectors of interest
sector_df = aiid_df.dropna(subset=['target_sector']).copy()

# Define Demographic Harm
# Looking for 'Demographic', 'Race', 'Gender', etc. 
# Based on AIID taxonomy, 'Harm Distribution Basis' usually contains 'Demographic' for discrimination issues.
def is_demographic(val):
    if pd.isna(val):
        return False
    val_lower = str(val).lower()
    keywords = ['demographic', 'race', 'gender', 'sex', 'ethnicity', 'age', 'religion', 'disability']
    return any(k in val_lower for k in keywords)

sector_df['is_demographic_harm'] = sector_df['Harm Distribution Basis'].apply(is_demographic)

# Create Contingency Table
contingency_table = pd.crosstab(sector_df['target_sector'], sector_df['is_demographic_harm'])

# Rename columns for clarity in output
contingency_table.columns = ['Other Harm', 'Demographic Harm']

print("--- Contingency Table ---")
print(contingency_table)
print("\n")

# Check sample sizes for statistical test selection
total_samples = contingency_table.sum().sum()
min_expected = 0
if contingency_table.size == 4:
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    min_expected = expected.min()
else:
    p = 1.0
    min_expected = 0

# Use Fisher's Exact Test if sample size is small or any expected cell count < 5
if min_expected < 5:
    print(f"Performing Fisher's Exact Test (Min expected count: {min_expected:.2f})...")
    oddsratio, p_value = stats.fisher_exact(contingency_table)
    test_name = "Fisher's Exact Test"
else:
    print(f"Performing Chi-Square Test (Min expected count: {min_expected:.2f})...")
    test_name = "Chi-Square Test"
    p_value = p

print(f"{test_name} Results:")
print(f"P-value: {p_value:.5f}")

# Calculate proportions
props = contingency_table.div(contingency_table.sum(axis=1), axis=0)
print("\n--- Proportions ---")
print(props)

# Visualization
ax = props.plot(kind='bar', stacked=True, figsize=(8, 6), color=['skyblue', 'salmon'])
plt.title(f'Harm Distribution Basis by Sector\n({test_name} p={p_value:.4f})')
plt.ylabel('Proportion of Incidents')
plt.xlabel('Sector')
plt.legend(title='Harm Basis', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Contingency Table ---
                Other Harm  Demographic Harm
target_sector                               
Financial                5                 2
Transportation          28                 0


Performing Fisher's Exact Test (Min expected count: 0.40)...
Fisher's Exact Test Results:
P-value: 0.03529

--- Proportions ---
                Other Harm  Demographic Harm
target_sector                               
Financial         0.714286          0.285714
Transportation    1.000000          0.000000


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

**1. Plot Type**
*   **Type:** Stacked Bar Plot.
*   **Purpose:** This plot compares the relative proportions of two categories of "Harm Basis" (Other Harm vs. Demographic Harm) across two distinct sectors (Financial and Transportation). The stacking allows for a visualization of the part-to-whole relationship within each sector.

**2. Axes**
*   **X-axis:**
    *   **Label:** "Sector"
    *   **Categories:** Two discrete categories are displayed: "Financial" and "Transportation". The labels are oriented vertically.
*   **Y-axis:**
    *   **Label:** "Proportion of Incidents"
    *   **Range:** The axis runs from 0.0 to 1.0, representing percentages from 0% to 100%.
    *   **Ticks:** Increments of 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).

**3. Data Trends**
*   **Financial Sector:** The bar shows a mix of harm types. Approximately 70-75% of the incidents fall under "Other Harm" (blue), while the remaining ~25-30% are attributed to "Demographic Harm" (red). This sector shows the highest proportion of Demographic Harm in the plot.
*   **Transportation Sector:** The bar is almost entirely composed of "Other Harm" (blue), reaching nearly 1.0 (100%). There is virtually no visible "Demographic Harm" (red) component for this sector.
*   **Comparison:** There is a distinct difference in composition between the two sectors. The Financial sector has a notable presence of Demographic Harm, whereas the Transportation sector is overwhelmingly characterized by Other Harm.

**4. Annotations and Legends**
*   **Main Title:** "Harm Distribution Basis by Sector"
*   **Subtitle/Statistical Annotation:** "(Fisher's Exact Test p=0.0353)" is included in the title area, indicating the statistical test used and the resulting p-value.
*   **Legend:** Located on the right side with the title "Harm Basis".
    *   **Sky Blue:** Represents "Other Harm".
    *   **Salmon/Red:** Represents "Demographic Harm".

**5. Statistical Insights**
*   **Significance:** The p-value provided in the title is **0.0353**. Since this value is less than the standard significance threshold of 0.05 (p < 0.05), the difference in the distribution of harm types between the Financial and Transportation sectors is statistically significant.
*   **Interpretation:** We can reject the null hypothesis that Sector and Harm Basis are independent. The data suggests that incidents in the Financial sector are significantly more likely to involve "Demographic Harm" compared to the Transportation sector, where such harm appears non-existent or extremely rare based on this dataset.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
