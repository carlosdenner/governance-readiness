# Experiment 11: node_3_2

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_2` |
| **ID in Run** | 11 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T01:30:12.496455+00:00 |
| **Runtime** | 291.4s |
| **Parent** | `node_2_3` |
| **Children** | `node_4_1`, `node_4_19`, `node_4_44` |
| **Creation Index** | 12 |

---

## Hypothesis

> Federal AI systems processing PII (Personally Identifiable Information) are not
statistically more likely to maintain a 'Data Catalog' than non-PII systems,
revealing a critical gap in privacy governance implementation.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.4194 (Maybe False) |
| **Posterior** | 0.8022 (Likely True) |
| **Surprise** | +0.4594 |
| **Surprise Interpretation** | Strong Positive (hypothesis strengthened) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 3.0 |
| Maybe True | 10.0 |
| Uncertain | 0.0 |
| Maybe False | 8.0 |
| Definitely False | 9.0 |

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

**Objective:** Evaluate if PII-flagged systems trigger better data documentation practices.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` (EO13960 subset).
- 2. Create a contingency table for `29_contains_pii` (Yes/No) vs. `31_data_catalog` (Yes/No).
- 3. Perform a Chi-Square Test of Independence or Fisher's Exact Test.
- 4. Calculate the Odds Ratio to quantify the association.

### Deliverables
- Contingency table; Chi-Square results; Bar chart of Data Catalog presence by PII status.

---

## Analysis

The experiment successfully analyzed 834 valid records from the EO13960 dataset
to evaluate the relationship between PII processing and data catalog
maintenance. The results support the hypothesis that systems processing PII are
not statistically more likely to maintain a data catalog than those that do not.

1.  **Statistical Significance**: The Chi-Square test yielded a p-value of 0.23,
indicating no statistically significant association between the two variables.
The null hypothesis could not be rejected.
2.  **Directionality**: Contrary to the expectation that sensitive data would
drive better governance, systems processing PII actually had a lower rate of
data catalog implementation (66.2%) compared to non-PII systems (71.5%).
3.  **Governance Gap**: The analysis reveals a critical gap, as 33.8% of federal
AI systems processing PII lack a data catalog, potentially hindering privacy
oversight and compliance.

---

## Review

The experiment successfully verified the hypothesis that federal AI systems
processing PII are not statistically more likely to maintain a Data Catalog than
those that do not. The implementation faithfully followed the plan, cleaning the
data to yield 834 valid records for analysis.

**Key Findings:**
1.  **Statistical Independence:** The Chi-Square test yielded a p-value of 0.23,
failing to reject the null hypothesis. There is no statistically significant
association between processing PII and having a Data Catalog.
2.  **Inverse Trend:** Contrary to the expectation that sensitive data warrants
better documentation, systems processing PII had a lower adoption rate of Data
Catalogs (66.2%) compared to non-PII systems (71.5%), though this difference was
not statistically significant.
3.  **Governance Gap:** The analysis highlights a critical compliance gap,
showing that 33.8% of systems identified as processing PII lack a Data Catalog,
a fundamental control for privacy governance.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_experiment():
    # Attempt to locate the file in current directory or parent directory
    filename = 'astalabs_discovery_all_data.csv'
    if os.path.exists(filename):
        file_path = filename
    elif os.path.exists(f'../{filename}'):
        file_path = f'../{filename}'
    else:
        print(f"Error: Could not find {filename} in current or parent directory.")
        return

    print(f"Loading dataset from {file_path}...")
    
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Filter for EO13960 subset
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO13960 subset size: {len(df_eo)} rows")

    # Columns of interest
    col_pii = '29_contains_pii'
    col_catalog = '31_data_catalog'

    if col_pii not in df_eo.columns or col_catalog not in df_eo.columns:
        print(f"Required columns not found. Available: {df_eo.columns.tolist()}")
        return

    # Data Cleaning
    def clean_response(val):
        s = str(val).strip().upper()
        if s in ['YES', 'Y', 'TRUE', '1']:
            return 'Yes'
        elif s in ['NO', 'N', 'FALSE', '0']:
            return 'No'
        return None

    df_eo['has_pii'] = df_eo[col_pii].apply(clean_response)
    df_eo['has_catalog'] = df_eo[col_catalog].apply(clean_response)

    # Drop invalid rows for analysis
    df_clean = df_eo.dropna(subset=['has_pii', 'has_catalog'])
    print(f"Valid records for analysis (after cleaning): {len(df_clean)}")

    if len(df_clean) == 0:
        print("Insufficient data for analysis.")
        return

    # 1. Contingency Table
    contingency = pd.crosstab(df_clean['has_pii'], df_clean['has_catalog'])
    print("\n--- Contingency Table (Counts) ---")
    print(contingency)

    # 2. Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")

    # 3. Odds Ratio Calculation
    try:
        def get_val(r, c):
            return contingency.loc[r, c] if r in contingency.index and c in contingency.columns else 0

        a = get_val('Yes', 'Yes') # PII=Yes, Cat=Yes
        b = get_val('Yes', 'No')  # PII=Yes, Cat=No
        c = get_val('No', 'Yes')  # PII=No, Cat=Yes
        d = get_val('No', 'No')   # PII=No, Cat=No
        
        print(f"\nCounts used for OR: PII_Yes/Cat_Yes={a}, PII_Yes/Cat_No={b}, PII_No/Cat_Yes={c}, PII_No/Cat_No={d}")

        if b == 0 or c == 0:
            print("Zero count detected in denominator, using Haldane-Anscombe correction (+0.5).")
            odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        else:
            odds_ratio = (a * d) / (b * c)
            
        print(f"Odds Ratio: {odds_ratio:.4f}")
        
    except Exception as e:
        print(f"Error calculating Odds Ratio: {e}")

    # Interpret results
    alpha = 0.05
    if p < alpha:
        print("\nResult: Statistically SIGNIFICANT association found.")
    else:
        print("\nResult: NO statistically significant association found.")

    # 4. Visualization
    # Prepare data for plotting (percentages)
    # Group by PII status and Catalog status to get counts
    plot_data = df_clean.groupby(['has_pii', 'has_catalog']).size().reset_index(name='count')
    
    # Calculate totals per PII group to normalize percentages
    # transform('sum') broadcasts the sum back to the original rows of the group
    plot_data['total_in_group'] = plot_data.groupby('has_pii')['count'].transform('sum')
    
    # Calculate percentage
    plot_data['percent'] = (plot_data['count'] / plot_data['total_in_group']) * 100

    print("\nPlot Data Preview:")
    print(plot_data)

    plt.figure(figsize=(8, 6))
    barplot = sns.barplot(data=plot_data, x='has_pii', y='percent', hue='has_catalog', palette='viridis')
    
    plt.title('Data Catalog Implementation by PII Status (EO13960)')
    plt.xlabel('System Processes PII?')
    plt.ylabel('Percentage of Systems within Group (%)')
    plt.legend(title='Has Data Catalog')
    plt.ylim(0, 100)
    
    # Add labels
    for container in barplot.containers:
        barplot.bar_label(container, fmt='%.1f%%', padding=3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
EO13960 subset size: 1757 rows
Valid records for analysis (after cleaning): 834

--- Contingency Table (Counts) ---
has_catalog   No  Yes
has_pii              
No           194  486
Yes           52  102

Chi-Square Statistic: 1.4136
P-value: 2.3446e-01

Counts used for OR: PII_Yes/Cat_Yes=102, PII_Yes/Cat_No=52, PII_No/Cat_Yes=486, PII_No/Cat_No=194
Odds Ratio: 0.7830

Result: NO statistically significant association found.

Plot Data Preview:
  has_pii has_catalog  count  total_in_group    percent
0      No          No    194             680  28.529412
1      No         Yes    486             680  71.470588
2     Yes          No     52             154  33.766234
3     Yes         Yes    102             154  66.233766


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Grouped Bar Chart (Clustered Bar Chart).
*   **Purpose:** The plot compares the implementation status of Data Catalogs across two distinct groups of systems: those that process Personally Identifiable Information (PII) and those that do not. It visualizes the relative percentage distribution within each group.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "System Processes PII?"
    *   **Labels:** "No" and "Yes". These categories represent whether the system handles sensitive PII data.
*   **Y-Axis:**
    *   **Title:** "Percentage of Systems within Group (%)"
    *   **Range:** 0 to 100.
    *   **Units:** Percentages (indicated by the % symbol in the title and the values on the bars).

### 3. Data Trends
*   **Overall Pattern:** Across both groups (systems that process PII and those that don't), the majority of systems **have** a data catalog implemented (indicated by the green bars).
*   **Group "No" (Systems that do not process PII):**
    *   **Tallest Bar:** "Yes" (Has Data Catalog) at **71.5%**.
    *   **Shortest Bar:** "No" (No Data Catalog) at **28.5%**.
*   **Group "Yes" (Systems that process PII):**
    *   **Tallest Bar:** "Yes" (Has Data Catalog) at **66.2%**.
    *   **Shortest Bar:** "No" (No Data Catalog) at **33.8%**.

### 4. Annotations and Legends
*   **Chart Title:** "Data Catalog Implementation by PII Status (EO13960)". This references Executive Order 13960, suggesting this data relates to federal agency compliance regarding AI or data usage.
*   **Legend:** Located in the top right corner with the header "Has Data Catalog".
    *   **Dark Blue/Slate:** Represents "No" (Data Catalog not present).
    *   **Green:** Represents "Yes" (Data Catalog present).
*   **Data Labels:** Specific percentage values are annotated directly on top of each bar (28.5%, 71.5%, 33.8%, 66.2%) for precise reading.

### 5. Statistical Insights
*   **Majority Adoption:** In both categories, over two-thirds of the systems have implemented a data catalog.
*   **Inverse Correlation with Sensitivity:** Surprisingly, systems that **do not** process PII have a higher rate of data catalog implementation (**71.5%**) compared to systems that **do** process PII (**66.2%**). One might typically expect systems handling sensitive PII to have higher governance and documentation standards.
*   **Gap Analysis:** There is a notable gap in coverage. Specifically, **33.8%** of systems that handle PII currently lack a data catalog. This represents a potential risk area for data governance and privacy compliance.
*   **Consistency:** The difference between the two groups is relatively small (approx. 5.3%), indicating that PII status is not a drastic differentiator in whether a catalog exists, though the trend leans slightly toward non-PII systems having better coverage.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
