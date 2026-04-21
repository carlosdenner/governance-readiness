# Experiment 75: node_4_34

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_34` |
| **ID in Run** | 75 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T04:33:51.612110+00:00 |
| **Runtime** | 209.8s |
| **Parent** | `node_3_14` |
| **Children** | `node_5_35`, `node_5_56` |
| **Creation Index** | 76 |

---

## Hypothesis

> Legacy Tech Debt: AI projects initiated before 2019 are significantly less
likely to maintain a 'Data Catalog' than projects initiated post-2019,
reflecting a lag in adopting modern data governance practices in older systems.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.2527 (Likely False) |
| **Surprise** | -0.5870 |
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
| Maybe False | 0.0 |
| Definitely False | 60.0 |

---

## Experiment Plan

**Objective:** Determine if the age of an AI system correlates with the maturity of its data governance.

### Steps
- 1. Load 'eo13960_scored'.
- 2. Parse '18_date_initiated' to datetime objects, handling errors/mixed formats.
- 3. Create a binary cohort variable: 'Legacy' (Start Date < 2019-01-01) vs. 'Modern' (Start Date >= 2019-01-01).
- 4. Compare the rate of affirmative responses in '31_data_catalog' between the two cohorts.
- 5. Perform a Chi-square test or Z-test.

### Deliverables
- Timeline plot of Data Catalog compliance; Cohort comparison statistics.

---

## Analysis

The experiment successfully tested the 'Legacy Tech Debt' hypothesis using the
EO 13960 dataset. After parsing dates and filtering for valid entries (n=1,191),
the data was split into 'Legacy' (<2019) and 'Modern' (>=2019) cohorts.

The results **contradict the hypothesis**:
1. **Observed Rates**: Legacy projects demonstrated a significantly *higher*
rate of Data Catalog maintenance (62.4%) compared to Modern projects (40.3%).
2. **Statistical Significance**: The Chi-Square test yielded a statistic of
27.56 with a p-value of < 0.001, indicating a strong dependency between project
age and data governance.
3. **Trend Analysis**: The timeline plot confirms a sharp decline in compliance
rates for recent projects, dropping from a peak of ~75% in 2015 and ~62% in 2019
down to ~33% in 2023.

These findings suggest that rather than suffering from tech debt, older systems
in this dataset appear to have more mature data governance documentation.
Conversely, newer 'Modern' deployments show a lag in adopting these controls,
potentially reflecting 'agile' governance gaps or a backlog in documenting
recently deployed systems.

---

## Review

The experiment successfully tested the 'Legacy Tech Debt' hypothesis using the
EO 13960 dataset. After parsing dates and filtering for valid entries (n=1,191),
the data was split into 'Legacy' (<2019) and 'Modern' (>=2019) cohorts.

The results **contradict the hypothesis**:
1. **Observed Rates**: Legacy projects demonstrated a significantly *higher*
rate of Data Catalog maintenance (62.4%) compared to Modern projects (40.3%).
2. **Statistical Significance**: The Chi-Square test yielded a statistic of
27.56 with a p-value of < 0.001, indicating a strong dependency between project
age and data governance.
3. **Trend Analysis**: The timeline plot confirms a sharp decline in compliance
rates for recent projects, dropping from a peak of ~75% in 2015 and ~62% in 2019
down to ~33% in 2023.

These findings suggest that rather than suffering from tech debt, older systems
in this dataset appear to have more mature data governance documentation.
Conversely, newer 'Modern' deployments show a lag in adopting these controls,
potentially reflecting 'agile' governance gaps or a backlog in documenting
recently deployed systems.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def run_experiment():
    # Load dataset
    file_path = "../astalabs_discovery_all_data.csv"
    
    print("Loading dataset...")
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        print("File not found in parent directory, trying current directory.")
        df = pd.read_csv("astalabs_discovery_all_data.csv", low_memory=False)

    # Filter EO 13960
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO 13960 records: {len(df_eo)}")

    # Columns
    date_col = '18_date_initiated'
    catalog_col = '31_data_catalog'

    if date_col not in df_eo.columns or catalog_col not in df_eo.columns:
        print(f"Error: Columns {date_col} or {catalog_col} not found.")
        return

    # Check unique values in catalog
    print(f"Unique values in '{catalog_col}': {df_eo[catalog_col].unique()}")

    # Clean catalog: 1 if YES, 0 otherwise
    # Being conservative: treat NaN as 0 (No). Only explicit 'Yes' counts as having a catalog.
    def is_affirmative(val):
        if pd.isna(val): return 0
        s = str(val).lower().strip()
        return 1 if s == 'yes' or s == 'true' else 0

    df_eo['has_catalog'] = df_eo[catalog_col].apply(is_affirmative)

    # Clean date
    # Convert to datetime using coerce to handle mixed formats
    df_eo['dt'] = pd.to_datetime(df_eo[date_col], errors='coerce')
    
    # Check how many dates were parsed
    parsed_count = df_eo['dt'].notna().sum()
    print(f"Parsed {parsed_count} valid dates out of {len(df_eo)} records.")

    if parsed_count < 10:
        print("Too few valid dates parsed. Aborting analysis.")
        print("Sample raw dates:", df_eo[date_col].dropna().head(10).tolist())
        return

    df_clean = df_eo.dropna(subset=['dt']).copy()
    
    # Define Cohorts
    # Legacy: Started before 2019-01-01
    cutoff = pd.Timestamp("2019-01-01")
    df_clean['cohort'] = df_clean['dt'].apply(lambda x: 'Legacy' if x < cutoff else 'Modern')

    # Stats
    stats_df = df_clean.groupby('cohort')['has_catalog'].agg(['count', 'mean', 'sum'])
    stats_df.rename(columns={'mean': 'compliance_rate', 'sum': 'num_compliant', 'count': 'n'}, inplace=True)
    
    print("\n--- Cohort Analysis ---")
    print(stats_df)

    # Contingency Table for Chi-Square
    # We need a 2x2 table of counts
    contingency = pd.crosstab(df_clean['cohort'], df_clean['has_catalog'])
    print("\n--- Contingency Table (0=No, 1=Yes) ---")
    print(contingency)

    # Run Chi-Square Test of Independence
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test: statistic={chi2:.4f}, p-value={p_val:.4f}")

    # Calculate Odds Ratio manually for interpretation
    # OR = (Modern_Yes / Modern_No) / (Legacy_Yes / Legacy_No)
    if 'Legacy' in contingency.index and 'Modern' in contingency.index:
        try:
            # Check if columns 0 and 1 exist (0=No, 1=Yes)
            leg_no = contingency.loc['Legacy', 0] if 0 in contingency.columns else 0
            leg_yes = contingency.loc['Legacy', 1] if 1 in contingency.columns else 0
            mod_no = contingency.loc['Modern', 0] if 0 in contingency.columns else 0
            mod_yes = contingency.loc['Modern', 1] if 1 in contingency.columns else 0
            
            if mod_no > 0 and leg_yes > 0 and leg_no > 0:
                 or_val = (mod_yes / mod_no) / (leg_yes / leg_no)
                 print(f"Odds Ratio (Modern vs Legacy): {or_val:.4f}")
            else:
                 print("Cannot calculate Odds Ratio due to zero counts in denominator.")
        except Exception as e:
            print(f"Could not calculate Odds Ratio: {e}")

    # Visualization
    df_clean['year'] = df_clean['dt'].dt.year
    # Group by year
    yearly = df_clean.groupby('year')['has_catalog'].mean()
    counts = df_clean.groupby('year')['has_catalog'].count()
    
    # Filter years with fewer than 5 records to reduce noise in the plot
    valid_years = counts[counts >= 5].index
    if len(valid_years) > 0:
        yearly_plot = yearly.loc[valid_years].sort_index()
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(yearly_plot.index, yearly_plot.values, color='cornflowerblue', label='Data Catalog Rate')
        
        # Add values on bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)

        plt.axvline(x=2018.5, color='red', linestyle='--', linewidth=2, label='2019 Cutoff')
        plt.title('Data Catalog Compliance Rate by Project Start Year')
        plt.xlabel('Year Initiated')
        plt.ylabel('Proportion with Data Catalog')
        plt.ylim(0, 1.15)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.show()
    else:
        print("Not enough data per year to plot.")

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset...
File not found in parent directory, trying current directory.
EO 13960 records: 1757
Unique values in '31_data_catalog': <StringArray>
['No', nan, 'Yes', 'Other', 'NO', ' ']
Length: 6, dtype: str
Parsed 1191 valid dates out of 1757 records.

--- Cohort Analysis ---
           n  compliance_rate  num_compliant
cohort                                      
Legacy   165         0.624242            103
Modern  1026         0.402534            413

--- Contingency Table (0=No, 1=Yes) ---
has_catalog    0    1
cohort               
Legacy        62  103
Modern       613  413

Chi-Square Test: statistic=27.5589, p-value=0.0000
Odds Ratio (Modern vs Legacy): 0.4055


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Chart.
*   **Purpose:** The plot visualizes the compliance rate (proportion) of projects possessing a data catalog, categorized by the year the project was initiated. It aims to show historical trends and how compliance has changed over time, specifically highlighting a cutoff point in 2019.

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Label:** "Year Initiated"
    *   **Range:** The axis spans from roughly **1900 to 2025**.
    *   **Scale:** The data is grouped into distinct years. There is a significant gap in data between the early 1900s and the cluster of data starting around 2015.
*   **Y-Axis (Vertical):**
    *   **Label:** "Proportion with Data Catalog"
    *   **Range:** **0.0 to 1.15** (though the grid lines stop at 1.0).
    *   **Units:** The values represent a decimal proportion (ratio), equivalent to 0% to 100%.

### 3. Data Trends
*   **Historical Outlier (approx. 1905):** There is a single isolated bar in the early 20th century (roughly 1905) with a high compliance rate of **0.73**. This likely represents legacy data or a specific archival dataset.
*   **Recent Cluster (2015–2023):** The majority of the data is clustered in the last decade.
    *   **Tallest Bar:** The highest recent compliance rate occurred in **2015** with a proportion of **0.75**.
    *   **Shortest Bar:** The lowest compliance rate in the recent cluster is the most recent year shown (likely **2023**), with a proportion of **0.33**.
    *   **Pattern of Decline:** There is a noticeable downward trend in the most recent years. After a spike in 2019 (0.62), the rate steadily drops: 0.58 (2020) -> 0.45 (2021) -> 0.40 (2022) -> 0.33 (2023).

### 4. Annotations and Legends
*   **Legend:** Located in the top-left corner.
    *   **"2019 Cutoff"**: Represented by a dashed red line.
    *   **"Data Catalog Rate"**: Represented by the blue bars.
*   **2019 Cutoff Line:** A vertical, dashed red line is drawn at the year 2019. This likely marks a policy change, a system implementation, or a deadline relevant to data cataloging.
*   **Value Labels:** Specific numerical values are annotated directly on top of each bar (e.g., 0.73, 0.75, 0.63, etc.) for precise reading of the data.

### 5. Statistical Insights
*   **Post-Cutoff Effect:** Immediately following the "2019 Cutoff," the compliance rate jumped from a local low of **0.38** (in 2018) to **0.62** (in 2019) and **0.58** (in 2020). This suggests that the event marked by the cutoff initially boosted compliance.
*   **Lag or Decay:** Following the 2019/2020 boost, compliance rates have degraded significantly, dropping by nearly half from 2019 (0.62) to 2023 (0.33). This could indicate that newer projects are lagging in documentation, or that enforcement of the requirement has relaxed.
*   **Volatility:** The data shows high volatility. Between 2015 and 2018, rates dropped from 0.75 to 0.38 before rebounding. This suggests inconsistent application of data cataloging practices over time.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
