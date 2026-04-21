# Experiment 279: node_6_76

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_76` |
| **ID in Run** | 279 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T14:20:14.081673+00:00 |
| **Runtime** | 369.2s |
| **Parent** | `node_5_16` |
| **Children** | None |
| **Creation Index** | 280 |

---

## Hypothesis

> Public Sector Intentionality: Incidents arising from 'Public Sector' deployments
are significantly more likely to be classified as 'Intentional Harm' (e.g.,
planned surveillance) compared to 'Private Sector' incidents (e.g., accidental
failures).

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7661 (Likely True) |
| **Posterior** | 0.2665 (Likely False) |
| **Surprise** | -0.5996 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 3.0 |
| Maybe True | 27.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 2.0 |
| Definitely False | 58.0 |

---

## Experiment Plan

**Objective:** Investigate if government AI failures are more often features (intentional) rather than bugs (accidents) compared to the private sector.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' (AIID subset).
- 2. Use `Public Sector Deployment` to categorize incidents into 'Public' (Yes) vs 'Private' (No).
- 3. Clean `Intentional Harm` column into binary: 'Intentional' (Yes/Designed to harm) vs 'Accidental/Unintentional' (No/Not designed to harm).
- 4. Create a contingency table and perform a Chi-square test.

### Deliverables
- Contingency table, Chi-square test results, and a comparison of intentional harm rates.

---

## Analysis

The experiment successfully tested the hypothesis regarding the intentionality
of AI harm in the Public vs. Private sectors using the AIID dataset. After
robust data cleaning to handle verbose labels, 188 incidents (149 Private, 39
Public) were available for analysis.

**Findings:**
1.  **Hypothesis Rejected:** The hypothesis that Public Sector incidents are
significantly more likely to be 'Intentional' was **rejected**. In fact, the
data showed the opposite directional trend, though not statistically
significant.
2.  **Sector Comparison:**
    - **Private Sector:** 94.6% Accidental vs. 5.4% Intentional (8 incidents).
    - **Public Sector:** 100% Accidental vs. 0% Intentional (0 incidents).
3.  **Statistical Significance:** The Chi-square test yielded a p-value of
**0.3015** (Statistic=1.0677), indicating no statistically significant
association between sector and harm intentionality. The null hypothesis could
not be rejected.
4.  **Dominant Pattern:** 'Accidental/Unintentional' harm is the overwhelming
norm for AI incidents across both sectors in this dataset.

---

## Review

The experiment was successfully executed and the hypothesis was tested
faithfully. After resolving initial data cleaning issues regarding verbose
labels, the analysis was performed on a subset of 188 incidents (149 Private, 39
Public).

**Hypothesis Evaluation:**
The hypothesis that Public Sector deployments are significantly more likely to
result in 'Intentional Harm' was **rejected**.

**Key Findings:**
1.  **Directionality:** The data showed the opposite trend of the hypothesis.
The Private sector had a higher rate of intentional harm (5.4%, 8 incidents)
compared to the Public sector (0%, 0 incidents).
2.  **Statistical Significance:** The difference was not statistically
significant (Chi-square p-value = 0.3015), indicating that for this dataset, the
sector does not reliably predict the intentionality of the harm.
3.  **Prevalence:** 'Accidental/Unintentional' harm is the dominant mode of
failure for both sectors (>94%).

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def analyze_public_sector_intentionality():
    # --- Load Dataset ---
    # Trying both current and parent directory to be safe, though previous run confirmed current.
    filename = 'astalabs_discovery_all_data.csv'
    if os.path.exists(filename):
        df = pd.read_csv(filename, low_memory=False)
    elif os.path.exists(os.path.join('..', filename)):
        df = pd.read_csv(os.path.join('..', filename), low_memory=False)
    else:
        print(f"Error: {filename} not found.")
        return

    # --- Filter AIID Data ---
    df_aiid = df[df['source_table'] == 'aiid_incidents'].copy()
    print(f"AIID Incidents loaded: {len(df_aiid)}")

    # --- Data Cleaning Logic ---
    col_public = 'Public Sector Deployment'
    col_intent = 'Intentional Harm'

    # Cleaning function for Public Sector
    def clean_sector(val):
        if pd.isna(val):
            return np.nan
        s = str(val).lower().strip()
        if s == 'yes':
            return 'Public'
        elif s == 'no':
            return 'Private'
        return np.nan

    # Cleaning function for Intentional Harm (handling verbose strings)
    def clean_intent(val):
        if pd.isna(val):
            return np.nan
        s = str(val).lower().strip()
        # Check starts with logic based on previous debug output
        if s.startswith('yes'):
            return 'Intentional'
        elif s.startswith('no'):
            return 'Accidental'
        return np.nan

    # Apply cleaning
    df_aiid['Sector_Clean'] = df_aiid[col_public].apply(clean_sector)
    df_aiid['Intent_Clean'] = df_aiid[col_intent].apply(clean_intent)

    # Filter valid rows
    df_clean = df_aiid.dropna(subset=['Sector_Clean', 'Intent_Clean']).copy()
    print(f"Valid rows for analysis: {len(df_clean)}")

    if len(df_clean) < 5:
        print("Insufficient data for analysis.")
        return

    # --- Statistical Analysis ---
    # Contingency Table
    ct = pd.crosstab(df_clean['Sector_Clean'], df_clean['Intent_Clean'])
    print("\n--- Contingency Table (Counts) ---")
    print(ct)
    
    # Proportions
    ct_prop = pd.crosstab(df_clean['Sector_Clean'], df_clean['Intent_Clean'], normalize='index')
    print("\n--- Contingency Table (Proportions) ---")
    print(ct_prop)
    
    # Chi-square Test
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    
    alpha = 0.05
    if p < alpha:
        print("Result: Statistically Significant (Reject H0)")
        print("There IS a significant difference in intentionality between Public and Private sectors.")
    else:
        print("Result: Not Significant (Fail to Reject H0)")
        print("There is NO significant difference in intentionality between Public and Private sectors.")

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ct_prop.plot(kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e'])
    
    ax.set_title('Intentional vs Accidental Harm by Sector (AIID)')
    ax.set_ylabel('Proportion')
    ax.set_xlabel('Sector')
    ax.set_ylim(0, 1.0)
    
    # Add labels
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1%', label_type='center', color='white', weight='bold')
    
    plt.legend(title='Harm Intent', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_public_sector_intentionality()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: AIID Incidents loaded: 1362
Valid rows for analysis: 188

--- Contingency Table (Counts) ---
Intent_Clean  Accidental  Intentional
Sector_Clean                         
Private              141            8
Public                39            0

--- Contingency Table (Proportions) ---
Intent_Clean  Accidental  Intentional
Sector_Clean                         
Private         0.946309     0.053691
Public          1.000000     0.000000

Chi-square Statistic: 1.0677
P-value: 0.3015
Result: Not Significant (Fail to Reject H0)
There is NO significant difference in intentionality between Public and Private sectors.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** 100% Stacked Bar Chart.
*   **Purpose:** The plot compares the relative proportions of two categories of harm intent ("Accidental" vs. "Intentional") across two different sectors ("Private" vs. "Public"). By normalizing the height of the bars to 1.0 (100%), it focuses on the composition of the data within each sector rather than the absolute number of incidents.

### 2. Axes
*   **X-axis:**
    *   **Label:** "Sector"
    *   **Categories:** "Private" and "Public".
*   **Y-axis:**
    *   **Label:** "Proportion"
    *   **Range:** 0.0 to 1.0 (representing 0% to 100%).
    *   **Ticks:** Intervals of 0.2.

### 3. Data Trends
*   **Dominant Trend:** In both the Private and Public sectors, "Accidental" harm (represented by the blue bars) makes up the overwhelming majority of incidents.
*   **Private Sector:**
    *   **Accidental:** Visually accounts for approximately 95% of the bar.
    *   **Intentional:** Represents a small minority, visually estimated at roughly 5% (the orange segment at the top).
*   **Public Sector:**
    *   **Accidental:** Visually accounts for nearly 100% of the bar.
    *   **Intentional:** There is almost no visible orange segment, suggesting that intentional harm in the Public sector is negligible or non-existent in this specific dataset.
*   **Comparison:** The Private sector has a noticeably larger proportion of intentional harm compared to the Public sector, though it remains a small fraction of the total.

### 4. Annotations and Legends
*   **Legend:** Located on the right side, titled "Harm Intent."
    *   **Blue:** Accidental
    *   **Orange:** Intentional
*   **Annotations (Data Labels):**
    *   There is white text centered within the bar segments reading `%.1%`.
    *   **Observation:** This indicates a coding/formatting error in the generation of the chart (likely a format string literal was printed instead of the actual numerical value). As a result, exact percentages cannot be read directly from the labels and must be estimated visually from the Y-axis.

### 5. Statistical Insights
*   **Nature of AI Incidents:** According to the AIID (Artificial Intelligence Incident Database) data presented here, AI-related harm is predominantly accidental rather than malicious or intentional.
*   **Sector Behavior:** The Public sector appears to operate with a near-zero rate of intentional harm incidents. In contrast, the Private sector, while still mostly accidental, demonstrates a non-zero incidence of intentional harm. This could suggest different risk profiles, use-cases, or reporting standards between commercial/private AI deployments and government/public ones.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
