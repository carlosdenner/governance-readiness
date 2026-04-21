# Experiment 128: node_5_41

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_41` |
| **ID in Run** | 128 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T07:03:45.285213+00:00 |
| **Runtime** | 351.0s |
| **Parent** | `node_4_23` |
| **Children** | `node_6_7`, `node_6_22`, `node_6_73` |
| **Creation Index** | 129 |

---

## Hypothesis

> Policy-Practice Decoupling: There is no significant correlation between self-
reported compliance with the 'Information Quality Act' (IQA) and the operational
implementation of 'Post-Deployment Monitoring'.

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

**Objective:** Test if bureaucratic compliance (IQA) translates to operational oversight (Monitoring).

### Steps
- 1. Load `eo13960_scored`.
- 2. Clean `28_iqa_compliance` (Yes/No) and `56_monitor_postdeploy` (Yes/No).
- 3. Create a confusion matrix/contingency table.
- 4. Calculate the Phi coefficient (correlation) and run a Chi-Square test.
- 5. Analyze the 'compliant but not monitoring' quadrant.

### Deliverables
- Contingency table; Phi coefficient; Chi-Square p-value; Analysis of the 'Paper Tiger' segment (IQA Yes / Monitor No).

---

## Analysis

The experiment was successfully executed. By implementing robust fuzzy matching
to categorize free-text responses in the 'eo13960_scored' dataset, the script
successfully isolated 37 valid data points for analysis. The Chi-Square test
yielded a p-value of 0.0137, which is statistically significant (p < 0.05).
Consequently, we **reject** the hypothesis of Policy-Practice Decoupling.

The data instead supports **Policy-Practice Alignment**: there is a moderate
positive correlation (Phi = 0.4053) between bureaucratic compliance (IQA) and
operational oversight (Monitoring). Specifically, 92.6% (25/27) of systems
claiming IQA compliance also reported active post-deployment monitoring
mechanisms. The 'Paper Tiger' rate—instances where policy exists without
practice—was found to be low at 7.4%. Additionally, the analysis revealed a
cluster of 'Proactive Engineering' (5 cases) where monitoring is performed even
without a formal IQA mandate.

---

## Review

The experiment was successfully executed. By implementing robust fuzzy matching
to categorize free-text responses in the 'eo13960_scored' dataset, the script
successfully isolated 37 valid data points for analysis. The Chi-Square test
yielded a p-value of 0.0137, which is statistically significant (p < 0.05).
Consequently, we **reject** the hypothesis of Policy-Practice Decoupling.

The data instead supports **Policy-Practice Alignment**: there is a moderate
positive correlation (Phi = 0.4053) between bureaucratic compliance (IQA) and
operational oversight (Monitoring). Specifically, 92.6% (25/27) of systems
claiming IQA compliance also reported active post-deployment monitoring
mechanisms. The 'Paper Tiger' rate—instances where policy exists without
practice—was found to be low at 7.4%. Additionally, the analysis revealed a
cluster of 'Proactive Engineering' (5 cases) where monitoring is performed even
without a formal IQA mandate.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
file_name = 'astalabs_discovery_all_data.csv'
if os.path.exists(file_name):
    file_path = file_name
elif os.path.exists(f'../{file_name}'):
    file_path = f'../{file_name}'
else:
    print("Dataset not found.")
    exit(1)

print(f"Loading dataset from: {file_path}")

try:
    df = pd.read_csv(file_path, low_memory=False)
    
    # Filter for the relevant source table
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    
    col_policy = '28_iqa_compliance'
    col_practice = '56_monitor_postdeploy'
    
    # --- MAPPING FUNCTIONS ---
    
    def map_iqa_compliance(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).lower()
        
        # Priority: Check for explicit "No" indicators first
        no_keywords = ['not applicable', 'non-public', 'proof of concept', "doesn't appear to meet", 'research']
        if any(k in val_str for k in no_keywords):
            return 'No'
            
        # Check for "Yes" indicators
        yes_keywords = ['policy', 'policies', 'compliance', 'practices', 'checks', 'standard', 'guidance', 'procedures']
        if any(k in val_str for k in yes_keywords):
            return 'Yes'
        
        return np.nan

    def map_monitoring(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).lower()
        
        # Priority: Check for explicit "No" indicators first
        no_keywords = ['no monitoring', 'not available', 'not safety', 'not rights-impacting', 'under development']
        if any(k in val_str for k in no_keywords):
            return 'No'
            
        # Check for "Yes" indicators
        yes_keywords = ['intermittent', 'automated', 'established process', 'manually updated', 'regularly scheduled', 'plan for monitoring']
        if any(k in val_str for k in yes_keywords):
            return 'Yes'
            
        return np.nan

    # Apply mappings
    df_eo['IQA_Mapped'] = df_eo[col_policy].apply(map_iqa_compliance)
    df_eo['Monitor_Mapped'] = df_eo[col_practice].apply(map_monitoring)
    
    # Drop rows where either value couldn't be mapped
    df_clean = df_eo.dropna(subset=['IQA_Mapped', 'Monitor_Mapped']).copy()
    
    print(f"\nTotal rows in source: {len(df_eo)}")
    print(f"Rows with valid mapped data: {len(df_clean)}")
    
    # Create Contingency Table
    contingency_table = pd.crosstab(
        df_clean['IQA_Mapped'], 
        df_clean['Monitor_Mapped']
    )
    
    print("\n--- Contingency Table (Mapped) ---")
    print(contingency_table)
    
    if contingency_table.empty:
        print("No valid data for analysis.")
    else:
        # Ensure 2x2 shape if possible by reindexing
        contingency_table = contingency_table.reindex(index=['Yes', 'No'], columns=['Yes', 'No']).fillna(0)
        
        print("\n--- Reindexed Contingency Table ---")
        print(contingency_table)

        # Chi-Square Test
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Phi Coefficient
        n = contingency_table.sum().sum()
        phi = np.sqrt(chi2 / n) if n > 0 else 0
        
        print("\n--- Statistical Results ---")
        print(f"N: {n}")
        print(f"Chi-Square Statistic: {chi2:.4f}")
        print(f"P-value: {p:.4e}")
        print(f"Phi Coefficient: {phi:.4f}")
        
        # Interpretations
        print("\n--- Analysis ---")
        if p < 0.05:
            print("Result: Statistically Significant Association (Reject Null Hypothesis)")
        else:
            print("Result: No Significant Association (Fail to Reject Null Hypothesis)")
            
        # Paper Tiger Analysis (IQA=Yes, Monitor=No)
        iqa_yes_total = contingency_table.loc['Yes'].sum()
        paper_tiger_count = contingency_table.loc['Yes', 'No']
        
        if iqa_yes_total > 0:
            paper_tiger_rate = (paper_tiger_count / iqa_yes_total) * 100
            print(f"\n'Paper Tiger' Rate: {paper_tiger_rate:.1f}% ({int(paper_tiger_count)}/{int(iqa_yes_total)}) of IQA-compliant systems lack operational monitoring.")
        else:
            print("\nNo IQA-compliant systems found to calculate 'Paper Tiger' rate.")
        
        # Visualization
        plt.figure(figsize=(7, 6))
        sns.heatmap(contingency_table, annot=True, fmt='.0f', cmap='Oranges', cbar=False)
        plt.title('Policy-Practice Decoupling:\nIQA Compliance vs. Post-Deployment Monitoring')
        plt.ylabel('IQA Compliance (Policy)')
        plt.xlabel('Post-Deployment Monitoring (Practice)')
        plt.tight_layout()
        plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: astalabs_discovery_all_data.csv

Total rows in source: 1757
Rows with valid mapped data: 37

--- Contingency Table (Mapped) ---
Monitor_Mapped  No  Yes
IQA_Mapped             
No               5    5
Yes              2   25

--- Reindexed Contingency Table ---
Monitor_Mapped  Yes  No
IQA_Mapped             
Yes              25   2
No                5   5

--- Statistical Results ---
N: 37
Chi-Square Statistic: 6.0768
P-value: 1.3697e-02
Phi Coefficient: 0.4053

--- Analysis ---
Result: Statistically Significant Association (Reject Null Hypothesis)

'Paper Tiger' Rate: 7.4% (2/27) of IQA-compliant systems lack operational monitoring.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Heatmap representing a **Contingency Table** (or Confusion Matrix).
*   **Purpose:** To visualize the relationship and overlap between two categorical variables: "IQA Compliance (Policy)" and "Post-Deployment Monitoring (Practice)." It aims to highlight instances where policy aligns with practice versus where "decoupling" (a mismatch) occurs.

### 2. Axes
*   **Y-Axis (Vertical):**
    *   **Label:** "IQA Compliance (Policy)"
    *   **Categories:** "Yes" (Top row) and "No" (Bottom row).
*   **X-Axis (Horizontal):**
    *   **Label:** "Post-Deployment Monitoring (Practice)"
    *   **Categories:** "Yes" (Left column) and "No" (Right column).
*   **Value Ranges:** Since the axes represent binary categories, there are no numerical ranges on the axes themselves. The values within the matrix represent frequency counts ranging from **2 to 25**.

### 3. Data Trends
*   **Highest Value (Darkest Area):** The top-left quadrant (Policy: Yes / Practice: Yes) contains the highest count of **25**. This indicates that in the majority of cases, IQA compliance policy exists, and post-deployment monitoring is actively practiced.
*   **Lowest Value (Lightest Area):** The top-right quadrant (Policy: Yes / Practice: No) has the lowest count of **2**. This represents a rare scenario where the policy exists, but the practice is not followed.
*   **Uniform Low Values:** The bottom row shows uniform counts of **5** for both "Yes" and "No" in the Practice category when the Policy is "No."

### 4. Annotations and Legends
*   **Title:** "Policy-Practice Decoupling: IQA Compliance vs. Post-Deployment Monitoring." This sets the context for the analysis, focusing on how well written rules (policy) match actual behavior (practice).
*   **Cell Values:** Each of the four cells contains a numerical annotation representing the count for that intersection:
    *   **25**: Yes Policy / Yes Practice
    *   **2**: Yes Policy / No Practice
    *   **5**: No Policy / Yes Practice
    *   **5**: No Policy / No Practice
*   **Color Scale:** A sequential color palette (shades of orange/brown) is used, where darker colors indicate higher frequency counts and lighter colors indicate lower counts.

### 5. Statistical Insights
*   **Total Sample Size:** The total number of observations is **37** ($25 + 2 + 5 + 5$).
*   **Strong Alignment (Coupling):** There is a high degree of alignment between policy and practice.
    *   **81%** of cases ($30/37$) show alignment (Yes/Yes or No/No).
    *   Specifically, the **Yes/Yes** category dominates, accounting for roughly **67.5%** of the total data, suggesting that when a policy is in place, it is usually followed.
*   **Analysis of Decoupling (Mismatch):**
    *   **Policy without Practice:** Only **2 cases** show a policy being present without the corresponding monitoring practice. This suggests non-compliance is rare.
    *   **Practice without Policy:** Interestingly, there are **5 cases** where monitoring occurs ("Yes") despite there being no IQA compliance policy ("No"). This suggests proactive behavior where teams are monitoring deployments even without a formal mandate.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
