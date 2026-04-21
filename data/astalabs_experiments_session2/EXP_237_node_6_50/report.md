# Experiment 237: node_6_50

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_50` |
| **ID in Run** | 237 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T12:10:52.202392+00:00 |
| **Runtime** | 201.8s |
| **Parent** | `node_5_1` |
| **Children** | None |
| **Creation Index** | 238 |

---

## Hypothesis

> The 'Code Sovereignty' Gap: AI systems developed by external contractors are
significantly less likely to grant government agencies full access to source
code compared to systems developed in-house, creating a 'black box' maintenance
risk.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9839 (Definitely True) |
| **Posterior** | 0.9945 (Definitely True) |
| **Surprise** | +0.0128 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 30.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

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

**Objective:** Determine if the development method (Outsourced vs. In-House) predicts the availability of code access.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `eo13960_scored`.
- 2. Create a grouping variable `dev_model` from `22_dev_method`: 'Contracted' vs 'In-House' (exclude others).
- 3. Clean the `38_code_access` column into a binary variable: 'Yes' (contains 'Yes' or 'Open') vs 'No' (contains 'No', 'Restricted', 'Proprietary').
- 4. Calculate the percentage of systems with code access for each group.
- 5. Perform a Chi-Square test of independence.

### Deliverables
- Bar chart of Code Access Rates by Development Model; Chi-Square test results.

---

## Analysis

The experiment successfully validated the 'Code Sovereignty' hypothesis.
Analyzing 781 records from the EO 13960 dataset, the study found a striking
disparity in source code availability based on the development model. Agencies
retain code access for 90.5% of systems developed in-house, compared to only
33.8% for systems developed by contractors. The Chi-Square test confirmed this
difference is highly statistically significant (p=1.74e-59). These findings
highlight a critical governance gap: when the US government outsources AI
development, it loses direct access to the source code in nearly two-thirds of
cases, potentially impeding auditability and long-term maintenance.

---

## Review

The experiment successfully validated the 'Code Sovereignty' hypothesis.
Analyzing 781 records from the EO 13960 dataset, the study found a striking
disparity in source code availability based on the development model. Agencies
retain code access for 90.5% of systems developed in-house, compared to only
33.8% for systems developed by contractors. The Chi-Square test confirmed this
difference is highly statistically significant (p=1.74e-59). These findings
highlight a critical governance gap: when the US government outsources AI
development, it loses direct access to the source code in nearly two-thirds of
cases, potentially impeding auditability and long-term maintenance.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import re
import sys

def load_data():
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        try:
            df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
        except FileNotFoundError:
            print("Dataset not found.")
            sys.exit(1)
    return df

def classify_code_access(val):
    if not isinstance(val, str):
        return np.nan
    val_lower = val.lower()
    
    # Negative indicators
    # Use word boundaries for 'no' to avoid matching inside words like 'innovation'
    # 'proprietary' and 'restricted' are distinctive enough
    is_negative = False
    if re.search(r'\bno\b', val_lower) or 'restricted' in val_lower or 'proprietary' in val_lower:
        is_negative = True
        
    # Positive indicators
    is_positive = False
    if re.search(r'\byes\b', val_lower) or 'open' in val_lower:
        is_positive = True
        
    # Classification Logic
    if is_positive and not is_negative:
        return 1
    elif is_negative and not is_positive:
        return 0
    elif is_positive and is_negative:
        # Conflict: usually 'No, but...' or 'Yes, however restricted...'
        # In context of 'Code Sovereignty', restrictions usually mean lack of full sovereignty.
        return 0
    else:
        return np.nan

def main():
    print("Loading dataset...")
    df = load_data()
    
    # Filter for EO 13960 Scored subset
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO 13960 subset size: {len(df_eo)}")
    
    # Map Development Method
    if '22_dev_method' not in df_eo.columns:
        print("Error: Column '22_dev_method' not found.")
        return
        
    dev_map = {
        'Developed with contracting resources.': 'Contracted',
        'Developed in-house.': 'In-House'
    }
    df_eo['dev_model'] = df_eo['22_dev_method'].map(dev_map)
    
    # Filter valid development models
    df_analysis = df_eo.dropna(subset=['dev_model']).copy()
    print(f"Records with valid Development Method: {len(df_analysis)}")
    
    # Process Code Access
    col_access = '38_code_access'
    if col_access not in df_analysis.columns:
        print(f"Error: Column '{col_access}' not found.")
        return
        
    df_analysis['access_binary'] = df_analysis[col_access].apply(classify_code_access)
    
    # Filter valid access scores
    df_clean = df_analysis.dropna(subset=['access_binary'])
    print(f"Records with valid Code Access data: {len(df_clean)}")
    
    # Calculate Statistics
    summary = df_clean.groupby('dev_model')['access_binary'].agg(['count', 'mean'])
    summary['percent'] = summary['mean'] * 100
    
    print("\nSummary Statistics (Code Access Rates):")
    print(summary)
    
    # Chi-Square Test
    contingency = pd.crosstab(df_clean['dev_model'], df_clean['access_binary'])
    print("\nContingency Table (0=No Access, 1=Access):")
    print(contingency)
    
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test Results:")
    print(f"Chi-Square Statistic: {chi2:.4f}")
    print(f"P-Value: {p:.4e}")
    
    if p < 0.05:
        print("Conclusion: Significant relationship between development model and code access.")
    else:
        print("Conclusion: No significant relationship detected.")
        
    # Visualization
    plt.figure(figsize=(8, 6))
    colors = ['#ff9999', '#66b3ff']
    ax = summary['percent'].plot(kind='bar', color=colors, edgecolor='black', rot=0)
    
    plt.title('Code Access Rates by Development Model')
    plt.ylabel('Percentage with Code Access (%)')
    plt.xlabel('Development Model')
    plt.ylim(0, 105)
    
    # Annotate bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
                    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset...
EO 13960 subset size: 1757
Records with valid Development Method: 920
Records with valid Code Access data: 781

Summary Statistics (Code Access Rates):
            count      mean    percent
dev_model                             
Contracted    390  0.338462  33.846154
In-House      391  0.905371  90.537084

Contingency Table (0=No Access, 1=Access):
access_binary  0.0  1.0
dev_model              
Contracted     258  132
In-House        37  354

Chi-Square Test Results:
Chi-Square Statistic: 264.5628
P-Value: 1.7376e-59
Conclusion: Significant relationship between development model and code access.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

**1. Plot Type**
*   **Type:** Vertical Bar Plot (or Bar Chart).
*   **Purpose:** The plot is designed to compare the percentage of code access between two distinct categorical groups: "Contracted" and "In-House" development models.

**2. Axes**
*   **X-Axis (Horizontal):**
    *   **Label:** "Development Model"
    *   **Categories:** The axis displays two discrete categories: "Contracted" and "In-House".
*   **Y-Axis (Vertical):**
    *   **Label:** "Percentage with Code Access (%)"
    *   **Range:** The scale ranges from 0 to roughly 105, with major tick marks at intervals of 20 (0, 20, 40, 60, 80, 100).
    *   **Units:** Percentage (%).

**3. Data Trends**
*   **Tallest Bar:** The "In-House" category represents the highest value, reaching near the top of the y-axis scale.
*   **Shortest Bar:** The "Contracted" category is significantly lower, reaching only about one-third of the height of the y-axis.
*   **Pattern:** There is a stark disparity between the two models. The plot visualizes a massive gap in code access, showing that In-House development is associated with significantly higher access rates compared to Contracted development.

**4. Annotations and Legends**
*   **Data Labels:** Specific percentage values are annotated directly on top of each bar in bold text to provide precise measurements:
    *   **Contracted:** 33.8%
    *   **In-House:** 90.5%
*   **Color Coding:**
    *   The "Contracted" bar is colored light red/salmon.
    *   The "In-House" bar is colored light blue.
*   **Title:** The chart is titled "Code Access Rates by Development Model".

**5. Statistical Insights**
*   **Significant Disparity:** There is a 56.7 percentage point difference between the two groups ($90.5\% - 33.8\% = 56.7\%$).
*   **Relative Comparison:** Code access is nearly **2.7 times more prevalent** in In-House development models compared to Contracted models.
*   **Operational Implication:** The data suggests a strong operational or security tendency where internal teams are granted almost ubiquitous access to codebases, whereas external contractors are frequently restricted, likely due to security protocols or intellectual property concerns.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
