# Experiment 58: node_5_2

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_2` |
| **ID in Run** | 58 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T03:45:45.968218+00:00 |
| **Runtime** | 379.9s |
| **Parent** | `node_4_11` |
| **Children** | `node_6_0`, `node_6_15` |
| **Creation Index** | 59 |

---

## Hypothesis

> The 'Commercial Opacity' Gap: Commercial (COTS) AI deployments in the federal
government are statistically less likely to possess 'Data Documentation' and
'Code Access' controls compared to Custom-developed systems.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9758 (Definitely True) |
| **Posterior** | 0.5027 (Uncertain) |
| **Surprise** | -0.5677 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 29.0 |
| Maybe True | 1.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 2.0 |
| Maybe False | 58.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Determine if sourcing method (Commercial vs. Custom) impacts transparency governance controls.

### Steps
- 1. Load 'eo13960_scored' from 'astalabs_discovery_all_data.csv'.
- 2. Create a binary grouping variable based on '10_commercial_ai' (e.g., 'Yes' vs 'No').
- 3. Convert '34_data_docs' and '38_code_access' into binary indicators (1 for Yes/True, 0 for No/False/Missing).
- 4. Compute the proportion of systems having data documentation and code access for both Commercial and Custom groups.
- 5. Perform a Chi-Square test of independence for both controls against the sourcing group.
- 6. Calculate the Odds Ratio to quantify the likelihood of transparency based on sourcing.

### Deliverables
- Contingency tables for Data Docs and Code Access; Chi-Square statistics and p-values; Bar chart comparing compliance rates by source.

---

## Analysis

The experiment was successfully executed. After correcting the data mapping
logic to accurately identify 'Commercial' systems (n=353) versus 'Custom/Gov'
systems (n=1,357) based on the specific text descriptors in the
'10_commercial_ai' column, the analysis revealed a universal lack of
transparency controls.

While the results directionally support the hypothesis—Commercial systems had
lower compliance rates for 'Data Documentation' (0.3% vs 1.2%) and 'Code Access'
(2.0% vs 3.0%)—the differences were not statistically significant (Chi-Square
p-values of 0.23 and 0.38, respectively). The overarching finding is a
'transparency void' across the federal inventory, where >97% of all systems,
regardless of sourcing, fail to provide these basic governance artifacts.

---

## Review

The experiment was successfully executed. After correcting the data mapping
logic to accurately identify 'Commercial' systems (n=353) versus 'Custom/Gov'
systems (n=1,357) based on the specific text descriptors in the
'10_commercial_ai' column, the analysis revealed a universal lack of
transparency controls. While the results directionally support the
hypothesis—Commercial systems had lower compliance rates for 'Data
Documentation' (0.3% vs 1.2%) and 'Code Access' (2.0% vs 3.0%)—the differences
were not statistically significant (Chi-Square p-values of 0.23 and 0.38,
respectively). The overarching finding is a 'transparency void' across the
federal inventory, where >97% of all systems, regardless of sourcing, fail to
provide these basic governance artifacts.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys
import os

# --- 1. Load Data ---
filename = 'astalabs_discovery_all_data.csv'
if os.path.exists(filename):
    filepath = filename
elif os.path.exists(f'../{filename}'):
    filepath = f'../{filename}'
else:
    filepath = filename

print(f"Loading dataset from: {filepath}")
try:
    df = pd.read_csv(filepath, low_memory=False)
except Exception as e:
    print(f"Failed to load csv: {e}")
    sys.exit(1)

# Filter for relevant subset
subset = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Subset 'eo13960_scored' loaded. Rows: {len(subset)}")

# --- 2. Data Cleaning & Mapping ---

# Map Commercial Status based on debug findings
# 'None of the above.' -> Custom/Gov (0)
# Any other specific commercial description -> Commercial (1)
# NaN -> Exclude

def map_commercial_status(val):
    s = str(val).strip()
    if pd.isna(val) or s.lower() == 'nan':
        return np.nan
    if s == 'None of the above.':
        return 0
    return 1

subset['is_commercial'] = subset['10_commercial_ai'].apply(map_commercial_status)

# Drop rows where commercial status is ambiguous (NaN)
analysis_df = subset.dropna(subset=['is_commercial']).copy()

# Map Transparency Controls to Binary (1/0)
def map_binary(val):
    s = str(val).lower().strip()
    if s in ['yes', 'true', '1', '1.0']:
        return 1
    return 0

analysis_df['has_data_docs'] = analysis_df['34_data_docs'].apply(map_binary)
analysis_df['has_code_access'] = analysis_df['38_code_access'].apply(map_binary)

# Create label column
analysis_df['source_label'] = analysis_df['is_commercial'].map({1: 'Commercial (COTS)', 0: 'Custom / Gov'})

print("\n--- Analysis Groups ---")
print(analysis_df['source_label'].value_counts())

# --- 3. Statistical Analysis ---

results = {}

for control_col, label in [('has_data_docs', 'Data Documentation'), ('has_code_access', 'Code Access')]:
    print(f"\n>>> Analyzing: {label} <<<")
    
    # Contingency Table
    contingency = pd.crosstab(analysis_df['source_label'], analysis_df[control_col])
    
    # Ensure 0 (Absent) and 1 (Present) columns exist
    for c in [0, 1]:
        if c not in contingency.columns:
            contingency[c] = 0
    contingency = contingency[[0, 1]]
    
    print("Contingency Table (0=Absent, 1=Present):")
    print(contingency)
    
    # Compliance Rates
    rates = analysis_df.groupby('source_label')[control_col].mean()
    print(f"Compliance Rates:\n{rates}")
    
    # Chi-Square Test
    chi2, p, dof, ex = stats.chi2_contingency(contingency)
    print(f"Chi-Square: {chi2:.4f}, p-value: {p:.4e}")
    
    # Odds Ratio Calculation
    # OR = (Comm_Present / Comm_Absent) / (Cust_Present / Cust_Absent)
    # To avoid division by zero, we add 0.5 to cells if any cell is 0 (Haldane-Anscombe correction)
    # but for simple reporting we can check directly.
    
    try:
        comm_p = contingency.loc['Commercial (COTS)', 1]
        comm_a = contingency.loc['Commercial (COTS)', 0]
        cust_p = contingency.loc['Custom / Gov', 1]
        cust_a = contingency.loc['Custom / Gov', 0]
        
        # Use correction if needed for infinity
        if comm_a == 0 or cust_p == 0 or cust_a == 0 or comm_p == 0:
             odds_cots = (comm_p + 0.5) / (comm_a + 0.5)
             odds_cust = (cust_p + 0.5) / (cust_a + 0.5)
        else:
             odds_cots = comm_p / comm_a
             odds_cust = cust_p / cust_a
            
        or_val = odds_cots / odds_cust
        print(f"Odds Ratio (Commercial / Custom): {or_val:.4f}")
        
        if p < 0.05:
            if or_val < 1:
                print(f"SIGNIFICANT: Commercial systems are {(1/or_val):.2f}x LESS likely to have {label}.")
            else:
                print(f"SIGNIFICANT: Commercial systems are {or_val:.2f}x MORE likely to have {label}.")
        else:
            print("Result: Not statistically significant.")
            
    except Exception as e:
        print(f"Error calculating odds ratio: {e}")
        
    results[label] = rates

# --- 4. Visualization ---

fig, ax = plt.subplots(figsize=(10, 6))
groups = ['Commercial (COTS)', 'Custom / Gov']
x = np.arange(len(groups))
width = 0.35

# Get values in correct order
docs_vals = [results['Data Documentation'].get(g, 0) * 100 for g in groups]
code_vals = [results['Code Access'].get(g, 0) * 100 for g in groups]

rects1 = ax.bar(x - width/2, docs_vals, width, label='Data Documentation', color='#4e79a7')
rects2 = ax.bar(x + width/2, code_vals, width, label='Code Access', color='#f28e2b')

ax.set_ylabel('Compliance Rate (%)')
ax.set_title('Transparency Gap: Commercial vs Custom AI (Federal Inventory)')
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.ylim(0, max(max(docs_vals), max(code_vals)) * 1.3 if max(code_vals) > 0 else 10)
plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: astalabs_discovery_all_data.csv
Subset 'eo13960_scored' loaded. Rows: 1757

--- Analysis Groups ---
source_label
Custom / Gov         1357
Commercial (COTS)     353
Name: count, dtype: int64

>>> Analyzing: Data Documentation <<<
Contingency Table (0=Absent, 1=Present):
has_data_docs         0   1
source_label               
Commercial (COTS)   352   1
Custom / Gov       1341  16
Compliance Rates:
source_label
Commercial (COTS)    0.002833
Custom / Gov         0.011791
Name: has_data_docs, dtype: float64
Chi-Square: 1.4643, p-value: 2.2624e-01
Odds Ratio (Commercial / Custom): 0.2381
Result: Not statistically significant.

>>> Analyzing: Code Access <<<
Contingency Table (0=Absent, 1=Present):
has_code_access       0   1
source_label               
Commercial (COTS)   346   7
Custom / Gov       1316  41
Compliance Rates:
source_label
Commercial (COTS)    0.019830
Custom / Gov         0.030214
Name: has_code_access, dtype: float64
Chi-Square: 0.7592, p-value: 3.8358e-01
Odds Ratio (Commercial / Custom): 0.6494
Result: Not statistically significant.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Grouped Bar Chart (also known as a Clustered Bar Chart).
*   **Purpose:** The chart is designed to compare two distinct metrics of transparency ("Data Documentation" and "Code Access") across two different categories of AI procurement ("Commercial" vs. "Custom/Gov"). This allows for a direct comparison of compliance rates both within a category and between categories.

### 2. Axes
*   **X-Axis:**
    *   **Title/Labels:** The axis represents the source of the AI technology. The categories are **"Commercial (COTS)"** (likely standing for Commercial Off-The-Shelf) and **"Custom / Gov"** (Custom-built or Government-developed).
*   **Y-Axis:**
    *   **Title:** **"Compliance Rate (%)"**.
    *   **Units:** Percentage points.
    *   **Range:** The visual axis ranges from **0.0 to roughly 4.0** (with major tick marks labeled every 0.5 units up to 3.5).

### 3. Data Trends
*   **Overall Low Values:** The most immediate trend is that compliance rates are universally low. The highest value on the entire chart is only 3.0%, indicating that the vast majority of AI systems in this inventory do not meet these transparency standards.
*   **Custom vs. Commercial:** Custom/Government AI systems show consistently higher transparency compliance than Commercial systems across both metrics.
    *   *Data Documentation:* Custom (1.2%) is **4x higher** than Commercial (0.3%).
    *   *Code Access:* Custom (3.0%) is **1.5x higher** than Commercial (2.0%).
*   **Metric Comparison:** Across both categories, compliance for **"Code Access"** (Orange) is significantly higher than for **"Data Documentation"** (Blue).
*   **Tallest and Shortest Bars:**
    *   **Tallest Bar:** "Code Access" for Custom/Gov AI at **3.0%**.
    *   **Shortest Bar:** "Data Documentation" for Commercial (COTS) AI at **0.3%**.

### 4. Annotations and Legends
*   **Legend:** Located in the top right corner.
    *   **Blue Square:** Represents **"Data Documentation"**.
    *   **Orange Square:** Represents **"Code Access"**.
*   **Annotations:** Each bar is annotated with its exact percentage value above the bar:
    *   Commercial / Data Documentation: **0.3%**
    *   Commercial / Code Access: **2.0%**
    *   Custom / Data Documentation: **1.2%**
    *   Custom / Code Access: **3.0%**
*   **Chart Title:** "Transparency Gap: Commercial vs Custom AI (Federal Inventory)" – This title sets the context, highlighting that the chart intends to show the disparity in transparency between these two groups.

### 5. Statistical Insights
*   **The "Black Box" Problem in COTS:** The Commercial (COTS) sector exhibits extreme opacity regarding data documentation (0.3%). This suggests that vendors are highly protective of their proprietary data or training sets, making it nearly impossible for federal inventory managers to document the data underlying these tools.
*   **The Transparency Gap:** There is a clear "Transparency Gap" where Custom/Government-developed tools are more transparent. Because the government likely owns the IP for custom tools, they face fewer legal barriers to providing code access and data documentation compared to commercial vendors.
*   **Code is Easier than Data:** The data suggests that for federal agencies, providing access to code is easier or more common than documenting data. This may be because "Code Access" can often be satisfied by sharing a repository link, whereas "Data Documentation" requires the creation of artifacts (like datasheets or system cards) which is labor-intensive and often neglected.
*   **Critical Non-Compliance:** Regardless of the relative differences, the primary statistical insight is that compliance is close to zero. Even the "best" performing category (Custom Code Access) has a non-compliance rate of 97%. This indicates a systemic failure in enforcing or achieving transparency standards within the Federal AI Inventory.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
