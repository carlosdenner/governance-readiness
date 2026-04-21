# Experiment 174: node_5_68

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_68` |
| **ID in Run** | 174 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T09:12:48.990501+00:00 |
| **Runtime** | 440.2s |
| **Parent** | `node_4_3` |
| **Children** | None |
| **Creation Index** | 175 |

---

## Hypothesis

> Commercial Opacity: Commercially procured AI systems (COTS) in federal agencies
exhibit significantly lower transparency compliance (measured by availability of
data documentation) compared to internally developed 'Custom' AI systems.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7661 (Likely True) |
| **Posterior** | 0.9203 (Definitely True) |
| **Surprise** | +0.1850 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
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
| Definitely True | 60.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Determine if reliance on commercial AI vendors correlates with reduced documentation transparency in government deployments.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for the 'eo13960_scored' source_table.
- 2. Clean the '10_commercial_ai' column: standardize values to 'Commercial' (Yes) and 'Custom' (No/Internal).
- 3. Clean the '34_data_docs' column: map values to Binary (1 for Yes, 0 for No/Unsure).
- 4. Create a contingency table of Commercial Status vs. Data Documentation.
- 5. Perform a Chi-Square test of independence.
- 6. Calculate the percentage of documented systems in each group.

### Deliverables
- Contingency table, Chi-square statistic, p-value, and bar chart comparing documentation rates.

---

## Analysis

The experiment successfully tested the 'Commercial Opacity' hypothesis using the
EO 13960 dataset.

1. **Data Classification**:
   - **Procurement Type**: Systems describing specific commercial tasks (e.g.,
'Scheduling meetings') were classified as 'Commercial' (n=353), while those
marked 'None of the above' (indicating mission-specific/internal development)
were classified as 'Custom' (n=1357).
   - **Transparency**: Documentation status was binarized based on the presence
of affirmative keywords ('complete', 'partial', 'available') versus negative
ones ('missing', 'not available').

2. **Statistical Results**:
   - The Chi-Square test yielded a statistic of **160.61** and a p-value of
**8.31e-37**, indicating an extremely significant relationship between
procurement type and transparency.
   - **Commercial systems** demonstrated a documentation rate of only
**17.28%**.
   - **Custom systems** demonstrated a documentation rate of **55.27%**.

3. **Conclusion**:
   - The hypothesis is **strongly supported**. Custom/internal AI systems in
federal agencies are over **3 times more likely** to have available data
documentation compared to commercially procured COTS solutions. This confirms a
significant transparency gap where reliance on commercial vendors correlates
with reduced governance visibility.

---

## Review

The experiment successfully tested the 'Commercial Opacity' hypothesis using the
EO 13960 dataset. The implementation correctly adapted to the specific text
values found in the survey data (classifying specific commercial use-case
descriptions as 'Commercial' and 'None of the above' as 'Custom').

**Hypothesis**: Supported.

**Findings**:
- **Statistical Significance**: The Chi-Square test revealed an extremely
significant relationship ($p < 0.001$, $\chi^2 = 160.6$) between procurement
type and documentation transparency.
- **Transparency Gap**:
  - **Custom/Internal Systems** (n=1,357) demonstrated a documentation
compliance rate of **55.3%**.
  - **Commercial (COTS) Systems** (n=353) demonstrated a much lower compliance
rate of **17.3%**.

**Conclusion**: The results confirm that commercially procured AI tools in the
federal government are significantly less likely to have available data
documentation compared to custom-developed systems, highlighting a major
governance gap in the supply chain.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import os

# Load dataset
file_path = '../astalabs_discovery_all_data.csv'
if not os.path.exists(file_path):
    file_path = 'astalabs_discovery_all_data.csv'

try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    print(f"Error: Could not find dataset at {file_path}")
    exit(1)

# Filter for EO 13960 Scored subset
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

# 1. Map Procurement Type (Commercial vs Custom)
# Logic: 'None of the above.' implies the use case is not a standard commercial tool -> Custom/Mission Specific.
# Specific descriptions (e.g., 'Scheduling meetings') -> Commercial.
def map_procurement(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == 'None of the above.':
        return 'Custom'
    else:
        return 'Commercial'

df_eo['Procurement_Type'] = df_eo['10_commercial_ai'].apply(map_procurement)

# Drop rows where procurement type is unknown
df_analysis = df_eo.dropna(subset=['Procurement_Type']).copy()

# 2. Map Documentation Transparency (Has_Documentation)
# Logic: Identify positive confirmation of documentation while excluding explicit negatives.
def map_documentation(x):
    if pd.isna(x):
        return 0
    s = str(x).lower().strip()
    
    # Explicit negatives
    if 'missing' in s or 'not available' in s or 'not reported' in s or s == 'no':
        return 0
    
    # Positives
    # "partial" is considered compliant/present for the purpose of "Availability of documentation"
    if ('complete' in s or 
        'partial' in s or 
        'widely' in s or 
        'yes' in s or 
        'documented' in s or 
        'is available' in s):
        return 1
        
    return 0

df_analysis['Has_Documentation'] = df_analysis['34_data_docs'].apply(map_documentation)

# 3. Generate Summary Stats
print(f"Analyzed rows: {len(df_analysis)}")
print("Distribution of Procurement Type:")
print(df_analysis['Procurement_Type'].value_counts())

print("\nDistribution of Documentation Status:")
print(df_analysis['Has_Documentation'].value_counts())

# 4. Contingency Table
contingency = pd.crosstab(df_analysis['Procurement_Type'], df_analysis['Has_Documentation'])
# Ensure columns are 0 and 1
if 0 not in contingency.columns: contingency[0] = 0
if 1 not in contingency.columns: contingency[1] = 0
contingency = contingency[[0, 1]]
contingency.columns = ['No Doc', 'Has Doc']

print("\nContingency Table:")
print(contingency)

# 5. Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# 6. Documentation Rates
rates = df_analysis.groupby('Procurement_Type')['Has_Documentation'].mean()
comm_rate = rates.get('Commercial', 0) * 100
cust_rate = rates.get('Custom', 0) * 100

print(f"\nDocumentation Rates:")
print(f"Commercial: {comm_rate:.2f}%")
print(f"Custom: {cust_rate:.2f}%")

# 7. Visualization
plt.figure(figsize=(8, 6))
bars = plt.bar(['Commercial', 'Custom'], [comm_rate, cust_rate], color=['orange', 'skyblue'], edgecolor='black')
plt.title('Transparency Gap: Commercial vs Custom AI Documentation')
plt.ylabel('Percentage of Systems with Data Documentation (%)')
plt.ylim(0, 100)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{height:.1f}%',
             ha='center', va='bottom')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Analyzed rows: 1710
Distribution of Procurement Type:
Procurement_Type
Custom        1357
Commercial     353
Name: count, dtype: int64

Distribution of Documentation Status:
Has_Documentation
0    899
1    811
Name: count, dtype: int64

Contingency Table:
                  No Doc  Has Doc
Procurement_Type                 
Commercial           292       61
Custom               607      750

Chi-Square Statistic: 160.6143
P-value: 8.3067e-37

Documentation Rates:
Commercial: 17.28%
Custom: 55.27%


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Chart.
*   **Purpose:** The plot is designed to compare the prevalence of data documentation between two distinct categories of AI systems: "Commercial" and "Custom."

### 2. Axes
*   **Y-Axis:**
    *   **Label:** "Percentage of Systems with Data Documentation (%)"
    *   **Range:** 0 to 100.
    *   **Scale:** Linear, with major tick marks at intervals of 20 (0, 20, 40, 60, 80, 100).
*   **X-Axis:**
    *   **Label:** Categorical labels representing the type of AI system: "Commercial" and "Custom."
    *   **Range:** N/A (Categorical data).

### 3. Data Trends
*   **Commercial Bar (Orange):** This represents the shortest bar in the plot. It indicates a low level of documentation compliance.
*   **Custom Bar (Light Blue):** This is the tallest bar, visually appearing more than three times the height of the commercial bar.
*   **Pattern:** There is a distinct and significant upward trend in documentation transparency when moving from Commercial systems to Custom systems.

### 4. Annotations and Legends
*   **Chart Title:** "Transparency Gap: Commercial vs Custom AI Documentation" – This annotation frames the data as a comparison of transparency levels.
*   **Value Labels:** Exact percentage values are annotated directly above each bar for precision:
    *   Commercial: **17.3%**
    *   Custom: **55.3%**
*   **Gridlines:** Horizontal dashed gridlines are included to assist in visually estimating the bar heights relative to the Y-axis scale.

### 5. Statistical Insights
*   **Significant Transparency Gap:** There is a 38 percentage point difference between the two categories ($55.3\% - 17.3\% = 38.0\%$).
*   **Relative Performance:** Custom AI systems are approximately **3.2 times more likely** to have data documentation compared to Commercial AI systems.
*   **Majority vs. Minority:** While a majority of Custom systems (over half) include data documentation, the vast majority of Commercial systems (82.7%) lack such documentation, highlighting a critical deficiency in transparency within the commercial sector.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
