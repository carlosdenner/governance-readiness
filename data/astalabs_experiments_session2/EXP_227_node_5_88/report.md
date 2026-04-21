# Experiment 227: node_5_88

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_88` |
| **ID in Run** | 227 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T11:40:26.032817+00:00 |
| **Runtime** | 228.0s |
| **Parent** | `node_4_42` |
| **Children** | None |
| **Creation Index** | 228 |

---

## Hypothesis

> Commercial Obscurity: Federal AI systems procured from commercial vendors are
significantly less likely to provide 'Open' code access compared to systems
developed internally by agencies or via custom code, creating a transparency
barrier.

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

**Objective:** Evaluate the impact of commercial procurement on AI transparency (code availability).

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'eo13960_scored'.
- 2. Define two groups based on '10_commercial_ai': 'Commercial' (Yes) vs 'Custom/Internal' (No).
- 3. Binarize '38_code_access' into 'Transparent' (Open Source/Available) vs 'Opaque' (No/Closed/Restricted).
- 4. Perform a Chi-square test of independence.
- 5. Compute the percentage of transparent projects in each group.

### Deliverables
- Frequency table, Chi-square results, and comparative percentages.

---

## Analysis

The experiment successfully tested the 'Commercial Obscurity' hypothesis using
the corrected data columns identified in the previous debug step.

1. **Data Processing**: The analysis utilized `22_dev_method` to distinguish
between 'Commercial' (contracting resources, n=394) and 'Custom/Internal' (in-
house, n=400) systems, and `38_code_access` to measure transparency. The final
analysis set included 794 valid cases.

2. **Findings**:
   - **Descriptive Statistics**: Systems developed in-house ('Custom/Internal')
had a transparency rate of **8.00%** (32/400), whereas commercially procured
systems had a significantly lower transparency rate of **2.79%** (11/394).
   - **Statistical Significance**: The Chi-Square test yielded a statistic of
9.5185 and a **p-value of 0.0020**, which is well below the significance
threshold of 0.05.

3. **Conclusion**: The results provide strong statistical evidence to
**support** the hypothesis. Federal AI systems procured from commercial vendors
are significantly less likely to provide open access to their source code
compared to systems developed internally by agencies.

---

## Review

The experiment was successfully re-executed using the correct independent
variable (`22_dev_method`) identified in the debugging phase. The analysis
confirms the hypothesis: federal AI systems procured from commercial vendors are
significantly less likely to provide open code access compared to systems
developed in-house.

**Key Findings:**
- **Sample Size:** 794 valid cases from the EO 13960 dataset were analyzed.
- **Descriptive Statistics:** Systems developed in-house (`Custom/Internal`)
showed a transparency rate of **8.00%** (32/400), whereas commercially procured
systems (`Commercial`) had a significantly lower transparency rate of **2.79%**
(11/394).
- **Statistical Significance:** A Chi-Square test of independence yielded a
statistic of 9.5185 and a **p-value of 0.0020** (p < 0.05), indicating a
statistically significant relationship.
- **Conclusion:** The data supports the hypothesis that commercial procurement
creates a 'transparency barrier,' reducing the likelihood of public code
availability by approximately 65% relative to internal development.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import sys

print("Starting Commercial Obscurity hypothesis test (Attempt 2)...\n")

# 1. Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

# Filter for 'eo13960_scored'
df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"Loaded {len(df)} rows from EO 13960 dataset.")

# 2. Define Variables and Mappings
col_dev_method = '22_dev_method'
col_code_access = '38_code_access'

# Check columns
if col_dev_method not in df.columns or col_code_access not in df.columns:
    print(f"Error: Columns '{col_dev_method}' or '{col_code_access}' missing.")
    sys.exit(1)

# Mapping for Procurement (Independent Variable)
def map_procurement(val):
    s = str(val).strip()
    if pd.isna(val) or s.lower() == 'nan':
        return None
    if s == 'Developed with contracting resources.':
        return 'Commercial'
    elif s == 'Developed in-house.':
        return 'Custom/Internal'
    # Exclude 'Developed with both...' and 'Data not reported...' to ensure clean groups
    return None

# Mapping for Code Access (Dependent Variable)
# 'Transparent' = Publicly Available
# 'Opaque' = No access, or Internal access only (not public)
def map_transparency(val):
    s = str(val).strip()
    if pd.isna(val) or s.lower() == 'nan':
        return None # Missing data, exclude
    
    # Strict check for Public availability
    if 'publicly available' in s.lower():
        return 'Transparent'
    
    # All other non-missing values are Opaque (Restricted/Closed)
    # Includes: 'No – agency does not have access...', 'Yes – agency has access... but it is not public', 'Yes', 'YES'
    return 'Opaque'

# Apply mappings
df['Procurement_Type'] = df[col_dev_method].apply(map_procurement)
df['Code_Transparency'] = df[col_code_access].apply(map_transparency)

# Filter for valid rows (where both fields are not None)
df_analysis = df.dropna(subset=['Procurement_Type', 'Code_Transparency']).copy()

print(f"\nAnalysis set size after filtering: {len(df_analysis)}")

# 3. Generate Statistics

# Contingency Table
contingency_table = pd.crosstab(df_analysis['Procurement_Type'], df_analysis['Code_Transparency'])

print("\n--- Contingency Table ---")
print(contingency_table)

# Percentages
row_sums = contingency_table.sum(axis=1)
percentages = contingency_table.div(row_sums, axis=0) * 100

print("\n--- Percentages (Row-wise) ---")
print(percentages.round(2))

# 4. Statistical Test (Chi-Square)
if contingency_table.size > 0 and contingency_table.sum().sum() > 0:
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print("\n--- Chi-Square Test Results ---")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"p-value: {p:.4e}")
    print(f"Degrees of freedom: {dof}")
    
    alpha = 0.05
    if p < alpha:
        print("\nResult: Statistically SIGNIFICANT difference found.")
        
        # Check directionality
        comm_trans = percentages.loc['Commercial', 'Transparent'] if 'Transparent' in percentages.columns else 0
        cust_trans = percentages.loc['Custom/Internal', 'Transparent'] if 'Transparent' in percentages.columns else 0
        
        print(f"Commercial Transparency: {comm_trans:.2f}%")
        print(f"Custom/Internal Transparency: {cust_trans:.2f}%")
        
        if comm_trans < cust_trans:
            print("The data SUPPORTS the hypothesis: Commercial systems are less transparent.")
        else:
            print("The data CONTRADICTS the hypothesis direction.")
    else:
        print("\nResult: No statistically significant difference found (Fail to reject Null).")
else:
    print("\nInsufficient data for Chi-Square test.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting Commercial Obscurity hypothesis test (Attempt 2)...

Loaded 1757 rows from EO 13960 dataset.

Analysis set size after filtering: 794

--- Contingency Table ---
Code_Transparency  Opaque  Transparent
Procurement_Type                      
Commercial            383           11
Custom/Internal       368           32

--- Percentages (Row-wise) ---
Code_Transparency  Opaque  Transparent
Procurement_Type                      
Commercial          97.21         2.79
Custom/Internal     92.00         8.00

--- Chi-Square Test Results ---
Chi-square statistic: 9.5185
p-value: 2.0341e-03
Degrees of freedom: 1

Result: Statistically SIGNIFICANT difference found.
Commercial Transparency: 2.79%
Custom/Internal Transparency: 8.00%
The data SUPPORTS the hypothesis: Commercial systems are less transparent.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
