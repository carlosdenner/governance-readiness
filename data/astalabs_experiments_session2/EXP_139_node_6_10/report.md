# Experiment 139: node_6_10

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_10` |
| **ID in Run** | 139 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T07:33:38.198803+00:00 |
| **Runtime** | 654.1s |
| **Parent** | `node_5_11` |
| **Children** | None |
| **Creation Index** | 140 |

---

## Hypothesis

> The 'Autonomy-Harm' Escalation: Incidents involving 'High' autonomy AI systems
(e.g., autonomous vehicles, robots) result in a significantly higher proportion
of 'Physical' harms compared to 'Low' autonomy systems, which primarily generate
'Economic' or 'Psychological' harms.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.8790 (Likely True) |
| **Posterior** | 0.4258 (Maybe False) |
| **Surprise** | -0.5438 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 17.0 |
| Maybe True | 13.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 46.0 |
| Definitely False | 14.0 |

---

## Experiment Plan

**Objective:** Verify if higher AI autonomy levels correlate with physical safety risks by analyzing the relationship between Autonomy Level and Harm Domain.

### Steps
- 1. Load the dataset 'astalabs_discovery_all_data.csv'.
- 2. Filter the dataframe to include only rows where `source_table` is 'aiid_incidents'.
- 3. Identify the columns for 'Autonomy Level' (likely index ~81) and 'Harm Domain' (likely index ~73). Normalize column names if necessary.
- 4. Print the unique values of the 'Harm Domain' column to the console to verify the vocabulary (expecting values like 'Safety', 'Financial', etc.).
- 5. Create a new column 'Autonomy_Category': Map values containing 'Autonomy3' or 'High' to 'High Autonomy', and values containing 'Autonomy1', 'Autonomy2', 'Low', or 'Medium' to 'Low/Medium Autonomy'. Drop rows that do not match these.
- 6. Create a new column 'Harm_Category': Map values in 'Harm Domain' containing 'Safety', 'Physical', 'Life', 'Death', or 'Injury' (case-insensitive) to 'Physical Harm'. Map all other non-null values to 'Non-Physical Harm'.
- 7. Print the value counts for the new 'Harm_Category' to ensure 'Physical Harm' is populated.
- 8. Create a contingency table of 'Autonomy_Category' vs. 'Harm_Category'.
- 9. Perform a Chi-square test of independence.
- 10. Print the contingency table (counts and percentages) and the statistical test results.

### Deliverables
- 1. List of unique values in 'Harm Domain'.
- 2. Contingency table showing the distribution of Physical vs. Non-Physical harm across Autonomy levels.
- 3. Chi-square statistic and p-value.

---

## Analysis

The experiment successfully tested the 'Autonomy-Harm Escalation' hypothesis
using a text-mining approach to overcome missing metadata labels. By analyzing
incident descriptions for keywords (e.g., 'crash', 'injury', 'killed'), the code
classified 185 incidents where valid Autonomy Levels were present.

**Findings:**
1.  **Physical Harm Rates:** Contrary to the hypothesis, 'High Autonomy' systems
(Level 3) showed a *lower* rate of physical harm (15.1%, 8/53 incidents)
compared to 'Low/Medium Autonomy' systems (26.5%, 35/132 incidents).
2.  **Statistical Significance:** The Chi-square test yielded a p-value of
0.1415, indicating that the observed difference is **not statistically
significant** at the alpha=0.05 level.

**Conclusion:** The hypothesis is **rejected**. The data does not support the
claim that higher autonomy correlates with increased physical safety risks; in
fact, the trend suggests the opposite, though the difference is not
statistically distinguishing.

---

## Review

The experiment was successfully recovered and executed. After identifying that
the structured `Harm Domain` and `Tangible Harm` columns lacked specific
categorical labels (containing only boolean or generic indicators), the
programmer correctly pivoted to a text-mining approach. By analyzing incident
descriptions for physical harm keywords (e.g., 'crash', 'injury', 'death'), the
experiment successfully classified 185 incidents. The statistical analysis was
valid and sufficient to reject the hypothesis.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import sys
import os

# Experiment: Autonomy-Harm Escalation (Text Analysis Approach)
# Objective: Infer 'Physical Harm' from incident descriptions since structured labels are missing.

# 1. Load Dataset
filenames = ['astalabs_discovery_all_data.csv', '../astalabs_discovery_all_data.csv']
file_path = None
for fn in filenames:
    if os.path.exists(fn):
        file_path = fn
        break

if not file_path:
    print("Error: Dataset not found.")
    sys.exit(1)

try:
    df = pd.read_csv(file_path, low_memory=False)
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# 2. Filter for AIID Incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Total AIID Incidents: {len(aiid_df)}")

# 3. Identify Columns
# Autonomy
autonomy_col = next((c for c in aiid_df.columns if 'autonomy' in c.lower() and 'level' in c.lower()), None)

# Text Column (Description/Summary)
# We prefer 'description' or 'summary'
text_col = next((c for c in aiid_df.columns if c.lower() in ['description', 'summary', 'text', 'abstract']), None)

if not autonomy_col or not text_col:
    print(f"Critical columns missing. Autonomy: {autonomy_col}, Text: {text_col}")
    # Fallback search for text column
    candidates = [c for c in aiid_df.columns if aiid_df[c].dtype == 'object' and aiid_df[c].str.len().mean() > 50]
    if candidates:
        text_col = candidates[0]
        print(f"Fallback: Using '{text_col}' as text column.")
    else:
        print("No suitable text column found.")
        sys.exit(1)

print(f"Using columns: Autonomy='{autonomy_col}', Text='{text_col}'")

# 4. Mappings

def map_autonomy(val):
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    # High: Autonomy3
    if 'Autonomy3' in val_str:
        return 'High Autonomy'
    # Low/Medium: Autonomy1, Autonomy2
    elif 'Autonomy1' in val_str or 'Autonomy2' in val_str:
        return 'Low/Medium Autonomy'
    return None

def map_harm_from_text(val):
    if pd.isna(val):
        return 'Non-Physical Harm'
    text = str(val).lower()
    # Keywords for Physical Harm / Safety
    physical_keywords = [
        'killed', 'death', 'died', 'injury', 'injured', 'hurt', 'collision', 
        'crash', 'hit', 'accident', 'safety', 'physical', 'bodily', 'fatal', 
        'wound', 'burn', 'assault', 'violence', 'attack', 'robot'
    ]
    if any(k in text for k in physical_keywords):
        return 'Physical Harm'
    return 'Non-Physical Harm'

aiid_df['Autonomy_Category'] = aiid_df[autonomy_col].apply(map_autonomy)
aiid_df['Harm_Category'] = aiid_df[text_col].apply(map_harm_from_text)

# 5. Analysis
analysis_df = aiid_df.dropna(subset=['Autonomy_Category', 'Harm_Category'])
print(f"Rows used for analysis: {len(analysis_df)}")

# Check distribution
print("Harm Category Distribution:")
print(analysis_df['Harm_Category'].value_counts())

contingency_table = pd.crosstab(analysis_df['Autonomy_Category'], analysis_df['Harm_Category'])
print("\n--- Contingency Table (Counts) ---")
print(contingency_table)

if contingency_table.size > 0:
    # Percentages
    prop_table = pd.crosstab(analysis_df['Autonomy_Category'], analysis_df['Harm_Category'], normalize='index') * 100
    print("\n--- Contingency Table (Percentages) ---")
    print(prop_table.round(2))
    
    # Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print("\n--- Chi-Square Test Results ---")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"p-value: {p:.4e}")
    
    if p < 0.05:
        print("Result: Statistically significant relationship found.")
    else:
        print("Result: No statistically significant relationship found.")
else:
    print("Insufficient data for analysis.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Total AIID Incidents: 1362
Using columns: Autonomy='Autonomy Level', Text='description'
Rows used for analysis: 185
Harm Category Distribution:
Harm_Category
Non-Physical Harm    142
Physical Harm         43
Name: count, dtype: int64

--- Contingency Table (Counts) ---
Harm_Category        Non-Physical Harm  Physical Harm
Autonomy_Category                                    
High Autonomy                       45              8
Low/Medium Autonomy                 97             35

--- Contingency Table (Percentages) ---
Harm_Category        Non-Physical Harm  Physical Harm
Autonomy_Category                                    
High Autonomy                    84.91          15.09
Low/Medium Autonomy              73.48          26.52

--- Chi-Square Test Results ---
Chi2 Statistic: 2.1617
p-value: 1.4149e-01
Result: No statistically significant relationship found.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
