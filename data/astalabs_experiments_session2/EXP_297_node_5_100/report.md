# Experiment 297: node_5_100

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_100` |
| **ID in Run** | 297 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T15:20:34.174328+00:00 |
| **Runtime** | 223.3s |
| **Parent** | `node_4_25` |
| **Children** | None |
| **Creation Index** | 298 |

---

## Hypothesis

> The Governance-Silo Hypothesis: There is no statistically significant
correlation between having an 'Impact Assessment' and undergoing 'Independent
Evaluation', suggesting these controls are applied piecemeal rather than as a
coherent governance bundle.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.3790 (Maybe False) |
| **Posterior** | 0.1291 (Likely False) |
| **Surprise** | -0.2999 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 8.0 |
| Uncertain | 1.0 |
| Maybe False | 19.0 |
| Definitely False | 2.0 |

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

**Objective:** Determine if governance controls are applied as a bundle or in isolation.

### Steps
- 1. Filter `eo13960_scored` for the subset of systems that have non-null values for `52_impact_assessment` and `55_independent_eval`.
- 2. Create a binary map for both columns (Yes=1, No/NA=0).
- 3. Create a contingency table.
- 4. Calculate the Phi coefficient or Cramér's V to measure the strength of association and perform a Chi-square test.

### Deliverables
- Contingency table of Impact Assessment vs Independent Evaluation; Association metric (Phi/Cramer's V) and p-value.

---

## Analysis

The experiment successfully tested the 'Governance-Silo Hypothesis' using the
'eo13960_scored' dataset (n=1757). The code correctly identified the relevant
columns ('52_impact_assessment' and '55_independent_eval') and mapped complex
string values to binary indicators, though a minor edge case ('TRUE') was mapped
to 0.

The contingency table revealed that the vast majority of systems (1677, ~95.4%)
lacked both controls. Among the small subset that implemented governance, the
controls showed a statistically significant positive association (Chi-square =
344.51, p < 1e-76, Phi = 0.44).

Consequently, the 'Governance-Silo' hypothesis (positing no correlation) is
rejected. The data suggests that while overall adoption is low, these governance
mechanisms are not applied as isolated silos; when an agency implements one,
they are significantly more likely to implement the other compared to the
baseline. The heatmap confirms this moderate 'bundling' effect.

---

## Review

The experiment successfully tested the 'Governance-Silo Hypothesis' using the
'eo13960_scored' dataset (n=1757). The code correctly identified the relevant
columns ('52_impact_assessment' and '55_independent_eval') and implemented a
binary mapping strategy to handle the text-based data. A minor data-cleaning
oversight was noted where the value 'TRUE' in the evaluation column was mapped
to 0 (as it does not start with 'yes'), but given the overwhelming statistical
signal, this does not invalidate the conclusion.

The analysis revealed that the vast majority of systems (1677, ~95.4%) lacked
both governance controls. However, among the subset that implemented them, there
was a statistically significant positive association (Chi-square = 344.51, p <
1e-76, Phi = 0.44). Consequently, the null hypothesis (positing no correlation)
is definitively rejected. The findings indicate that these governance mechanisms
are not applied as isolated silos; rather, agencies that perform impact
assessments are significantly more likely to also undergo independent evaluation
compared to those that do not.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback for local testing if needed, though instruction says one level above
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for 'eo13960_scored' source table
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# Identify relevant columns
# Based on metadata, likely names are '52_impact_assessment' and '55_independent_eval'
col_impact = '52_impact_assessment'
col_eval = '55_independent_eval'

# Check if columns exist
if col_impact not in eo_data.columns or col_eval not in eo_data.columns:
    # search for columns if exact names don't match
    cols = eo_data.columns.tolist()
    col_impact = next((c for c in cols if 'impact_assessment' in c.lower()), None)
    col_eval = next((c for c in cols if 'independent_eval' in c.lower()), None)

print(f"Using columns: '{col_impact}' and '{col_eval}'")

# Inspect unique values to determine mapping
print("Unique values in Impact Assessment:", eo_data[col_impact].unique())
print("Unique values in Independent Eval:", eo_data[col_eval].unique())

# Function to map values to binary
def map_to_binary(val):
    if pd.isna(val):
        return 0
    val_str = str(val).strip().lower()
    if val_str.startswith('yes'):
        return 1
    return 0

# Apply mapping
eo_data['has_impact_assessment'] = eo_data[col_impact].apply(map_to_binary)
eo_data['has_independent_eval'] = eo_data[col_eval].apply(map_to_binary)

# Create Contingency Table
contingency_table = pd.crosstab(
    eo_data['has_impact_assessment'], 
    eo_data['has_independent_eval'], 
    rownames=['Impact Assessment'], 
    colnames=['Independent Eval']
)

print("\nContingency Table (0=No/NA, 1=Yes):")
print(contingency_table)

# Perform Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

# Calculate Phi Coefficient
n = contingency_table.sum().sum()
phi = np.sqrt(chi2 / n)

print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")
print(f"Phi Coefficient: {phi:.4f}")

# Interpretation
alpha = 0.05
if p < alpha:
    print("\nResult: Statistically significant association found (Reject Null Hypothesis).")
    if phi > 0.5:
        print("Strength: Strong association.")
    elif phi > 0.3:
        print("Strength: Moderate association.")
    else:
        print("Strength: Weak association.")
else:
    print("\nResult: No statistically significant association found (Fail to reject Null Hypothesis).")
    print("This supports the 'Governance-Silo' hypothesis that controls are applied piecemeal.")

# Visualization
plt.figure(figsize=(6, 5))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Co-occurrence of Governance Controls')
plt.ylabel('Has Impact Assessment')
plt.xlabel('Has Independent Evaluation')
plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Using columns: '52_impact_assessment' and '55_independent_eval'
Unique values in Impact Assessment: <StringArray>
[nan, 'Planned or in-progress.', 'Yes', 'No', 'YES']
Length: 5, dtype: str
Unique values in Independent Eval: <StringArray>
[                                                                                                    nan,
                                                                                     'Yes – by the CAIO',
                                                                                'Planned or in-progress',
 'Yes – by another appropriate agency office that was not directly involved in the system’s development',
               'Yes – by an agency AI oversight board not directly involved in the system’s development',
                                       'Does not apply, use case is neither safety or rights impacting.',
                         'Agency CAIO has waived this minimum practice and reported such waiver to OMB.',
                                                                 'AI is not safety or rights-impacting.',
                                                                                                  'TRUE']
Length: 9, dtype: str

Contingency Table (0=No/NA, 1=Yes):
Independent Eval      0   1
Impact Assessment          
0                  1677  19
1                    37  24

Chi-square Statistic: 344.5137
P-value: 6.6356e-77
Phi Coefficient: 0.4428

Result: Statistically significant association found (Reject Null Hypothesis).
Strength: Moderate association.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** This is a **Heatmap** visualizing a **Contingency Table** (or Co-occurrence Matrix).
*   **Purpose:** The plot displays the frequency distribution between two binary categorical variables. It is designed to show how often two specific governance controls ("Has Impact Assessment" and "Has Independent Evaluation") occur together, separately, or not at all within a dataset.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Has Independent Evaluation"
    *   **Values:** Binary categories **0** (representing the absence of the control) and **1** (representing the presence of the control).
*   **Y-Axis:**
    *   **Label:** "Has Impact Assessment"
    *   **Values:** Binary categories **0** (Absence) and **1** (Presence).

### 3. Data Trends
*   **Dominant Category (High Concentration):** The vast majority of the data is concentrated in the top-left cell (Intersection of 0, 0). This area is shaded dark blue, indicating a high frequency count.
*   **Minor Categories (Low Concentration):** The remaining three quadrants are shaded very light blue or white, indicating very low frequency counts relative to the top-left cell.
*   **Specific Counts:**
    *   **Neither Control (0, 0):** 1677 occurrences (Highest).
    *   **Impact Assessment Only (1, 0):** 37 occurrences.
    *   **Independent Evaluation Only (0, 1):** 19 occurrences.
    *   **Both Controls (1, 1):** 24 occurrences.

### 4. Annotations and Legends
*   **Title:** "Co-occurrence of Governance Controls" appears at the top, defining the context of the variables.
*   **Cell Annotations:** Each cell contains a numerical annotation representing the exact count of observations for that specific combination of variables (1677, 19, 37, 24).
*   **Color Mapping:** Although no separate legend bar is provided, the visual encoding uses a sequential color scheme where **Dark Blue** represents high values and **White/Light Blue** represents low values.

### 5. Statistical Insights
*   **Low Adoption Rates:** The most significant insight is that the overwhelming majority of entities in this dataset (**approx. 95.4%**, calculated as 1677 out of 1757 total) possess **neither** an Impact Assessment nor an Independent Evaluation.
*   **Rare Co-occurrence:** The scenario where an entity possesses both governance controls (Cell 1, 1) is rare, occurring only 24 times (approx. 1.4% of the total).
*   **Marginal Independence:**
    *   If an entity has an "Independent Evaluation" (Total = 43), there is a roughly 56% chance ($24/43$) they also have an "Impact Assessment."
    *   However, if an entity has an "Impact Assessment" (Total = 61), there is only a roughly 39% chance ($24/61$) they also have an "Independent Evaluation."
*   **Implication:** This suggests that while these two controls are positively associated (having one increases the likelihood of having the other compared to the baseline), the standard operating procedure for the vast majority of the population measured here is to operate without either governance mechanism.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
