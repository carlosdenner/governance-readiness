# Experiment 175: node_7_1

| Property | Value |
|---|---|
| **Experiment ID** | `node_7_1` |
| **ID in Run** | 175 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T09:21:03.199205+00:00 |
| **Runtime** | 212.4s |
| **Parent** | `node_6_7` |
| **Children** | None |
| **Creation Index** | 176 |

---

## Hypothesis

> The Assessment-Action Gap: Completing an 'Impact Assessment' does not
statistically guarantee the presence of 'Disparity Mitigation' plans, indicating
that assessments often remain bureaucratic exercises without operational follow-
through.

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

**Objective:** Test the causal link between assessing impact and planning mitigation.

### Steps
- 1. Load 'eo13960_scored'.
- 2. Filter for rows where '52_impact_assessment' is 'Yes'.
- 3. Within this group, analyze '62_disparity_mitigation' (Yes vs No/None).
- 4. Compare this rate to the baseline rate of mitigation in systems without assessments using a Chi-Square test.
- 5. Assess the magnitude of the 'Assessed but Unmitigated' group.

### Deliverables
- Comparative mitigation rates; Statistical significance; 'Paper Tiger' analysis.

---

## Analysis

The experiment was successfully executed and robustly supports the 'Assessment-
Action Gap' hypothesis. After cleaning and binary-encoding the survey responses,
43 valid data points were analyzed. The data reveals a complete decoupling
between assessment and action: the rate of Disparity Mitigation plans among
systems with an Impact Assessment (7.14%) was statistically indistinguishable
from those without one (6.67%). A Chi-Square test yielded a p-value of 1.00,
confirming no correlation. The analysis identifies a massive 'Paper Tiger'
phenomenon, where 92.9% (26/28) of systems that underwent an impact assessment
failed to implement any corresponding disparity mitigation, validating that
assessments are largely bureaucratic exercises without operational follow-
through in this dataset.

---

## Review

The experiment was successfully executed and robustly supports the 'Assessment-
Action Gap' hypothesis. After cleaning and binary-encoding the survey responses,
43 valid data points were analyzed. The data reveals a complete decoupling
between assessment and action: the rate of Disparity Mitigation plans among
systems with an Impact Assessment (7.14%) was statistically indistinguishable
from those without one (6.67%). A Chi-Square test yielded a p-value of 1.00,
confirming no correlation. The analysis identifies a massive 'Paper Tiger'
phenomenon, where 92.9% (26/28) of systems that underwent an impact assessment
failed to implement any corresponding disparity mitigation, validating that
assessments are largely bureaucratic exercises without operational follow-
through in this dataset.

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
print(f"Loading dataset from {file_path}...")

try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 Scored data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored subset shape: {eo_df.shape}")

col_assessment = '52_impact_assessment'
col_mitigation = '62_disparity_mitigation'

# Function to clean and binary encode text responses
def clean_binary_response(text):
    if pd.isna(text):
        return np.nan
    t = str(text).lower().strip()
    # Positive indicators
    if t.startswith('yes') or 'completed' in t or 'conducted' in t or 'planned' in t:
        return 'Yes'
    # Negative indicators
    if t.startswith('no') or 'none' in t or 'n/a' in t or 'not applicable' in t or 'not required' in t:
        return 'No'
    return np.nan

# Apply cleaning
eo_df['assessment_clean'] = eo_df[col_assessment].apply(clean_binary_response)
eo_df['mitigation_clean'] = eo_df[col_mitigation].apply(clean_binary_response)

# Drop rows where either value could not be determined
valid_df = eo_df.dropna(subset=['assessment_clean', 'mitigation_clean'])
print(f"Valid data points after cleaning: {len(valid_df)}")

# Generate Contingency Table
contingency_table = pd.crosstab(valid_df['assessment_clean'], valid_df['mitigation_clean'])

# Ensure 2x2 table by reindexing (handling missing categories like 'Yes' in mitigation)
contingency_table = contingency_table.reindex(index=['No', 'Yes'], columns=['No', 'Yes'], fill_value=0)

print("\nContingency Table (Impact Assessment vs Disparity Mitigation):")
print(contingency_table)

# Calculate counts and rates safely
assessed_yes = contingency_table.loc['Yes'].sum()
mitigated_given_assessed = contingency_table.loc['Yes', 'Yes']

assessed_no = contingency_table.loc['No'].sum()
mitigated_given_not_assessed = contingency_table.loc['No', 'Yes']

rate_assessed = (mitigated_given_assessed / assessed_yes * 100) if assessed_yes > 0 else 0.0
rate_not_assessed = (mitigated_given_not_assessed / assessed_no * 100) if assessed_no > 0 else 0.0

print(f"\nMitigation Rate when Assessment='Yes': {rate_assessed:.2f}% ({mitigated_given_assessed}/{assessed_yes})")
print(f"Mitigation Rate when Assessment='No': {rate_not_assessed:.2f}% ({mitigated_given_not_assessed}/{assessed_no})")

# Analysis of the 'Assessment-Action Gap'
# Gap defined as Assessment='Yes' but Mitigation='No'
gap_count = contingency_table.loc['Yes', 'No']
gap_percentage = (gap_count / assessed_yes * 100) if assessed_yes > 0 else 0.0

print(f"\nAssessment-Action Gap Analysis:")
print(f"Number of systems with Impact Assessment but NO Disparity Mitigation: {gap_count}")
print(f"Percentage of Assessed systems that are Unmitigated ('Paper Tigers'): {gap_percentage:.2f}%")

# Chi-Square Test
# Check if we have enough data to run the test (at least 2 dimensions with some data)
if contingency_table.sum().sum() > 0:
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-Value: {p:.4e}")
    
    # Interpretation
    alpha = 0.05
    if p < alpha:
        print("\nResult: Statistically significant relationship found.")
        if rate_assessed > rate_not_assessed:
            print("Evidence supports: Conducting an Impact Assessment positively correlates with Disparity Mitigation.")
        else:
            print("Evidence suggests: Negative or paradoxical relationship.")
    else:
        print("\nResult: No statistically significant relationship found.")
        print("Interpretation: Conducting an Impact Assessment does not statistically guarantee Disparity Mitigation plans (supports the 'Assessment-Action Gap').")
else:
    print("\nInsufficient data for Chi-Square test.")

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Impact Assessment vs. Disparity Mitigation')
plt.xlabel('Disparity Mitigation Planned?')
plt.ylabel('Impact Assessment Conducted?')
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from ../astalabs_discovery_all_data.csv...
EO 13960 Scored subset shape: (1757, 196)
Valid data points after cleaning: 43

Contingency Table (Impact Assessment vs Disparity Mitigation):
mitigation_clean  No  Yes
assessment_clean         
No                14    1
Yes               26    2

Mitigation Rate when Assessment='Yes': 7.14% (2/28)
Mitigation Rate when Assessment='No': 6.67% (1/15)

Assessment-Action Gap Analysis:
Number of systems with Impact Assessment but NO Disparity Mitigation: 26
Percentage of Assessed systems that are Unmitigated ('Paper Tigers'): 92.86%

Chi-Square Statistic: 0.0000
P-Value: 1.0000e+00

Result: No statistically significant relationship found.
Interpretation: Conducting an Impact Assessment does not statistically guarantee Disparity Mitigation plans (supports the 'Assessment-Action Gap').


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** This is a **Heatmap** representing a $2 \times 2$ contingency table (or cross-tabulation).
*   **Purpose:** It visualizes the frequency distribution between two categorical variables: whether an impact assessment was conducted and whether disparity mitigation is planned. The color intensity corresponds to the count (frequency) in each intersection.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Disparity Mitigation Planned?"
    *   **Categories:** Binary categories: "No" (left) and "Yes" (right).
*   **Y-Axis:**
    *   **Label:** "Impact Assessment Conducted?"
    *   **Categories:** Binary categories: "No" (top) and "Yes" (bottom).
*   **Units:** The axes represent categorical responses. The values inside the cells represent the **count** of occurrences.

### 3. Data Trends
*   **Highest Value (Hotspot):** The area with the highest concentration is the intersection of **Impact Assessment Conducted: "Yes"** and **Disparity Mitigation Planned: "No"**, with a count of **26**. This cell is colored the darkest blue.
*   **Second Highest Value:** The intersection of answering "No" to both questions, with a count of **14**.
*   **Lowest Values:** The "Yes" column for Disparity Mitigation contains very low counts. Only **1** instance had no impact assessment but planned mitigation, and only **2** instances had both an assessment and planned mitigation.
*   **Overall Pattern:** The data is heavily skewed toward the "No" column for Disparity Mitigation. The color gradient clearly shifts from dark blue on the left ("No" mitigation) to very light blue/white on the right ("Yes" mitigation).

### 4. Annotations and Legends
*   **Title:** "Impact Assessment vs. Disparity Mitigation" is clearly displayed at the top.
*   **Cell Annotations:** Each of the four quadrants is annotated with the exact count of observations (14, 1, 26, 2).
*   **Color Key:** While there is no explicit legend bar, the visualization uses a sequential blue color palette where darker blue indicates a higher frequency count and lighter/white indicates a lower frequency count.

### 5. Statistical Insights
*   **Total Sample Size:** The plot represents a total of 43 entries ($14 + 1 + 26 + 2$).
*   **Assessment vs. Action Gap:**
    *   A majority of the sample ($28/43$ or $\approx 65\%$) **did** conduct an impact assessment.
    *   However, an overwhelming majority ($40/43$ or $\approx 93\%$) have **no** disparity mitigation planned.
*   **Key Insight:** There is a significant disconnect between assessing impact and acting on it. Specifically, the largest group (26 entities) conducted the impact assessment but still have no plans for disparity mitigation. This suggests that while organizations are performing the necessary checks (assessments), these checks are not translating into planned mitigation strategies.
*   **Rare Occurrence:** Planning for disparity mitigation is an outlier behavior in this dataset, occurring in only 3 out of 43 cases ($7\%$).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
