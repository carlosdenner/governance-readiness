# Experiment 214: node_6_38

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_38` |
| **ID in Run** | 214 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T10:59:35.497049+00:00 |
| **Runtime** | 343.7s |
| **Parent** | `node_5_70` |
| **Children** | None |
| **Creation Index** | 215 |

---

## Hypothesis

> The 'Biometric Rights' Correlation: AI incidents involving Facial Recognition or
Biometrics are significantly more likely to fall under the 'Civil Rights' harm
domain compared to other AI technologies.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9839 (Definitely True) |
| **Posterior** | 0.7381 (Likely True) |
| **Surprise** | -0.2949 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | Yes |

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
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 32.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Investigate the relationship between Biometric technology and Civil Rights harms.

### Steps
- 1. Load 'aiid_incidents'.
- 2. Create a 'Tech Category' variable by searching '84_Known AI Technology' for terms like 'Face', 'Biometric', 'Recognition'.
- 3. Create a 'Harm Category' variable focusing on 'Civil Rights' (from '73_Harm Domain').
- 4. Perform a Chi-square test comparing Biometric vs Non-Biometric incidents.

### Deliverables
- Distribution of Civil Rights harms across Tech Categories; Statistical test results.

---

## Analysis

The experiment successfully executed the code to test the 'Biometric Rights'
correlation using the AIID dataset. The script processed 1,362 incidents. The
technology categorization logic successfully identified 52 incidents related to
Biometrics or Facial Recognition. However, the harm categorization logic—relying
on specific keywords like 'civil rights', 'liberty', and 'privacy'—failed to
identify a single matching record in the 'Harm Domain' column, classifying all
1,362 incidents as 'Other'. As a result, the contingency table contained no
'Civil Rights' entries, rendering the Chi-square test trivial (p-value = 1.0).
While the code execution was technically successful (exit code 0), the lack of
matching harm categories suggests that the keyword list did not align with the
dataset's specific taxonomy for harm domains (likely requiring broader terms
like 'Societal' or 'Fairness'), resulting in a null finding.

---

## Review

The experiment successfully executed the analysis pipeline to test the
'Biometric Rights' correlation using the AIID dataset (N=1,362). The
implementation correctly identified and loaded the dataset, and successfully
categorized 52 incidents as involving Biometric or Facial Recognition
technologies. However, the keyword-based categorization for the 'Civil Rights'
harm domain (using terms like 'civil rights', 'privacy', 'discrimination')
failed to identify any matching records in the 'Harm Domain' column. As a
result, the contingency table contained zero entries for 'Civil Rights', leading
to a trivial Chi-square test result (p-value = 1.0).

Finding: The experiment yields a null result, finding no statistical association
between Biometric technologies and Civil Rights harms under the current
definitions. However, this result is heavily qualified by the fact that the
'Harm Domain' column likely uses a different taxonomy (e.g., broad categories
like 'Societal' rather than specific terms like 'Civil Rights'), which caused
the dependent variable detection to fail. Therefore, while the code functioned
correctly, the hypothesis was not supported by the data as operationalized,
pointing to a schema mismatch rather than a definitive disproof of the
theoretical link.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

# --- Load Data ---
filename = 'astalabs_discovery_all_data.csv'
if not os.path.exists(filename):
    if os.path.exists('../' + filename):
        filename = '../' + filename

print(f"Loading dataset from: {filename}")
df = pd.read_csv(filename, low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid_df)} records")

# --- 1. Define Tech Categories (Biometric vs Other) ---
# Correct Column Name found in debug: 'Known AI Technology'
tech_col = 'Known AI Technology'

# keywords for Biometrics/Facial Recognition
bio_keywords = ['face', 'facial', 'biometric', 'recognition', 'gait', 'iris', 'voice print']

def categorize_tech(text):
    if pd.isna(text):
        return 'Other'
    text_lower = str(text).lower()
    if any(k in text_lower for k in bio_keywords):
        return 'Biometric'
    return 'Other'

aiid_df['Tech_Category'] = aiid_df[tech_col].apply(categorize_tech)

# --- 2. Define Harm Categories (Civil Rights vs Other) ---
# Correct Column Name found in debug: 'Harm Domain'
# Mapping AIID harm domains to 'Civil Rights' broadly.
harm_col = 'Harm Domain'

civil_rights_keywords = ['civil rights', 'liberty', 'privacy', 'discrimination', 'allocative', 'representation']

def categorize_harm(text):
    if pd.isna(text):
        return 'Other'
    text_lower = str(text).lower()
    if any(k in text_lower for k in civil_rights_keywords):
        return 'Civil Rights'
    return 'Other'

aiid_df['Harm_Category'] = aiid_df[harm_col].apply(categorize_harm)

# --- 3. Analysis ---

# Generate Contingency Table
contingency_table = pd.crosstab(aiid_df['Tech_Category'], aiid_df['Harm_Category'])

print("\n--- Contingency Table (Counts) ---")
print(contingency_table)

# Calculate Percentages
# Row-wise normalization to see: Of Biometric techs, what % are Civil Rights harms?
row_pct = pd.crosstab(aiid_df['Tech_Category'], aiid_df['Harm_Category'], normalize='index') * 100
print("\n--- Contingency Table (Percentages by Tech) ---")
print(row_pct.round(2))

# Perform Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\n--- Chi-Square Test Results ---")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-Value: {p:.4e}")

# Interpretation
alpha = 0.05
print("\n--- Conclusion ---")
if p < alpha:
    print("Result: Statistically Significant.")
    print("The data supports the hypothesis that Biometric technologies have a different harm profile (likely skewed towards Civil Rights/Liberties) compared to other AI technologies.")
else:
    print("Result: Not Statistically Significant.")
    print("The data does not show a significant association between Biometric technologies and Civil Rights harms compared to other technologies.")

# Additional context
print("\n[Debug Info] Top 5 Tech strings classified as Biometric:")
print(aiid_df[aiid_df['Tech_Category']=='Biometric'][tech_col].value_counts().head(5))

print("\n[Debug Info] Top 5 Harm Domains classified as Civil Rights:")
print(aiid_df[aiid_df['Harm_Category']=='Civil Rights'][harm_col].value_counts().head(5))

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: astalabs_discovery_all_data.csv
AIID Incidents loaded: 1362 records

--- Contingency Table (Counts) ---
Harm_Category  Other
Tech_Category       
Biometric         52
Other           1310

--- Contingency Table (Percentages by Tech) ---
Harm_Category  Other
Tech_Category       
Biometric      100.0
Other          100.0

--- Chi-Square Test Results ---
Chi2 Statistic: 0.0000
P-Value: 1.0000e+00

--- Conclusion ---
Result: Not Statistically Significant.
The data does not show a significant association between Biometric technologies and Civil Rights harms compared to other technologies.

[Debug Info] Top 5 Tech strings classified as Biometric:
Known AI Technology
Face Detection, Visual Object Detection, Image Segmentation                          7
Optical Character Recognition, Visual Object Detection                               4
Face Detection, Visual Object Detection                                              3
Convolutional Neural Network, Face Detection                                         2
Image Classification, Face Detection, Visual Object Detection, Image Segmentation    2
Name: count, dtype: int64

[Debug Info] Top 5 Harm Domains classified as Civil Rights:
Series([], Name: count, dtype: int64)

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
