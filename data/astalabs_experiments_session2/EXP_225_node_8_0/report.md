# Experiment 225: node_8_0

| Property | Value |
|---|---|
| **Experiment ID** | `node_8_0` |
| **ID in Run** | 225 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T11:34:22.110033+00:00 |
| **Runtime** | 360.6s |
| **Parent** | `node_7_6` |
| **Children** | None |
| **Creation Index** | 226 |

---

## Hypothesis

> The Generative Governance Gap: AI use cases involving 'Language' or 'Chat' (Text
modality) are significantly less likely to have undergone an 'Impact Assessment'
compared to 'Vision' or 'Image' based systems.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6774 (Maybe True) |
| **Posterior** | 0.3956 (Maybe False) |
| **Surprise** | -0.3382 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 26.0 |
| Uncertain | 0.0 |
| Maybe False | 4.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 60.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Compare governance maturity (Impact Assessments) across AI modalities (Text vs. Vision) using a Chi-square test.

### Steps
- 1. Load the dataset `astalabs_discovery_all_data.csv`.
- 2. Filter for rows where `source_table` is 'eo13960_scored'.
- 3. Create a 'Modality' classification based on keywords in columns `2_use_case_name` and `11_purpose_benefits`:
    - Text Keywords: 'chat', 'language', 'nlp', 'text', 'summariz', 'translate'.
    - Vision Keywords: 'vision', 'image', 'video', 'camera', 'detect', 'surveil'.
    - Classification Logic: Assign 'Text' if text-keywords exist and vision-keywords do not. Assign 'Vision' if vision-keywords exist and text-keywords do not. Ignore 'Mixed' or 'Other'.
- 4. Create a binary target variable `has_impact_assessment` derived from `52_impact_assessment`. treat values containing 'yes' (case-insensitive) as True, others as False.
- 5. Generate a contingency table for Modality (Text/Vision) vs. Impact Assessment (True/False). Ensure the table is explicitly 2x2 even if counts are zero.
- 6. Perform a Chi-square test of independence.
- 7. Calculate and print the assessment rates for both groups.

### Deliverables
- 1. Loaded and filtered dataset.
- 2. Contingency table of Modality vs. Impact Assessment.
- 3. Chi-square test results (statistic, p-value).
- 4. Assessment rates for Text vs. Vision systems.

---

## Analysis

The experiment was successfully executed with a corrected contingency table.
Analysis of 723 federal AI use cases (413 Text-based, 310 Vision-based) revealed
the following:

1.  **Low Governance Baseline**: Documented Impact Assessments are extremely
rare for both modalities. Only 21 out of 723 systems (2.9%) had an affirmative
assessment.
2.  **Comparison**: Text-based systems showed a lower assessment rate (2.18%)
compared to Vision-based systems (3.87%).
3.  **Statistical Conclusion**: The Chi-Square test yielded a p-value of 0.2641,
which is above the standard alpha of 0.05. Therefore, the difference is **not
statistically significant**.

While the data directionally suggests a 'governance gap' where text systems are
less scrutinized, the hypothesis is technically **rejected** due to lack of
statistical significance. The primary insight is the systemic absence of impact
assessments across both AI modalities in the current federal inventory.

---

## Review

The experiment was successfully executed. The file loading issue was resolved,
and the contingency table was correctly generated with valid dimensions (2x2),
allowing for a proper Chi-Square test. The analysis of 723 federal AI use cases
(413 Text vs 310 Vision) revealed a directionally consistent but statistically
insignificant difference in governance maturity. Text-based systems had a lower
impact assessment rate (2.18%) compared to Vision-based systems (3.87%), but the
p-value (0.26) indicates this could be due to chance. The hypothesis is
rejected, highlighting a broader finding of systemic low governance
documentation (<4%) across all AI modalities in the federal inventory.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

# 1. Load Dataset
filename = 'astalabs_discovery_all_data.csv'
possible_paths = [filename, f'../{filename}']
file_path = None

for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        break

if file_path is None:
    # Fallback default
    file_path = filename

df = pd.read_csv(file_path, low_memory=False)

# 2. Filter for EO13960
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO13960 records loaded: {len(eo_data)}")

# 3. Text Processing & Modality Classification
# Combine columns for keyword search
eo_data['combined_text'] = (eo_data['2_use_case_name'].fillna('') + ' ' + eo_data['11_purpose_benefits'].fillna('')).str.lower()

# Define keywords
text_keywords = ['chat', 'language', 'nlp', 'text', 'summariz', 'translat']
vision_keywords = ['vision', 'image', 'video', 'camera', 'detect', 'surveil']

def classify_modality(text):
    if not isinstance(text, str):
        return 'Other'
    has_text = any(k in text for k in text_keywords)
    has_vision = any(k in text for k in vision_keywords)
    
    if has_text and not has_vision:
        return 'Text'
    elif has_vision and not has_text:
        return 'Vision'
    elif has_text and has_vision:
        return 'Mixed'
    else:
        return 'Other'

eo_data['modality'] = eo_data['combined_text'].apply(classify_modality)

# 4. Filter for specific modalities
analysis_df = eo_data[eo_data['modality'].isin(['Text', 'Vision'])].copy()
print(f"\nModality Counts:\n{analysis_df['modality'].value_counts()}")

# 5. Target Variable Creation
# Check raw values first
print(f"\nUnique values in '52_impact_assessment': {analysis_df['52_impact_assessment'].unique()}")

# Map to Boolean: True if contains 'yes', else False
analysis_df['has_assessment'] = analysis_df['52_impact_assessment'].astype(str).str.lower().str.contains('yes')

# 6. Generate Contingency Table
# Explicitly ensure both rows (Text, Vision) and columns (False, True) exist
# We use pd.Categorical to enforce structure even if counts are zero
analysis_df['modality'] = pd.Categorical(analysis_df['modality'], categories=['Text', 'Vision'])
analysis_df['has_assessment'] = pd.Categorical(analysis_df['has_assessment'], categories=[False, True])

contingency_table = pd.crosstab(analysis_df['modality'], analysis_df['has_assessment'], dropna=False)

print("\nContingency Table (Modality x Has Assessment):")
print(contingency_table)

# 7. Statistical Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"\nChi-Square Results:")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")
print(f"Degrees of Freedom: {dof}")

# 8. Rate Calculation & Conclusion
rates = analysis_df.groupby('modality', observed=False)['has_assessment'].apply(lambda x: (x == True).mean())
text_rate = rates['Text']
vision_rate = rates['Vision']

print(f"\nAssessment Rates:")
print(f"Text:   {text_rate:.2%}")
print(f"Vision: {vision_rate:.2%}")

alpha = 0.05
if p < alpha:
    print("\nConclusion: Statistically Significant Difference.")
    if text_rate < vision_rate:
        print("Hypothesis Supported: Text systems have significantly lower assessment rates than Vision systems.")
    else:
        print("Hypothesis Rejected (Direction): Text systems have higher assessment rates.")
else:
    print("\nConclusion: No Statistically Significant Difference found.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: EO13960 records loaded: 1757

Modality Counts:
modality
Text      413
Vision    310
Name: count, dtype: int64

Unique values in '52_impact_assessment': <StringArray>
[nan, 'Planned or in-progress.', 'Yes', 'No', 'YES']
Length: 5, dtype: str

Contingency Table (Modality x Has Assessment):
has_assessment  False  True
modality                   
Text              404     9
Vision            298    12

Chi-Square Results:
Chi2 Statistic: 1.2473
P-value: 2.6406e-01
Degrees of Freedom: 1

Assessment Rates:
Text:   2.18%
Vision: 3.87%

Conclusion: No Statistically Significant Difference found.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
