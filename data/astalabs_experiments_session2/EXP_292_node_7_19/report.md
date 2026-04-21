# Experiment 292: node_7_19

| Property | Value |
|---|---|
| **Experiment ID** | `node_7_19` |
| **ID in Run** | 292 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T14:58:15.240036+00:00 |
| **Runtime** | 204.9s |
| **Parent** | `node_6_24` |
| **Children** | None |
| **Creation Index** | 293 |

---

## Hypothesis

> The 'Sector-Tech' Specificity: 'Healthcare' sector incidents are predominantly
driven by 'Computer Vision' failures, whereas 'Financial' sector incidents are
driven by 'Language/Text' processing failures.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.7143 (Likely True) |
| **Surprise** | -0.0332 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

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
| Uncertain | 4.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Analyze the technical modalities associated with failures in distinct sectors.

### Steps
- 1. Filter `aiid_incidents` for rows where `Sector of Deployment` is 'Healthcare' or 'Financial'.
- 2. Text-mine `Known AI Technology` to classify into 'Vision' (keywords: image, cv, recognition, vision) vs 'Language' (keywords: nlp, text, language, chat, llm).
- 3. Create a contingency table of Sector vs. Technology Modality.
- 4. Run a Chi-Square test.

### Deliverables
- 1. Contingency table of Sector vs. Tech Modality.
- 2. Chi-Square test results.
- 3. Heatmap of counts.

---

## Analysis

The experiment aimed to test the 'Sector-Tech Specificity' hypothesis by
correlating sector (Healthcare vs. Financial) with technology modality (Vision
vs. Language).

**Findings:**
1. **Data Sparsity**: Despite having 1,366 incidents in the AIID dataset,
filtering by the `Sector of Deployment` column yielded only **21** relevant
incidents (15 Healthcare, 6 Financial). This suggests the sector metadata is
highly sparse or unpopulated in this version of the dataset.
2. **Classification Failure**: Further filtering these 21 incidents by `Known AI
Technology` resulted in only **2** classifiable cases (both Healthcare/Computer
Vision).
3. **Test Viability**: The resulting sample size (N=2) was insufficient to
generate a contingency table or perform the planned Chi-Square test.

**Conclusion**: The experiment successfully executed the logic but failed to
produce statistical results due to missing metadata. The hypothesis cannot be
validated using the pre-coded columns (`Sector of Deployment`, `Known AI
Technology`). Future attempts should utilize NLP techniques to extract sector
and technology information from the unstructured `description` or `summary`
fields.

---

## Review

Hypothesis: The 'Sector-Tech' Specificity hypothesis posited that 'Healthcare'
incidents are driven by 'Computer Vision' failures, while 'Financial' incidents
are driven by 'Language/Text' failures.

Experiment Results:
1. Data Sparsity: Despite processing 1,362 incidents, only 21 records contained
values in the 'Sector of Deployment' column matching the target sectors (15
Healthcare, 6 Financial).
2. Classification Failure: Further filtering these 21 records by 'Known AI
Technology' yielded only 2 classifiable instances (both Healthcare / Computer
Vision).
3. Statistical Outcome: The sample size (N=2) was insufficient to generate a
contingency table or perform the planned Chi-Square test.

Findings: The hypothesis remains untested due to extreme sparsity in the
structured metadata columns ('Sector of Deployment' and 'Known AI Technology').
The experiment demonstrates that these specific columns in the current dataset
version are not populated sufficiently to support the proposed correlation
analysis without supplementary NLP on unstructured text fields.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

print(f"Total AIID incidents: {len(aiid_df)}")

# 1. Define Sector Filtering
# We need to identify Healthcare vs Financial sectors
# Let's clean the column first
aiid_df['Sector of Deployment'] = aiid_df['Sector of Deployment'].fillna('').astype(str)

def classify_sector(val):
    val_lower = val.lower()
    if 'health' in val_lower or 'medicine' in val_lower or 'medical' in val_lower:
        return 'Healthcare'
    elif 'financ' in val_lower or 'bank' in val_lower or 'trading' in val_lower:
        return 'Financial'
    else:
        return 'Other'

aiid_df['target_sector'] = aiid_df['Sector of Deployment'].apply(classify_sector)

# Filter only for the target sectors
analysis_df = aiid_df[aiid_df['target_sector'].isin(['Healthcare', 'Financial'])].copy()
print(f"Incidents in target sectors (Healthcare/Financial): {len(analysis_df)}")
print(analysis_df['target_sector'].value_counts())

# 2. Define Technology Classification
# Classify 'Known AI Technology' into 'Vision', 'Language', or 'Other'
analysis_df['Known AI Technology'] = analysis_df['Known AI Technology'].fillna('').astype(str)

def classify_tech(val):
    val_lower = val.lower()
    
    # Vision keywords
    vision_keywords = ['vision', 'image', 'video', 'facial', 'face', 'recognition', 'detection', 'surveillance', 'camera']
    is_vision = any(k in val_lower for k in vision_keywords)
    
    # Language keywords
    language_keywords = ['language', 'text', 'nlp', 'chat', 'llm', 'generative', 'translation', 'speech', 'voice', 'chatbot']
    is_language = any(k in val_lower for k in language_keywords)
    
    if is_vision and not is_language:
        return 'Computer Vision'
    elif is_language and not is_vision:
        return 'Language/Text'
    elif is_vision and is_language:
        return 'Multimodal/Both' # Or prioritize one based on hypothesis, but let's keep separate for clarity
    else:
        return 'Unspecified/Other'

analysis_df['tech_modality'] = analysis_df['Known AI Technology'].apply(classify_tech)

# Filter out Unspecified/Other/Multimodal to test the specific hypothesis strictly between Vision and Language
# Hypothesis: Healthcare -> Vision, Financial -> Language
strict_df = analysis_df[analysis_df['tech_modality'].isin(['Computer Vision', 'Language/Text'])].copy()

print(f"Incidents with clear Vision/Language classification: {len(strict_df)}")
print(strict_df.groupby(['target_sector', 'tech_modality']).size())

# 3. Create Contingency Table
contingency_table = pd.crosstab(strict_df['target_sector'], strict_df['tech_modality'])
print("\nContingency Table:")
print(contingency_table)

# 4. Chi-Square Test
if not contingency_table.empty and contingency_table.shape == (2, 2):
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-Square Test Results:")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-Value: {p:.4e}")
    
    # 5. Visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues')
    plt.title('AI Incident Counts: Sector vs. Technology Modality')
    plt.ylabel('Sector')
    plt.xlabel('Technology Modality')
    plt.show()
    
    # Calculate percentages for interpretation
    row_sums = contingency_table.sum(axis=1)
    props = contingency_table.div(row_sums, axis=0)
    print("\nProportions (Row-wise):")
    print(props)
else:
    print("Insufficient data for Chi-Square test (need 2x2 table).")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Total AIID incidents: 1362
Incidents in target sectors (Healthcare/Financial): 21
target_sector
Healthcare    15
Financial      6
Name: count, dtype: int64
Incidents with clear Vision/Language classification: 2
target_sector  tech_modality  
Healthcare     Computer Vision    2
dtype: int64

Contingency Table:
tech_modality  Computer Vision
target_sector                 
Healthcare                   2
Insufficient data for Chi-Square test (need 2x2 table).

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
