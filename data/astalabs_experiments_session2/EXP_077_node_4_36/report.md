# Experiment 77: node_4_36

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_36` |
| **ID in Run** | 77 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T04:39:15.281332+00:00 |
| **Runtime** | 211.1s |
| **Parent** | `node_3_0` |
| **Children** | `node_5_20`, `node_5_65` |
| **Creation Index** | 78 |

---

## Hypothesis

> The Generative Malice Gap: AI Incidents involving 'Generative' technologies
(LLMs, Diffusion, GANs) are significantly more likely to be classified as
'Intentional' harms (e.g., deepfakes, disinformation) compared to
'Predictive/Discriminative' AI incidents.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7742 (Likely True) |
| **Posterior** | 0.4286 (Maybe False) |
| **Surprise** | -0.4147 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 4.0 |
| Maybe True | 26.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
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

**Objective:** Analyze the relationship between AI technology type and the intentionality of the harm caused.

### Steps
- 1. Filter for `aiid_incidents`.
- 2. Create a new column `tech_category` by searching `Known AI Technology` for keywords (e.g., 'Generative', 'GPT', 'LLM', 'Diffusion' -> 'Generative'; others -> 'Predictive/Other').
- 3. Map `Intentional Harm` to a binary Yes/No variable.
- 4. Compare the proportion of Intentional incidents in Generative vs. Predictive categories.
- 5. Run a Chi-square or Fisher's Exact test.

### Deliverables
- Stacked bar chart of Intentionality by Technology Type; Statistical test outcome.

---

## Analysis

The experiment successfully tested the 'Generative Malice Gap' hypothesis using
a subset of 112 fully coded incidents from the AIID dataset (filtered from 1,362
total records based on available metadata). The analysis **rejected the
hypothesis** that Generative AI is more frequently associated with intentional
harm.

Results showed that **0%** (0/15) of the analyzed Generative AI incidents
involved intentional harm, compared to **2.1%** (2/97) of Predictive/Other AI
incidents. The statistical analysis (Chi-Square test) yielded a p-value of 1.00,
indicating no significant difference between the groups—likely due to the
extreme rarity of 'Intentional' harms in the dataset (only 2 total cases).

The findings suggest that, within this specific dataset, AI incidents are
overwhelmingly characterized by **unintentional failures** (accidents, errors,
or oversight) rather than malicious use, regardless of the technology
architecture. The extremely small sample size of fully coded 'Intentional' cases
is a notable limitation.

---

## Review

The experiment successfully tested the 'Generative Malice Gap' hypothesis using
a subset of 112 fully coded incidents from the AIID dataset. The analysis
**rejected the hypothesis** that Generative AI is more frequently associated
with intentional harm.

Results showed that **0%** (0/15) of the analyzed Generative AI incidents
involved intentional harm, compared to **2.1%** (2/97) of Predictive/Other AI
incidents. The statistical analysis (Chi-Square test) yielded a p-value of 1.00,
indicating no significant difference between the groups—likely due to the
extreme rarity of 'Intentional' harms in the dataset (only 2 total cases).

The findings suggest that, within this specific dataset, AI incidents are
overwhelmingly characterized by **unintentional failures** (accidents, errors,
or oversight) rather than malicious use, regardless of the technology
architecture. The extremely small sample size of fully coded 'Intentional' cases
is a notable limitation.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# [debug]
print("Starting experiment: The Generative Malice Gap (Attempt 2)")

# Load dataset
file_name = 'astalabs_discovery_all_data.csv'
paths = [f'../{file_name}', file_name]
ds_path = next((p for p in paths if os.path.exists(p)), None)

if not ds_path:
    print(f"Error: Dataset {file_name} not found.")
    exit(1)

df = pd.read_csv(ds_path, low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Total AIID incidents: {len(aiid_df)}")

# Find correct column names dynamically
cols = df.columns.tolist()
col_tech = next((c for c in cols if 'Known AI Technology' in str(c)), None)
col_intent = next((c for c in cols if 'Intentional Harm' in str(c)), None)

print(f"Identified Tech Column: {col_tech}")
print(f"Identified Intent Column: {col_intent}")

if not col_tech or not col_intent:
    print("Could not identify required columns. Available columns:")
    # Print columns that might be relevant or a sample
    print([c for c in cols if 'AI' in str(c) or 'Harm' in str(c)])
    exit(1)

# Define Generative keywords
gen_keywords = [
    'generative', 'llm', 'gpt', 'diffusion', 'gan', 'transformer', 
    'chatbot', 'language model', 'text-to-image', 'deepfake', 'synthetic media',
    'stable diffusion', 'midjourney', 'dall-e', 'bard', 'chatgpt', 'gemini'
]

def categorize_tech(val):
    if pd.isna(val):
        return 'Unknown'
    val_lower = str(val).lower()
    if any(k in val_lower for k in gen_keywords):
        return 'Generative'
    return 'Predictive/Other'

aiid_df['tech_category'] = aiid_df[col_tech].apply(categorize_tech)

# Define Intentionality mapping
def categorize_intent(val):
    if pd.isna(val):
        return 'Unclear'
    val_lower = str(val).lower()
    # Check for 'yes', 'true' for Intentional
    # Check for 'no', 'false' for Unintentional
    if val_lower in ['yes', 'true', 'intentional']:
        return 'Intentional'
    elif val_lower in ['no', 'false', 'unintentional', 'accidental']:
        return 'Unintentional'
    # Sometimes values are sentences, so simpler check:
    if 'yes' in val_lower or 'true' in val_lower:
        return 'Intentional'
    if 'no' in val_lower or 'false' in val_lower:
        return 'Unintentional'
    return 'Unclear'

aiid_df['intent_category'] = aiid_df[col_intent].apply(categorize_intent)

# Filter for analysis (exclude Unknown tech and Unclear intent)
analysis_df = aiid_df[
    (aiid_df['tech_category'] != 'Unknown') & 
    (aiid_df['intent_category'] != 'Unclear')
].copy()

print(f"\nData points for analysis: {len(analysis_df)}")
print("Distribution by Tech Category:")
print(analysis_df['tech_category'].value_counts())
print("Distribution by Intent Category:")
print(analysis_df['intent_category'].value_counts())

if len(analysis_df) < 5:
    print("Not enough data points for statistical analysis.")
else:
    # Contingency Table
    contingency = pd.crosstab(analysis_df['tech_category'], analysis_df['intent_category'])
    print("\nContingency Table:")
    print(contingency)

    # Proportions (Row-wise to see % Intentional per Tech Category)
    props = pd.crosstab(analysis_df['tech_category'], analysis_df['intent_category'], normalize='index')
    print("\nProportions (Row-normalized):")
    print(props)

    # Statistical Test (Chi-Square)
    # We are testing if there is an association between Tech Type and Intentionality
    chi2, p, dof, ex = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test Results:")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Plotting
    # Reorder for visualization if needed
    # We want a stacked bar of Intentional vs Unintentional
    ax = props.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#ff9999', '#66b3ff'])
    plt.title('Intentionality of Harm by AI Technology Type')
    plt.ylabel('Proportion of Incidents')
    plt.xlabel('Technology Category')
    plt.xticks(rotation=0)
    plt.legend(title='Intentional Harm', loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting experiment: The Generative Malice Gap (Attempt 2)
Total AIID incidents: 1362
Identified Tech Column: Known AI Technology
Identified Intent Column: Intentional Harm

Data points for analysis: 112
Distribution by Tech Category:
tech_category
Predictive/Other    97
Generative          15
Name: count, dtype: int64
Distribution by Intent Category:
intent_category
Unintentional    110
Intentional        2
Name: count, dtype: int64

Contingency Table:
intent_category   Intentional  Unintentional
tech_category                               
Generative                  0             15
Predictive/Other            2             95

Proportions (Row-normalized):
intent_category   Intentional  Unintentional
tech_category                               
Generative           0.000000       1.000000
Predictive/Other     0.020619       0.979381

Chi-Square Test Results:
Chi2 Statistic: 0.0000
P-value: 1.0000e+00


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is a detailed analysis:

### 1. Plot Type
*   **Type:** 100% Stacked Bar Chart.
*   **Purpose:** This chart compares the relative proportions of two sub-groups ("Intentional" vs. "Unintentional" harm) across two distinct categories of AI technology ("Generative" and "Predictive/Other"). It is designed to show the composition of incidents rather than the total volume of incidents.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Technology Category"
    *   **Categories:** Two discrete categories are displayed: "Generative" and "Predictive/Other".
*   **Y-Axis:**
    *   **Label:** "Proportion of Incidents"
    *   **Range:** 0.0 to 1.0 (representing 0% to 100%).
    *   **Scale:** Linear scale with major tick marks every 0.2 units.

### 3. Data Trends
*   **Dominant Pattern:** For both technology categories, the vast majority of incidents are categorized as **"Unintentional"** (represented by the blue bars).
*   **Generative AI:** The bar for "Generative" appears to be entirely blue. Visually, this indicates that nearly **100%** of the recorded incidents in this category were unintentional, with no visible proportion of intentional harm.
*   **Predictive/Other AI:** While still dominated by "Unintentional" incidents (blue), this bar shows a very thin strip of pink (light red) at the bottom. This indicates a small, non-zero proportion of **"Intentional"** harm, roughly estimated visually to be between 1% and 3% (0.01 - 0.03).

### 4. Annotations and Legends
*   **Chart Title:** "Intentionality of Harm by AI Technology Type" – clear and descriptive of the data subject.
*   **Legend:** Located on the right side of the plot with the header "Intentional Harm".
    *   **Pink/Light Red:** Represents "Intentional" incidents.
    *   **Blue:** Represents "Unintentional" incidents.

### 5. Statistical Insights
*   **Prevalence of Unintentional Harm:** The data strongly suggests that regardless of the AI technology type (Generative vs. Predictive), harm caused by these systems is overwhelmingly unintentional (likely due to errors, bias, or unforeseen consequences) rather than the result of malicious intent.
*   **Category Comparison:** There is a slight divergence between the two technologies. While Generative AI incidents in this dataset appear completely devoid of intentionality, the "Predictive/Other" category involves a marginal risk of intentional misuse or harm, though it remains a statistical outlier compared to unintentional incidents.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
