# Experiment 19: node_3_9

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_9` |
| **ID in Run** | 19 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T01:56:47.687143+00:00 |
| **Runtime** | 302.0s |
| **Parent** | `node_2_3` |
| **Children** | `node_4_12`, `node_4_41` |
| **Creation Index** | 20 |

---

## Hypothesis

> Incidents involving Generative AI technologies result in 'Intangible' harms
(e.g., bias, reputation) significantly more often than traditional
Discriminative AI, which skews towards 'Tangible' harms.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.5824 (Maybe True) |
| **Surprise** | -0.1914 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | Yes |

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
| Uncertain | 60.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Compare harm domains between Generative and Discriminative AI incidents.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` (AIID subset).
- 2. Classify rows as 'GenAI' or 'Traditional' based on keywords in `Known AI Technology` (e.g., 'LLM', 'Generative', 'GPT' vs 'Regression', 'Vision').
- 3. Group `Harm Domain` into 'Tangible' (Physical, Economic) and 'Intangible' (Cultural, Psychological).
- 4. Perform a Chi-Square test on the Technology vs. Harm Type contingency table.

### Deliverables
- Stacked bar chart of Harm Types by AI Technology; Chi-Square test results.

---

## Analysis

The experiment successfully analyzed the AIID dataset to test if Generative AI
incidents skew towards 'Intangible' harms compared to Traditional AI. Utilizing
the metadata-verified columns 'Tangible Harm' and 'Special Interest Intangible
Harm' for classification, the code identified 70 relevant incidents. The
analysis revealed that while 100% of the Generative AI incidents (n=3) were
classified as 'Intangible', compared to 64.2% for Traditional AI (n=67), the
sample size for Generative AI was too small to yield statistical significance.
The Chi-Square test resulted in a p-value of 0.5111. Consequently, the
hypothesis is not supported by this dataset; the extreme sparsity of Generative
AI incidents with definitive harm flags prevents drawing a statistically valid
conclusion despite the visual trend.

---

## Review

The experiment was successfully executed. The programmer correctly adapted the
data extraction strategy after identifying through debugging that the 'Harm
Domain' column contained non-descriptive boolean flags ('yes/no/maybe'),
utilizing the 'Tangible Harm' and 'Special Interest Intangible Harm' columns
instead.

### Hypothesis Verification:
The hypothesis that **Generative AI incidents result in 'Intangible' harms
significantly more often than Traditional AI** was **not statistically
supported**, primarily due to data limitations.

*   **Data Skew**: The dataset contained only 3 coded Generative AI incidents
compared to 67 Traditional AI incidents.
*   **Observed Trend**: Visually, the data aligned with the hypothesis: 100% of
Generative AI incidents (3/3) caused Intangible harm, compared to 64.2% of
Traditional AI incidents.
*   **Statistical Significance**: A Chi-Square test yielded a p-value of 0.5111,
indicating the observed difference is not statistically significant given the
extremely small sample size of Generative AI cases.

While the code functioned correctly and the visualization highlights a potential
trend, the scarcity of coded Generative AI data in this specific snapshot
prevents a definitive confirmation of the hypothesis.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Check file existence
file_path = '../astalabs_discovery_all_data.csv'
if not os.path.exists(file_path):
    file_path = 'astalabs_discovery_all_data.csv'

print(f"Loading dataset from {file_path}...")
df = pd.read_csv(file_path, low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid_df)} rows")

# 1. Classify Technology
def classify_technology(text):
    if pd.isna(text):
        return None
    text = str(text).lower()
    genai_keywords = ['generative', 'llm', 'large language model', 'gpt', 'diffusion', 
                      'dall-e', 'midjourney', 'stable diffusion', 'transformer', 'chatbot', 
                      'foundation model', 'chatgpt', 'bert', 'palm', 'llama', 'gan', 'stylegan']
    
    if any(k in text for k in genai_keywords):
        return 'Generative AI'
    return 'Traditional AI'

aiid_df['Tech_Type'] = aiid_df['Known AI Technology'].apply(classify_technology)

# 2. Classify Harm
# Logic: 
# - Tangible if 'Tangible Harm' says it definitively occurred AND 'Special Interest Intangible Harm' is NOT yes.
# - Intangible if 'Special Interest Intangible Harm' says yes AND 'Tangible Harm' did NOT definitively occur.
# - Exclude mixed/ambiguous cases to strictly test the skew.

def classify_harm_composite(row):
    tangible_val = str(row.get('Tangible Harm', '')).lower()
    intangible_val = str(row.get('Special Interest Intangible Harm', '')).lower()
    
    is_tangible = 'tangible harm definitively occurred' in tangible_val
    is_intangible = 'yes' in intangible_val
    
    if is_tangible and not is_intangible:
        return 'Tangible'
    elif is_intangible and not is_tangible:
        return 'Intangible'
    else:
        return None # Mixed or None

aiid_df['Harm_Type'] = aiid_df.apply(classify_harm_composite, axis=1)

# 3. Filter Data for Analysis
analysis_df = aiid_df.dropna(subset=['Tech_Type', 'Harm_Type']).copy()

print(f"\nData points for analysis: {len(analysis_df)}")
print("Distribution by Tech Type:")
print(analysis_df['Tech_Type'].value_counts())
print("\nDistribution by Harm Type:")
print(analysis_df['Harm_Type'].value_counts())

# 4. Statistical Test
contingency_table = pd.crosstab(analysis_df['Tech_Type'], analysis_df['Harm_Type'])
print("\nContingency Table (Observed):")
print(contingency_table)

if not contingency_table.empty and contingency_table.size >= 4:
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Interpret results
    alpha = 0.05
    if p < alpha:
        print("Result: Significant association between AI Technology and Harm Type.")
    else:
        print("Result: No significant association found.")

    # 5. Visualization
    # Normalize to get percentages for stacked bar to visualize the 'skew'
    ct_pct = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
    
    ax = ct_pct.plot(kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'salmon'])
    plt.title('Proportion of Tangible vs Intangible Harms by AI Technology')
    plt.xlabel('AI Technology Type')
    plt.ylabel('Percentage')
    plt.legend(title='Harm Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    
    # Annotate bars
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center')
        
    plt.tight_layout()
    plt.show()
else:
    print("Insufficient data for Chi-Square test.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
AIID Incidents loaded: 1362 rows

Data points for analysis: 70
Distribution by Tech Type:
Tech_Type
Traditional AI    67
Generative AI      3
Name: count, dtype: int64

Distribution by Harm Type:
Harm_Type
Intangible    46
Tangible      24
Name: count, dtype: int64

Contingency Table (Observed):
Harm_Type       Intangible  Tangible
Tech_Type                           
Generative AI            3         0
Traditional AI          43        24

Chi-Square Statistic: 0.4319
P-value: 5.1108e-01
Result: No significant association found.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** 100% Stacked Bar Chart.
*   **Purpose:** The plot compares the relative composition of two categories ("Tangible" vs. "Intangible" harms) across two different groups ("Generative AI" and "Traditional AI"). It is designed to visualize how the distribution of harm types differs between these two AI technologies.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "AI Technology Type"
    *   **Labels:** "Generative AI" and "Traditional AI"
*   **Y-Axis:**
    *   **Title:** "Percentage"
    *   **Range:** 0 to 100
    *   **Units:** Percent (%)

### 3. Data Trends
*   **Generative AI:** The bar is monochromatic, indicating a uniform distribution. The data suggests that **100%** of the harms associated with Generative AI in this dataset are classified as **Intangible**. There is a 0.0% share of Tangible harms.
*   **Traditional AI:** The bar shows a mixed distribution. While **Intangible** harms still constitute the majority at **64.2%**, there is a significant portion of **Tangible** harms at **35.8%**.
*   **Overall Pattern:** Intangible harms are the dominant category for both technologies, but Traditional AI demonstrates a notable presence of tangible risks that is entirely absent in the Generative AI data shown.

### 4. Annotations and Legends
*   **Chart Title:** "Proportion of Tangible vs Intangible Harms by AI Technology" positioned at the top center.
*   **Legend:** Located on the right side, titled "Harm Type".
    *   **Light Blue:** Represents "Intangible" harms.
    *   **Salmon/Red:** Represents "Tangible" harms.
*   **Annotations:** Specific percentage values are labeled directly onto the bar segments:
    *   Generative AI: 100.0% (Intangible), 0.0% (Tangible).
    *   Traditional AI: 64.2% (Intangible), 35.8% (Tangible).

### 5. Statistical Insights
*   **Nature of Risks:** The plot indicates a distinct dichotomy in the nature of risks posed by these technologies. Generative AI appears to be exclusively associated with non-physical or indirect harms (intangible), such as misinformation, bias, or privacy violations. In contrast, Traditional AI retains a significant component of physical or direct asset damage (tangible).
*   **Prevalence of Intangible Harm:** Regardless of the technology type, "Intangible" harms are the primary concern, accounting for the majority of incidents in both categories.
*   **Risk Profile Shift:** Moving from Traditional to Generative AI shifts the risk profile completely away from tangible consequences to purely intangible ones, according to this specific dataset.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
