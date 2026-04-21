# Experiment 186: node_6_27

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_27` |
| **ID in Run** | 186 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T09:47:36.795715+00:00 |
| **Runtime** | 251.7s |
| **Parent** | `node_5_32` |
| **Children** | `node_7_18` |
| **Creation Index** | 187 |

---

## Hypothesis

> Modality-Driven Malice: Incidents involving Computer Vision technologies are
significantly more likely to be classified as 'Intentional' harm (e.g.,
surveillance, deepfakes) compared to NLP/Language technologies.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.4176 (Maybe False) |
| **Surprise** | -0.3892 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
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
| Uncertain | 0.0 |
| Maybe False | 60.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Explore if visual AI modalities are more prone to malicious use than text-based AI.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'aiid_incidents'.
- 2. Classify 'Known AI Technology' into 'Vision' (keywords: Vision, Image, Face, Video) and 'Language' (keywords: NLP, Text, Language, Chatbot, LLM).
- 3. Binarize 'Intentional Harm' (True/False).
- 4. Create a contingency table (Modality vs Intent).
- 5. Perform a Chi-Square test.

### Deliverables
- Bar chart of Intentional Harm rates by Technology Modality and statistical results.

---

## Analysis

The experiment successfully analyzed 108 AI incidents from the AIID dataset,
classifying them into Computer Vision (n=97) and Language/NLP (n=11) modalities.
The investigation into 'Intentional Harm' revealed an extremely low prevalence
of malicious use in this subset: only 1 incident (linked to Vision) was
classified as intentional (~1.0%), while 0 incidents were intentional for
Language (0.0%).

A Fisher's Exact Test yielded a p-value of 1.0000, confirming that there is no
statistically significant difference between the two groups. Therefore, the
hypothesis 'Modality-Driven Malice' is **not supported** by this dataset.

It is important to note that the result is heavily influenced by the sparsity of
the data: the sample size for Language incidents was very small (11), and the
total number of intentional harm cases identified in the filtered subset was
only one. This lack of statistical power suggests that while no difference was
observed here, a larger dataset would be required to definitively rule out
modality-based differences in malicious use.

---

## Review

The experiment was successfully executed and the analysis was robustly handled
despite data limitations. The code correctly filtered the AIID incidents and
classified them into 'Vision' (n=97) and 'Language' (n=11) modalities. The text
processing logic for identifying 'Intentional Harm' was improved to capture
variations in the data, identifying exactly one intentional incident (in the
Vision category) and zero in Language.

The statistical analysis correctly pivoted to Fisher's Exact Test due to the low
expected frequencies (violating Chi-Square assumptions). The resulting p-value
of 1.00 indicates no statistically significant difference between the groups.
Consequently, the 'Modality-Driven Malice' hypothesis is **not supported** by
this dataset. The analysis highlights that intentional harm is a rare event in
this specific data subset (~1% for Vision, 0% for Language), and the small
sample size for Language technologies limits the ability to draw broader
generalizations.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact
import numpy as np

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID Incidents
df_incidents = df[df['source_table'] == 'aiid_incidents'].copy()

# Identify relevant columns
tech_col = next((c for c in df_incidents.columns if 'Known AI Technology' in c), None)
intent_col = next((c for c in df_incidents.columns if 'Intentional Harm' in c), None)

if not tech_col or not intent_col:
    print(f"Critical columns not found. Available: {list(df_incidents.columns)}")
else:
    print(f"Using columns: Technology='{tech_col}', Intent='{intent_col}'")

    # 1. Classify Modality
    def classify_modality(val):
        if not isinstance(val, str):
            return 'Other'
        val_lower = val.lower()
        # Vision keywords
        if any(x in val_lower for x in ['vision', 'image', 'face', 'facial', 'video', 'surveillance', 'recognition']):
            return 'Vision'
        # Language keywords
        if any(x in val_lower for x in ['nlp', 'text', 'language', 'chatbot', 'llm', 'translation', 'generative text', 'gpt', 'bert']):
            return 'Language'
        return 'Other'

    df_incidents['modality'] = df_incidents[tech_col].apply(classify_modality)
    
    # Filter for only Vision and Language
    df_analysis = df_incidents[df_incidents['modality'].isin(['Vision', 'Language'])].copy()
    
    print("\n--- Modality Counts ---")
    print(df_analysis['modality'].value_counts())

    # 2. Clean Intentional Harm (Fixed Logic)
    def is_intentional(val):
        if pd.isna(val):
            return False
        val_str = str(val).lower().strip()
        # Check for presence of 'yes' or 'intentionally'
        if val_str.startswith('yes'):
            return True
        if 'intentionally designed' in val_str and 'not intentionally' not in val_str:
             # Handle cases like "Yes. Intentionally..."
             return True
        return False

    df_analysis['is_intentional'] = df_analysis[intent_col].apply(is_intentional)
    
    print("\n--- Intentional vs Unintentional Counts (Processed) ---")
    print(df_analysis['is_intentional'].value_counts())

    # 3. Contingency Table
    contingency = pd.crosstab(df_analysis['modality'], df_analysis['is_intentional'])
    
    # Reindex to ensure both False and True columns exist
    contingency = contingency.reindex(columns=[False, True], fill_value=0)
    contingency.columns = ['Unintentional', 'Intentional']
    
    print("\n--- Contingency Table (Final) ---")
    print(contingency)

    # Check if we have any intentional cases at all
    if contingency['Intentional'].sum() == 0:
        print("\nNo intentional cases found in the filtered subset. Statistical test cannot be performed.")
    else:
        # 4. Statistical Test
        # Use Fisher's Exact Test if sample sizes are small (< 5 in any cell of expected), otherwise Chi2
        # Given the previous debug output, counts might be very low.
        # Let's try Chi2 first, catch error, fallback to Fisher if needed or if appropriate.
        
        try:
            chi2, p, dof, ex = chi2_contingency(contingency)
            print(f"\nChi-Square Statistic: {chi2:.4f}")
            print(f"P-Value: {p:.4e}")
            
            # Check expected frequencies for validity of Chi2
            if (ex < 5).any():
                print("Warning: Some expected frequencies are < 5. Fisher's Exact Test is recommended.")
                res = fisher_exact(contingency)
                print(f"Fisher's Exact Test P-Value: {res[1]:.4e}")
                p = res[1] # Update p for conclusion

            if p < 0.05:
                print("Result: Statistically Significant difference found.")
            else:
                print("Result: No statistically significant difference found.")
        except Exception as e:
            print(f"Statistical test failed: {e}")

        # 5. Visualization
        # Calculate rates
        total = contingency.sum(axis=1)
        # Avoid division by zero
        rates = (contingency['Intentional'] / total.replace(0, 1)) * 100
        
        print("\n--- Intentional Harm Rates ---")
        print(rates)

        plt.figure(figsize=(8, 6))
        bars = plt.bar(rates.index, rates.values, color=['skyblue', 'salmon'])
        plt.title('Rate of Intentional Harm by AI Modality')
        plt.ylabel('Percentage of Intentional Incidents (%)')
        plt.xlabel('AI Modality')
        # Adjust ylim slightly above max value or default to 10 if 0
        top_val = max(rates.values) if len(rates) > 0 else 0
        plt.ylim(0, top_val * 1.2 if top_val > 0 else 10)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, 
                     f'{height:.1f}%',
                     ha='center', va='bottom')
        
        plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Using columns: Technology='Known AI Technology', Intent='Intentional Harm'

--- Modality Counts ---
modality
Vision      97
Language    11
Name: count, dtype: int64

--- Intentional vs Unintentional Counts (Processed) ---
is_intentional
False    107
True       1
Name: count, dtype: int64

--- Contingency Table (Final) ---
          Unintentional  Intentional
modality                            
Language             11            0
Vision               96            1

Chi-Square Statistic: 0.0000
P-Value: 1.0000e+00
Warning: Some expected frequencies are < 5. Fisher's Exact Test is recommended.
Fisher's Exact Test P-Value: 1.0000e+00
Result: No statistically significant difference found.

--- Intentional Harm Rates ---
modality
Language    0.000000
Vision      1.030928
dtype: float64


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot.
*   **Purpose:** This plot is designed to compare a quantitative metric (the rate of intentional harm) across two distinct categorical groups (AI modalities: Language and Vision).

### 2. Axes
*   **X-Axis:**
    *   **Title:** "AI Modality"
    *   **Categories:** The axis displays two discrete categories: "Language" and "Vision".
*   **Y-Axis:**
    *   **Title:** "Percentage of Intentional Incidents (%)"
    *   **Range:** The scale ranges from **0.0 to 1.2**, with tick marks at intervals of 0.2 (0.0, 0.2, 0.4, ... 1.2).
    *   **Units:** The data is presented as a percentage.

### 3. Data Trends
*   **Highest Value:** The "Vision" category represents the tallest bar (and the only visible bar), indicating a non-zero rate of intentional incidents.
*   **Lowest Value:** The "Language" category has no visible bar height, indicating a value of zero.
*   **Pattern:** There is a stark contrast between the two modalities. While Language AI models show no recorded intentional harm incidents in this dataset, Vision AI models show a measurable presence of such incidents.

### 4. Annotations and Legends
*   **Data Labels:**
    *   Above the **Language** category, there is an annotation reading **"0.0%"**, confirming the absence of recorded incidents for this group.
    *   Above the **Vision** category, there is an annotation reading **"1.0%"**, providing the exact value for the bar height.
*   **Legend:** There is no separate legend provided, as the x-axis labels clearly identify the categories corresponding to the data.

### 5. Statistical Insights
*   **Discrepancy in Risk:** The data indicates a significant discrepancy in the rate of intentional harm between the two modalities. Vision-based AI systems appear to be the sole contributor to intentional harm incidents in this specific comparison, with a rate of **1.0%**.
*   **Safety Profile:** Language models, with a rate of **0.0%**, demonstrate a "safer" profile regarding intentional harm in the context of this specific metric and dataset compared to their Vision counterparts.
*   **Magnitude:** While the visual difference is dramatic due to the comparison with zero, the absolute rate for Vision models is still relatively low at 1.0% (one out of every hundred incidents). However, relative to Language models, the risk is infinitely higher.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
