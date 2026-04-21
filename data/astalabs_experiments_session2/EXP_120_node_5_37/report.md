# Experiment 120: node_5_37

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_37` |
| **ID in Run** | 120 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T06:43:04.674624+00:00 |
| **Runtime** | 289.9s |
| **Parent** | `node_4_30` |
| **Children** | `node_6_46` |
| **Creation Index** | 121 |

---

## Hypothesis

> The 'Generative-Harm Specificity': Incidents involving Generative AI
technologies (LLMs, Diffusion) are significantly more likely to involve
'Reputational' or 'Psychological' harm compared to Predictive AI, which is more
strongly associated with 'Allocative' or 'Economic' harm.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7984 (Likely True) |
| **Posterior** | 0.7665 (Likely True) |
| **Surprise** | -0.0383 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 7.0 |
| Maybe True | 23.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 60.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Determine if the rise of Generative AI is shifting the nature of harm towards reputational and psychological domains.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `aiid_incidents`.
- 2. Parse `Known AI Technology` to classify incidents as 'Generative' (keywords: GPT, LLM, diffusion, GAN, chat) or 'Predictive/Discriminative' (keywords: regression, classifier, vision, recognition).
- 3. Categorize `Harm Domain` into 'Psych/Reputational' vs 'Economic/Allocative'.
- 4. Create a contingency table and run a Fisher's Exact Test or Chi-square test.

### Deliverables
- 1. Bar chart of Harm Domains split by AI Technology type.
- 2. Statistical test results verifying the association.

---

## Analysis

The experiment tested the 'Generative-Harm Specificity' hypothesis, which posits
that Generative AI incidents are more likely to cause
'Psychological/Reputational' harm, whereas Predictive AI is more associated with
'Economic/Allocative' harm.

**Methodology:**
Due to the absence of structured labels in the 'Harm Domain' column (which
contained boolean values), the analysis relied on keyword-based classification
of incident descriptions to categorize harm types. Similarly, 'Known AI
Technology' was parsed to distinguish 'Generative' (e.g., LLMs, Diffusion) from
'Predictive' (e.g., Classification, Regression) systems.

**Findings:**
1.  **Distribution:** The data showed a clear directional trend supporting the
hypothesis.
    *   **Generative AI (n=16):** 93.8% of incidents were classified as
Psychological/Reputational, with only 6.2% (1 case) involving
Economic/Allocative harm.
    *   **Predictive AI (n=42):** While also predominantly Psychological
(69.0%), Predictive AI showed a much stronger association with
Economic/Allocative harm (31.0%) compared to Generative AI.

2.  **Statistical Significance:** A Fisher's Exact Test yielded a **p-value of
0.0836**. While this falls short of the standard significance threshold (p <
0.05), it suggests a strong potential association that is likely constrained by
the small sample size (n=58).

**Conclusion:**
The hypothesis is **directionally supported** but **statistically inconclusive**
at the 95% confidence level. The data indicates that while Generative AI
incidents in this dataset almost exclusively impact reputation or psychological
well-being, Predictive AI incidents have a notably higher propensity for
economic impact.

---

## Review

The experiment was successfully executed. The programmer correctly adapted to
data limitations (missing structured labels for harm types) by implementing a
keyword-based classification on incident descriptions. The heuristic for
classifying AI technology and harm domains was logical. The statistical analysis
(Fisher's Exact Test) was appropriate for the small sample size (n=58). Although
the result (p=0.08) did not meet the standard significance threshold (p<0.05),
the experiment provided a valid test of the hypothesis.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Dataset not found.")
        sys.exit(1)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# 1. Classify AI Technology
def classify_tech(tech_str):
    if pd.isna(tech_str):
        return 'Unknown'
    tech_str = str(tech_str).lower()
    
    gen_keywords = ['generative', 'gan', 'gpt', 'llm', 'diffusion', 'transformer', 'chatbot', 'language model', 'text generation', 'image generation', 'content generation', 'synthes']
    pred_keywords = ['regression', 'classification', 'classifier', 'recognition', 'detection', 'predictive', 'recommendation', 'filtering', 'scoring', 'assessment', 'computer vision', 'clustering']
    
    # Prioritize Generative because some predictive systems use transformers now, but in AIID context usually implies GenAI if explicit
    if any(k in tech_str for k in gen_keywords):
        return 'Generative'
    elif any(k in tech_str for k in pred_keywords):
        return 'Predictive'
    else:
        return 'Other'

aiid_df['Tech_Type'] = aiid_df['Known AI Technology'].apply(classify_tech)

# 2. Classify Harm Domain from Description (since structured labels are missing/boolean)
def classify_harm(text):
    if pd.isna(text):
        return 'Unknown'
    text = str(text).lower()
    
    # Keywords for Psych/Reputational
    psych_keywords = [
        'reputation', 'defamation', 'slander', 'libel', 'bias', 'discrimination', 'racist', 'sexist', 
        'slur', 'offensive', 'toxic', 'harassment', 'bullying', 'psychological', 'mental', 'stress', 
        'anxiety', 'trauma', 'dignity', 'privacy', 'surveillance', 'shaming', 'stereotype', 'misinformation', 'hallucination'
    ]
    
    # Keywords for Economic/Allocative
    econ_keywords = [
        'economic', 'financial', 'monetary', 'money', 'job', 'employment', 'hiring', 'firing', 
        'credit', 'loan', 'insurance', 'benefits', 'welfare', 'housing', 'allocation', 'resource', 
        'opportunity', 'access', 'price', 'market', 'fraud', 'theft', 'loss of funds', 'payment'
    ]
    
    has_psych = any(k in text for k in psych_keywords)
    has_econ = any(k in text for k in econ_keywords)
    
    if has_psych and not has_econ:
        return 'Psychological/Reputational'
    elif has_econ and not has_psych:
        return 'Economic/Allocative'
    elif has_psych and has_econ:
        return 'Mixed'
    else:
        return 'Other'

# Apply harm classification on 'description' column
aiid_df['Harm_Class'] = aiid_df['description'].apply(classify_harm)

# Filter out Unknown/Other/Mixed for cleaner analysis, or keep them if sample size is too small
analysis_df = aiid_df[
    (aiid_df['Tech_Type'].isin(['Generative', 'Predictive'])) & 
    (aiid_df['Harm_Class'].isin(['Psychological/Reputational', 'Economic/Allocative']))
]

# Create Contingency Table
contingency_table = pd.crosstab(analysis_df['Tech_Type'], analysis_df['Harm_Class'])

print("--- Contingency Table ---")
print(contingency_table)

# Perform Statistical Test
if contingency_table.size >= 4:
    # Fisher's exact test for 2x2, Chi2 for larger (though we filtered to 2x2)
    if contingency_table.shape == (2, 2):
        oddsratio, pvalue = stats.fisher_exact(contingency_table)
        test_name = "Fisher's Exact Test"
    else:
        chi2, pvalue, dof, expected = stats.chi2_contingency(contingency_table)
        test_name = "Chi-Square Test"
        
    print(f"\n{test_name} Results:")
    print(f"P-value: {pvalue:.5f}")
    if pvalue < 0.05:
        print("Result: Statistically Significant Association")
    else:
        print("Result: No Statistically Significant Association")
else:
    print("\nNot enough data for statistical test.")

# Plotting
if not contingency_table.empty:
    # Calculate percentages for better comparison
    contingency_pct = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100
    
    ax = contingency_pct.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#1f77b4', '#ff7f0e'])
    plt.title('Harm Domain Distribution by AI Technology Type')
    plt.xlabel('AI Technology Type')
    plt.ylabel('Percentage of Incidents')
    plt.legend(title='Harm Domain', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Add counts to bars
    for i, (idx, row) in enumerate(contingency_table.iterrows()):
        total = row.sum()
        if total > 0:
            y_pos = 0
            for col in contingency_table.columns:
                count = row[col]
                pct = (count / total) * 100
                if count > 0:
                    plt.text(i, y_pos + pct/2, f"{count}\n({pct:.1f}%)", ha='center', va='center', color='white')
                y_pos += pct
    
    plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Contingency Table ---
Harm_Class  Economic/Allocative  Psychological/Reputational
Tech_Type                                                  
Generative                    1                          15
Predictive                   13                          29

Fisher's Exact Test Results:
P-value: 0.08360
Result: No Statistically Significant Association


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** 100% Stacked Bar Chart.
*   **Purpose:** This plot compares the proportional distribution of different "Harm Domains" (categories of negative impact) across two distinct types of AI technology ("Generative" and "Predictive"). By normalizing the height of the bars to 100%, it allows for an easy comparison of the *ratio* of harm types within each technology category, regardless of the total number of incidents.

### 2. Axes
*   **X-axis:**
    *   **Title:** "AI Technology Type".
    *   **Categories:** Two discrete categories labeled "Generative" and "Predictive".
*   **Y-axis:**
    *   **Title:** "Percentage of Incidents".
    *   **Range:** 0 to 100 (representing 0% to 100%).
    *   **Increments:** Marks are placed every 20 units (0, 20, 40, 60, 80, 100).

### 3. Data Trends
*   **Dominant Category:** Across both AI technology types, "Psychological/Reputational" harm (represented by the orange bars) is the majority category.
*   **Generative AI:** This bar is overwhelmingly dominated by the orange segment. It shows a very high concentration of "Psychological/Reputational" harm compared to a negligible amount of "Economic/Allocative" harm.
*   **Predictive AI:** While still led by "Psychological/Reputational" harm, this bar shows a much more significant proportion of "Economic/Allocative" harm (the blue segment) compared to the Generative bar. The distribution is roughly 70/30.

### 4. Annotations and Legends
*   **Legend:** Located on the right side, titled "Harm Domain."
    *   **Blue:** Represents "Economic/Allocative" harm.
    *   **Orange:** Represents "Psychological/Reputational" harm.
*   **Data Labels (Annotations within bars):**
    *   **Generative / Economic (Blue):** Count: 1, Percentage: 6.2%.
    *   **Generative / Psychological (Orange):** Count: 15, Percentage: 93.8%.
    *   **Predictive / Economic (Blue):** Count: 13, Percentage: 31.0%.
    *   **Predictive / Psychological (Orange):** Count: 29, Percentage: 69.0%.

### 5. Statistical Insights
*   **Technology-Specific Risks:** There is a clear correlation between the type of AI and the type of harm produced.
    *   **Generative AI** incidents in this dataset are almost exclusively associated with **Psychological or Reputational** damage (93.8%). Economic harm is an outlier in this category (only 6.2%).
    *   **Predictive AI** poses a significantly higher risk of **Economic or Allocative** harm (31.0%) compared to Generative AI. This suggests that predictive systems (likely used for decision-making in finance, hiring, or resource distribution) are more prone to affecting people's economic standing than generative systems are.
*   **Sample Size:** While the bars appear equal in height due to the percentage scaling, the raw counts indicate that the "Predictive" dataset is larger (42 total incidents: 13 + 29) compared to the "Generative" dataset (16 total incidents: 1 + 15).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
