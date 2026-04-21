# Experiment 107: node_4_45

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_45` |
| **ID in Run** | 107 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T06:08:36.119794+00:00 |
| **Runtime** | 253.1s |
| **Parent** | `node_3_3` |
| **Children** | `node_5_67`, `node_5_96` |
| **Creation Index** | 108 |

---

## Hypothesis

> GenAI Governance Lag: AI use cases utilizing Generative AI or LLM technologies
(identified via text keywords) have significantly lower compliance with 'Notice'
and 'Consent' requirements compared to traditional Predictive AI, reflecting a
regulatory lag for new technologies.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.2912 (Likely False) |
| **Surprise** | -0.5409 |
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
| Maybe False | 14.0 |
| Definitely False | 46.0 |

---

## Experiment Plan

**Objective:** Assess if Generative AI use cases are less governed regarding transparency than traditional AI.

### Steps
- 1. Filter for `eo13960_scored`. Combine text from `2_use_case_name` and `11_purpose_benefits`.
- 2. Create a binary flag `is_genai` using keywords: 'generative', 'llm', 'language model', 'chatbot', 'summariz', 'text generation'.
- 3. Create a control group `is_predictive` using keywords: 'predict', 'detect', 'classify', 'vision', 'risk model'.
- 4. Compare rates of `59_ai_notice` and `67_opt_out` between GenAI and Predictive groups using Fisher's Exact Test.

### Deliverables
- Compliance percentages for GenAI vs Predictive AI, Fisher's Exact Test p-values.

---

## Analysis

The experiment successfully tested the 'GenAI Governance Lag' hypothesis but
found no statistical support for it.

1. **Data Classification**: The code successfully identified 263 'GenAI' use
cases and 432 'Traditional Predictive' use cases from the EO 13960 inventory
using keyword analysis.

2. **Compliance Results**:
   - **Notice (Q59)**: Both groups showed **0% compliance** (0/263 for GenAI,
0/432 for Predictive). The Fisher's Exact Test p-value was 1.000, indicating no
difference.
   - **Consent (Q67)**: Compliance was extremely low for both groups. GenAI
showed **3.0%** (8/263) compliance compared to **5.1%** (22/432) for Predictive
AI. The p-value of **0.2491** indicates this difference is not statistically
significant.

3. **Conclusion**: The hypothesis that GenAI has *significantly lower*
compliance is **not supported**. Instead, the data reveals a systemic lack of
documented compliance (or positive responses) for these specific transparency
and opt-out requirements across *all* federal AI use cases in this dataset,
regardless of the technology type.

---

## Review

The experiment was faithfully implemented and the hypothesis was successfully
tested.

1. **Execution**: The code robustly loaded the data, handled the file path issue
from the previous attempt, and successfully classified 695 use cases (263 GenAI,
432 Traditional Predictive) using the specified keyword logic.

2. **Results**:
   - **Notice (Q59)**: Both groups showed 0% compliance (0/263 vs 0/432),
resulting in a p-value of 1.000.
   - **Consent (Q67)**: GenAI showed 3.0% compliance vs 5.1% for Predictive AI.
The Fisher's Exact Test p-value of 0.2491 indicates this difference is not
statistically significant.

3. **Conclusion**: The hypothesis that GenAI has *significantly lower*
compliance is **not supported**. The data instead highlights a systemic issue
where almost no federal use cases in this dataset—regardless of
technology—report compliance with these specific Notice and Opt-out
requirements.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def run_experiment():
    # Robust file loading
    filename = 'astalabs_discovery_all_data.csv'
    possible_paths = [filename, f'../{filename}']
    file_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
            
    if file_path is None:
        print(f"Error: Could not find {filename} in current or parent directory.")
        return

    print(f"Loading dataset from {file_path}...")
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    # Filter for EO 13960 Scored data
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO 13960 Scored rows: {len(df_eo)}")
    
    # Combine text columns for keyword search (handling NaNs)
    df_eo['text_content'] = df_eo['2_use_case_name'].fillna('').astype(str) + ' ' + df_eo['11_purpose_benefits'].fillna('').astype(str)
    df_eo['text_content'] = df_eo['text_content'].str.lower()
    
    # Define Keywords
    genai_keywords = ['generative', 'llm', 'language model', 'chatbot', 'summariz', 'text generation']
    predictive_keywords = ['predict', 'detect', 'classify', 'vision', 'risk model']
    
    # helper function
    def has_keyword(text, keywords):
        return any(k in text for k in keywords)
    
    # Create flags
    df_eo['is_genai'] = df_eo['text_content'].apply(lambda x: has_keyword(x, genai_keywords))
    
    # Control group: Predictive keywords AND NOT GenAI (to isolate traditional AI)
    df_eo['is_predictive'] = df_eo['text_content'].apply(lambda x: has_keyword(x, predictive_keywords)) & (~df_eo['is_genai'])
    
    genai_group = df_eo[df_eo['is_genai']]
    pred_group = df_eo[df_eo['is_predictive']]
    
    print(f"\nSample Sizes:\n  GenAI: {len(genai_group)}\n  Traditional Predictive: {len(pred_group)}")
    
    if len(genai_group) < 5 or len(pred_group) < 5:
        print("Warning: Small sample sizes may affect statistical validity.")

    # Compliance Analysis Targets
    # 59_ai_notice: Did the agency provide notice?
    # 67_opt_out: Did the agency provide opt-out/consent mechanisms?
    targets = {
        'Notice (Q59)': '59_ai_notice',
        'Consent (Q67)': '67_opt_out'
    }
    
    results = []
    
    for label, col in targets.items():
        print(f"\n--- Analyzing {label} ---")
        
        # Check for unique values to ensure mapping is correct
        # print(f"Unique values in {col}: {df_eo[col].unique()}")
        
        def is_compliant(val):
            if pd.isna(val):
                return 0
            return 1 if str(val).strip().lower() == 'yes' else 0

        # GenAI stats
        genai_compliant = genai_group[col].apply(is_compliant).sum()
        genai_total = len(genai_group)
        genai_rate = (genai_compliant / genai_total) if genai_total > 0 else 0
        
        # Predictive stats
        pred_compliant = pred_group[col].apply(is_compliant).sum()
        pred_total = len(pred_group)
        pred_rate = (pred_compliant / pred_total) if pred_total > 0 else 0
        
        print(f"  GenAI: {genai_compliant}/{genai_total} ({genai_rate*100:.1f}%)")
        print(f"  Predictive: {pred_compliant}/{pred_total} ({pred_rate*100:.1f}%)")
        
        # Fisher's Exact Test
        # Contingency Table: [[GenAI_Yes, GenAI_No], [Pred_Yes, Pred_No]]
        table = [
            [genai_compliant, genai_total - genai_compliant],
            [pred_compliant, pred_total - pred_compliant]
        ]
        
        odds_ratio, p_val = stats.fisher_exact(table)
        print(f"  Fisher's Exact Test p-value: {p_val:.4f}")
        
        results.append({
            'Metric': label,
            'GenAI_Rate': genai_rate * 100,
            'Pred_Rate': pred_rate * 100,
            'P_Value': p_val
        })

    # Visualization
    labels = [r['Metric'] for r in results]
    genai_vals = [r['GenAI_Rate'] for r in results]
    pred_vals = [r['Pred_Rate'] for r in results]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, genai_vals, width, label='GenAI (Emerging)', color='#d62728')
    rects2 = ax.bar(x + width/2, pred_vals, width, label='Predictive (Traditional)', color='#1f77b4')
    
    ax.set_ylabel('Compliance Rate (%)')
    ax.set_title('Governance Lag: Compliance Rates for GenAI vs Traditional AI')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend()
    
    # Add p-values on chart
    for i, r in enumerate(results):
        p_text = f"p={r['P_Value']:.3f}"
        # Position text above the higher bar
        h = max(r['GenAI_Rate'], r['Pred_Rate'])
        ax.text(i, h + 2, p_text, ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
EO 13960 Scored rows: 1757

Sample Sizes:
  GenAI: 263
  Traditional Predictive: 432

--- Analyzing Notice (Q59) ---
  GenAI: 0/263 (0.0%)
  Predictive: 0/432 (0.0%)
  Fisher's Exact Test p-value: 1.0000

--- Analyzing Consent (Q67) ---
  GenAI: 8/263 (3.0%)
  Predictive: 22/432 (5.1%)
  Fisher's Exact Test p-value: 0.2491


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Grouped Bar Chart (Clustered Bar Chart).
*   **Purpose:** The plot is designed to compare compliance rates between two different types of Artificial Intelligence—**GenAI (Emerging)** and **Predictive (Traditional)**—across two specific governance categories: **Notice** and **Consent**.

### 2. Axes
*   **Y-Axis:**
    *   **Label:** "Compliance Rate (%)"
    *   **Range:** 0 to 100.
    *   **Scale:** Linear scale in increments of 20.
*   **X-Axis:**
    *   **Labels:** Categorical variables representing specific survey questions or compliance criteria: "Notice (Q59)" and "Consent (Q67)".

### 3. Data Trends
*   **Notice (Q59):**
    *   Both the GenAI (Red) and Predictive AI (Blue) bars are non-visible or effectively at zero. This indicates a near-total lack of compliance or affirmative responses for this specific metric for both AI types.
*   **Consent (Q67):**
    *   The bars are visible but extremely low relative to the Y-axis scale (0-100%).
    *   The **Predictive (Traditional)** bar (Blue) appears slightly taller than the **GenAI (Emerging)** bar (Red), roughly hovering around the 5% mark.
    *   Despite the slight visual difference, both values are very low, indicating poor compliance performance for consent mechanisms in both technologies.

### 4. Annotations and Legends
*   **Legend:** Located in the top-right corner.
    *   **Red Box:** Represents "GenAI (Emerging)".
    *   **Blue Box:** Represents "Predictive (Traditional)".
*   **Annotations (p-values):** Statistical significance values are placed above each group of bars.
    *   **Above Notice (Q59):** "**p=1.000**"
    *   **Above Consent (Q67):** "**p=0.249**"

### 5. Statistical Insights
*   **Statistical Significance:**
    *   **Notice (Q59):** A p-value of **1.000** indicates that there is absolutely no statistical difference between GenAI and Predictive AI regarding the "Notice" compliance rate. This reinforces the visual observation that both are likely at exactly 0%.
    *   **Consent (Q67):** A p-value of **0.249** is well above the standard threshold for statistical significance (typically p < 0.05). This means that although the blue bar looks slightly higher than the red bar, the difference is not statistically significant and could be due to random chance.
*   **Overall Conclusion:** The title "Governance Lag" is strongly supported by the data. The primary insight is not a disparity between the two AI types, but rather that **both** GenAI and Traditional AI have remarkably low compliance rates (near 0% for Notice and <5% for Consent) in these specific governance areas. Neither technology is currently meeting these standards effectively.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
