# Experiment 138: node_5_45

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_45` |
| **ID in Run** | 138 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T07:30:00.091890+00:00 |
| **Runtime** | 175.9s |
| **Parent** | `node_4_46` |
| **Children** | `node_6_30` |
| **Creation Index** | 139 |

---

## Hypothesis

> The 'Evasion' vs 'Injection' Split: Adversarial cases involving Computer Vision
systems are significantly more likely to employ 'Evasion' tactics, while cases
involving Large Language Models (LLMs) are more likely to employ 'impact' or
'exfiltration' tactics.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.5968 (Maybe True) |
| **Posterior** | 0.3681 (Maybe False) |
| **Surprise** | -0.2744 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 21.0 |
| Uncertain | 0.0 |
| Maybe False | 9.0 |
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

**Objective:** Map adversarial tactics to the underlying AI technology modality (Vision vs Language) using ATLAS case data.

### Steps
- 1. Load 'atlas_cases' from the dataset.
- 2. Use keyword matching on the 'summary' or 'name' columns to classify cases as 'Vision' (e.g., image, face, object) or 'Language' (e.g., text, chat, translation).
- 3. Extract tactics from the 'tactics' column (checking for 'Evasion', 'Exfiltration', etc.).
- 4. Create a contingency table of Modality vs Tactic Category.
- 5. Run a Chi-Square or Fisher's test.

### Deliverables
- Stacked bar chart of tactics by AI modality; Statistical validation of the modality-tactic relationship.

---

## Analysis

The experiment successfully tested the 'Evasion vs Injection' hypothesis using
the ATLAS adversarial case data. The code utilized keyword analysis to
categorize 52 ATLAS cases, identifying 25 relevant records: 15 classified as
'Language' and 10 as 'Vision'.

1. **Hypothesis 1 (Vision -> Evasion)**: The hypothesis predicted that Vision
systems are more likely to employ Evasion tactics. The results contradicted
this, showing a higher evasion rate for Language systems (53.33%) compared to
Vision systems (40.00%). A Fisher's Exact Test (p=0.6882) confirmed the
difference was not statistically significant.

2. **Hypothesis 2 (Language -> Impact/Exfiltration)**: The hypothesis predicted
that Language systems are more likely to employ Impact or Exfiltration tactics.
The data directionally supported this (Language: 86.67% vs. Vision: 80.00%), but
the difference was minimal and not statistically significant (p=1.0000).

Overall, the hypothesis is not supported. The analysis was limited by the small
sample size (n=25), suggesting that while distinct tactic patterns may exist,
the current dataset does not provide sufficient evidence to claim a
statistically significant divergence based on modality.

---

## Review

The experiment was executed perfectly according to the plan. The programmer
correctly filtered the data, implemented robust keyword-based classification for
the modalities, and applied the appropriate statistical test (Fisher's Exact
Test) given the small sample size. The analysis is complete.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

# [debug] # Set up simple debug print
# def debug_print(msg):
#     print(f"[DEBUG] {msg}")

try:
    # Load the dataset
    file_path = '../astalabs_discovery_all_data.csv'
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

    # Filter for ATLAS cases
    atlas_df = df[df['source_table'] == 'atlas_cases'].copy()
    
    print(f"Loaded ATLAS cases: {len(atlas_df)} records")

    # Define keywords for modality classification
    # Combining name and summary for search
    atlas_df['combined_text'] = (atlas_df['name'].fillna('') + ' ' + atlas_df['summary'].fillna('')).str.lower()
    
    vision_keywords = ['image', 'face', 'facial', 'recognition', 'camera', 'video', 'vision', 'pixel', 
                       'surveillance', 'biometric', 'object detection', 'yolo', 'glasses', 'patch', 
                       'traffic', 'sign', 'autonomous', 'driving', 'vehicle', 'tesla', 'lidar']
    
    language_keywords = ['text', 'language', 'translation', 'chat', 'bot', 'gpt', 'bert', 'llm', 
                         'dialogue', 'speech', 'email', 'phishing', 'spam', 'tweet', 'twitter', 
                         'sentiment', 'word', 'translate', 'completion']

    def classify_modality(text):
        is_vision = any(k in text for k in vision_keywords)
        is_language = any(k in text for k in language_keywords)
        
        if is_vision and not is_language:
            return 'Vision'
        elif is_language and not is_vision:
            return 'Language'
        elif is_vision and is_language:
            # Conflict resolution: check counts or default to Multimodal, but for this experiment let's try to discern
            # heuristic: if 'vision' or 'image' appears, it's likely vision even if it has text
            # For now mark as 'Mixed'
            return 'Mixed'
        else:
            return 'Other'

    atlas_df['modality'] = atlas_df['combined_text'].apply(classify_modality)
    
    # Classify tactics
    # Target: Evasion vs (Impact OR Exfiltration)
    atlas_df['tactics'] = atlas_df['tactics'].fillna('').str.lower()
    
    def check_tactic(tactic_str, target_list):
        return any(t in tactic_str for t in target_list)

    # "Evasion" usually appears as "Defense Evasion" in ATLAS or just "Evasion"
    atlas_df['has_evasion'] = atlas_df['tactics'].apply(lambda x: 'evasion' in x)
    # "Impact" or "Exfiltration"
    atlas_df['has_impact_exfil'] = atlas_df['tactics'].apply(lambda x: 'impact' in x or 'exfiltration' in x)

    # Filter for only Vision and Language for the hypothesis test
    analysis_df = atlas_df[atlas_df['modality'].isin(['Vision', 'Language'])].copy()
    
    # Generate Summary Stats
    modality_counts = analysis_df['modality'].value_counts()
    print("\n--- Modality Distribution ---")
    print(modality_counts)
    
    # Create Contingency Table for Evasion
    print("\n--- Hypothesis 1: Vision -> Evasion ---")
    ct_evasion = pd.crosstab(analysis_df['modality'], analysis_df['has_evasion'])
    print(ct_evasion)
    
    # Fisher's Exact Test for Evasion
    # We want to see if Vision has higher rate of True than Language
    # Contingency table structure roughly:
    #           False   True
    # Language  A       B
    # Vision    C       D
    # We compare proportions B/(A+B) vs D/(C+D)
    if 'Vision' in ct_evasion.index and 'Language' in ct_evasion.index:
        odds_ratio, p_value_evasion = stats.fisher_exact(ct_evasion)
        # Note: fisher_exact is for 2x2. 
        # Check if Vision is significantly different from Language
        vision_evasion_rate = analysis_df[analysis_df['modality']=='Vision']['has_evasion'].mean()
        language_evasion_rate = analysis_df[analysis_df['modality']=='Language']['has_evasion'].mean()
        print(f"Vision Evasion Rate: {vision_evasion_rate:.2%}")
        print(f"Language Evasion Rate: {language_evasion_rate:.2%}")
        print(f"Fisher's Exact Test p-value: {p_value_evasion:.4f}")
    else:
        print("Insufficient data for Evasion test.")

    # Create Contingency Table for Impact/Exfiltration
    print("\n--- Hypothesis 2: Language -> Impact/Exfiltration ---")
    ct_impact = pd.crosstab(analysis_df['modality'], analysis_df['has_impact_exfil'])
    print(ct_impact)
    
    if 'Vision' in ct_impact.index and 'Language' in ct_impact.index:
        odds_ratio_imp, p_value_impact = stats.fisher_exact(ct_impact)
        vision_impact_rate = analysis_df[analysis_df['modality']=='Vision']['has_impact_exfil'].mean()
        language_impact_rate = analysis_df[analysis_df['modality']=='Language']['has_impact_exfil'].mean()
        print(f"Vision Impact/Exfil Rate: {vision_impact_rate:.2%}")
        print(f"Language Impact/Exfil Rate: {language_impact_rate:.2%}")
        print(f"Fisher's Exact Test p-value: {p_value_impact:.4f}")
    else:
        print("Insufficient data for Impact/Exfil test.")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Evasion
    if not ct_evasion.empty:
        # Normalize to get percentages
        ct_evasion_norm = pd.crosstab(analysis_df['modality'], analysis_df['has_evasion'], normalize='index')
        if True in ct_evasion_norm.columns:
            ct_evasion_norm[True].plot(kind='bar', ax=axes[0], color=['orange', 'blue'], alpha=0.7)
        else:
             # Handle case where no True values exist
             ct_evasion_norm.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Rate of Evasion Tactics by Modality')
        axes[0].set_ylabel('Proportion of Cases')
        axes[0].set_ylim(0, 1.0)

    # Plot 2: Impact/Exfil
    if not ct_impact.empty:
        ct_impact_norm = pd.crosstab(analysis_df['modality'], analysis_df['has_impact_exfil'], normalize='index')
        if True in ct_impact_norm.columns:
            ct_impact_norm[True].plot(kind='bar', ax=axes[1], color=['orange', 'blue'], alpha=0.7)
        else:
             ct_impact_norm.plot(kind='bar', ax=axes[1])
        axes[1].set_title('Rate of Impact/Exfiltration by Modality')
        axes[1].set_ylabel('Proportion of Cases')
        axes[1].set_ylim(0, 1.0)

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded ATLAS cases: 52 records

--- Modality Distribution ---
modality
Language    15
Vision      10
Name: count, dtype: int64

--- Hypothesis 1: Vision -> Evasion ---
has_evasion  False  True 
modality                 
Language         7      8
Vision           6      4
Vision Evasion Rate: 40.00%
Language Evasion Rate: 53.33%
Fisher's Exact Test p-value: 0.6882

--- Hypothesis 2: Language -> Impact/Exfiltration ---
has_impact_exfil  False  True 
modality                      
Language              2     13
Vision                2      8
Vision Impact/Exfil Rate: 80.00%
Language Impact/Exfil Rate: 86.67%
Fisher's Exact Test p-value: 1.0000


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** The image displays **two side-by-side bar plots**.
*   **Purpose:** The plots compare the proportion of cases exhibiting specific behaviors ("Evasion Tactics" on the left and "Impact/Exfiltration" on the right) across two different AI modalities ("Language" and "Vision").

### 2. Axes
*   **X-Axis:**
    *   **Label:** "modality"
    *   **Categories:** Two discrete categories, "Language" and "Vision". The labels are oriented vertically.
*   **Y-Axis:**
    *   **Label:** "Proportion of Cases"
    *   **Units:** The scale is a normalized proportion ranging from 0.0 to 1.0 (representing 0% to 100%).
    *   **Range:** The axis ticks are marked at 0.0, 0.2, 0.4, 0.6, 0.8, and 1.0.

### 3. Data Trends
*   **Left Plot (Rate of Evasion Tactics):**
    *   **Language (Orange Bar):** This is the tallest bar in this plot, reaching approximately **0.53** (or 53%).
    *   **Vision (Blue Bar):** This bar is shorter, sitting exactly at the **0.40** (40%) line.
    *   **Trend:** Evasion tactics are observed more frequently in the Language modality compared to Vision.

*   **Right Plot (Rate of Impact/Exfiltration):**
    *   **Language (Orange Bar):** This is the tallest bar, reaching approximately **0.87** (or 87%).
    *   **Vision (Blue Bar):** This bar reaches exactly the **0.80** (80%) line.
    *   **Trend:** Similar to the left plot, the Language modality shows a higher rate than Vision, though the gap is narrower. Additionally, the absolute values for both modalities are significantly higher in this category compared to Evasion Tactics.

### 4. Annotations and Legends
*   **Titles:**
    *   Left Chart: "Rate of Evasion Tactics by Modality"
    *   Right Chart: "Rate of Impact/Exfiltration by Modality"
*   **Color Coding:** While there is no explicit legend box, the color usage is consistent:
    *   **Orange:** Represents "Language" modality.
    *   **Blue/Purple:** Represents "Vision" modality.

### 5. Statistical Insights
*   **Prevalence of Behavior:** Impact/Exfiltration behaviors are far more common than Evasion Tactics across both modalities. While evasion rates hover between 40-53%, impact/exfiltration rates are very high, between 80-87%.
*   **Modality Risk Factor:** The **Language modality consistently exhibits higher rates** for both metric categories compared to the Vision modality.
    *   In *Evasion Tactics*, Language exceeds Vision by approximately 13 percentage points.
    *   In *Impact/Exfiltration*, Language exceeds Vision by approximately 7 percentage points.
*   **Conclusion:** Based on this data, text-based (Language) models appear to be more prone to—or more frequently detected performing—these specific adversarial or failure behaviors than image-based (Vision) models.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
