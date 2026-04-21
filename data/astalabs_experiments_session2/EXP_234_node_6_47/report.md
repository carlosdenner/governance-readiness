# Experiment 234: node_6_47

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_47` |
| **ID in Run** | 234 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T11:57:30.700680+00:00 |
| **Runtime** | 432.5s |
| **Parent** | `node_5_29` |
| **Children** | None |
| **Creation Index** | 235 |

---

## Hypothesis

> Technology-Driven Autonomy: AI incidents involving 'Robotics' technologies
exhibit significantly higher autonomy levels than incidents involving 'Language'
or 'Computer Vision' technologies.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.6190 (Maybe True) |
| **Surprise** | -0.1475 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
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
| Uncertain | 32.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Investigate the relationship between AI technology type and system autonomy in real-world failure scenarios.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'aiid_incidents'.
- 2. Clean '84_Known AI Technology': Categorize entries into 'Robotics', 'Language' (NLP, Text), 'Vision' (CV, Image), and 'Other'.
- 3. Clean '81_Autonomy Level': Map string values (e.g., 'Low', 'High') to an ordinal numeric scale.
- 4. Perform a Kruskal-Wallis H-test to compare the distributions of Autonomy scores across the Technology categories.

### Deliverables
- Kruskal-Wallis statistic, p-value, and a boxplot of Autonomy Levels by Technology.

---

## Analysis

The experiment successfully processed the dataset but encountered a critical
data limitation that prevented a full test of the hypothesis.

1. **Data Availability**: The keyword-based classification identified 49
incidents for 'Computer Vision' and 13 for 'Language', but **zero** incidents
for 'Robotics'. Despite using an extensive list of robotics-related keywords
(e.g., 'drone', 'vehicle', 'tesla', 'robot'), no matches were found in the
'Known AI Technology' column of the provided AIID subset.

2. **Statistical Findings**: A Kruskal-Wallis H-test was performed on the
available groups ('Computer Vision' vs. 'Language'). The result (H=0.737,
p=0.391) indicated **no statistically significant difference** in autonomy
levels between these two domains.

3. **Visual Evidence**: The generated boxplot confirms that both Computer Vision
and Language incidents exhibit similar autonomy profiles, primarily ranging
between scores of 1 (Assistive/Low) and 3 (Conditional/Human-in-the-loop), with
no data visible for Robotics.

4. **Conclusion**: The hypothesis is **not supported**. The absence of Robotics
data prevents the primary comparison, and the secondary comparison between
Vision and Language revealed no significant distinction in autonomy levels.

---

## Review

The experiment was faithfully implemented. The programmer successfully adapted
to data quality issues (incorrect column names and non-standard value formats
like 'Autonomy1') to extract usable data. However, the hypothesis could not be
fully validated due to a lack of relevant data in the dataset (0 Robotics
incidents found). The analysis was correctly pivoted to compare the available
groups.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
import numpy as np
import re
import os

# Load dataset
filename = 'astalabs_discovery_all_data.csv'
filepath = filename if os.path.exists(filename) else os.path.join('..', filename)

print(f"Loading dataset from {filepath}...")
try:
    df = pd.read_csv(filepath, low_memory=False)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid_df)} rows")

# Identify columns
cols = aiid_df.columns
autonomy_col = next((c for c in cols if 'Autonomy Level' in c), None)
tech_col = next((c for c in cols if 'Known AI Technology' in c), None)

print(f"Using columns: Autonomy='{autonomy_col}', Tech='{tech_col}'")

# --- 1. Clean Autonomy Level ---
# Observed values: "Autonomy1", "Autonomy3", etc.
def map_autonomy(val):
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    
    # Regex to capture "Autonomy" followed by a digit
    match = re.search(r'Autonomy(\d+)', val_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Fallback: simple digit check if regex fails
    if val_str.isdigit():
        return int(val_str)
        
    return None

if autonomy_col:
    aiid_df['Autonomy_Score'] = aiid_df[autonomy_col].apply(map_autonomy)
else:
    aiid_df['Autonomy_Score'] = np.nan

# --- 2. Clean Technology ---
# Observed values: "Transformer", "Face Detection", "Visual Object Detection, Image Segmentation"
def map_technology(val):
    if pd.isna(val):
        return 'Other'
    val_str = str(val).lower()
    
    # Vision Keywords
    vision_keys = ['vision', 'image', 'face', 'facial', 'camera', 'video', 'detection', 'segmentation', 'recognition', 'ocr', 'optical character']
    # Language Keywords
    lang_keys = ['language', 'text', 'nlp', 'translation', 'transformer', 'bert', 'gpt', 'llm', 'chat', 'dialogue', 'document', 'summary', 'speech', 'voice']
    # Robotics Keywords (Physical agents)
    robot_keys = ['robot', 'drone', 'vehicle', 'car', 'autonomous driving', 'self-driving', 'uav', 'physical', 'manipulation', 'navigation', 'tesla', 'waymo', 'cruise']

    # Check Robotics first (often involves vision, but distinct by physical nature)
    if any(k in val_str for k in robot_keys):
        return 'Robotics'
    # Check Vision
    elif any(k in val_str for k in vision_keys):
        return 'Computer Vision'
    # Check Language
    elif any(k in val_str for k in lang_keys):
        return 'Language'
    else:
        return 'Other'

if tech_col:
    aiid_df['Tech_Category'] = aiid_df[tech_col].apply(map_technology)
else:
    aiid_df['Tech_Category'] = 'Other'

# --- 3. Analysis Filter ---
analysis_df = aiid_df.dropna(subset=['Autonomy_Score']).copy()
analysis_df = analysis_df[analysis_df['Tech_Category'].isin(['Robotics', 'Computer Vision', 'Language'])]

print(f"\nData points available for analysis: {len(analysis_df)}")
counts = analysis_df['Tech_Category'].value_counts()
print(counts)

# --- 4. Statistical Test & Visualization ---
groups = [analysis_df[analysis_df['Tech_Category'] == t]['Autonomy_Score'].values for t in ['Robotics', 'Computer Vision', 'Language']]

# Only proceed if we have data for comparison
if len(analysis_df) > 0 and len(groups) == 3:
    # Perform Kruskal-Wallis only if each group has at least one sample, otherwise print warning
    valid_groups = [g for g in groups if len(g) > 0]
    if len(valid_groups) > 1:
        stat, p = kruskal(*valid_groups)
        print(f"\nKruskal-Wallis H-test results:")
        print(f"Statistic: {stat:.4f}")
        print(f"p-value: {p:.4e}")
        if p < 0.05:
            print("Result: Significant difference detected.")
        else:
            print("Result: No significant difference.")
    else:
        print("\nNot enough groups for statistical comparison.")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Tech_Category', y='Autonomy_Score', data=analysis_df, 
                order=['Robotics', 'Computer Vision', 'Language'], palette='viridis')
    plt.title('Distribution of Autonomy Levels by AI Technology')
    plt.ylabel('Autonomy Score')
    plt.xlabel('Technology Domain')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
else:
    print("\nInsufficient data to generate plot or stats.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
AIID Incidents loaded: 1362 rows
Using columns: Autonomy='Autonomy Level', Tech='Known AI Technology'

Data points available for analysis: 62
Tech_Category
Computer Vision    49
Language           13
Name: count, dtype: int64

Kruskal-Wallis H-test results:
Statistic: 0.7370
p-value: 3.9061e-01
Result: No significant difference.

STDERR:
<ipython-input-1-ef88b3d5934a>:117: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.boxplot(x='Tech_Category', y='Autonomy_Score', data=analysis_df,


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Box Plot (also known as a box-and-whisker plot).
*   **Purpose:** The plot is designed to visualize and compare the distribution of statistical data (likely minimum, first quartile, median, third quartile, and maximum) for a continuous variable ("Autonomy Score") across distinct categorical groups ("Technology Domain").

### 2. Axes
*   **X-axis:**
    *   **Label:** "Technology Domain".
    *   **Categories:** The axis displays three specific domains: "Robotics", "Computer Vision", and "Language".
*   **Y-axis:**
    *   **Label:** "Autonomy Score".
    *   **Value Range:** The visible markings range from **1.00 to 3.00**, with major intervals marked every 0.25 units.

### 3. Data Trends
*   **Robotics:**
    *   There is **no visible data distribution** plotted for the "Robotics" category. The area above this label is empty, indicating either missing data, null values, or that the values fall outside the current plot parameters.
*   **Computer Vision (Teal Box):**
    *   **Range:** The distribution spans from a minimum of **1.0** to a maximum of **3.0** (indicated by the top whisker).
    *   **Interquartile Range (IQR):** The box itself, representing the middle 50% of the data, spans from **1.0 to 2.0**.
    *   **Pattern:** There is no lower whisker visible, suggesting the minimum value and the first quartile (Q1) are identical (both at 1.0).
*   **Language (Light Green Box):**
    *   **Range & IQR:** The distribution for "Language" appears visually identical to "Computer Vision." The box spans from **1.0 to 2.0**, and the top whisker extends to **3.0**.
    *   **Pattern:** Similar to Computer Vision, the lack of a lower whisker suggests a concentration of data at the lower end of the scale (1.0).

### 4. Annotations and Legends
*   **Title:** The chart is titled **"Distribution of Autonomy Levels by AI Technology"** at the top center.
*   **Gridlines:** Horizontal dashed grey lines are provided across the plot area to assist in reading the Y-axis values.
*   **Color Coding:**
    *   **Teal:** Represents Computer Vision.
    *   **Light Green:** Represents Language.
    *   (No legend box is provided, as the X-axis labels serve as the legend).

### 5. Statistical Insights
*   **Uniformity between Vision and Language:** The data suggests that, within this specific dataset or study, **Computer Vision** and **Language** technologies share nearly identical distributions regarding their Autonomy Scores. Both exhibit a variability ranging from score 1 to 3.
*   **Skewness:** The distributions for both Computer Vision and Language appear right-skewed (or positively skewed). Since the "box" (1.0 to 2.0) is at the bottom of the total range (1.0 to 3.0) and there is no bottom whisker, it implies a high density of lower scores (around 1.0), with fewer instances reaching the higher score of 3.0.
*   **Missing Robotics Data:** A key insight is the notable absence of data for **Robotics**. This gap suggests that Robotics was either not measured in this specific experiment, yielded invalid results, or had a sample size of zero.
*   **Ceiling Effect:** The maximum score observed is 3.0, which acts as a hard ceiling for the data presented in the visible domains.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
