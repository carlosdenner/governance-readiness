# Experiment 271: node_6_72

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_72` |
| **ID in Run** | 271 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T13:54:35.838184+00:00 |
| **Runtime** | 363.2s |
| **Parent** | `node_5_83` |
| **Children** | None |
| **Creation Index** | 272 |

---

## Hypothesis

> The 'Physicality' of Failure Severity: Incidents involving 'Robotics' or
'Autonomous Vehicles' have a significantly higher 'Harm Severity Score' (derived
from tangible harm descriptions) compared to 'Generative AI' incidents, which
are voluminous but lower-severity.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9758 (Definitely True) |
| **Posterior** | 0.9918 (Definitely True) |
| **Surprise** | +0.0191 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 29.0 |
| Maybe True | 1.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 60.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Quantify and compare harm severity across AI technologies.

### Steps
- 1. Load `aiid_incidents`.
- 2. Create a 'Severity Score' ordinal variable based on `Tangible Harm`: 3 (Death/Severe Injury), 2 (Property Damage/Financial Loss), 1 (Intangible/Near Miss), 0 (None/Unclear).
- 3. Categorize `Known AI Technology` into 'Robotics/AV' vs 'Generative/Language'.
- 4. Compare the distribution of Severity Scores using a Mann-Whitney U test (non-parametric).

### Deliverables
- Severity score distributions (histograms/boxplots), Mann-Whitney U test statistics, and median severity for each technology group.

---

## Analysis

The experiment successfully validated the hypothesis that 'Robotics/AV'
incidents are associated with significantly higher harm severity scores compared
to 'Generative AI' incidents.

**Methodology & Data:**
Due to sparse metadata in the 'Known AI Technology' column, the experiment
employed a text-mining strategy on incident titles and descriptions. This
successfully classified 668 incidents: 133 as 'Robotics/AV' and 535 as
'Generative/LLM'.

**Findings:**
1.  **Statistical Significance:** The Mann-Whitney U test yielded a p-value of
`5.42e-08`, well below the 0.05 threshold, confirming that the severity
distributions of the two groups are statistically distinct.
2.  **Severity Disparity:**
    - **Robotics/AV:** Median Severity = 1.0. High-severity outcomes (Level 2:
Property/Injury or Level 3: Death) accounted for **48.1%** of incidents
(64/133).
    - **Generative AI:** Median Severity = 0.0. The vast majority (**78.7%**) of
incidents were Low Severity (Level 0: Unclear or Level 1: Intangible), with only
~5% resulting in severe physical harm or death.

**Conclusion:**
The 'Physicality' of Failure Severity hypothesis is confirmed. While Generative
AI incidents are more voluminous in the recent dataset, they predominantly
result in intangible or low-severity harms. In contrast, Robotics and Autonomous
Vehicles, though less frequent in this sample, pose a significantly higher risk
of physical injury, property damage, and loss of life.

---

## Review

The experiment successfully validated the hypothesis regarding the 'Physicality'
of failure severity. The implementation correctly adapted to data quality issues
(sparsity in the 'Known AI Technology' column) by implementing a text-mining
strategy on incident descriptions, which successfully categorized 668 incidents.

**Findings:**
1.  **Statistical Significance:** The Mann-Whitney U test yielded a p-value of
5.42e-08, confirming a statistically significant difference in harm severity
between the two groups.
2.  **Severity Profile:** 'Robotics/AV' incidents (n=133) demonstrated a higher
median severity (1.0) compared to 'Generative/LLM' (n=535, Median 0.0).
3.  **Risk Distribution:** High-severity outcomes (Level 2: Property/Injury and
Level 3: Death) constituted ~48% of Robotics incidents, compared to only ~21%
for Generative AI. Conversely, 79% of Generative AI incidents were low severity
(None/Unclear or Intangible), validating the hypothesis that Generative AI risks
are voluminous but less physically severe than those of autonomous systems.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Load dataset
file_path = 'astalabs_discovery_all_data.csv'
if not os.path.exists(file_path):
    file_path = '../astalabs_discovery_all_data.csv'

try:
    df = pd.read_csv(file_path, low_memory=False)
except:
    # Fallback for very sparse CSVs if engine='c' fails (rare but possible)
    df = pd.read_csv(file_path, low_memory=False, engine='python')

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()

# Create a consolidated text field for classification
# We combine title, description, and Known AI Technology to maximize signal
text_cols = ['title', 'description', 'Known AI Technology']
for col in text_cols:
    if col not in aiid.columns:
        aiid[col] = ''
    else:
        aiid[col] = aiid[col].fillna('')

aiid['full_text'] = (aiid['title'] + ' ' + aiid['description'] + ' ' + aiid['Known AI Technology']).str.lower()

# --- Classification Functions ---

def classify_tech(text):
    # Robotics / Autonomous Vehicles
    if any(k in text for k in ['robot', 'autonomous vehicle', 'self-driving', 'drone', 'tesla', 'autopilot', 'waymo', 'cruise', 'uber', 'driverless']):
        return 'Robotics/AV'
    # Generative AI / LLM / Chatbots
    if any(k in text for k in ['generative', 'llm', 'gpt', 'chatgpt', 'diffusion', 'chatbot', 'midjourney', 'dall-e', 'language model', 'deepfake', 'text-to-image', 'stable diffusion', 'bert', 'transformer', 'hallucinat']):
        return 'Generative/LLM'
    return 'Other'

def classify_severity(text):
    # Level 3: Severe Physical Harm / Death
    if any(k in text for k in ['kill', 'death', 'fatal', 'died', 'suicide', 'loss of life', 'murder']):
        return 3
    # Level 2: Physical Injury / Property / Financial
    if any(k in text for k in ['injur', 'crash', 'collision', 'accident', 'damage', 'property', 'financial', 'money', 'theft', 'arrest', 'physical']):
        return 2
    # Level 1: Intangible / Social / Psychological
    if any(k in text for k in ['bias', 'discriminat', 'racis', 'sexis', 'privacy', 'surveillance', 'offensive', 'inappropriate', 'nudity', 'copyright', 'plagiaris', 'reputation', 'stereotyp', 'wrongful']):
        return 1
    # Level 0: Unclear / None specified (default base)
    return 0

# Apply Classifications
aiid['tech_category'] = aiid['full_text'].apply(classify_tech)
aiid['severity_score'] = aiid['full_text'].apply(classify_severity)

# Filter for comparison groups
comp_df = aiid[aiid['tech_category'].isin(['Robotics/AV', 'Generative/LLM'])].copy()

# Stats
robotics_scores = comp_df[comp_df['tech_category'] == 'Robotics/AV']['severity_score']
genai_scores = comp_df[comp_df['tech_category'] == 'Generative/LLM']['severity_score']

print(f"--- Classification Results ---")
print(f"Robotics/AV Samples: {len(robotics_scores)}")
print(f"Generative/LLM Samples: {len(genai_scores)}")
print(f"Total Classified: {len(comp_df)}")

print(f"\n--- Severity Statistics (Median) ---")
print(f"Robotics/AV: {robotics_scores.median()}")
print(f"Generative/LLM: {genai_scores.median()}")

# Mann-Whitney U Test
u_stat, p_val = stats.mannwhitneyu(robotics_scores, genai_scores, alternative='greater')
print(f"\n--- Hypothesis Test (Mann-Whitney U) ---")
print(f"Null: Severity(Robotics) <= Severity(GenAI)")
print(f"Alternative: Severity(Robotics) > Severity(GenAI)")
print(f"U-Statistic: {u_stat}")
print(f"P-Value: {p_val:.5e}")

# Visualization
# 1. Stacked Bar Chart (Proportions)
counts = comp_df.groupby(['tech_category', 'severity_score']).size().unstack(fill_value=0)
props = counts.div(counts.sum(axis=1), axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Stacked Bar
props.plot(kind='bar', stacked=True, ax=axes[0], colormap='RdYlBu_r', alpha=0.85)
axes[0].set_title('Proportion of Severity Levels by Tech')
axes[0].set_ylabel('Proportion')
axes[0].set_xlabel('Technology')
axes[0].legend(title='Severity Score', labels=['0: None/Unclear', '1: Intangible', '2: Property/Injury', '3: Death/Severe'])

# 2. Boxplot (Distribution)
# We use a list of arrays for boxplot
data_to_plot = [robotics_scores, genai_scores]
axes[1].boxplot(data_to_plot, labels=['Robotics/AV', 'Generative/LLM'], patch_artist=True, 
                boxprops=dict(facecolor='lightblue'))
axes[1].set_title('Distribution of Severity Scores')
axes[1].set_ylabel('Severity Score (0-3)')
axes[1].yaxis.grid(True)

plt.tight_layout()
plt.show()

# Print counts for verification
print("\n--- Detailed Counts ---")
print(counts)

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Classification Results ---
Robotics/AV Samples: 133
Generative/LLM Samples: 535
Total Classified: 668

--- Severity Statistics (Median) ---
Robotics/AV: 1.0
Generative/LLM: 0.0

--- Hypothesis Test (Mann-Whitney U) ---
Null: Severity(Robotics) <= Severity(GenAI)
Alternative: Severity(Robotics) > Severity(GenAI)
U-Statistic: 44603.0
P-Value: 5.42404e-08

--- Detailed Counts ---
severity_score    0   1   2   3
tech_category                  
Generative/LLM  364  57  85  29
Robotics/AV      65   4  37  27

STDERR:
<ipython-input-1-767356daee9b>:104: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  axes[1].boxplot(data_to_plot, labels=['Robotics/AV', 'Generative/LLM'], patch_artist=True,


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
The image contains two distinct subplots side-by-side:
*   **Left Subplot:** A **Stacked Bar Plot**. Its purpose is to compare the relative proportions of different categorical levels (Severity Scores) within two distinct groups (Technologies).
*   **Right Subplot:** A **Box Plot** (or Box-and-Whisker Plot). Its purpose is to visualize the statistical distribution (median, quartiles, and range) of the numerical severity scores for the two technology groups.

### 2. Axes
**Left Plot (Stacked Bar):**
*   **X-axis:** Labeled **"Technology"**. Categories are "Generative/LLM" and "Robotics/AV".
*   **Y-axis:** Labeled **"Proportion"**. The range is **0.0 to 1.0**, representing 0% to 100% of the total cases for each technology.

**Right Plot (Box Plot):**
*   **X-axis:** Implicitly labeled by category. Categories are **"Robotics/AV"** and **"Generative/LLM"**.
*   **Y-axis:** Labeled **"Severity Score (0-3)"**. The range is **-0.1 to 3.1** (visual buffer), covering the discrete score values of 0, 1, 2, and 3.

### 3. Data Trends
**Left Plot (Stacked Bar Trends):**
*   **Generative/LLM:** This column is dominated by the lowest severity score (0: None/Unclear), which takes up roughly 70% of the bar. The highest severity (3: Death/Severe) represents a very small fraction (approx. 5%).
*   **Robotics/AV:** This column shows a much more even distribution across severity levels. While score 0 is still significant (approx. 50%), there is a substantial proportion of score 2 (Property/Injury) and score 3 (Death/Severe). The "Death/Severe" portion is visibly much larger here (approx. 20%) compared to the LLM group.

**Right Plot (Box Plot Trends):**
*   **Robotics/AV:**
    *   **Median:** The median line is at **1.0**.
    *   **Spread:** The Interquartile Range (the blue box) spans from 0.0 to 2.0. The whiskers extend fully to 3.0, indicating a wide variance in incident severity.
*   **Generative/LLM:**
    *   **Median:** The median line is at **0.0** (indicated by the orange line at the bottom of the box).
    *   **Spread:** The box spans from 0.0 to 1.0.
    *   **Outliers:** There is a distinct outlier circle at **3.0**, indicating that while severe incidents are rare for LLMs, they do occur.

### 4. Annotations and Legends
*   **Legend (Left Plot):** A legend titled **"Severity Score"** is provided in the bottom left corner, mapping colors to severity levels:
    *   **Dark Purple/Blue:** 0: None/Unclear
    *   **Light Blue:** 1: Intangible
    *   **Light Orange:** 2: Property/Injury
    *   **Dark Red:** 3: Death/Severe
*   **Titles:**
    *   Left: "Proportion of Severity Levels by Tech"
    *   Right: "Distribution of Severity Scores"
*   **Gridlines:** The right plot uses horizontal gridlines to assist in reading specific score values.

### 5. Statistical Insights
The plots suggest a fundamental difference in the risk profiles of the two technologies:

1.  **Physical vs. Non-Physical Risk:** **Robotics/AV** technologies are associated with significantly higher physical risks. The stacked bar chart shows that roughly half of the incidents recorded for Robotics/AV involve property damage, injury, or death (scores 2 and 3). In contrast, **Generative/LLM** incidents are overwhelmingly categorized as "None/Unclear" or "Intangible."
2.  **Severity Distribution:** The box plot confirms that the "average" (median) incident for Robotics/AV has a severity score of 1, whereas the median for Generative/LLM is 0.
3.  **High-Severity Frequency:** While both technologies *can* result in maximum severity incidents (score 3), they are outliers for Generative AI but a common component of the distribution for Robotics/AV (representing roughly 20% of its cases).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
