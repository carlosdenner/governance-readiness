# Experiment 273: node_6_73

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_73` |
| **ID in Run** | 273 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T14:00:42.071071+00:00 |
| **Runtime** | 361.5s |
| **Parent** | `node_5_41` |
| **Children** | None |
| **Creation Index** | 274 |

---

## Hypothesis

> Intentionality vs. Severity: Incidents involving 'Intentional Harm' (Malicious
Use) result in statistically higher 'AI Harm Levels' than incidents caused by
accidental or unintentional failures.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.3548 (Maybe False) |
| **Posterior** | 0.3297 (Maybe False) |
| **Surprise** | -0.0302 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 12.0 |
| Maybe False | 18.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 16.0 |
| Maybe False | 44.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Determine if malicious attacks produce more severe consequences than accidents, informing threat modeling priorities.

### Steps
- 1. Load 'aiid_incidents'.
- 2. Filter for valid '82_Intentional Harm' (True/False) and '75_AI Harm Level'.
- 3. Convert 'AI Harm Level' to an ordinal scale (e.g., Near Miss=0, Minor=1, Severe=2) or keep categorical.
- 4. Perform a Mann-Whitney U test (if ordinal) or Chi-Square test (if categorical).
- 5. Compare the distributions of harm severity for Intentional vs Unintentional incidents.

### Deliverables
- Distribution plots (e.g., Violin plot), Statistical test results.

---

## Analysis

The experiment was successfully executed. By implementing robust keyword
matching and heuristic scoring, the analysis isolated 183 valid incidents (175
Unintentional, 8 Intentional) from the AIID dataset. The Mann-Whitney U test
yielded a p-value of 0.266, which is not statistically significant (p > 0.05).
Therefore, we **fail to reject the null hypothesis**; the data does not support
the claim that intentional/malicious AI incidents result in higher severity
harms than accidental ones.

Descriptive statistics show a slightly higher mean severity score for
intentional incidents (0.56) compared to unintentional ones (0.41), but the
median for both groups is 0.0, indicating that the majority of recorded
incidents in both categories are 'Near Misses' or low-severity events.

A critical limitation of this finding is the severe class imbalance and small
sample size for intentional incidents (n=8). This suggests that either malicious
use is rare in the AIID dataset, or the unstructured text fields require more
advanced NLP than keyword matching to accurately categorize harm severity and
intent for the majority of rows.

---

## Review

The experiment was successfully executed. By implementing robust keyword
matching and heuristic scoring, the analysis isolated 183 valid incidents (175
Unintentional, 8 Intentional) from the AIID dataset. The Mann-Whitney U test
yielded a p-value of 0.266, which is not statistically significant (p > 0.05).
Therefore, we **fail to reject the null hypothesis**; the data does not support
the claim that intentional/malicious AI incidents result in higher severity
harms than accidental ones.

Descriptive statistics show a slightly higher mean severity score for
intentional incidents (0.56) compared to unintentional ones (0.41), but the
median for both groups is 0.0, indicating that the majority of recorded
incidents in both categories are 'Near Misses' or low-severity events.

A critical limitation of this finding is the severe class imbalance and small
sample size for intentional incidents (n=8). This suggests that either malicious
use is rare in the AIID dataset, or the unstructured text fields require more
advanced NLP than keyword matching to accurately categorize harm severity and
intent for the majority of rows.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# Load data
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Loaded {len(aiid)} AIID incidents.")

# Dynamic column finding
cols = aiid.columns.tolist()
intent_col = next((c for c in cols if 'Intentional Harm' in str(c)), 'Intentional Harm')
harm_level_col = next((c for c in cols if 'AI Harm Level' in str(c)), 'AI Harm Level')
tangible_harm_col = next((c for c in cols if 'Tangible Harm' in str(c)), 'Tangible Harm')

print(f"Columns identified:\n - Intent: {intent_col}\n - Harm Level: {harm_level_col}\n - Tangible Harm: {tangible_harm_col}")

# --- 1. Clean Intentionality ---
def clean_intent(val):
    s = str(val).lower().strip()
    if s.startswith('yes'):
        return True
    elif s.startswith('no'):
        return False
    return np.nan

aiid['is_intentional'] = aiid[intent_col].apply(clean_intent)

# --- 2. Construct Harm Severity Score ---
# Strategy: Use 'Tangible Harm' for granularity. If specific keywords found, assign score.
# If not found, look at 'AI Harm Level' as fallback context.

def calculate_severity(row):
    # Get strings
    t_harm = str(row[tangible_harm_col]).lower() if pd.notna(row[tangible_harm_col]) else ''
    h_level = str(row[harm_level_col]).lower() if pd.notna(row[harm_level_col]) else ''
    
    # Priority 1: High Severity Keywords in Tangible Harm description
    if any(x in t_harm for x in ['death', 'killed', 'loss of life', 'fatal']):
        return 4
    if any(x in t_harm for x in ['injury', 'physical', 'hospital', 'safety']):
        return 3
    if any(x in t_harm for x in ['financial', 'economic', 'property', 'monetary', 'theft']):
        return 2
    if any(x in t_harm for x in ['reputation', 'psychological', 'bias', 'discrimination', 'privacy', 'civil rights']):
        return 1
        
    # Priority 2: Fallback to Broad Categories in AI Harm Level
    if 'event' in h_level:
        # Default for an event with unspecified tangible harm
        return 1.5 # Treat as generic harm
    if 'issue' in h_level:
        return 1
    if 'near-miss' in h_level or 'near miss' in h_level:
        return 0
    if 'none' in h_level:
        return 0
        
    return np.nan

aiid['severity_score'] = aiid.apply(calculate_severity, axis=1)

# --- 3. Analysis ---
valid = aiid.dropna(subset=['is_intentional', 'severity_score'])
print(f"\nValid rows for analysis: {len(valid)}")

if len(valid) > 10:
    # Descriptive Stats
    stats = valid.groupby('is_intentional')['severity_score'].agg(['count', 'mean', 'median', 'std'])
    print("\n--- Descriptive Statistics by Intentionality ---")
    print(stats)
    
    # Mann-Whitney U Test
    group_intent = valid[valid['is_intentional'] == True]['severity_score']
    group_unintent = valid[valid['is_intentional'] == False]['severity_score']
    
    stat, p_val = mannwhitneyu(group_intent, group_unintent, alternative='greater') 
    # Hypothesis: Intentional > Unintentional (one-sided 'greater')
    
    print(f"\nMann-Whitney U Test (Intentional > Unintentional):\nStatistic={stat:.2f}, p-value={p_val:.5f}")
    
    if p_val < 0.05:
        print("RESULT: Statistically Significant. Intentional incidents have higher severity.")
    else:
        print("RESULT: Not Statistically Significant.")
        
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='is_intentional', y='severity_score', data=valid, inner='box', palette='Set2')
    plt.xticks([0, 1], ['Unintentional', 'Intentional'])
    plt.ylabel('Harm Severity Score (0=Near Miss -> 4=Fatal)')
    plt.title('Comparison of Harm Severity: Intentional vs Unintentional AI Incidents')
    plt.show()
    
else:
    print("Insufficient data. Printing sample of Tangible Harm for debugging:")
    print(aiid[tangible_harm_col].unique()[:20])

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded 1362 AIID incidents.
Columns identified:
 - Intent: Intentional Harm
 - Harm Level: AI Harm Level
 - Tangible Harm: Tangible Harm

Valid rows for analysis: 183

--- Descriptive Statistics by Intentionality ---
                count      mean  median       std
is_intentional                                   
False             175  0.408571     0.0  0.647884
True                8  0.562500     0.0  0.776324

Mann-Whitney U Test (Intentional > Unintentional):
Statistic=773.50, p-value=0.26601
RESULT: Not Statistically Significant.

STDERR:
<ipython-input-1-dec495019639>:99: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.violinplot(x='is_intentional', y='severity_score', data=valid, inner='box', palette='Set2')


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is a detailed analysis:

### 1. Plot Type
*   **Type:** Violin Plot.
*   **Purpose:** This plot is used to visualize the distribution of numerical data (`Harm Severity Score`) across different categories (`is_intentional`). It combines features of a box plot and a kernel density plot. The width of the "violin" at any given y-value represents the frequency or probability density of data points at that value.

### 2. Axes
*   **X-axis:**
    *   **Label:** `is_intentional`
    *   **Categories:** Two discrete categories are plotted: "Unintentional" (left) and "Intentional" (right).
*   **Y-axis:**
    *   **Label:** `Harm Severity Score (0=Near Miss -> 4=Fatal)`
    *   **Units/Scale:** A numerical score representing severity. While the label indicates a theoretical range of 0 to 4, the plotted y-axis ticks range from **-1.0 to 2.5**.
    *   **Note on Range:** The extension of the plot shape below 0 and above the highest data points is likely an artifact of the kernel density estimation (smoothing) used to generate the violin shape, rather than representing actual negative harm scores.

### 3. Data Trends
*   **Unintentional Incidents (Teal/Green):**
    *   **Shape:** The distribution is heavily weighted towards the bottom. The widest part of the violin is at the 0 mark (Near Miss), indicating the majority of unintentional incidents result in low or no harm.
    *   **Secondary Peak:** There is a smaller, secondary bulge around the 1.5 mark, suggesting a smaller subset of incidents that result in moderate harm.
    *   **Range:** The data is more compact vertically, suggesting less variability in severity compared to intentional incidents.
*   **Intentional Incidents (Orange/Salmon):**
    *   **Shape:** While there is still a wide base near 0, the distribution is much more "top-heavy" compared to the unintentional group.
    *   **High Severity Cluster:** There is a significant swelling of the plot between the 1.5 and 2.5 marks. This indicates a much higher density of incidents occurring at higher severity levels compared to the unintentional category.
    *   **Range:** The plot extends higher vertically, indicating a broader range of outcomes including more severe ones.

### 4. Annotations and Legends
*   **Title:** "Comparison of Harm Severity: Intentional vs Unintentional AI Incidents".
*   **Internal Box Plot Elements:** Inside each violin is a grey box plot representation:
    *   **White Dot:** Represents the **median** value. For both categories, the median appears to be 0 (or extremely close to it).
    *   **Thick Grey Bar:** Represents the **Interquartile Range (IQR)** (from the 25th to the 75th percentile). The IQR for "Intentional" extends significantly higher than for "Unintentional," reaching up to approximately 1.5.
    *   **Thin Grey Line:** Represents the "whiskers," showing the rest of the distribution, excluding outliers.

### 5. Statistical Insights
*   **Base Rate of Severity:** For both intentional and unintentional AI incidents, the **median severity is low** (likely 0/Near Miss). This suggests that the most common outcome for any AI incident, regardless of intent, is a near miss or low harm.
*   **Risk Profile Variance:** The "Intentional" category exhibits significantly higher variance and a higher upper quartile. Statistically, this means that while the "average" outcome is similar, **intentional incidents carry a much higher risk of escalating to moderate or high severity** compared to unintentional ones.
*   **Distribution Shape:** The "Unintentional" distribution is clearly bimodal but dominated by the lower mode. The "Intentional" distribution is also bimodal but shows a much stronger secondary mode at the higher severity end, suggesting that when intentional harm occurs, it is frequently aimed at or results in a specific higher tier of severity (around the 1.5–2.0 score range).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
