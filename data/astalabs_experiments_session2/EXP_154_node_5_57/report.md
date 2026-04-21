# Experiment 154: node_5_57

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_57` |
| **ID in Run** | 154 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T08:16:19.659600+00:00 |
| **Runtime** | 469.2s |
| **Parent** | `node_4_47` |
| **Children** | None |
| **Creation Index** | 155 |

---

## Hypothesis

> Physical Severity Escalation: AI incidents classified under the 'Physical
Safety' harm domain are associated with significantly higher severity scores
(Harm Level) than those in 'Bias & Discrimination' domains.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.9066 (Definitely True) |
| **Surprise** | +0.1976 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
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
| Definitely True | 58.0 |
| Maybe True | 2.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Quantify the severity differential between physical and sociotechnical AI failures.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'aiid_incidents'.
- 2. Filter rows where 'Harm Domain' contains 'Physical' or 'Bias/Discrimination'.
- 3. Map 'AI Harm Level' to an ordinal scale (e.g., 1=Minor, 2=Moderate, 3=Severe/Critical).
- 4. Create two groups based on Harm Domain.
- 5. Perform a Mann-Whitney U test to compare the distributions of Harm Level scores between the two groups.

### Deliverables
- Median Harm Level for each group; Mann-Whitney U test statistic and p-value; Box plot visualization.

---

## Analysis

The experiment successfully tested the 'Physical Severity Escalation' hypothesis
using the AIID dataset. Faced with sparse structured metadata for harm domains,
the code successfully implemented a fallback text-mining strategy to categorize
incidents based on keywords in their descriptions, identifying 33 'Physical
Safety' and 41 'Bias & Discrimination' cases. The statistical analysis supports
the hypothesis: a Mann-Whitney U test yielded a p-value of 0.005, significantly
below the 0.05 threshold. This indicates that Physical Safety incidents are
statistically more severe (more likely to be realized 'events' vs. 'issues' or
'near-misses') than Bias & Discrimination incidents. The generated boxplot
confirms this, showing a distribution for physical incidents that extends to the
maximum severity level, whereas bias incidents are heavily concentrated at lower
severity levels.

---

## Review

The experiment successfully tested the 'Physical Severity Escalation' hypothesis
using the AIID dataset. Faced with sparse structured metadata for harm domains,
the code successfully implemented a fallback text-mining strategy to categorize
incidents based on keywords in their descriptions, identifying 33 'Physical
Safety' and 41 'Bias & Discrimination' cases. The statistical analysis supports
the hypothesis: a Mann-Whitney U test yielded a p-value of 0.005, significantly
below the 0.05 threshold. This indicates that Physical Safety incidents are
statistically more severe (more likely to be realized 'events' vs. 'issues' or
'near-misses') than Bias & Discrimination incidents. The generated boxplot
confirms this, showing a distribution for physical incidents that extends to the
maximum severity level, whereas bias incidents are heavily concentrated at lower
severity levels.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load dataset
print("Loading dataset...")
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Check unified_evidence_base for comparison
unified_df = df[df['source_table'] == 'unified_evidence_base'].copy()
print(f"Unified rows: {len(unified_df)}")

# Check AIID incidents again
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID rows: {len(aiid_df)}")

# Determine which dataframe has better data for 'Harm Domain' or similar
# We will prioritize AIID but fallback to search if needed.

# Define categorization function based on keywords in text if structured column fails
def categorize_from_text(row):
    # Combine relevant text columns
    text = str(row.get('title', '')) + " " + str(row.get('description', '')) + " " + str(row.get('summary', '')) + " " + str(row.get('Harm Domain', ''))
    text = text.lower()
    
    # Classification logic
    is_physical = any(w in text for w in ['death', 'injury', 'kill', 'physical', 'safety', 'accident', 'collision', 'robot', 'autonomous vehicle', 'drone'])
    is_bias = any(w in text for w in ['bias', 'discrimination', 'racist', 'sexist', 'gender', 'race', 'unfair', 'stereotype', 'facial recognition', 'demographic'])
    
    if is_physical and not is_bias:
        return 'Physical Safety'
    if is_bias and not is_physical:
        return 'Bias & Discrimination'
    return 'Other/Mixed'

# Define Severity Mapping
# Based on previous debug: 'AI tangible harm event', 'AI tangible harm near-miss', 'AI tangible harm issue'
severity_map = {
    'ai tangible harm event': 3,
    'ai tangible harm near-miss': 2,
    'ai tangible harm issue': 1,
    'none': 0,
    'unclear': 0
}

def map_severity(val):
    if pd.isna(val): return None
    s = str(val).lower().strip()
    return severity_map.get(s, None)

# Apply logic to AIID incidents
# We use the text-based classification because 'Harm Domain' column was shown to be sparse/boolean
print("Classifying domains based on text analysis...")
aiid_df['domain_group'] = aiid_df.apply(categorize_from_text, axis=1)

# Map Severity
# We assume 'AI Harm Level' is the column name based on previous debug output
print("Mapping severity scores...")
# Ensure we use the correct column name. Previous debug showed 'AI Harm Level'.
aiid_df['severity_score'] = aiid_df['AI Harm Level'].apply(map_severity)

# Filter for analysis
analysis_df = aiid_df.dropna(subset=['severity_score', 'domain_group'])
physical_group = analysis_df[analysis_df['domain_group'] == 'Physical Safety']['severity_score']
bias_group = analysis_df[analysis_df['domain_group'] == 'Bias & Discrimination']['severity_score']

print(f"Physical Safety N={len(physical_group)}")
print(f"Bias & Discrimination N={len(bias_group)}")

# Perform Statistics
if len(physical_group) > 5 and len(bias_group) > 5:
    print("Performing Mann-Whitney U Test...")
    u_stat, p_val = stats.mannwhitneyu(physical_group, bias_group, alternative='greater')
    
    print(f"Mann-Whitney U statistic: {u_stat}")
    print(f"P-value (Physical > Bias): {p_val:.5f}")
    
    print(f"Median Severity (Physical): {physical_group.median()}")
    print(f"Median Severity (Bias): {bias_group.median()}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    data = [physical_group, bias_group]
    plt.boxplot(data, labels=['Physical Safety', 'Bias & Discrimination'])
    plt.title('AI Harm Severity: Physical vs Bias Incidents')
    plt.ylabel('Severity Level (0=None, 1=Issue, 2=Near-Miss, 3=Event)')
    plt.yticks([0, 1, 2, 3], ['None/Unclear', 'Issue', 'Near-Miss', 'Event'])
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()
else:
    print("Insufficient data for statistical analysis after text-based classification.")
    print("Sample of Domain Grouping:", analysis_df['domain_group'].value_counts())
    print("Sample of Severity Mapping:", analysis_df['severity_score'].value_counts())

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset...
Unified rows: 3171
AIID rows: 1362
Classifying domains based on text analysis...
Mapping severity scores...
Physical Safety N=33
Bias & Discrimination N=41
Performing Mann-Whitney U Test...
Mann-Whitney U statistic: 864.0
P-value (Physical > Bias): 0.00498
Median Severity (Physical): 0.0
Median Severity (Bias): 0.0

STDERR:
<ipython-input-1-9a3d3101b171>:88: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(data, labels=['Physical Safety', 'Bias & Discrimination'])


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis of the plot:

### 1. Plot Type
*   **Type:** Box Plot (Box-and-Whisker Plot).
*   **Purpose:** To compare the distribution of severity levels between two different categories of AI harm ("Physical Safety" and "Bias & Discrimination"). It visualizes the median, interquartile range (IQR), and outliers for each category.

### 2. Axes
*   **Y-Axis (Vertical):**
    *   **Label:** "Severity Level (0=None, 1=Issue, 2=Near-Miss, 3=Event)".
    *   **Units:** An ordinal scale mapped to integers 0 through 3.
    *   **Range:** The axis spans four discrete levels: "None/Unclear", "Issue", "Near-Miss", and "Event".
*   **X-Axis (Horizontal):**
    *   **Label:** None explicitly (implied categories).
    *   **Categories:** "Physical Safety" and "Bias & Discrimination".

### 3. Data Trends
*   **Physical Safety (Left Box):**
    *   **Distribution:** This category shows a very wide Interquartile Range (IQR). The "box" extends from the baseline ("None/Unclear") all the way to the top ("Event").
    *   **Median:** The orange line (median) sits at the "None/Unclear" level.
    *   **Pattern:** This suggests a polarized or high-variance distribution. While the median incident is low severity, the upper quartile (the top of the box) hits the maximum severity level ("Event"). This implies that a significant portion (at least the top 25%) of physical safety incidents are classified as major events.
*   **Bias & Discrimination (Right Box):**
    *   **Distribution:** The distribution is heavily compressed at the bottom. The box is flattened at the "None/Unclear" line, indicating that the 1st Quartile (Q1), Median, and 3rd Quartile (Q3) are likely all situated at 0.
    *   **Outliers:** There are distinct circular markers (outliers) visible at the "Issue," "Near-Miss," and "Event" levels. This indicates that while the vast majority of bias incidents are low severity (or unclear), there are sporadic instances of high severity.

### 4. Annotations and Legends
*   **Title:** "AI Harm Severity: Physical vs Bias Incidents".
*   **Grid:** Horizontal dashed gray lines are provided at each severity level to assist in reading the discrete values.
*   **Outlier Markers:** Small circles ($\circ$) represent data points that fall statistically outside the typical range for the "Bias & Discrimination" category.

### 5. Statistical Insights
*   **Severity Disparity:** Physical Safety incidents in this dataset have a much higher ceiling for severity compared to Bias & Discrimination incidents. The Physical Safety data indicates that when things go wrong, they often result in a full "Event," whereas Bias incidents rarely escalate to that level.
*   **Commonality of Low Severity:** For both categories, the median is "None/Unclear." This suggests that the most common outcome for both Physical and Bias reports in this dataset is either no harm or an unclear result.
*   **Predictability:** The "Bias & Discrimination" data is more consistent (clustered at low severity) with occasional exceptions. The "Physical Safety" data is highly variable, making it statistically "riskier" in terms of potential maximum harm, as a significant percentage of its distribution reaches the highest possible severity rating.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
