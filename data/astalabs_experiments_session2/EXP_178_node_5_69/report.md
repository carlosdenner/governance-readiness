# Experiment 178: node_5_69

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_69` |
| **ID in Run** | 178 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T09:28:14.183836+00:00 |
| **Runtime** | 238.8s |
| **Parent** | `node_4_43` |
| **Children** | `node_6_32` |
| **Creation Index** | 179 |

---

## Hypothesis

> The 'Adversarial Mirage' Hypothesis: Despite the research focus on adversarial
attacks (ATLAS), real-world AI failures (AIID) are overwhelmingly non-
adversarial (e.g., reliability/performance failures), with 'Adversarial' causes
accounting for less than 5% of recorded incidents.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9839 (Definitely True) |
| **Posterior** | 0.3352 (Maybe False) |
| **Surprise** | -0.7784 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 30.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 60.0 |

---

## Experiment Plan

**Objective:** Determine the prevalence of adversarial vs. non-adversarial causes in real-world AI incidents.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'aiid_incidents'.
- 2. Define a list of adversarial keywords (e.g., 'attack', 'adversarial', 'poison', 'evasion', 'extraction', 'hack').
- 3. Text-mine the 'Known AI Technical Failure' and 'description/summary' columns (if available, else use 'title') to flag incidents containing these keywords.
- 4. Classify remaining incidents as 'Non-Adversarial/Failure'.
- 5. Calculate the proportion of Adversarial incidents.
- 6. Compare this proportion against a 5% threshold using a one-sample Z-test.

### Deliverables
- 1. Count and percentage of Adversarial vs Non-Adversarial incidents.
- 2. List of identified adversarial keywords found.
- 3. Pie chart of failure types.

---

## Analysis

The experiment successfully tested the 'Adversarial Mirage' hypothesis using the
AIID dataset.

1.  **Data Processing**: The code identified 1,362 incidents from the AIID
source table. It successfully utilized text columns ('title', 'description',
'summary', 'Known AI Technical Failure') to perform keyword matching.

2.  **Findings**:
    *   **Adversarial Count**: 174 incidents were flagged as adversarial based
on the keyword list (including terms like 'attack', 'hack', 'poison',
'security').
    *   **Proportion**: The observed proportion was **12.8%** (174/1362).

3.  **Hypothesis Test**:
    *   **Target**: The hypothesis claimed adversarial incidents would be **<
5%**.
    *   **Result**: The actual proportion (12.8%) is significantly *higher* than
the hypothesized threshold (5%).
    *   **Statistical Outcome**: The one-sample Z-test (checking for p < 0.05)
yielded a p-value of 1.0, meaning we **fail to reject the null hypothesis**. In
fact, the data strongly contradicts the specific claim that adversarial failures
are negligible (<5%) in this dataset.

4.  **Conclusion**: The 'Adversarial Mirage' hypothesis is **refuted** by this
analysis. While still a minority compared to non-adversarial failures (87.2%),
adversarial/security-related incidents constitute a substantial portion (~13%)
of the reported AI failures, more than double the expected 5%.

---

## Review

The experiment was executed successfully and faithfully followed the plan.

**Findings:**
1.  **Hypothesis Refutation:** The 'Adversarial Mirage' hypothesis, which
claimed that adversarial incidents constitute less than 5% of real-world AI
failures, was **refuted**.
2.  **Prevalence Analysis:** Out of 1,362 analyzed incidents from the AI
Incident Database (AIID), **174 (12.8%)** were flagged as adversarial based on
keyword text mining (e.g., 'attack', 'hack', 'poison', 'security'), while 1,188
(87.2%) were non-adversarial (reliability/safety failures).
3.  **Statistical Significance:** The observed proportion (12.8%) was
significantly higher than the hypothesized threshold (5%). Consequently, the
one-sample Z-test (checking for p < 0.05) yielded a p-value of 1.0, failing to
reject the null hypothesis in the direction of the claim.

**Conclusion:** Contrary to the hypothesis, adversarial and security-related AI
failures are a substantial category (over 1 in 8 incidents) in the reported
data, rather than a negligible 'mirage'.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest

# Define dataset path - trying current directory based on previous successful runs in context
file_path = 'astalabs_discovery_all_data.csv'

try:
    # Load dataset
    # Using low_memory=False to avoid dtype warnings on mixed columns
    df = pd.read_csv(file_path, low_memory=False)
    
    # Filter for AIID incidents
    aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
    
    # Define Adversarial Keywords
    adversarial_keywords = [
        'attack', 'adversarial', 'poison', 'evasion', 'extraction', 'hack', 
        'backdoor', 'trojan', 'inference', 'inversion', 'manipulation', 'security',
        'exploit', 'breach'
    ]
    
    # Identify potential text columns
    # Check for columns that might contain text data describing the incident
    potential_cols = ['title', 'description', 'summary', 'Known AI Technical Failure', 'incident_title', 'incident_description']
    available_cols = [c for c in potential_cols if c in aiid_df.columns]
    
    print(f"Analyzing {len(aiid_df)} AIID incidents using columns: {available_cols}")
    
    if not available_cols:
        # Fallback: if no specific text columns found, try to use all object columns (risky but better than nothing)
        print("Warning: Expected text columns not found. Searching all object columns.")
        obj_cols = aiid_df.select_dtypes(include=['object']).columns
        available_cols = obj_cols

    # Combine text for search
    # creating a temporary column for searching
    aiid_df['combined_text'] = aiid_df[available_cols].fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1).str.lower()
    
    # Flag Adversarial Incidents
    pattern = '|'.join(adversarial_keywords)
    aiid_df['is_adversarial'] = aiid_df['combined_text'].str.contains(pattern, case=False, regex=True)
    
    # Calculate Statistics
    total_incidents = len(aiid_df)
    adversarial_count = aiid_df['is_adversarial'].sum()
    non_adversarial_count = total_incidents - adversarial_count
    adversarial_prop = adversarial_count / total_incidents
    
    print(f"\n--- Results ---")
    print(f"Total Incidents: {total_incidents}")
    print(f"Adversarial Incidents: {adversarial_count}")
    print(f"Non-Adversarial Incidents: {non_adversarial_count}")
    print(f"Adversarial Proportion: {adversarial_prop:.4%}")
    
    # One-sample Z-test
    # Null Hypothesis (H0): p = 0.05
    # Alternative Hypothesis (H1): p < 0.05
    # We want to see if the real proportion is significantly LESS than 5%
    
    if adversarial_count < total_incidents and total_incidents > 0:
        stat, p_value = proportions_ztest(count=adversarial_count, nobs=total_incidents, value=0.05, alternative='smaller')
        print(f"\nOne-sample Z-test (Test Value=0.05, Alternative='smaller'):")
        print(f"Z-statistic: {stat:.4f}")
        print(f"P-value: {p_value:.4e}")
        
        if p_value < 0.05:
            print("Conclusion: REJECT H0. The proportion of adversarial incidents is significantly less than 5%.")
        else:
            print("Conclusion: FAIL TO REJECT H0. Evidence does not support that the proportion is less than 5%.")
    else:
        print("Cannot perform Z-test due to data constraints (e.g., 0 incidents or 100% match).")

    # Visualization
    labels = ['Non-Adversarial', 'Adversarial']
    sizes = [non_adversarial_count, adversarial_count]
    colors = ['lightgray', 'red']
    explode = (0, 0.1) 

    plt.figure(figsize=(10, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=45)
    plt.title(f'Prevalence of Adversarial Incidents in AIID (n={total_incidents})')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: File '{file_path}' not found in current directory.")
except Exception as e:
    print(f"An error occurred: {e}")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Analyzing 1362 AIID incidents using columns: ['title', 'description', 'summary', 'Known AI Technical Failure']

--- Results ---
Total Incidents: 1362
Adversarial Incidents: 174
Non-Adversarial Incidents: 1188
Adversarial Proportion: 12.7753%

One-sample Z-test (Test Value=0.05, Alternative='smaller'):
Z-statistic: 8.5961
P-value: 1.0000e+00
Conclusion: FAIL TO REJECT H0. Evidence does not support that the proportion is less than 5%.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Exploded Pie Chart.
*   **Purpose:** The plot illustrates the proportional distribution of a binary classification ("Adversarial" vs. "Non-Adversarial") within a specific dataset. The "exploded" feature (where one slice is separated from the center) is used to visually emphasize the "Adversarial" category.

### 2. Axes
*   **Titles and Labels:** As is standard for pie charts, there are no X or Y axes. Instead, the data is represented by circular sectors (slices).
*   **Value Ranges:** The total area represents 100% of the dataset. The values shown are percentages that sum to 100%.
    *   **Metric:** Percentage (%) of total incidents.
    *   **Sample Size:** The title indicates the total sample size is **n=1362**.

### 3. Data Trends
*   **Dominant Category:** The "Non-Adversarial" category (colored grey) dominates the chart, occupying the vast majority of the area.
*   **Minority Category:** The "Adversarial" category (colored red) represents a significantly smaller portion of the whole.
*   **Pattern:** There is a clear imbalance in the dataset, with non-adversarial incidents outnumbering adversarial ones by a ratio of roughly 7 to 1.

### 4. Annotations and Legends
*   **Chart Title:** "Prevalence of Adversarial Incidents in AIID (n=1362)". This contextualizes the data as coming from the "AIID" (likely the AI Incident Database) and provides the total count of incidents analyzed.
*   **Slice Labels:**
    *   **Non-Adversarial:** Labeled directly on the grey slice with the value **87.2%**.
    *   **Adversarial:** Labeled next to the red slice with the value **12.8%**.
*   **Visual Emphasis:** The "Adversarial" slice is colored bright red and "exploded" (pulled away from the center). This is a visual annotation technique used to draw the viewer's eye specifically to this minority statistic, highlighting it as the primary subject of interest despite its smaller size.

### 5. Statistical Insights
*   **Rarity of Adversarial Attacks:** The data suggests that within the AIID dataset, adversarial incidents are the exception rather than the norm. Only about 1 in 8 recorded incidents are classified as adversarial.
*   **Absolute Numbers:** Based on the sample size ($n=1362$) and the provided percentages:
    *   There are approximately **1,188** Non-Adversarial incidents ($1362 \times 0.872$).
    *   There are approximately **174** Adversarial incidents ($1362 \times 0.128$).
*   **Implication:** While the majority of reported AI incidents are likely due to accidents, negligence, or unintended bias (Non-Adversarial), a notable subset (12.8%) involves active adversarial actions, which is a significant enough portion to warrant specific study and visualization.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
