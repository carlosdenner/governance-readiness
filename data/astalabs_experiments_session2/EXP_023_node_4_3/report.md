# Experiment 23: node_4_3

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_3` |
| **ID in Run** | 23 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T02:11:28.452115+00:00 |
| **Runtime** | 213.9s |
| **Parent** | `node_3_3` |
| **Children** | `node_5_3`, `node_5_68` |
| **Creation Index** | 24 |

---

## Hypothesis

> Autonomy-Failure Correlation: High-autonomy systems (Level 3+) primarily fail
due to 'Robustness' issues (e.g., environmental adaptation), whereas Low-
autonomy systems (Level 1) fail due to 'Human-Machine Interaction' or 'Operator
Error'.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7823 (Likely True) |
| **Posterior** | 0.2995 (Likely False) |
| **Surprise** | -0.5794 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 5.0 |
| Maybe True | 25.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 12.0 |
| Definitely False | 48.0 |

---

## Experiment Plan

**Objective:** Investigate the relationship between system autonomy levels and technical failure mechanisms.

### Steps
- 1. Filter for `aiid_incidents`. Map `Autonomy Level` to ordinal (1-5).
- 2. Categorize `Known AI Technical Failure` into 'Robustness' (distribution shift, noise, sensor) vs. 'Operator/HMI' (human error, misuse).
- 3. Compare the distribution of Autonomy Levels for Robustness failures vs. Operator failures using a Mann-Whitney U test.
- 4. Visualize the density of failure types across the autonomy spectrum.

### Deliverables
- Violin plot of Autonomy Level by Failure Category, Mann-Whitney U statistics.

---

## Analysis

The experiment successfully tested the relationship between autonomy levels and
technical failure types. The code correctly loaded the dataset, filtered for
AIID incidents, and mapped autonomy levels to an ordinal scale (1-3) and failure
types to 'Robustness' (n=32) or 'Operator/HMI' (n=13) categories.

**Key Findings:**
1.  **Statistical Significance:** The Mann-Whitney U test yielded a p-value of
**0.018**, indicating a statistically significant difference between the groups.
2.  **Hypothesis Rejection:** The results **contradict** the stated hypothesis.
The hypothesis predicted that High Autonomy corresponds to 'Robustness' failures
and Low Autonomy to 'Operator Error'. The data shows the opposite:
    *   **Operator/HMI** failures are associated with **higher** autonomy levels
(Mean: 2.23, Median: 2.0).
    *   **Robustness** failures are associated with **lower** autonomy levels
(Mean: 1.56, Median: 1.0).

This suggests that as autonomy increases, the primary risk shifts from basic
technical reliability (Robustness) to complex human-system interaction issues,
likely due to phenomena like automation surprise or mode confusion.

---

## Review

The experiment was faithfully implemented and the analysis is statistically
sound. The code successfully filtered the AI Incident Database, mapped autonomy
levels to an ordinal scale (1-3), and categorized failures into 'Robustness'
(n=32) and 'Operator/HMI' (n=13).

**Hypothesis Evaluation**:
The hypothesis stated that high-autonomy systems fail due to 'Robustness' while
low-autonomy systems fail due to 'Operator Error'. The results **contradict**
this hypothesis with statistical significance (p=0.018, Mann-Whitney U).

**Findings**:
- **Operator/HMI failures** are associated with **higher** autonomy levels
(Mean: 2.23, Median: 2.0).
- **Robustness failures** are associated with **lower** autonomy levels (Mean:
1.56, Median: 1.0).

This suggests that as systems become more autonomous, the primary failure mode
shifts from intrinsic technical reliability (Robustness) to breakdowns in human-
system interaction (Operator/HMI), effectively inverting the proposed
relationship.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import os
import re

# [debug]
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir('.'))
try:
    print("Files in parent directory:", os.listdir('..'))
except Exception as e:
    print("Cannot list parent directory:", e)

# Load dataset
filename = 'astalabs_discovery_all_data.csv'
filepath = f'../{filename}' if os.path.exists(f'../{filename}') else filename

try:
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Successfully loaded {filepath}")
except FileNotFoundError:
    print(f"Error: Could not find {filename} in . or ..")
    exit(1)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid)}")

# Clean Autonomy Level
# Expecting values that can be mapped to ordinal. 
# Previous context suggests 'Autonomy1', 'Autonomy2', etc.
print("\nUnique Autonomy Levels (raw):", aiid['Autonomy Level'].unique())

def parse_autonomy(val):
    if pd.isna(val):
        return None
    val_str = str(val).lower()
    # Look for digits
    digits = re.findall(r'\d+', val_str)
    if digits:
        return int(digits[0])
    # Fallback keyword mapping if needed (though datasets usually use numeric codes)
    if 'high' in val_str: return 3
    if 'medium' in val_str: return 2
    if 'low' in val_str: return 1
    return None

aiid['Autonomy_Ordinal'] = aiid['Autonomy Level'].apply(parse_autonomy)

# Clean Technical Failure
print("\nUnique Technical Failures (raw top 20):", aiid['Known AI Technical Failure'].value_counts().head(20).index.tolist())

def classify_failure(val):
    if pd.isna(val):
        return None
    val_str = str(val).lower()
    
    # Keywords for Robustness
    robust_keys = ['robust', 'reliability', 'dependability', 'sensor', 'noise', 'environment', 'shift', 'distribution', 'failure of mechanism']
    # Keywords for Operator/HMI
    operator_keys = ['human', 'operator', 'user', 'mistake', 'misuse', 'hmi', 'interaction', 'training']
    
    is_robust = any(k in val_str for k in robust_keys)
    is_operator = any(k in val_str for k in operator_keys)
    
    if is_robust and not is_operator:
        return 'Robustness'
    elif is_operator and not is_robust:
        return 'Operator/HMI'
    elif is_robust and is_operator:
        return 'Mixed'
    else:
        return 'Other'

aiid['Failure_Category'] = aiid['Known AI Technical Failure'].apply(classify_failure)

# Filter data for analysis
analysis_df = aiid.dropna(subset=['Autonomy_Ordinal', 'Failure_Category'])
analysis_df = analysis_df[analysis_df['Failure_Category'].isin(['Robustness', 'Operator/HMI'])]

print("\nData for Analysis:")
print(analysis_df['Failure_Category'].value_counts())
print(analysis_df.groupby('Failure_Category')['Autonomy_Ordinal'].describe())

# Statistical Test
robustness_scores = analysis_df[analysis_df['Failure_Category'] == 'Robustness']['Autonomy_Ordinal']
operator_scores = analysis_df[analysis_df['Failure_Category'] == 'Operator/HMI']['Autonomy_Ordinal']

if len(robustness_scores) > 0 and len(operator_scores) > 0:
    u_stat, p_val = stats.mannwhitneyu(robustness_scores, operator_scores, alternative='two-sided')
    print(f"\nMann-Whitney U Test:\nU-statistic: {u_stat}\nP-value: {p_val}")
else:
    print("\nInsufficient data for statistical test.")

# Visualization
plt.figure(figsize=(10, 6))
sns.violinplot(data=analysis_df, x='Failure_Category', y='Autonomy_Ordinal', inner='stick', palette='muted')
plt.title('Distribution of Autonomy Levels by Failure Category')
plt.ylabel('Autonomy Level (Ordinal)')
plt.xlabel('Failure Category')
plt.grid(axis='y', alpha=0.3)
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Current working directory: /data
Files in current directory: ['.placeholder', 'astalabs_discovery_all_data.csv', 'context_construct_definitions.md', 'context_crosswalk_evidence.json', 'context_propositions.json', 'context_propositions.md', 'context_step1_evidence.json', 'context_validation_report.md']
Files in parent directory: ['tmp', 'usr', 'media', 'boot', 'lib', 'root', 'dev', 'run', 'proc', 'etc', 'srv', 'bin', 'sys', 'home', '.uv', 'sbin', '__modal', 'var', 'opt', 'lib64', 'mnt', 'data']
Successfully loaded astalabs_discovery_all_data.csv
AIID Incidents loaded: 1362

Unique Autonomy Levels (raw): <StringArray>
['Autonomy1', 'Autonomy2', 'Autonomy3', 'unclear', nan]
Length: 5, dtype: str

Unique Technical Failures (raw top 20): ['Generalization Failure', 'Misinformation Generation Hazard, Unsafe Exposure or Access', 'Distributional Bias', 'Context Misidentification', 'Lack of Transparency', 'Generalization Failure, Context Misidentification', 'Harmful Application', 'Unsafe Exposure or Access', 'Unsafe Exposure or Access, Misinformation Generation Hazard', 'Misinformation Generation Hazard', 'Latency Issues', 'Algorithmic Bias', 'Context Misidentification, Generalization Failure', 'Human Error', 'Hardware Failure', 'Generalization Failure, Lack of Safety Protocols', 'Distributional Bias, Limited Dataset', 'Algorithmic Bias, Problematic Features', 'Unauthorized Data', 'Gaming Vulnerability']

Data for Analysis:
Failure_Category
Robustness      32
Operator/HMI    13
Name: count, dtype: int64
                  count      mean       std  min  25%  50%   75%  max
Failure_Category                                                     
Operator/HMI       13.0  2.230769  0.832050  1.0  2.0  2.0  3.00  3.0
Robustness         32.0  1.562500  0.877588  1.0  1.0  1.0  2.25  3.0

Mann-Whitney U Test:
U-statistic: 123.0
P-value: 0.017736642601064015

STDERR:
<ipython-input-1-40714259be72>:102: FutureWarning: 

Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.

  sns.violinplot(data=analysis_df, x='Failure_Category', y='Autonomy_Ordinal', inner='stick', palette='muted')


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** This is a **Violin Plot**.
*   **Purpose:** A violin plot is used to visualize the distribution of numerical data across different categorical variables. It is similar to a box plot, but it also includes a Kernel Density Estimation (KDE) on each side, showing the probability density of the data at different values. This allows for a deeper understanding of the data's distribution shape (e.g., peaks, skewness) beyond just summary statistics.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Failure Category"
    *   **Labels:** The axis contains two categorical variables: "Robustness" and "Operator/HMI".
*   **Y-Axis:**
    *   **Title:** "Autonomy Level (Ordinal)"
    *   **Units/Range:** The axis represents ordinal values ranging from **0.0 to 4.0**. The grid lines are spaced at intervals of 0.5. The term "Ordinal" suggests the data points likely correspond to specific integer levels (e.g., Level 1, 2, 3, 4) of autonomy.

### 3. Data Trends
*   **Robustness (Left, Blue Violin):**
    *   **Distribution Shape:** The distribution is heavily bottom-heavy. The widest part of the violin (indicating the highest frequency of data points) is centered around **Autonomy Level 1**.
    *   **Secondary Feature:** There is a significant narrowing (neck) around Level 2, followed by a smaller, secondary bulge around Level 3.
    *   **Summary:** Failures categorized as "Robustness" occur most frequently at lower levels of autonomy.
*   **Operator/HMI (Right, Orange Violin):**
    *   **Distribution Shape:** The distribution is top-heavy. The violin is widest around **Autonomy Level 3**, with substantial width extending down to Level 2.
    *   **Taper:** The shape tapers off significantly towards the bottom, indicating very few Operator/HMI failures at Level 1 or 0.
    *   **Summary:** Failures categorized as "Operator/HMI" (Human-Machine Interface) are most prevalent at higher levels of autonomy.

### 4. Annotations and Legends
*   **Internal Lines:** Inside each violin shape, there are horizontal lines located at integer values (y=1, y=2, y=3). In the context of violin plots, these lines typically represent the **quartiles** (25th percentile, median, and 75th percentile) or mark the specific **ordinal integer levels** where the discrete data points lie. Given the "Ordinal" label, these lines help identify that the data is clustered at levels 1, 2, and 3.
*   **Color Coding:** The plot uses distinct colors to differentiate the categories: Blue for "Robustness" and Orange/Brown for "Operator/HMI".

### 5. Statistical Insights
*   **Inverse Relationship:** There is a clear inverse relationship between the failure categories and autonomy levels.
    *   **Low Autonomy Risks:** As the autonomy level decreases (moving towards Level 1), the system is more prone to **Robustness** failures. This implies that simpler or manual-assist systems struggle more with basic mechanical or software robustness issues rather than interaction issues.
    *   **High Autonomy Risks:** As the autonomy level increases (moving towards Level 3), the prevalence of **Operator/HMI** failures rises sharply. This suggests that as systems become more autonomous, the complexity of the interaction between the human operator and the machine increases, leading to a higher likelihood of interface-related errors or confusion.
*   **Bimodality:** The "Robustness" category shows some signs of bimodality (peaks at 1 and roughly 3), whereas the "Operator/HMI" category is more unimodal, centering broadly on the higher end of the scale.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
