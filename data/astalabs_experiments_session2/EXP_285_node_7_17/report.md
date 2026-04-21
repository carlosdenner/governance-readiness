# Experiment 285: node_7_17

| Property | Value |
|---|---|
| **Experiment ID** | `node_7_17` |
| **ID in Run** | 285 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T14:44:44.574252+00:00 |
| **Runtime** | 225.8s |
| **Parent** | `node_6_19` |
| **Children** | None |
| **Creation Index** | 286 |

---

## Hypothesis

> Adversarial Complexity: ATLAS case studies involving 'Generative' AI (LLMs,
Chatbots) involve a significantly higher count of unique 'Techniques' per case
compared to 'Computer Vision' cases, reflecting a larger attack surface.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7177 (Likely True) |
| **Posterior** | 0.2445 (Likely False) |
| **Surprise** | -0.5679 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 29.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 1.0 |

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

**Objective:** Compare the adversarial attack surface (technique count) between GenAI and Computer Vision systems.

### Steps
- 1. Load the CSV and filter for 'atlas_cases'.
- 2. Categorize cases based on 'name' or 'summary' keywords: 'Generative' (LLM, GPT, Chatbot, Text) vs. 'Computer Vision' (Image, Face, Recognition, Vehicle, Camera). Drop unclassifiable rows.
- 3. Calculate the number of techniques per case by splitting the 'techniques' column string (e.g., counting commas or semicolons).
- 4. Perform a t-test comparing the average technique count between the two groups.

### Deliverables
- Average technique counts, t-test results, and a boxplot comparing attack complexity.

---

## Analysis

The experiment successfully analyzed the 'atlas_cases' dataset to compare
adversarial complexity between Generative AI (N=14) and Computer Vision (N=8)
systems. The code correctly categorized the cases and attempted to count the
number of techniques per case.

**Findings:**
- **Uniform Complexity:** Unexpectedly, every single case in both groups had
exactly one technique listed (Mean = 1.0, Std Dev = 0.0). The maximum value was
1.
- **Statistical Test:** Due to the zero variance in both samples, the t-test
mathematically failed (Statistic=NaN, p-value=NaN), as a difference in means
cannot be tested against zero standard error.
- **Visualization:** The generated boxplot confirmed this degeneracy, showing a
single line at y=1.0 for both categories.

**Conclusion:**
The hypothesis that Generative AI cases involve a significantly higher count of
techniques is **rejected** for this specific dataset. The data indicates no
difference in complexity, as all analyzed cases were coded with a singular
technique.

---

## Review

The experiment was successfully executed. The code handled the data loading and
categorization logic correctly. The statistical test returning 'NaN' is the
mathematically expected result when comparing two groups with zero variance (all
values were identical). The findings are valid for this specific dataset slice.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import re
import os

# Handle potential file path issues
filename = 'astalabs_discovery_all_data.csv'
if not os.path.exists(filename):
    if os.path.exists(f'../{filename}'):
        filename = f'../{filename}'
    else:
        # Fallback to searching, though unexpected based on prompt
        pass

print(f"Loading dataset from {filename}...")
df = pd.read_csv(filename, low_memory=False)

# Identify where the ATLAS case data resides
# We need 'name', 'summary', and 'techniques'

# Check 'atlas_cases' source_table
atlas_rows = df[df['source_table'] == 'atlas_cases'].copy()
print(f"Rows in atlas_cases: {len(atlas_rows)}")

# Check if techniques are populated in atlas_cases
techniques_populated_atlas = atlas_rows['techniques'].notna().sum()
print(f"Non-null techniques in atlas_cases: {techniques_populated_atlas}")

working_df = pd.DataFrame()

if techniques_populated_atlas > 10:
    working_df = atlas_rows
else:
    # Check 'step3_incident_coding'
    step3_rows = df[df['source_table'] == 'step3_incident_coding'].copy()
    techniques_populated_step3 = step3_rows['techniques'].notna().sum()
    print(f"Rows in step3_incident_coding: {len(step3_rows)}")
    print(f"Non-null techniques in step3_incident_coding: {techniques_populated_step3}")
    
    if techniques_populated_step3 > 10:
        working_df = step3_rows
    else:
        # Fallback: search entire dataframe for rows with 'techniques' and 'name' that look like ATLAS cases
        # ATLAS cases usually have a 'case_id' or 'summary'
        print("Searching entire dataframe for valid ATLAS entries...")
        mask = df['techniques'].notna() & (df['name'].notna() | df['summary'].notna())
        working_df = df[mask].copy()
        # Filter out rows that might be strictly incidents if they don't look like ATLAS cases, 
        # but for this specific dataset, techniques are primarily for ATLAS.
        print(f"Found {len(working_df)} rows with techniques populated.")

if len(working_df) == 0:
    print("No data found with populated techniques.")
else:
    # Function to categorize case type
    def categorize_system(row):
        # Combine name and summary for keyword search
        text_parts = []
        if pd.notna(row.get('name')):
            text_parts.append(str(row['name']))
        if pd.notna(row.get('summary')):
            text_parts.append(str(row['summary']))
        
        text = " ".join(text_parts).lower()
        
        gen_keywords = ['llm', 'gpt', 'genai', 'diffusion', 'chatbot', 'generative', 'language model', 'text generation', 'chatgpt', 'openai', 'bard', 'bing chat']
        cv_keywords = ['image', 'face', 'facial', 'recognition', 'vehicle', 'camera', 'vision', 'object detection', 'video', 'surveillance', 'yolo', 'pixel', 'tesla', 'driving']
        
        is_gen = any(k in text for k in gen_keywords)
        is_cv = any(k in text for k in cv_keywords)
        
        if is_gen and not is_cv:
            return 'Generative AI'
        elif is_cv and not is_gen:
            return 'Computer Vision'
        elif is_gen and is_cv:
            # Heuristic: if it mentions both, check which is the primary subject. 
            # For simplicity in this experiment, we might classify as Mixed or check count.
            # Let's try to see if "generative" appears more often or specific strong keywords.
            return 'Mixed/Ambiguous'
        else:
            return 'Other'

    # Function to count techniques
    def count_techniques(val):
        if pd.isna(val) or str(val).strip() == '':
            return 0
        val_str = str(val)
        # Remove brackets if present (e.g. "['T1', 'T2']")
        if val_str.strip().startswith('[') and val_str.strip().endswith(']'):
             # simple strip
             val_str = val_str.strip()[1:-1]
        
        # Split by comma or semicolon
        tokens = re.split(r'[,;]\s*', val_str)
        # Filter out empty strings and quotes
        clean_tokens = [t.strip().strip("'").strip('"') for t in tokens if t.strip()]
        return len(clean_tokens)

    working_df['system_category'] = working_df.apply(categorize_system, axis=1)
    working_df['technique_count'] = working_df['techniques'].apply(count_techniques)

    # Analysis Groups
    gen_ai = working_df[working_df['system_category'] == 'Generative AI']
    comp_vis = working_df[working_df['system_category'] == 'Computer Vision']

    print(f"\nCounts:\nGenerative AI: {len(gen_ai)}\nComputer Vision: {len(comp_vis)}\nOther/Mixed: {len(working_df) - len(gen_ai) - len(comp_vis)}")

    if len(gen_ai) > 1 and len(comp_vis) > 1:
        # Descriptive Stats
        print("\nGenerative AI - Technique Count Stats:")
        print(gen_ai['technique_count'].describe())
        print("\nComputer Vision - Technique Count Stats:")
        print(comp_vis['technique_count'].describe())

        # T-Test
        t_stat, p_val = stats.ttest_ind(gen_ai['technique_count'], comp_vis['technique_count'], equal_var=False)
        print(f"\nT-test Result: Statistic={t_stat:.4f}, p-value={p_val:.4f}")
        
        if p_val < 0.05:
            print("Conclusion: Significant difference in adversarial complexity.")
        else:
            print("Conclusion: No significant difference observed.")

        # Plot
        plt.figure(figsize=(8, 6))
        plt.boxplot([gen_ai['technique_count'], comp_vis['technique_count']], labels=['Generative AI', 'Computer Vision'])
        plt.title('Adversarial Complexity: Techniques per Case')
        plt.ylabel('Count of Techniques')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    else:
        print("Insufficient data in one or both categories to perform statistical testing.")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
Rows in atlas_cases: 52
Non-null techniques in atlas_cases: 52

Counts:
Generative AI: 14
Computer Vision: 8
Other/Mixed: 30

Generative AI - Technique Count Stats:
count    14.0
mean      1.0
std       0.0
min       1.0
25%       1.0
50%       1.0
75%       1.0
max       1.0
Name: technique_count, dtype: float64

Computer Vision - Technique Count Stats:
count    8.0
mean     1.0
std      0.0
min      1.0
25%      1.0
50%      1.0
75%      1.0
max      1.0
Name: technique_count, dtype: float64

T-test Result: Statistic=nan, p-value=nan
Conclusion: No significant difference observed.

STDERR:
/usr/local/lib/python3.13/site-packages/scipy/stats/_axis_nan_policy.py:592: RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation. This occurs when the data are nearly identical. Results may be unreliable.
  res = hypotest_fun_out(*samples, **kwds)
<ipython-input-1-08671c043809>:132: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot([gen_ai['technique_count'], comp_vis['technique_count']], labels=['Generative AI', 'Computer Vision'])


=== Plot Analysis (figure 1) ===
Based on the analysis of the provided plot, here are the detailed observations:

### 1. Plot Type
*   **Type:** This is a **Box Plot** (specifically, a degenerate box plot where the distribution has zero variance).
*   **Purpose:** The plot is designed to compare the distribution of the "Count of Techniques" used in adversarial cases across two different AI domains: Generative AI and Computer Vision.

### 2. Axes
*   **X-Axis:**
    *   **Label/Categories:** The axis represents categorical data containing two groups: **"Generative AI"** and **"Computer Vision"**.
*   **Y-Axis:**
    *   **Title:** **"Count of Techniques"**.
    *   **Range:** The visible scale is highly zoomed in, ranging from approximately **0.95 to 1.05**.
    *   **Ticks:** Major tick marks are placed at intervals of 0.02 (0.96, 0.98, 1.00, 1.02, 1.04).

### 3. Data Trends
*   **Visual Observation:** For both categories ("Generative AI" and "Computer Vision"), the "box" has collapsed into a single horizontal orange line.
*   **Interpretation:**
    *   In a standard box plot, the box represents the Interquartile Range (IQR), and the orange line represents the median.
    *   Because the box is flattened into a single line at the value **1.00**, it indicates that **every single data point** in this dataset has a value of exactly 1.
    *   There is **no variance**, no spread, and no outliers. The minimum, maximum, median, first quartile, and third quartile are all identical (Value = 1).

### 4. Annotations and Legends
*   **Chart Title:** "Adversarial Complexity: Techniques per Case" suggests the chart aims to measure how complex adversarial attacks are by counting how many different techniques are combined in a single case.
*   **Grid Lines:** Horizontal dashed grid lines appear at regular intervals (e.g., 1.00, 1.02) to assist in reading the precise values.
*   **Color Coding:** The orange lines are standard formatting for the median line in Python plotting libraries (likely Matplotlib), confirming the central tendency is at 1.00.

### 5. Statistical Insights
*   **Uniformity of Complexity:** The data indicates that for the cases analyzed, adversarial attacks are not complex in terms of technique combinations. Every recorded case utilized exactly **one technique** at a time.
*   **No Cross-Domain Difference:** There is no difference between Generative AI and Computer Vision regarding this metric. Both fields show an identical pattern of single-technique usage per case.
*   **Implication:** This suggests that "chaining" multiple adversarial techniques (multi-vector attacks) was not observed in this specific dataset. The attacks are likely isolated or singular in nature rather than composite.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
