# Experiment 12: node_3_3

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_3` |
| **ID in Run** | 12 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T21:52:11.672710+00:00 |
| **Runtime** | 202.4s |
| **Parent** | `node_2_0` |
| **Children** | `node_4_2`, `node_4_14` |
| **Creation Index** | 13 |

---

## Hypothesis

> The presence of 'Prompt Injection' techniques in an incident is a significant
predictor that the primary failure mode is a 'Prevention Failure' rather than a
'Detection' or 'Response' failure.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9597 (Definitely True) |
| **Posterior** | 0.4318 (Maybe False) |
| **Surprise** | -0.6126 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 27.0 |
| Maybe True | 3.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 90.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Test if specific attack techniques force specific failure modes.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Create a boolean column 'has_prompt_injection' by checking if 'Prompt Injection' (or similar string) exists in the 'techniques_used' column.
- 3. Create a contingency table of 'has_prompt_injection' vs. 'failure_mode' (grouping Detection/Response together if counts are low).
- 4. Perform a Chi-Square test or Fisher's Exact Test.
- 5. Visualize the proportion of Prevention Failures for incidents with vs. without Prompt Injection.

### Deliverables
- Grouped bar chart and statistical test results.

---

## Analysis

The experiment successfully analyzed the relationship between 'Prompt Injection'
techniques and 'Prevention Failure' modes. The data reveals a significant class
imbalance: 51 out of 52 incidents (98%) are classified as 'Prevention Failure',
with only a single incident classified as 'Detection/Response'. Consequently,
while 100% of Prompt Injection cases (n=20) were indeed Prevention Failures, 97%
of non-Prompt Injection cases (31/32) were also Prevention Failures. Fisher's
Exact Test yielded a p-value of 1.000, indicating no statistically significant
difference between the groups. The hypothesis is not supported in a meaningful
way; while Prompt Injection incidents do result in Prevention Failures, this is
simply the dominant baseline for all incidents in the dataset rather than a
specific outcome of the attack technique.

---

## Review

The experiment was successfully executed, confirming that while 'Prompt
Injection' incidents are indeed exclusively associated with 'Prevention
Failures' (20/20), this is not a distinguishing predictor. The dataset is
overwhelmingly dominated by Prevention Failures (51/52 incidents, 98%)
regardless of the attack technique used. Consequently, the Fisher's Exact Test
yielded a p-value of 1.000, indicating no statistically significant association.
The hypothesis that Prompt Injection *specifically* forces this failure mode is
not supported, as Prevention Failure is the baseline for nearly all observed
incidents.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
import os

# Define file path (one level above as per instructions)
dataset_path = '../step3_incident_coding.csv'

# Check if file exists, fallback to current dir if not (for robustness)
if not os.path.exists(dataset_path):
    dataset_path = 'step3_incident_coding.csv'

try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    print(f"Error: Could not find dataset at {dataset_path}")
    sys.exit(1)

print("=== Loading and Preprocessing Data ===")
print(f"Dataset loaded: {dataset_path}")
print(f"Shape: {df.shape}")

# --- Step 1: Feature Engineering (Prompt Injection) ---
# We check 'techniques_used' for the string 'Prompt Injection'. 
# We also check for 'AML.T0051' (ATLAS ID for LLM Prompt Injection) just in case the column uses IDs.
df['techniques_used'] = df['techniques_used'].fillna('')
df['has_prompt_injection'] = df['techniques_used'].astype(str).str.contains('Prompt Injection|AML.T0051', case=False, regex=True)

print("\n--- Distribution of Prompt Injection ---")
print(df['has_prompt_injection'].value_counts())

# Debug: Show a few techniques to confirm format
print("\n[Debug] First 5 'techniques_used' entries:")
print(df['techniques_used'].head().tolist())

# --- Step 2: Feature Engineering (Failure Mode) ---
# Categorize into Prevention vs. Non-Prevention (Detection/Response)
def categorize_failure(mode):
    if pd.isna(mode):
        return 'Unknown'
    mode_str = str(mode).lower()
    if 'prevention' in mode_str:
        return 'Prevention'
    elif 'detection' in mode_str or 'response' in mode_str:
        return 'Detection/Response'
    else:
        return 'Other'

df['failure_category'] = df['failure_mode'].apply(categorize_failure)

print("\n--- Failure Category Distribution ---")
print(df['failure_category'].value_counts())

# --- Step 3: Contingency Table ---
# Rows: Has Prompt Injection (False/True)
# Cols: Failure Category (Prevention/Detection+Response)
contingency_table = pd.crosstab(df['has_prompt_injection'], df['failure_category'])

# Ensure we have the specific columns we want to test
expected_cols = ['Prevention', 'Detection/Response']
for col in expected_cols:
    if col not in contingency_table.columns:
        contingency_table[col] = 0

# Reorder for consistency
contingency_table = contingency_table[expected_cols]

print("\n--- Contingency Table (Observed) ---")
print(contingency_table)

# --- Step 4: Statistical Test ---
# Using Fisher's Exact Test due to potential small sample sizes in the 'Detection/Response' column
try:
    # Fisher's Exact Test requires a 2x2 table
    # Table structure: [[No_Prev, No_Det], [Yes_Prev, Yes_Det]]
    if contingency_table.shape == (2, 2):
        odds_ratio, p_value = stats.fisher_exact(contingency_table)
        print("\n=== Statistical Test Results (Fisher's Exact Test) ===")
        print(f"Odds Ratio: {odds_ratio}")
        print(f"P-value: {p_value:.4f}")
        
        alpha = 0.05
        if p_value < alpha:
            print("Conclusion: Statistically SIGNIFICANT association between Prompt Injection and Failure Mode.")
        else:
            print("Conclusion: NO statistically significant association found.")
    else:
        print("\nCannot perform Fisher's Exact Test: Contingency table is not 2x2 (likely missing one category entirely).")
except Exception as e:
    print(f"\nError performing statistical test: {e}")

# --- Step 5: Visualization ---
# Grouped bar chart to visualize the counts
# We plot the contingency table directly
ax = contingency_table.plot(kind='bar', figsize=(10, 6), rot=0, color=['#1f77b4', '#ff7f0e'])

plt.title('Failure Mode Distribution by Presence of Prompt Injection')
plt.xlabel('Has Prompt Injection')
plt.ylabel('Count of Incidents')
plt.xticks([0, 1], ['No', 'Yes'])
plt.legend(title='Failure Category')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: === Loading and Preprocessing Data ===
Dataset loaded: step3_incident_coding.csv
Shape: (52, 22)

--- Distribution of Prompt Injection ---
has_prompt_injection
False    32
True     20
Name: count, dtype: int64

[Debug] First 5 'techniques_used' entries:
['AML.T0000.001; AML.T0002.000; AML.T0005; AML.T0015; AML.T0042; AML.T0043.003', 'AML.T0000; AML.T0002; AML.T0015; AML.T0017.000; AML.T0042; AML.T0043.001', 'AML.T0010.002; AML.T0016.000; AML.T0020; AML.T0043', 'AML.T0000; AML.T0015; AML.T0017.000; AML.T0043.003; AML.T0047; AML.T0063', 'AML.T0008.001; AML.T0015; AML.T0016.000; AML.T0016.001; AML.T0021; AML.T0047; AML.T0048.000; AML.T0087']

--- Failure Category Distribution ---
failure_category
Prevention            51
Detection/Response     1
Name: count, dtype: int64

--- Contingency Table (Observed) ---
failure_category      Prevention  Detection/Response
has_prompt_injection                                
False                         31                   1
True                          20                   0

=== Statistical Test Results (Fisher's Exact Test) ===
Odds Ratio: 0.0
P-value: 1.0000
Conclusion: NO statistically significant association found.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Grouped Bar Chart (or Clustered Bar Chart).
*   **Purpose:** This chart compares the frequency of specific failure categories ("Prevention" and "Detection/Response") across two different conditions regarding the presence of prompt injection ("No" and "Yes"). It allows for easy comparison of categorical data distributions.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Has Prompt Injection"
    *   **Categories:** Two categorical values: "No" and "Yes".
*   **Y-Axis:**
    *   **Label:** "Count of Incidents"
    *   **Range:** The axis ranges from 0 to roughly 32, with grid lines marked at intervals of 5 (0, 5, 10, 15, 20, 25, 30).
    *   **Units:** Integer counts of incidents.

### 3. Data Trends
*   **Dominant Category:** The "Prevention" failure category (blue bars) is significantly higher than "Detection/Response" (orange bars) in both scenarios.
*   **"No" Prompt Injection Group:**
    *   This group has the highest single bar on the chart. The "Prevention" count is slightly above the 30 grid line (estimated at **31**).
    *   The "Detection/Response" count is very low (estimated at **1**).
*   **"Yes" Prompt Injection Group:**
    *   The "Prevention" count is exactly on the **20** grid line.
    *   The "Detection/Response" bar is not visible, implying a count of **0**.
*   **Overall Volume:** There appear to be more total incidents recorded for cases without prompt injection (approx. 32) compared to cases with prompt injection (20).

### 4. Annotations and Legends
*   **Chart Title:** "Failure Mode Distribution by Presence of Prompt Injection" – situated at the top center.
*   **Legend:** Located in the top right corner with the title "Failure Category".
    *   **Blue:** Represents "Prevention".
    *   **Orange:** Represents "Detection/Response".
*   **Grid:** Horizontal dashed grid lines are present to assist with estimating the height of the bars.

### 5. Statistical Insights
*   **Imbalance in Failure Modes:** The data suggests a massive imbalance in failure modes. "Prevention" failures account for nearly 100% of the incidents shown (approx. 51 out of 52 total incidents). This indicates that the system struggles primarily with preventing issues rather than detecting or responding to them—or, alternatively, that detection/response mechanisms are rarely tested or logged as failures in this dataset.
*   **Counter-intuitive Volume:** One might expect "Prompt Injection" scenarios to generate more failures, but the data shows higher incident counts when prompt injection is *not* present (approx. 32 vs. 20).
*   **Absence of Detection Failures in Injection Scenarios:** The fact that there are zero "Detection/Response" failures when prompt injection is present ("Yes") is notable. This could mean detection was perfect (no failures), or perhaps more likely, that the focus of these specific incidents was entirely on the prevention layer, rendering detection metrics moot.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
