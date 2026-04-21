# Experiment 198: node_5_77

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_77` |
| **ID in Run** | 198 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T10:19:56.456121+00:00 |
| **Runtime** | 240.6s |
| **Parent** | `node_4_15` |
| **Children** | None |
| **Creation Index** | 199 |

---

## Hypothesis

> Intentionality of Failure Modes: Incidents classified as 'Intentional Harm' are
significantly more likely to manifest as 'Security' or 'Privacy' failures,
whereas Unintentional incidents are dominated by 'Safety' and 'Reliability'
failures.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9839 (Definitely True) |
| **Posterior** | 0.9054 (Definitely True) |
| **Surprise** | -0.0942 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | No |

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
| Uncertain | 6.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Map the intent of AI incidents to specific technical failure modes.

### Steps
- 1. Load 'aiid_incidents'.
- 2. Clean 'Intentional Harm' to binary (True/Yes vs False/No).
- 3. Categorize 'Known AI Technical Failure' into broad groups: 'Security' (Adversarial, Privacy, Leakage) vs 'Reliability' (Safety, Error, Bias, Robustness). Use keyword mapping.
- 4. Create a contingency table: Intent vs Failure Category.
- 5. Perform a Chi-square test.
- 6. Identify the most overrepresented failure type for Intentional harms using standardized residuals.

### Deliverables
- Heatmap of Intent vs Failure Category, Chi-square results, and residual analysis.

---

## Analysis

The experiment was executed, but the hypothesis could not be tested due to data
limitations. After processing the 'aiid_incidents' dataset (N=1,362) and
cleaning the 'Intentional Harm' column, the analysis revealed that **0
incidents** were classified as 'Intentional'. All 1,362 records were categorized
as 'Unintentional' (likely due to the column being empty, sparse, or containing
nulls treated as false in this specific dataset snapshot).

Because the independent variable ('Intent') lacked variance (containing only one
level), the Chi-square test of independence could not be meaningfully performed
(Statistic=0.0, p=1.0).

**Findings:**
- **Hypothesis Status:** **Untestable**. It is impossible to determine if
intentional incidents differ from unintentional ones without a comparison group.
- **Descriptive Statistics:** Among the 'Unintentional' incidents, the majority
of technical failures were classified as 'Other' (51.8%) or 'Safety/Reliability'
(40.3%). Failures related to 'Security/Privacy' were comparatively rare,
accounting for only 7.9% of the unintentional cases.

---

## Review

The experiment was faithfully implemented in code, but the results reveal a
critical data limitation that prevented the testing of the hypothesis.
Specifically, the data cleaning process for the 'Intentional Harm' column
resulted in 100% of the 1,362 incidents being classified as 'Unintentional'.
Because the independent variable ('Intent') had zero variance (only one
category), the Chi-square test of independence was mathematically vacuous
(p=1.0) and could not evaluate the relationship.

While the code execution was successful, the scientific outcome is that the
hypothesis is **untestable** with the current dataset configuration. The
descriptive statistics provided for the 'Unintentional' group are valid, showing
a dominance of 'Safety/Reliability' and 'Other' failures over 'Security'
failures, but the comparative claim regarding 'Intentional' incidents remains
unverified.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# [debug] Start of experiment execution
print("Starting experiment: Intentionality of Failure Modes")

# 1. Load Data
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    # Fallback if file is in current directory
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    print("Dataset loaded from current directory.")

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid_df)} rows")

# 2. Clean 'Intentional Harm' (Column index 82 usually, name 'Intentional Harm' or similar)
# Let's find the exact column name
intent_col = [c for c in aiid_df.columns if 'Intentional Harm' in c]
if not intent_col:
    # Fallback based on known schema from previous steps
    # The column might be named '82_Intentional Harm' or just 'Intentional Harm' depending on header processing
    # In the provided previous output, it was '82: Intentional Harm'. 
    # However, pandas usually reads the header. Let's look for partial match.
    intent_col = [c for c in aiid_df.columns if 'Intentional' in c]

if intent_col:
    intent_col = intent_col[0]
    print(f"Using column for Intent: '{intent_col}'")
else:
    print("Could not find Intentional Harm column. Listing columns:")
    print(aiid_df.columns.tolist()[:20])
    exit(1)

# Normalize Intent
# Assuming values like 'Yes', 'No', 'True', 'False', or boolean
aiid_df['intent_clean'] = aiid_df[intent_col].astype(str).str.lower().map({
    'yes': 'Intentional',
    'true': 'Intentional',
    '1': 'Intentional',
    '1.0': 'Intentional',
    'no': 'Unintentional',
    'false': 'Unintentional',
    '0': 'Unintentional',
    '0.0': 'Unintentional'
}).fillna('Unintentional') # Treat NaNs as Unintentional for now, or exclude. 
# Let's verify distribution
print("Intent distribution:")
print(aiid_df['intent_clean'].value_counts())

# 3. Categorize 'Known AI Technical Failure' 
# Find column
tech_col = [c for c in aiid_df.columns if 'Technical Failure' in c and 'Known' in c]
if tech_col:
    tech_col = tech_col[0]
    print(f"Using column for Technical Failure: '{tech_col}'")
else:
    print("Could not find Technical Failure column.")
    exit(1)

def categorize_failure(val):
    if pd.isna(val):
        return 'Unknown'
    val_str = str(val).lower()
    
    # Security / Privacy keywords
    security_keywords = ['adversarial', 'attack', 'privacy', 'security', 'leakage', 
                         'extraction', 'inversion', 'poisoning', 'evasion', 'model theft']
    for kw in security_keywords:
        if kw in val_str:
            return 'Security/Privacy'
            
    # Safety / Reliability keywords
    safety_keywords = ['error', 'bias', 'fairness', 'robustness', 'safety', 'accident', 
                       'hallucination', 'malfunction', 'performance', 'reliability', 'unsafe']
    for kw in safety_keywords:
        if kw in val_str:
            return 'Safety/Reliability'
            
    return 'Other'

aiid_df['failure_category'] = aiid_df[tech_col].apply(categorize_failure)

# Filter out 'Unknown' and 'Other' to focus on the hypothesis comparison if needed, 
# but let's keep 'Other' to see the full picture, remove 'Unknown' for stats.
analysis_df = aiid_df[aiid_df['failure_category'] != 'Unknown'].copy()

print("\nFailure Category Distribution:")
print(analysis_df['failure_category'].value_counts())

# 4. Contingency Table
contingency = pd.crosstab(analysis_df['intent_clean'], analysis_df['failure_category'])
print("\nContingency Table:")
print(contingency)

# 5. Chi-square Test
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# 6. Residual Analysis (Standardized Residuals)
# (Observed - Expected) / sqrt(Expected)
residuals = (contingency - expected) / np.sqrt(expected)
print("\nStandardized Residuals:")
print(residuals)

# Identify overrepresented pairs
print("\nInterpretation:")
if p < 0.05:
    print("Significant relationship found between Intent and Failure Mode.")
    # Check specific hypothesis cells
    try:
        sec_resid = residuals.loc['Intentional', 'Security/Privacy']
        safe_resid = residuals.loc['Unintentional', 'Safety/Reliability']
        print(f"Residual for Intentional -> Security/Privacy: {sec_resid:.2f}")
        print(f"Residual for Unintentional -> Safety/Reliability: {safe_resid:.2f}")
        
        if sec_resid > 1.96:
            print("Confirmed: Intentional incidents are significantly associated with Security/Privacy failures.")
        else:
            print("Intentional incidents are NOT significantly associated with Security/Privacy failures.")
            
        if safe_resid > 1.96:
            print("Confirmed: Unintentional incidents are significantly associated with Safety/Reliability failures.")
    except KeyError:
        print("Could not compute specific residuals due to missing categories in data.")
else:
    print("No significant relationship found.")

# Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(contingency, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Heatmap of Intent vs Technical Failure Mode')
plt.ylabel('Intent')
plt.xlabel('Failure Category')
plt.tight_layout()
plt.show()

# Bar chart of percentages
contingency_pct = pd.crosstab(analysis_df['intent_clean'], analysis_df['failure_category'], normalize='index') * 100
contingency_pct.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Technical Failure Modes by Intent')
plt.ylabel('Percentage')
plt.xlabel('Intent')
plt.legend(title='Failure Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting experiment: Intentionality of Failure Modes
Dataset loaded from current directory.
AIID Incidents loaded: 1362 rows
Using column for Intent: 'Intentional Harm'
Intent distribution:
intent_clean
Unintentional    1362
Name: count, dtype: int64
Using column for Technical Failure: 'Known AI Technical Failure'

Failure Category Distribution:
failure_category
Other                 144
Safety/Reliability    112
Security/Privacy       22
Name: count, dtype: int64

Contingency Table:
failure_category  Other  Safety/Reliability  Security/Privacy
intent_clean                                                 
Unintentional       144                 112                22

Chi-square Statistic: 0.0000
P-value: 1.0000e+00

Standardized Residuals:
failure_category  Other  Safety/Reliability  Security/Privacy
intent_clean                                                 
Unintentional       0.0                 0.0               0.0

Interpretation:
No significant relationship found.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis of the plot:

### 1. Plot Type
*   **Type:** Heatmap (specifically, an annotated confusion matrix or frequency count visualization).
*   **Purpose:** The plot visualizes the relationship between "Intent" (on the y-axis) and "Technical Failure Mode" (on the x-axis) by using color intensity and numerical labels to represent the frequency or count of occurrences for each intersection of categories.

### 2. Axes
*   **X-Axis:**
    *   **Title:** Failure Category
    *   **Labels:** The axis represents categorical data with three distinct groups: "Other", "Safety/Reliability", and "Security/Privacy".
*   **Y-Axis:**
    *   **Title:** Intent
    *   **Labels:** The axis represents categorical data, showing a single category: "Unintentional".
*   **Color Scale (Z-Axis equivalent):**
    *   Represented by a vertical color bar on the right side.
    *   **Range:** The scale runs from approximately 20 (light yellow) to roughly 145 (dark blue).

### 3. Data Trends
*   **High Values:** The area of highest density is the intersection of "Unintentional" intent and the "Other" failure category, indicated by the darkest blue color and the value **144**.
*   **Medium Values:** The intersection of "Unintentional" and "Safety/Reliability" shows a moderate to high frequency with a count of **112**, represented by a medium blue shade.
*   **Low Values:** The area of lowest density is the intersection of "Unintentional" and "Security/Privacy", indicated by the light yellow color and a significantly lower value of **22**.

### 4. Annotations and Legends
*   **Chart Title:** "Heatmap of Intent vs Technical Failure Mode" clearly defines the scope of the visualization.
*   **Cell Annotations:** Each cell contains a specific numerical value (144, 112, 22) overlaid on the color, providing exact counts rather than requiring the viewer to estimate based on the color bar.
*   **Color Legend:** The bar on the right guides the interpretation of the colors, where light yellow represents low frequency and dark blue represents high frequency.

### 5. Statistical Insights
*   **Dominant Failure Modes:** When the intent is "Unintentional," the vast majority of technical failures fall under the "Other" category (51.8% of the displayed data), followed closely by "Safety/Reliability" (40.3%).
*   **Disparity in Categories:** There is a significant disparity between "Security/Privacy" failures and the other two categories. "Other" failures occur approximately **6.5 times** more frequently than "Security/Privacy" failures in this dataset.
*   **Safety vs. Security:** For unintentional incidents, issues related to Safety/Reliability are far more common than those related to Security/Privacy (112 vs 22 counts). This suggests that unintentional errors are much more likely to result in reliability or safety glitches than in security breaches.
==================================================

=== Plot Analysis (figure 2) ===
Based on the provided image, here is the analysis:

### 1. Plot Type
*   **Type:** Stacked Bar Chart (specifically a 100% stacked bar chart for a single category).
*   **Purpose:** The plot visualizes the composition of "Technical Failure Modes" within the specific context of "Unintentional" intent. It breaks down the total failures (100%) into three distinct sub-categories to show their relative proportions.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Intent"
    *   **Category:** There is a single category displayed: "Unintentional" (text is rotated vertically).
*   **Y-Axis:**
    *   **Label:** "Percentage"
    *   **Range:** 0 to 100.
    *   **Scale:** Linear increments of 20 (0, 20, 40, 60, 80, 100).

### 3. Data Trends
*   **Dominant Category:** The **"Other"** category (blue) is the largest segment, appearing to occupy slightly more than half of the total bar (approximately 52%).
*   **Secondary Category:** **"Safety/Reliability"** (orange) is the second largest component. It spans from roughly the 52% mark up to the 92% mark, representing about 40% of the total.
*   **Minor Category:** **"Security/Privacy"** (green) is the smallest segment, occupying the top portion of the bar (roughly from 92% to 100%), representing approximately 8%.

### 4. Annotations and Legends
*   **Plot Title:** "Technical Failure Modes by Intent"
*   **Legend:** Located on the right side, titled "Failure Category," mapping colors to specific failure types:
    *   **Blue:** Other
    *   **Orange:** Safety/Reliability
    *   **Green:** Security/Privacy

### 5. Statistical Insights
*   **Majority of Unintentional Failures are Miscellaneous:** More than 50% of the technical failure modes classified as "Unintentional" fall into the "Other" category. This suggests a wide variety of miscellaneous errors that do not fit strictly into safety or security buckets.
*   **Significant Safety Component:** A substantial portion (approx. 40%) of unintentional failures are attributed to "Safety/Reliability," indicating that when systems fail unintentionally, operational reliability is a major factor.
*   **Low Security Correlation:** Only a small fraction (approx. <10%) of unintentional failures are categorized as "Security/Privacy." This implies that security and privacy breaches are rarely the primary mode of failure when the intent is unintentional; conversely, it suggests that security/privacy failures might be more strongly associated with *intentional* acts (though that data is not shown here for comparison).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
