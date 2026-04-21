# Experiment 18: node_3_8

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_8` |
| **ID in Run** | 18 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T01:48:56.606527+00:00 |
| **Runtime** | 469.4s |
| **Parent** | `node_2_4` |
| **Children** | `node_4_9`, `node_4_22`, `node_4_48` |
| **Creation Index** | 19 |

---

## Hypothesis

> The Intentionality of Greed: AI incidents resulting in 'Financial' harm are
significantly more likely to be intentional acts compared to incidents resulting
in 'Physical' or 'Civil Rights' harm.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7500 (Likely True) |
| **Posterior** | 0.2995 (Likely False) |
| **Surprise** | -0.5407 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 1.0 |
| Maybe True | 29.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 16.0 |
| Definitely False | 44.0 |

---

## Experiment Plan

**Objective:** Determine if the domain of harm predicts the intentionality of the AI incident.

### Steps
- 1. Filter for `aiid_incidents`.
- 2. Group records by `73: Harm Domain` (e.g., Financial, Physical, Civil Rights).
- 3. Convert `82: Intentional Harm` to a binary variable (Intentional vs. Unintentional/Accidental).
- 4. Calculate the percentage of intentional incidents per harm domain.
- 5. Perform a Chi-Square test or pairwise Z-tests for proportions.

### Deliverables
- Bar chart of Intentionality Rate by Harm Domain; Statistical summary showing which domains attract more malicious actors.

---

## Analysis

The experiment was successfully executed using a text-mining approach to
overcome missing structured labels for Harm Domains. The analysis processed 191
AIID incidents that had valid 'Intentional Harm' labels.

Contrary to the hypothesis that 'Financial' harm is more likely to be
intentional (the 'Intentionality of Greed'), the results showed a **0%
intentionality rate** for Financial incidents (0 out of 6). In comparison,
'Physical' harm had the highest intentionality rate at 5.7% (3 out of 53),
followed by 'Civil Rights' at 4.3% (3 out of 70).

The overall Chi-Square test (p=0.87) indicated no statistically significant
relationship between harm domain and intentionality, largely due to the sparsity
of the data (only 8 confirmed intentional incidents in the entire valid subset).
Consequently, the hypothesis is rejected based on this dataset; there is no
evidence here that financial gain drives higher rates of intentional AI harm
compared to other domains.

---

## Review

The experiment was faithfully executed, adapting to significant data quality
issues (specifically the lack of structured 'Harm Domain' labels in the subset
of records with valid 'Intentionality' tags) by employing text analysis to infer
categories. The analysis correctly identified that the available data does not
support the hypothesis.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import sys
import numpy as np

print("Starting experiment: The Intentionality of Greed (Attempt 5 - Text Analysis)...")

# 1. Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

# 2. Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Loaded {len(aiid_df)} AIID incidents.")

# 3. Clean Intentionality
# Using the logic validated in previous steps
intent_col = 'Intentional Harm'
if intent_col not in aiid_df.columns:
    cols = [c for c in aiid_df.columns if 'Intentional Harm' in str(c)]
    if cols: 
        intent_col = cols[0]
    else:
        print("Column 'Intentional Harm' not found.")
        sys.exit(1)

def map_intentionality(val):
    s = str(val).lower()
    if 'yes. intentionally' in s:
        return 1
    elif 'no. not intentionally' in s:
        return 0
    return None

aiid_df['is_intentional'] = aiid_df[intent_col].apply(map_intentionality)
analysis_df = aiid_df.dropna(subset=['is_intentional']).copy()
print(f"Records with valid intentionality: {len(analysis_df)}")

# 4. Infer Harm Domain from Description/Title
# Since structured columns failed, we mine the text.
analysis_df['text_content'] = analysis_df['title'].fillna('') + " " + analysis_df['description'].fillna('')
analysis_df['text_content'] = analysis_df['text_content'].str.lower()

def infer_domain(text):
    # Keywords
    financial_keys = ['financial', 'money', 'cost', 'dollar', 'bank', 'fraud', 'theft', 
                      'market', 'stock', 'economy', 'economic', 'credit', 'price', 'fund', 
                      'wage', 'salary', 'billing', 'fee', 'crypto', 'currency']
    
    physical_keys = ['death', 'dead', 'kill', 'die', 'injury', 'injure', 'hurt', 'physical', 
                     'crash', 'collision', 'accident', 'safety', 'robot', 'autonomous vehicle', 
                     'drone', 'weapon', 'assault', 'violence', 'hospital', 'medical']
    
    civil_keys = ['discrimination', 'discriminat', 'bias', 'racist', 'sexist', 'gender', 'race', 
                  'black', 'white', 'woman', 'man', 'arrest', 'police', 'surveillance', 'privacy', 
                  'facial recognition', 'civil rights', 'liberties', 'censorship', 'profile', 'profiling']
    
    # Check presence
    has_fin = any(k in text for k in financial_keys)
    has_phy = any(k in text for k in physical_keys)
    has_civ = any(k in text for k in civil_keys)
    
    # Priority resolution if multiple match (Physical > Civil > Financial for categorization purposes, 
    # though the hypothesis focuses on Financial. We assign primarily based on what the text *likely* is about.)
    # Let's count matches to be smarter? No, simple priority for now to avoid over-complication.
    
    if has_phy: return 'Physical'
    if has_civ: return 'Civil Rights'
    if has_fin: return 'Financial'
    return 'Other'

analysis_df['inferred_domain'] = analysis_df['text_content'].apply(infer_domain)

# 5. Generate Statistics
summary = analysis_df.groupby('inferred_domain')['is_intentional'].agg(['count', 'mean', 'sum'])
summary.columns = ['Total Incidents', 'Intentionality Rate', 'Intentional Count']
print("\nSummary Statistics by Inferred Harm Domain:")
print(summary)

# 6. Statistical Tests
contingency_table = pd.crosstab(analysis_df['inferred_domain'], analysis_df['is_intentional'])
print("\nContingency Table (0=Unintentional, 1=Intentional):")
print(contingency_table)

# Chi-Square
if len(summary) > 1 and contingency_table.sum().sum() > 0:
    try:
        chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
        print(f"\nOverall Chi-Square Test: Chi2={chi2:.4f}, p-value={p:.4e}")
    except ValueError:
        print("Chi-square failed.")

# Pairwise Comparisons (Financial vs Others)
target = 'Financial'
others = ['Physical', 'Civil Rights']

print(f"\nPairwise Comparisons ({target} vs X):")
for other in others:
    if target in summary.index and other in summary.index:
        subset = analysis_df[analysis_df['inferred_domain'].isin([target, other])]
        ct_sub = pd.crosstab(subset['inferred_domain'], subset['is_intentional'])
        
        # Check if we have both 0s and 1s in the subset to avoid errors
        if ct_sub.shape == (2, 2):
            c2, pv, _, _ = stats.chi2_contingency(ct_sub)
            r1 = summary.loc[target, 'Intentionality Rate']
            r2 = summary.loc[other, 'Intentionality Rate']
            print(f"  {target} ({r1:.1%}) vs {other} ({r2:.1%}): Chi2={c2:.4f}, p={pv:.4e}")
        else:
            print(f"  {target} vs {other}: Insufficient variance (likely 0 intentional in one group).")
    else:
        print(f"  {other} category missing from data.")

# 7. Visualization
if not summary.empty:
    plt.figure(figsize=(10, 6))
    
    # Standard Error
    summary['se'] = summary.apply(lambda row: 
        stats.sem(analysis_df[analysis_df['inferred_domain'] == row.name]['is_intentional']) 
        if row['Total Incidents'] > 1 else 0, axis=1)
    
    sns.barplot(x=summary.index, y=summary['Intentionality Rate'], hue=summary.index, palette='coolwarm', legend=False)
    plt.errorbar(x=range(len(summary)), y=summary['Intentionality Rate'], 
                 yerr=summary['se'], fmt='none', c='black', capsize=5)
    
    plt.title('Intentionality Rate by Inferred Harm Domain')
    plt.ylabel('Proportion of Intentional Incidents')
    plt.xlabel('Harm Domain (Inferred)')
    plt.axhline(y=analysis_df['is_intentional'].mean(), color='gray', linestyle='--', label='Global Average')
    plt.legend()
    plt.tight_layout()
    plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting experiment: The Intentionality of Greed (Attempt 5 - Text Analysis)...
Loaded 1362 AIID incidents.
Records with valid intentionality: 191

Summary Statistics by Inferred Harm Domain:
                 Total Incidents  Intentionality Rate  Intentional Count
inferred_domain                                                         
Civil Rights                  70             0.042857                3.0
Financial                      6             0.000000                0.0
Other                         62             0.032258                2.0
Physical                      53             0.056604                3.0

Contingency Table (0=Unintentional, 1=Intentional):
is_intentional   0.0  1.0
inferred_domain          
Civil Rights      67    3
Financial          6    0
Other             60    2
Physical          50    3

Overall Chi-Square Test: Chi2=0.6932, p-value=8.7479e-01

Pairwise Comparisons (Financial vs X):
  Financial (0.0%) vs Physical (5.7%): Chi2=0.0000, p=1.0000e+00
  Financial (0.0%) vs Civil Rights (4.3%): Chi2=0.0000, p=1.0000e+00


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Chart with Error Bars.
*   **Purpose:** The plot is designed to compare the "Intentionality Rate" (proportion of incidents deemed intentional) across four different categories of inferred harm domains. The inclusion of error bars indicates the variability or uncertainty (likely standard error or confidence intervals) associated with the mean measurement for each category.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Harm Domain (Inferred)"
    *   **Categories:** The axis displays categorical data representing four domains: "Civil Rights," "Financial," "Other," and "Physical."
*   **Y-Axis:**
    *   **Title:** "Proportion of Intentional Incidents"
    *   **Units:** The values represent a ratio/proportion (0.0 to 1.0 scale), effectively equivalent to percentages if multiplied by 100.
    *   **Range:** The axis ticks range from **0.00 to 0.08**, though the error bar for the "Physical" category extends slightly higher, reaching approximately **0.09**.

### 3. Data Trends
*   **Tallest Bar:** The **"Physical"** harm domain has the highest mean intentionality rate, reaching approximately **0.057**.
*   **Shortest Bar:** The **"Financial"** harm domain has the lowest rate, with a bar that is barely visible, suggesting a value very close to **0.00**.
*   **Patterns:**
    *   **"Civil Rights"** shows a moderate rate (approx. **0.043**), very closely aligned with the global average.
    *   **"Other"** falls below the global average at approximately **0.032**.
*   **Variability (Error Bars):**
    *   The **"Physical"** domain has the largest error bar (spanning from ~0.025 to ~0.09), indicating the highest uncertainty or variance in the data for this category.
    *   The **"Civil Rights"** and **"Other"** domains also show significant error margins.
    *   The **"Financial"** domain shows a very small error bar, consistent with its near-zero value.

### 4. Annotations and Legends
*   **Legend:** A legend in the top-left corner identifies the horizontal dashed grey line as the **"Global Average."**
*   **Global Average Line:** This horizontal dashed line runs across the entire plot at a y-value of approximately **0.042**. It serves as a baseline to see which categories perform above or below the overall mean of the dataset.

### 5. Statistical Insights
*   **High Uncertainty in Physical Harm:** While "Physical" harm incidents have the highest *average* rate of intentionality, the large error bar overlaps significantly with the "Civil Rights" and "Other" categories. This suggests that while physical harm is seemingly more intentional on average, there is wide variability in the incidents recorded.
*   **Financial Incidents are Unintentional:** The data suggests a strong statistical finding that incidents falling under the "Financial" harm domain are almost never classified as intentional (the rate is near zero).
*   **Civil Rights as the Baseline:** The "Civil Rights" domain sits almost exactly on the Global Average line. This implies that the frequency of intentionality in Civil Rights cases is representative of the dataset as a whole.
*   **Significance:** The error bars for "Civil Rights," "Other," and "Physical" all overlap with the Global Average line. This suggests that, statistically, the deviation of these specific domains from the global average might not be highly significant, unlike the "Financial" domain, which is clearly distinct from the others.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
