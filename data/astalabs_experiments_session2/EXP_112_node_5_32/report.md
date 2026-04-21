# Experiment 112: node_5_32

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_32` |
| **ID in Run** | 112 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T06:18:58.084185+00:00 |
| **Runtime** | 429.2s |
| **Parent** | `node_4_41` |
| **Children** | `node_6_27` |
| **Creation Index** | 113 |

---

## Hypothesis

> Commercial Opacity: Government AI systems procured from commercial vendors are
significantly less likely to have available data documentation compared to
custom/internally developed systems.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7984 (Likely True) |
| **Posterior** | 0.9313 (Definitely True) |
| **Surprise** | +0.1595 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 7.0 |
| Maybe True | 23.0 |
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

**Objective:** Investigate if commercial 'black-box' procurement hinders transparency compliance.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `eo13960_scored`.
- 2. Create a binary variable `is_commercial` from column `10_commercial_ai` (Yes vs No).
- 3. Create a binary variable `has_data_docs` from column `34_data_docs` (checking for affirmative responses).
- 4. Generate a contingency table.
- 5. Perform a Chi-Square test of independence.
- 6. Calculate the odds ratio to quantify the likelihood of documentation availability.

### Deliverables
- Contingency table; Bar chart of Data Documentation rates by Source (Commercial vs Internal); Chi-Square statistics.

---

## Analysis

The experiment successfully tested the 'Commercial Opacity' hypothesis using the
EO13960 dataset. After correcting initial variable selection errors during
debugging, the analysis focused on 920 AI systems, distinguishing between those
developed 'in-house' (Internal, n=439) and those using 'contracting resources'
(Commercial, n=481).

The results strongly support the hypothesis. Internal systems demonstrated a
significantly higher data documentation rate of 82.69% compared to 62.16% for
Commercial systems. A Chi-Square test confirmed this difference is statistically
significant (Chi2=46.91, p < 0.001). The Odds Ratio of 0.344 indicates that
Commercial systems are roughly one-third as likely to have available data
documentation (in terms of odds) compared to Internal systems. The visualization
clearly illustrates this gap, with the 'No Docs' portion being more than double
for Commercial systems (37.8%) compared to Internal ones (17.3%). Validating the
hypothesis suggests that commercial procurement acts as a barrier to
transparency compliance in government AI inventories.

---

## Review

The experiment was successfully executed and the results are statistically
significant. The code robustly handled the text-based classification of both the
procurement source (using '22_dev_method') and documentation status (using
'34_data_docs'), overcoming previous issues with data sparsity and mixed types.
The statistical analysis (Chi-Square) and visualization (Stacked Bar Chart)
directly addressed the research question.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import numpy as np

# Define file path
file_name = 'astalabs_discovery_all_data.csv'
file_path = f'../{file_name}'
if not os.path.exists(file_path):
    file_path = file_name

# Load dataset
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    print(f"Error: Could not find {file_name}")
    exit(1)

# Filter for EO13960 scored data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# --- 1. Define Independent Variable: Procurement Source ---
# Column: '22_dev_method'

def classify_source(val):
    s = str(val).lower().strip()
    is_contracting = 'contracting' in s
    is_in_house = 'in-house' in s
    
    if is_contracting and not is_in_house:
        return 'Commercial'
    elif is_in_house and not is_contracting:
        return 'Internal'
    else:
        return None

eo_df['procurement_source'] = eo_df['22_dev_method'].apply(classify_source)

# Drop rows where source is undefined
analysis_df = eo_df.dropna(subset=['procurement_source']).copy()

# --- 2. Define Dependent Variable: Documentation Availability ---
# Column: '34_data_docs'

def check_docs(val):
    if pd.isna(val):
        return False
    s = str(val).lower().strip()
    # explicit negatives
    if 'missing' in s or 'not available' in s or 'not reported' in s or s == 'no' or s == '':
        return False
    # explicit positives
    if 'complete' in s or 'available' in s or 'partial' in s or 'yes' in s or 'public' in s:
        return True
    return False

analysis_df['has_docs'] = analysis_df['34_data_docs'].apply(check_docs)

# --- 3. Analysis ---

# Contingency Table
raw_contingency = pd.crosstab(analysis_df['procurement_source'], analysis_df['has_docs'])

# Robustly ensure 2x2 shape using reindex
# Index: Internal first (reference), then Commercial
# Columns: False (No Docs), True (Has Docs)
contingency = raw_contingency.reindex(index=['Internal', 'Commercial'], columns=[False, True], fill_value=0)
contingency.columns = ['No Docs', 'Has Docs']

print("--- Contingency Table (Source vs Documentation) ---")
print(contingency)

# Calculate Rates
internal_total = contingency.loc['Internal'].sum()
commercial_total = contingency.loc['Commercial'].sum()

if internal_total > 0:
    int_rate = (contingency.loc['Internal', 'Has Docs'] / internal_total) * 100
else:
    int_rate = 0

if commercial_total > 0:
    comm_rate = (contingency.loc['Commercial', 'Has Docs'] / commercial_total) * 100
else:
    comm_rate = 0

print(f"\nInternal Systems with Docs:   {int_rate:.2f}% (N={internal_total})")
print(f"Commercial Systems with Docs: {comm_rate:.2f}% (N={commercial_total})")

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Interpretation
alpha = 0.05
print(f"\nInterpretation (alpha={alpha}):")
if p < alpha:
    print("Result: Statistically SIGNIFICANT difference found.")
    if comm_rate < int_rate:
        print("Hypothesis SUPPORTED: Commercial systems have significantly LOWER documentation rates.")
    else:
        print("Hypothesis REFUTED: Commercial systems have significantly HIGHER documentation rates.")
else:
    print("Result: NO statistically significant difference found.")

# Odds Ratio Calculation
# OR = (Odds of Docs given Commercial) / (Odds of Docs given Internal)
# Odds = P / (1-P)
odds_comm = comm_rate / (100 - comm_rate) if comm_rate != 100 else np.inf
odds_int = int_rate / (100 - int_rate) if int_rate != 100 else np.inf

if odds_int == 0:
    or_val = np.inf
else:
    or_val = odds_comm / odds_int
    
print(f"Odds Ratio (Commercial / Internal): {or_val:.4f}")

# --- 4. Visualization ---
plt.figure(figsize=(10, 6))
# Normalize to get percentages
props = contingency.div(contingency.sum(axis=1), axis=0)

ax = props.plot(kind='bar', stacked=True, color=['#d9534f', '#5bc0de'], ax=plt.gca())

plt.title('Data Documentation Availability: Commercial vs. Internal AI', fontsize=14)
plt.xlabel('Procurement Source', fontsize=12)
plt.ylabel('Proportion of Systems', fontsize=12)
plt.ylim(0, 1.15) # Extra space for labels
plt.legend(title='Documentation Status', loc='upper right')
plt.xticks(rotation=0)

# Annotate bars
for c in ax.containers:
    # Only label if segment is big enough
    labels = [f'{v.get_height()*100:.1f}%' if v.get_height() > 0.05 else '' for v in c]
    ax.bar_label(c, labels=labels, label_type='center', fontweight='bold')

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Contingency Table (Source vs Documentation) ---
                    No Docs  Has Docs
procurement_source                   
Internal                 76       363
Commercial              182       299

Internal Systems with Docs:   82.69% (N=439)
Commercial Systems with Docs: 62.16% (N=481)

Chi-Square Statistic: 46.9084
P-value: 7.4383e-12

Interpretation (alpha=0.05):
Result: Statistically SIGNIFICANT difference found.
Hypothesis SUPPORTED: Commercial systems have significantly LOWER documentation rates.
Odds Ratio (Commercial / Internal): 0.3440


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Stacked Bar Chart (specifically a 100% stacked bar chart).
*   **Purpose:** The chart compares the proportional distribution of a categorical variable (Documentation Status) across two distinct groups (Procurement Sources: Internal vs. Commercial). It allows for an easy visual comparison of the ratio of documented to undocumented systems between the two sources.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Procurement Source"
    *   **Categories:** Two discrete categories labeled "Internal" and "Commercial".
*   **Y-Axis:**
    *   **Label:** "Proportion of Systems"
    *   **Range:** 0.0 to approximately 1.15 (though the data effectively tops out at 1.0 or 100%).
    *   **Units:** Decimals representing proportions (0.0 to 1.0).

### 3. Data Trends
*   **Internal Source:**
    *   **Dominant Trend:** The vast majority of systems procured internally have data documentation. The blue "Has Docs" segment is the tallest, accounting for **82.7%** of the bar.
    *   **Minority Trend:** Only a small fraction (**17.3%**) lack documentation.
*   **Commercial Source:**
    *   **Dominant Trend:** A majority still have documentation (**62.2%**), but the proportion is noticeably smaller compared to the Internal source.
    *   **Minority Trend:** A significant portion (**37.8%**) lack documentation.
*   **Comparison:** The visual pattern shows that the red segment ("No Docs") is more than twice as large in the "Commercial" column as it is in the "Internal" column, indicating a higher prevalence of undocumented systems in the commercial sector.

### 4. Annotations and Legends
*   **Chart Title:** "Data Documentation Availability: Commercial vs. Internal AI"
*   **Legend:** Located in the top right corner, titled "Documentation Status".
    *   **Red (Salmon color):** Represents "No Docs".
    *   **Blue (Cyan color):** Represents "Has Docs".
*   **Bar Annotations:** Each segment of the bars is annotated with its specific percentage value in bold black text:
    *   Internal / No Docs: **17.3%**
    *   Internal / Has Docs: **82.7%**
    *   Commercial / No Docs: **37.8%**
    *   Commercial / Has Docs: **62.2%**

### 5. Statistical Insights
*   **Documentation Gap:** There is a **20.5 percentage point gap** in documentation availability between the two sources. Internal AI systems are significantly more likely to be documented (82.7%) than Commercial AI systems (62.2%).
*   **Risk Assessment:** The likelihood of encountering a system *without* documentation is roughly **2.2 times higher** for Commercial systems (37.8%) compared to Internal systems (17.3%).
*   **Conclusion:** The data suggests that internal procurement or development processes enforce stricter documentation standards than those found in the commercial market for AI systems.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
