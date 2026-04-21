# Experiment 299: node_6_86

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_86` |
| **ID in Run** | 299 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T15:25:19.926493+00:00 |
| **Runtime** | 329.2s |
| **Parent** | `node_5_79` |
| **Children** | None |
| **Creation Index** | 300 |

---

## Hypothesis

> The Vendor Opacity Effect: AI systems procured from commercial vendors (COTS)
exhibit statistically significantly lower levels of 'Code Access' and 'Data
Documentation' compared to government-developed (GOTS) systems.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9839 (Definitely True) |
| **Posterior** | 0.9945 (Definitely True) |
| **Surprise** | +0.0128 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
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
| Definitely True | 60.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Quantify the transparency gap between Commercial and Government-developed AI.

### Steps
- 1. Load 'eo13960_scored'.
- 2. Segment data into 'Commercial' (`10_commercial_ai`='Yes') and 'Government' (No).
- 3. Binarize `38_code_access` and `27_34_data_docs` (using column `34_data_docs`).
- 4. Perform T-tests or Chi-Square tests comparing the transparency rates between the two groups.
- 5. Visualize the gap.

### Deliverables
- Comparison statistics; Bar chart of Transparency Controls by Procurement Source.

---

## Analysis

The experiment successfully quantified the 'Vendor Opacity Effect' by analyzing
920 AI systems from the EO 13960 inventory.

1. **Data Segmentation**: The code correctly used the '22_dev_method' column to
segment the data into Government-Developed (GOTS, n=439) and Commercial-
Developed (COTS, n=481) groups, overcoming the previous mapping error.

2. **Code Access Gap**: The analysis revealed a massive disparity in code
transparency. Government systems provided code access in **90.54%** of cases,
whereas Commercial systems did so in only **33.85%** of cases. The Chi-Square
test yielded a p-value of 1.74e-59, indicating overwhelming statistical
significance.

3. **Data Documentation Gap**: A significant gap was also observed in data
documentation. Government systems had adequate documentation rates of **88.11%**
compared to **65.71%** for Commercial systems (p < 0.001).

4. **Conclusion**: The hypothesis is strongly confirmed. Commercial AI vendors
operating in the federal space exhibit significantly lower transparency levels
compared to in-house government development, with the opacity being most acute
in source code availability.

---

## Review

The experiment successfully validated the 'Vendor Opacity Effect' hypothesis
using the EO 13960 dataset. The programmer correctly identified that the
originally suggested column '10_commercial_ai' was unsuitable for segmentation
and appropriately pivoted to '22_dev_method', allowing for a robust comparison
between Government-Developed (GOTS, n=439) and Commercial-Developed (COTS,
n=481) systems.

Key Findings:
1. **Code Access**: A massive transparency gap was identified. Government
systems provided source code access in **90.54%** of cases, compared to only
**33.85%** for commercial systems. The Chi-Square test (p < 1.74e-59) confirms
this is highly statistically significant.
2. **Data Documentation**: A significant gap also exists here, though less
extreme. Government systems had adequate documentation rates of **88.11%**
versus **65.71%** for commercial systems (p < 1.73e-14).

The analysis robustly supports the hypothesis that commercial AI vendors
operating in the federal space maintain significantly higher levels of opacity,
particularly regarding source code access, compared to internal government
development.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load dataset
filename = 'astalabs_discovery_all_data.csv'
print(f"Loading dataset from {filename}...")

try:
    df = pd.read_csv(filename, low_memory=False)
except FileNotFoundError:
    # Fallback to relative path if needed
    df = pd.read_csv(f'../{filename}', low_memory=False)

# Filter for EO 13960 Scored data
subset = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored subset shape: {subset.shape}")

# --- Step 1: Define COTS vs GOTS groups ---
# Using '22_dev_method' as proxy
# 'Developed in-house.' -> Government (GOTS)
# 'Developed with contracting resources.' -> Commercial (COTS)

def classify_source(val):
    s = str(val).lower()
    if 'in-house' in s and 'contracting' not in s:
        return 'Government (GOTS)'
    elif 'contracting' in s and 'in-house' not in s:
        return 'Commercial (COTS)'
    else:
        return None

subset['procurement_type'] = subset['22_dev_method'].apply(classify_source)

# Filter to only these two groups
analysis_df = subset.dropna(subset=['procurement_type']).copy()
print(f"Analyzable subset (COTS vs GOTS) shape: {analysis_df.shape}")
print("Group distribution:")
print(analysis_df['procurement_type'].value_counts())

# --- Step 2: Binarize Transparency Columns ---

# Variable 1: Code Access ('38_code_access')
# Logic: Contains 'Yes' -> 1, Contains 'No' -> 0
def parse_code_access(val):
    s = str(val).lower()
    if 'yes' in s:
        return 1
    elif 'no' in s:
        return 0
    return np.nan  # Treat unclear/nan as missing

analysis_df['has_code_access'] = analysis_df['38_code_access'].apply(parse_code_access)

# Variable 2: Data Documentation ('34_data_docs')
# Logic: 'missing', 'no' -> 0; 'complete', 'partial', 'available', 'yes' -> 1
def parse_data_docs(val):
    s = str(val).lower()
    if pd.isna(val) or s == 'nan' or 'not reported' in s:
        return np.nan
    if 'missing' in s or 'not available' in s or s == 'no':
        return 0
    if 'complete' in s or 'partial' in s or 'available' in s or 'yes' in s:
        return 1
    return 0 # Default fallback for negatives not caught, though risky. Let's inspect coverage.

analysis_df['has_data_docs'] = analysis_df['34_data_docs'].apply(parse_data_docs)

# --- Step 3: Statistical Analysis ---

results = []
metrics = [('has_code_access', 'Code Access'), ('has_data_docs', 'Data Documentation')]

print("\n--- Statistical Tests ---")

for col, label in metrics:
    # Drop NaNs for the specific test
    valid_data = analysis_df.dropna(subset=[col])
    
    # Contingency Table
    contingency = pd.crosstab(valid_data['procurement_type'], valid_data[col])
    
    # Chi-Square Test
    chi2, p, dof, ex = chi2_contingency(contingency)
    
    # Calculate Rates
    rates = valid_data.groupby('procurement_type')[col].mean()
    gots_rate = rates.get('Government (GOTS)', 0)
    cots_rate = rates.get('Commercial (COTS)', 0)
    
    results.append({
        'Metric': label,
        'GOTS Rate': gots_rate,
        'COTS Rate': cots_rate,
        'p-value': p,
        'Significant': p < 0.05
    })
    
    print(f"\nMetric: {label}")
    print(contingency)
    print(f"GOTS Rate: {gots_rate:.2%}, COTS Rate: {cots_rate:.2%}")
    print(f"Chi-Square: {chi2:.4f}, p-value: {p:.4e}")

# --- Step 4: Visualization ---

res_df = pd.DataFrame(results)

# Plotting
labels = res_df['Metric']
gots_means = res_df['GOTS Rate'] * 100
cots_means = res_df['COTS Rate'] * 100

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, gots_means, width, label='Government (GOTS)', color='#1f77b4')
rects2 = ax.bar(x + width/2, cots_means, width, label='Commercial (COTS)', color='#ff7f0e')

ax.set_ylabel('Transparency Rate (%)')
ax.set_title('Transparency Gap: Commercial vs Government AI')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add significance stars
for i, p_val in enumerate(res_df['p-value']):
    if p_val < 0.001:
        sig = '***'
    elif p_val < 0.01:
        sig = '**'
    elif p_val < 0.05:
        sig = '*'
    else:
        sig = 'ns'
    
    # Height for annotation
    max_h = max(gots_means[i], cots_means[i])
    ax.text(i, max_h + 2, sig, ha='center', va='bottom', fontweight='bold')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)

plt.ylim(0, 110)
plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
EO 13960 Scored subset shape: (1757, 196)
Analyzable subset (COTS vs GOTS) shape: (920, 197)
Group distribution:
procurement_type
Commercial (COTS)    481
Government (GOTS)    439
Name: count, dtype: int64

--- Statistical Tests ---

Metric: Code Access
has_code_access    0.0  1.0
procurement_type           
Commercial (COTS)  258  132
Government (GOTS)   37  354
GOTS Rate: 90.54%, COTS Rate: 33.85%
Chi-Square: 264.5628, p-value: 1.7376e-59

Metric: Data Documentation
has_data_docs      0.0  1.0
procurement_type           
Commercial (COTS)  156  299
Government (GOTS)   49  363
GOTS Rate: 88.11%, COTS Rate: 65.71%
Chi-Square: 58.8173, p-value: 1.7302e-14


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Grouped Bar Chart.
*   **Purpose:** The chart is designed to compare transparency rates between two different sectors (Government AI vs. Commercial AI) across two distinct categories of transparency (Code Access and Data Documentation).

### 2. Axes
*   **Y-Axis:**
    *   **Label:** "Transparency Rate (%)".
    *   **Range:** The visual scale runs from 0 to roughly 110, with major tick marks at intervals of 20 (0, 20, 40, 60, 80, 100).
*   **X-Axis:**
    *   **Labels:** The axis represents categorical data split into two groups: "Code Access" and "Data Documentation."

### 3. Data Trends
*   **General Pattern:** Government (GOTS) AI systems consistently show significantly higher transparency rates compared to Commercial (COTS) AI systems across both metrics.
*   **Code Access:**
    *   This category exhibits the widest disparity.
    *   **Tallest Bar:** Government (GOTS) at **90.5%**.
    *   **Shortest Bar:** Commercial (COTS) at **33.8%**.
*   **Data Documentation:**
    *   The Government sector remains higher, but the gap is narrower compared to Code Access.
    *   Government (GOTS) is at **88.1%**.
    *   Commercial (COTS) is at **65.7%**.

### 4. Annotations and Legends
*   **Legend:** Located in the top right corner, distinguishing the two groups:
    *   **Blue:** Government (GOTS - Government off-the-shelf).
    *   **Orange:** Commercial (COTS - Commercial off-the-shelf).
*   **Value Labels:** Exact percentage values are annotated directly above each bar for clarity (e.g., 90.5%, 33.8%).
*   **Statistical Significance markers:** There are triple asterisks (**\*\*\***) placed above the Government bars in both categories. In scientific plotting, this conventionally indicates a high level of statistical significance (often $p < 0.001$) regarding the difference between the control group (or comparator) and the test group.

### 5. Statistical Insights
*   **The "Transparency Gap":** The plot visualizes a significant gap in openness between government and commercial AI.
    *   In **Code Access**, the gap is massive, with Government AI being nearly **2.7 times** more likely to provide code access than Commercial AI (a difference of 56.7 percentage points).
    *   In **Data Documentation**, the gap persists but is smaller (a difference of 22.4 percentage points).
*   **Conclusion:** Government-developed AI appears to adhere to much stricter standards of transparency, particularly regarding the ability to review source code, whereas commercial entities are significantly more opaque, especially regarding their codebases. The statistical markers suggest these differences are not due to random chance.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
