# Experiment 202: node_7_5

| Property | Value |
|---|---|
| **Experiment ID** | `node_7_5` |
| **ID in Run** | 202 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T10:28:57.153054+00:00 |
| **Runtime** | 195.1s |
| **Parent** | `node_6_29` |
| **Children** | None |
| **Creation Index** | 203 |

---

## Hypothesis

> The 'Procurement' Transparency Barrier: Systems developed via external contracts
are significantly less likely to have accessible 'Code Access' or technical
documentation compared to systems developed in-house.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7661 (Likely True) |
| **Posterior** | 0.9203 (Definitely True) |
| **Surprise** | +0.1850 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 3.0 |
| Maybe True | 27.0 |
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

**Objective:** Analyze the relationship between development method and technical transparency.

### Steps
- 1. Filter `eo13960_scored`.
- 2. Categorize `22_dev_method` into 'Contracted' vs 'In-house/Government'.
- 3. Binarize `38_code_access` (or `37_custom_code` availability).
- 4. Perform a Chi-square test to compare transparency/access rates.

### Deliverables
- Contingency table; Statistical results; Bar chart.

---

## Analysis

The experiment successfully tested the 'Procurement Transparency Barrier'
hypothesis using the EO 13960 dataset.

**Data Processing:**
- The analysis focused on 1,107 AI systems where the development method was
clearly identified.
- Rows were categorized into 'Contracted' (n=668, including mixed methods) and
'In-house' (n=439).
- Accessibility was defined as the agency having access to the source code or
technical documentation (regardless of public availability).

**Findings:**
- **In-house Development:** 80.6% (354/439) of systems have accessible
code/documentation.
- **Contracted Development:** Only 35.6% (238/668) of systems have accessible
code/documentation.
- **Statistical Significance:** The Chi-square test yielded a p-value of
**1.94e-48** (Statistic=213.9), indicating an extremely significant difference.

**Conclusion:**
The hypothesis is **strongly supported**. Systems developed via external
contracts are less than half as likely to provide the government with code
access compared to in-house developed systems. This quantifies a severe 'black
box' procurement risk, where agencies fail to secure intellectual property
rights or technical transparency from vendors, potentially leading to vendor
lock-in and reduced auditability.

---

## Review

The experiment successfully tested the 'Procurement Transparency Barrier'
hypothesis using the EO 13960 dataset.

**Data Processing:**
- The analysis focused on 1,107 AI systems where the development method was
clearly identified.
- Rows were categorized into 'Contracted' (n=668, including mixed methods) and
'In-house' (n=439).
- Accessibility was defined as the agency having access to the source code or
technical documentation (regardless of public availability).

**Findings:**
- **In-house Development:** 80.6% (354/439) of systems have accessible
code/documentation.
- **Contracted Development:** Only 35.6% (238/668) of systems have accessible
code/documentation.
- **Statistical Significance:** The Chi-square test yielded a p-value of
**1.94e-48** (Statistic=213.9), indicating an extremely significant difference.

**Conclusion:**
The hypothesis is **strongly supported**. Systems developed via external
contracts are less than half as likely to provide the government with code
access compared to in-house developed systems. This quantifies a severe 'black
box' procurement risk, where agencies fail to secure intellectual property
rights or technical transparency from vendors.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import sys

# Load dataset
file_name = 'astalabs_discovery_all_data.csv'
# Try checking one level above first as per instructions
file_path = f'../{file_name}'
if not os.path.exists(file_path):
    file_path = file_name
    if not os.path.exists(file_path):
        print(f"Error: Dataset {file_name} not found in ../ or current directory.")
        sys.exit(1)

print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path, low_memory=False)

# Filter for EO 13960 Scored data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored rows: {len(df_eo)}")

# --- Step 1: Process Development Method ---
col_dev = '22_dev_method'

# Normalize and inspect
df_eo[col_dev] = df_eo[col_dev].astype(str).fillna('')

print("\n--- Unique values in '22_dev_method' (Top 10) ---")
print(df_eo[col_dev].value_counts().head(10))

def classify_dev_method(val):
    val_lower = val.lower()
    # Priority to Contracted if mixed (assuming external involvement adds the barrier)
    if any(x in val_lower for x in ['contract', 'vendor', 'commercial', 'external', 'private']):
        return 'Contracted'
    elif any(x in val_lower for x in ['agency', 'government', 'in-house', 'federal', 'staff']):
        return 'In-house'
    else:
        return 'Unknown'

df_eo['dev_category'] = df_eo[col_dev].apply(classify_dev_method)

# Filter out Unknowns
df_analysis = df_eo[df_eo['dev_category'] != 'Unknown'].copy()
print(f"\nRows after filtering Dev Method: {len(df_analysis)}")
print(df_analysis['dev_category'].value_counts())

# --- Step 2: Process Code Access ---
col_access = '38_code_access'

# Normalize and inspect
df_analysis[col_access] = df_analysis[col_access].astype(str).fillna('')

print("\n--- Unique values in '38_code_access' (Top 10) ---")
print(df_analysis[col_access].value_counts().head(10))

def classify_access(val):
    val_lower = val.lower()
    # Look for affirmative keywords indicating availability
    if any(x in val_lower for x in ['yes', 'public', 'open', 'available', 'github', 'repo']):
        return 1
    # Treat 'No', 'N/A', 'Restricted', nan as 0
    return 0

df_analysis['is_accessible'] = df_analysis[col_access].apply(classify_access)

# --- Step 3: Statistical Analysis ---
contingency = pd.crosstab(df_analysis['dev_category'], df_analysis['is_accessible'])
print("\n--- Contingency Table (Code Accessibility) ---")
print(contingency)

# Calculate rates
rates = df_analysis.groupby('dev_category')['is_accessible'].mean()
counts = df_analysis['dev_category'].value_counts()
print("\n--- Accessibility Rates ---")
print(rates)

# Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# --- Step 4: Visualization ---
plt.figure(figsize=(8, 6))
bar_colors = ['skyblue', 'lightcoral']
ax = rates.plot(kind='bar', color=bar_colors, edgecolor='black')
plt.title('Code/Technical Documentation Accessibility by Development Method')
plt.xlabel('Development Method')
plt.ylabel('Proportion with Accessible Code/Docs')
plt.ylim(0, max(rates.max() * 1.2, 0.1))  # Ensure some headroom

# Add labels
for i, v in enumerate(rates):
    count = counts[rates.index[i]]
    plt.text(i, v + 0.005, f"{v:.1%}\n(n={count})", ha='center', va='bottom')

plt.axhline(0, color='black', linewidth=1)
plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: astalabs_discovery_all_data.csv
EO 13960 Scored rows: 1757

--- Unique values in '22_dev_method' (Top 10) ---
22_dev_method
                                                                                               629
Developed with contracting resources.                                                          481
Developed in-house.                                                                            439
Developed with both contracting and in-house resources.                                        187
                                                                                                15
Data not reported by submitter and will be updated once additional information is collected      6
Name: count, dtype: int64

Rows after filtering Dev Method: 1107
dev_category
Contracted    668
In-house      439
Name: count, dtype: int64

--- Unique values in '38_code_access' (Top 10) ---
38_code_access
Yes – agency has access to source code, but it is not public.    497
No – agency does not have access to source code.                 324
                                                                 175
Yes – source code is publicly available.                          47
Yes                                                               47
                                                                  16
YES                                                                1
Name: count, dtype: int64

--- Contingency Table (Code Accessibility) ---
is_accessible    0    1
dev_category           
Contracted     430  238
In-house        85  354

--- Accessibility Rates ---
dev_category
Contracted    0.356287
In-house      0.806378
Name: is_accessible, dtype: float64

Chi-square Statistic: 213.8995
P-value: 1.9371e-48


=== Plot Analysis (figure 1) ===
Based on the provided image, here is a detailed analysis of the plot:

### 1. Plot Type
*   **Type:** Vertical Bar Chart.
*   **Purpose:** The plot compares categorical data, specifically the proportion of code or technical documentation that is accessible, split between two different software development methods: "Contracted" versus "In-house."

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Title:** "Development Method"
    *   **Labels:** The axis represents two discrete categories: "Contracted" and "In-house."
*   **Y-Axis (Vertical):**
    *   **Title:** "Proportion with Accessible Code/Docs"
    *   **Range:** The scale runs from **0.0 to approximately 0.95**, with major tick marks at 0.0, 0.2, 0.4, 0.6, and 0.8.
    *   **Units:** The values represent a proportion (decimal format), which corresponds to percentages (0.0 = 0%, 1.0 = 100%).

### 3. Data Trends
*   **Tallest Bar:** The "In-house" category is significantly taller, reaching slightly above the 0.8 mark.
*   **Shortest Bar:** The "Contracted" category is much shorter, reaching just below the 0.4 mark.
*   **Pattern:** There is a substantial disparity between the two methods. In-house development shows a much higher rate of accessibility compliance compared to contracted development. The In-house bar is visually more than double the height of the Contracted bar.

### 4. Annotations and Legends
*   **Bar Annotations:** Specific values are printed directly above each bar to provide exact data points:
    *   **Contracted:**
        *   Percentage: **35.6%**
        *   Sample Size: **(n=668)**
    *   **In-house:**
        *   Percentage: **80.6%**
        *   Sample Size: **(n=439)**
*   **Title:** The chart is titled "Code/Technical Documentation Accessibility by Development Method."

### 5. Statistical Insights
*   **Significant Performance Gap:** There is a dramatic difference in accessibility outcomes. **80.6%** of In-house projects had accessible code/documentation, compared to only **35.6%** of Contracted projects. This suggests In-house teams are **more than twice as likely** to produce accessible documentation compared to contractors in this dataset.
*   **Sample Size Validity:** The sample sizes for both groups are robust ($n=668$ for Contracted and $n=439$ for In-house). Interestingly, despite having a larger sample size (and thus potentially more opportunities for variance), the Contracted group consistently underperforms the In-house group in this metric.
*   **Implication:** The data strongly suggests that direct oversight or internal standards present in "In-house" development environments may lead to better adherence to accessibility standards than outsourcing ("Contracted").
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
