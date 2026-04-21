# Experiment 87: node_6_7

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_7` |
| **ID in Run** | 87 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-21T00:22:12.187716+00:00 |
| **Runtime** | 172.2s |
| **Parent** | `node_5_2` |
| **Children** | None |
| **Creation Index** | 88 |

---

## Hypothesis

> Requirements originating from 'Normative' frameworks (e.g., EU AI Act, NIST AI
RMF) are significantly more likely to be classified as 'Trust Readiness',
whereas 'Technical' sources (e.g., OWASP, NIST GenAI Profile) map to
'Integration Readiness'.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7581 (Likely True) |
| **Posterior** | 0.7521 (Likely True) |
| **Surprise** | -0.0070 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 2.0 |
| Maybe True | 28.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 90.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Validate whether the source framework type predicts the resulting competency bundle.

### Steps
- 1. Load 'step2_crosswalk_matrix.csv'.
- 2. Create a new variable 'source_type': classify 'EU AI Act' and 'NIST AI RMF 1.0' as 'Normative', and 'OWASP Top 10 LLM' and 'NIST GenAI Profile' as 'Technical'.
- 3. Create a contingency table of 'source_type' vs. 'bundle'.
- 4. Run a Chi-square test of independence.

### Deliverables
- Contingency table showing source distribution across bundles, Chi-square test statistics, and a mosaic plot.

---

## Analysis

The experiment successfully tested the hypothesis that the source framework type
(Normative vs. Technical) predicts the competency bundle (Trust vs.
Integration).

**Descriptive Statistics:**
- **Normative Sources (n=28):** Contributed 12 'Integration Readiness' and 16
'Trust Readiness' requirements. The distribution is fairly balanced (43%
Integration / 57% Trust).
- **Technical Sources (n=14):** Contributed 11 'Integration Readiness' and only
3 'Trust Readiness' requirements. This group is heavily skewed toward
Integration (79% Integration / 21% Trust).

**Statistical Inference:**
- The Chi-square test of independence yielded a statistic of 3.4720 and a
**p-value of 0.0624**.
- Since the p-value is slightly above the standard significance level of 0.05,
we **fail to reject the null hypothesis** at the 95% confidence level.

**Conclusion:**
While there is a visible trend suggesting 'Technical' sources are more likely to
map to 'Integration Readiness', the relationship is not statistically
significant ($p > 0.05$) likely due to the small sample size (n=42). The data
shows that 'Normative' sources are actually quite versatile, contributing nearly
equally to both governance and technical controls, whereas 'Technical' sources
are more specialized.

---

## Review

The experiment successfully tested the hypothesis that the source framework type
(Normative vs. Technical) predicts the competency bundle (Trust vs.
Integration).

**Descriptive Statistics:**
- **Normative Sources (n=28):** Contributed 12 'Integration Readiness' and 16
'Trust Readiness' requirements. The distribution is fairly balanced (43%
Integration / 57% Trust).
- **Technical Sources (n=14):** Contributed 11 'Integration Readiness' and only
3 'Trust Readiness' requirements. This group is heavily skewed toward
Integration (79% Integration / 21% Trust).

**Statistical Inference:**
- The Chi-square test of independence yielded a statistic of 3.4720 and a
**p-value of 0.0624**.
- Since the p-value is slightly above the standard significance level of 0.05,
we **fail to reject the null hypothesis** at the 95% confidence level.

**Conclusion:**
While there is a visible trend suggesting 'Technical' sources are more likely to
map to 'Integration Readiness', the relationship is not statistically
significant ($p > 0.05$) likely due to the small sample size (n=42). The data
shows that 'Normative' sources are actually quite versatile, contributing nearly
equally to both governance and technical controls, whereas 'Technical' sources
are more specialized.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import sys
import subprocess

# Function to install statsmodels if not present, as mosaic plot is requested
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

try:
    from statsmodels.graphics.mosaicplot import mosaic
except ImportError:
    install('statsmodels')
    from statsmodels.graphics.mosaicplot import mosaic

# Load dataset
file_path = '../step2_crosswalk_matrix.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {file_path}")
except FileNotFoundError:
    # Fallback for local testing if directory structure differs
    file_path = 'step2_crosswalk_matrix.csv'
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}")
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

# Inspect unique sources to ensure correct mapping
print("\nUnique sources in dataset:")
print(df['source'].unique())

# 2. Create 'source_type' variable
# Normative: EU AI Act, NIST AI RMF 1.0
# Technical: OWASP Top 10 LLM, NIST GenAI Profile
def classify_source(source):
    source = str(source).strip()
    if 'EU AI Act' in source or 'NIST AI RMF 1.0' in source:
        return 'Normative'
    elif 'OWASP' in source or 'NIST GenAI Profile' in source:
        return 'Technical'
    else:
        return 'Other'

df['source_type'] = df['source'].apply(classify_source)

# Check for unclassified sources
if 'Other' in df['source_type'].values:
    print("\nWarning: Some sources were classified as 'Other':")
    print(df[df['source_type'] == 'Other']['source'].unique())

# Filter to only Normative and Technical for the test (though 'Other' shouldn't exist based on metadata)
df_analysis = df[df['source_type'] != 'Other'].copy()

# 3. Create Contingency Table
contingency_table = pd.crosstab(df_analysis['source_type'], df_analysis['bundle'])
print("\nContingency Table (Source Type vs Bundle):")
print(contingency_table)

# 4. Chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\nChi-Square Test Results:")
print(f"Chi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")
print("\nExpected Frequencies:")
print(expected)

# Interpret results
alpha = 0.05
if p < alpha:
    print("\nConclusion: Reject the null hypothesis. There is a statistically significant association between source type and competency bundle.")
else:
    print("\nConclusion: Fail to reject the null hypothesis. No statistically significant association found.")

# 5. Visualization: Mosaic Plot
plt.figure(figsize=(10, 6))
mosaic(df_analysis, ['source_type', 'bundle'], 
       title='Mosaic Plot: Source Type vs Competency Bundle',
       labelizer=lambda k: '',  # Remove internal labels if too cluttered, or keep default
       gap=0.02)
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Successfully loaded step2_crosswalk_matrix.csv

Unique sources in dataset:
<StringArray>
[      'NIST AI RMF 1.0',    'NIST GenAI Profile', 'EU AI Act (2024/1689)',
      'OWASP Top 10 LLM']
Length: 4, dtype: str

Contingency Table (Source Type vs Bundle):
bundle       Integration Readiness  Trust Readiness
source_type                                        
Normative                       12               16
Technical                       11                3

Chi-Square Test Results:
Chi-square Statistic: 3.4720
P-value: 0.0624
Degrees of Freedom: 1

Expected Frequencies:
[[15.33333333 12.66666667]
 [ 7.66666667  6.33333333]]

Conclusion: Fail to reject the null hypothesis. No statistically significant association found.


=== Plot Analysis (figure 2) ===
Based on the provided image, here is the analysis:

**1. Plot Type**
*   **Type:** Mosaic Plot (also known as a Marimekko chart).
*   **Purpose:** This plot is used to visualize the relationship between two categorical variables ("Source Type" and "Competency Bundle") by displaying a contingency table. The area of each rectangle is proportional to the frequency (count) of that specific category combination within the dataset.

**2. Axes**
*   **Titles and Labels:**
    *   **X-axis (Bottom):** Represents the "Source Type" variable. The categories are labeled "Normative" and "Technical".
    *   **Y-axis (Left):** Represents the "Competency Bundle" variable. The categories are labeled "Integration Readiness" and "Trust Readiness".
*   **Value Ranges:**
    *   There are no numerical scales explicitly shown. However, in a mosaic plot:
        *   The **width** of the columns on the X-axis represents the marginal proportion of each Source Type in the total population (0 to 100%).
        *   The **height** of the segments within a column represents the conditional proportion of the Competency Bundle given that Source Type (0 to 100%).

**3. Data Trends**
*   **Column Widths (Source Type Distribution):** The "Normative" column is significantly wider than the "Technical" column. This indicates that the dataset contains a much larger number of "Normative" observations compared to "Technical" observations.
*   **Block Heights (Competency Bundle Distribution):**
    *   **Within Normative Sources (Left/Red):** The vertical split is relatively balanced. The area for "Trust Readiness" (bottom brown block) appears slightly larger or roughly equal to "Integration Readiness" (top dark red block). This suggests that Normative sources are split fairly evenly between these two competencies.
    *   **Within Technical Sources (Right/Green):** The vertical split is highly uneven. The "Integration Readiness" block (top bright green) is much taller than the "Trust Readiness" block (bottom dark green). This indicates that Technical sources are overwhelmingly associated with "Integration Readiness."

**4. Annotations and Legends**
*   **Title:** "Mosaic Plot: Source Type vs Competency Bundle" identifies the variables being analyzed.
*   **Color Coding:**
    *   **Red/Brown tones:** Represent the "Normative" category.
    *   **Green tones:** Represent the "Technical" category.
*   **Separators:** Thick white lines separate the rectangles, clearly defining the boundaries of the four distinct groups formed by the intersection of the two variables.

**5. Statistical Insights**
*   **Dependence of Variables:** The variables "Source Type" and "Competency Bundle" appear to be dependent (correlated). If they were independent, the horizontal dividing lines in both columns would align at the same height. The fact that the line is much lower in the Technical column indicates a difference in distribution.
*   **Key Association:** "Technical" sources have a much higher probability of being associated with "Integration Readiness" compared to "Normative" sources.
*   **Prevalence:** The "Normative" source type is the dominant category in this dataset, comprising the majority of the data (likely 60-70% based on visual width estimation).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
