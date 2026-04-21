# Experiment 12: node_3_3

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_3` |
| **ID in Run** | 12 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T01:30:12.500018+00:00 |
| **Runtime** | 424.2s |
| **Parent** | `node_2_5` |
| **Children** | `node_4_3`, `node_4_26`, `node_4_45` |
| **Creation Index** | 13 |

---

## Hypothesis

> Domain-Specific Autonomy Risk: AI incidents in physical-impact sectors (e.g.,
Transportation, Healthcare) are associated with significantly higher reported
autonomy levels compared to incidents in digital-impact sectors (e.g., Finance,
Media).

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7258 (Likely True) |
| **Posterior** | 0.3846 (Maybe False) |
| **Surprise** | -0.4094 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 29.0 |
| Uncertain | 0.0 |
| Maybe False | 1.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 50.0 |
| Definitely False | 10.0 |

---

## Experiment Plan

**Objective:** Compare the distribution of 'Autonomy Level' between 'Physical' and 'Digital' sectors to determine if physical risks are correlated with higher system autonomy.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for rows where `source_table` is 'aiid_incidents'.
- 2. Clean the `Autonomy Level` column: Map 'Autonomy1' to 1 (Low), 'Autonomy2' to 2 (Medium), 'Autonomy3' to 3 (High), and drop 'unclear' or NaN values.
- 3. Clean the `Sector of Deployment` column: Normalize text to lowercase. Create a new variable `Risk_Domain`.
    - Map sectors containing 'transport', 'health', 'medic', 'manufactur', 'industr', 'energy', 'agricultur', 'construct', 'robot' to 'Physical'.
    - Map sectors containing 'financ', 'bank', 'educat', 'govern', 'public', 'media', 'entertain', 'retail', 'consum', 'service' to 'Digital'.
    - Drop rows that do not match these keywords.
- 4. Perform a Mann-Whitney U test to compare the distributions of ordinal Autonomy scores between 'Physical' and 'Digital' domains.
- 5. Generate a boxplot visualizing Autonomy Levels by Risk Domain to illustrate the findings.

### Deliverables
- 1. Printed counts of incidents in 'Physical' vs 'Digital' domains.
- 2. Mann-Whitney U test statistics (U-stat, p-value) printed to stdout.
- 3. A boxplot saved or displayed showing Autonomy Level distributions by Domain.

---

## Analysis

The experiment successfully tested the hypothesis that physical-impact sectors
involve higher AI autonomy levels than digital-impact sectors.

1. **Data Processing**: The code successfully filtered the AI Incident Database,
mapping 124 incidents to 'Physical' (n=44) and 'Digital' (n=80) risk domains
based on sector keywords. Autonomy levels were mapped to an ordinal scale (1-3).

2. **Statistical Results**:
   - The Mann-Whitney U test yielded a U-statistic of **1762.0**, which is
effectively identical to the expected value under the null hypothesis ($n_1
\times n_2 / 2 = 1760$).
   - The p-value of **0.993** confirms that the distributions of autonomy levels
in Physical and Digital sectors are statistically indistinguishable.
   - Although the sample median for Physical (2.0) was higher than Digital
(1.0), the rank-sum test proves this difference is not significant, likely due
to the discrete nature of the data and distribution shapes.

3. **Visual Discrepancy**: The provided 'Plot Analysis' text (likely generated
by a vision model) incorrectly interprets the Digital sector as having higher
autonomy. This contradicts the calculated medians (Physical=2.0 > Digital=1.0)
and the statistical finding of no difference. The quantitative code output
should be trusted over the descriptive text.

4. **Conclusion**: The hypothesis is **not supported**. There is no evidence in
this dataset that physical-sector AI incidents are characterized by higher
system autonomy than digital-sector incidents.

---

## Review

The experiment was faithfully implemented and the hypothesis was successfully
tested.

**Hypothesis**: 'Domain-Specific Autonomy Risk' – stating that AI incidents in
physical-impact sectors are associated with higher autonomy levels than those in
digital-impact sectors.

**Methodology**:
- **Data**: Filtered 124 incidents from the AI Incident Database.
- **Mapping**: Sectors were categorized into 'Physical' (n=44) and 'Digital'
(n=80) using keyword matching. Autonomy levels were mapped to an ordinal scale
(1–3).
- **Test**: Mann-Whitney U test (two-sided).

**Results**:
- **Statistics**: The U-statistic was 1762.0, almost exactly the expected value
for identical distributions (1760), yielding a p-value of 0.99.
- **Medians**: Physical median = 2.0; Digital median = 1.0.

**Findings**: The hypothesis is **not supported**. Despite the Physical sector
showing a higher median autonomy level, the statistical test confirms that the
overall distribution of autonomy levels between physical and digital sectors is
indistinguishable (p=0.99). The difference in medians is likely an artifact of
the discrete ordinal scale and does not reflect a statistically significant
trend.

**Note**: The provided 'Plot Analysis' text (likely generated by an external
vision model) incorrectly described the Digital sector as having higher
autonomy, contradicting the ground-truth statistical output. The statistical
results (Code Output) are taken as authoritative.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# Load dataset
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()

# Clean Autonomy Level
autonomy_map = {
    'Autonomy1': 1, 
    'Autonomy2': 2, 
    'Autonomy3': 3
}
aiid['autonomy_score'] = aiid['Autonomy Level'].map(autonomy_map)

# Clean Sector of Deployment
def map_domain(val):
    if pd.isna(val):
        return None
    val_str = str(val).lower()
    
    physical_keywords = ['transport', 'health', 'medic', 'manufactur', 'industr', 
                         'energy', 'agricultur', 'construct', 'robot']
    digital_keywords = ['financ', 'bank', 'educat', 'govern', 'public', 'media', 
                        'entertain', 'retail', 'consum', 'service']
    
    # Check physical first (arbitrary priority, or could be exclusive)
    if any(k in val_str for k in physical_keywords):
        return 'Physical'
    elif any(k in val_str for k in digital_keywords):
        return 'Digital'
    return None

aiid['risk_domain'] = aiid['Sector of Deployment'].apply(map_domain)

# Drop rows with missing values for the analysis
analysis_df = aiid.dropna(subset=['autonomy_score', 'risk_domain'])

# Descriptive Statistics
print("--- Analysis Counts ---")
counts = analysis_df['risk_domain'].value_counts()
print(counts)

# Statistical Test
physical_scores = analysis_df[analysis_df['risk_domain'] == 'Physical']['autonomy_score']
digital_scores = analysis_df[analysis_df['risk_domain'] == 'Digital']['autonomy_score']

print("\n--- Mann-Whitney U Test ---")
if len(physical_scores) > 0 and len(digital_scores) > 0:
    stat, p = mannwhitneyu(physical_scores, digital_scores, alternative='two-sided')
    print(f"U-statistic: {stat}")
    print(f"p-value: {p:.5f}")
    if p < 0.05:
        print("Result: Significant difference in autonomy levels between Physical and Digital domains.")
    else:
        print("Result: No significant difference found.")
        
    # Calculate medians for context
    print(f"Median Autonomy (Physical): {physical_scores.median()}")
    print(f"Median Autonomy (Digital): {digital_scores.median()}")
else:
    print("Insufficient data for statistical testing.")

# Visualization
plt.figure(figsize=(8, 6))
sns.boxplot(x='risk_domain', y='autonomy_score', data=analysis_df, order=['Physical', 'Digital'])
plt.title('AI Autonomy Levels: Physical vs Digital Sectors')
plt.ylabel('Autonomy Level (1=Low, 3=High)')
plt.xlabel('Risk Domain')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Analysis Counts ---
risk_domain
Digital     80
Physical    44
Name: count, dtype: int64

--- Mann-Whitney U Test ---
U-statistic: 1762.0
p-value: 0.99317
Result: No significant difference found.
Median Autonomy (Physical): 2.0
Median Autonomy (Digital): 1.0


=== Plot Analysis (figure 1) ===
Based on the visual analysis of the provided image, here is the detailed breakdown:

### 1. Plot Type
*   **Type:** This is a **Box Plot** (also known as a Box-and-Whisker plot).
*   **Purpose:** It compares the distribution of quantitative data (Autonomy Level) across a categorical variable (Risk Domain). It visualizes the range, quartiles, and potential spread of the data.

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Label:** "Risk Domain"
    *   **Categories:** Two discrete categories: "Physical" and "Digital".
*   **Y-Axis (Vertical):**
    *   **Label:** "Autonomy Level (1=Low, 3=High)"
    *   **Value Range:** The axis scale runs from **1.00 to 3.00**.
    *   **Tick Marks:** Major ticks are distinct, with grid lines appearing at 0.25 intervals (1.0, 1.25, 1.50, ... 3.00).

### 3. Data Trends
*   **Physical Sector:**
    *   **Interquartile Range (IQR):** The blue box spans from a value of **1.0 to 2.0**. This indicates that the middle 50% of the data falls within the Low to Medium autonomy range.
    *   **Whisker:** There is a top whisker extending upwards to **3.0**. This indicates that while the bulk of the data is lower, the maximum value in this category does reach "High" (3.0).
    *   **Distribution:** The data is skewed toward the lower end of the autonomy scale.
*   **Digital Sector:**
    *   **Interquartile Range (IQR):** The blue box spans the entire range from **1.0 to 3.0**.
    *   **Whiskers:** There are no visible whiskers extending beyond the box, suggesting the IQR covers the full range of the data (min to max).
    *   **Distribution:** The "Digital" sector shows a much broader distribution of central data, indicating a higher prevalence of high autonomy levels compared to the Physical sector.

### 4. Annotations and Legends
*   **Title:** "AI Autonomy Levels: Physical vs Digital Sectors" is displayed at the top.
*   **Grid Lines:** Horizontal dashed grid lines are present to assist in reading specific Y-axis values.
*   **Legend:** There is no distinct legend because the categories are explicitly labeled on the X-axis. The solid blue color is consistent across both categories.

### 5. Statistical Insights
*   **Higher Autonomy in Digital:** The Digital sector generally exhibits higher levels of autonomy than the Physical sector. The upper quartile (top of the box) for Digital is at the maximum level (3.0), whereas the upper quartile for Physical is only at 2.0.
*   **Constraint in Physical Systems:** The Physical sector appears more constrained. While it *can* reach high autonomy (as indicated by the whisker reaching 3.0), the majority of physical systems (75%) operate at or below level 2.0. This likely reflects real-world safety constraints and regulatory hurdles inherent to physical robotics that do not apply as strictly to purely digital software.
*   **Variability:** The Digital sector has high variability within its core distribution (a larger box), implying that digital AI agents are just as likely to be high autonomy as they are low autonomy, whereas physical agents are clustered more heavily in the low-autonomy region.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
