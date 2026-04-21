# Experiment 158: node_6_14

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_14` |
| **ID in Run** | 158 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T08:29:03.639413+00:00 |
| **Runtime** | 367.1s |
| **Parent** | `node_5_12` |
| **Children** | None |
| **Creation Index** | 159 |

---

## Hypothesis

> Sector Harm Fingerprints: AI incidents in the 'Financial' sector are
statistically more likely to involve 'Economic' or 'Intangible' harms (e.g.,
discrimination), whereas 'Transportation' incidents are more likely to involve
'Physical' harms.

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

**Objective:** Compare the distribution of Harm Domains across Financial and Transportation sectors.

### Steps
- 1. Load `aiid_incidents` data.
- 2. Filter for `Sector of Deployment` containing 'Finance'/'Bank' vs 'Transport'/'Auto'/'Aviation'.
- 3. Categorize `Harm Domain` (or `Tangible Harm`) into 'Physical' vs 'Non-Physical' (Economic/Social/Rights).
- 4. Create a contingency table.
- 5. Run a Fisher's Exact Test or Chi-square test.
- 6. Calculate the percentage of Physical harm for each sector.

### Deliverables
- Contingency table, statistical test results, and harm type distribution per sector.

---

## Analysis

The experiment successfully tested the 'Sector Harm Fingerprints' hypothesis.
Facing sparse metadata in the structured columns (as revealed in the previous
debug step), the analysis effectively pivoted to keyword-based classification on
unstructured text fields (`title`, `description`, `summary`). This strategy
successfully categorized 207 incidents. The results strongly support the
hypothesis: the Financial sector is characterized by 'Non-Physical' harms
(93.88% of cases, e.g., economic loss, bias), while the Transportation sector is
dominated by 'Physical' harms (72.48% of cases, e.g., safety accidents). The
difference is highly statistically significant (Chi-Square = 91.16, p < 1e-21),
confirming distinct risk profiles for these industries.

---

## Review

The experiment successfully tested the 'Sector Harm Fingerprints' hypothesis.
Facing sparse metadata in the structured columns (as revealed in the previous
debug step), the analysis effectively pivoted to keyword-based classification on
unstructured text fields (`title`, `description`, `summary`). This strategy
successfully categorized 207 incidents. The results strongly support the
hypothesis: the Financial sector is characterized by 'Non-Physical' harms
(93.88% of cases, e.g., economic loss, bias), while the Transportation sector is
dominated by 'Physical' harms (72.48% of cases, e.g., safety accidents). The
difference is highly statistically significant (Chi-Square = 91.16, p < 1e-21),
confirming distinct risk profiles for these industries.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
df_aiid = df[df['source_table'] == 'aiid_incidents'].copy()

# Create a text corpus for classification from potential text columns
# We prioritize columns known to exist or likely to contain descriptive text
text_cols = ['title', 'description', 'summary', 'reports', 'Sector of Deployment', 'Infrastructure Sectors', 'Harm Domain', 'Tangible Harm', 'Alleged harmed or nearly harmed parties']

# Combine available columns into a single string for keyword searching
df_aiid['text_corpus'] = ''
for col in text_cols:
    if col in df_aiid.columns:
        df_aiid['text_corpus'] += ' ' + df_aiid[col].fillna('').astype(str)

df_aiid['text_corpus'] = df_aiid['text_corpus'].str.lower()

# --- 1. Classify Sectors ---
sector_map = {
    'Financial': ['financ', 'bank', 'credit', 'loan', 'insurance', 'trading', 'mortgage', 'lending', 'crypto'],
    'Transportation': ['transport', 'auto', 'vehicle', 'car', 'aviation', 'flight', 'drone', 'driverless', 'tesla', 'uber', 'collision', 'autopilot', 'self-driving']
}

def classify_sector(text):
    scores = {cat: 0 for cat in sector_map}
    for cat, keywords in sector_map.items():
        for k in keywords:
            if k in text:
                scores[cat] += 1
    
    # Return the category with the highest non-zero score
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return None

df_aiid['derived_sector'] = df_aiid['text_corpus'].apply(classify_sector)

# --- 2. Classify Harms ---
harm_map = {
    'Physical': ['death', 'kill', 'injur', 'hurt', 'accident', 'crash', 'safety', 'physical', 'died', 'fatal', 'bodily'],
    'Non-Physical': ['economic', 'money', 'financ', 'cost', 'discrimin', 'bias', 'racist', 'sexist', 'privacy', 'surveillance', 'reputation', 'credit score', 'denied', 'unfair']
}

def classify_harm(text):
    # Priority check: If 'death' or 'injury' is explicitly mentioned, it's Physical (safety critical)
    # However, we'll use a scoring system to be robust
    scores = {cat: 0 for cat in harm_map}
    for cat, keywords in harm_map.items():
        for k in keywords:
            if k in text:
                scores[cat] += 1
    
    # Heuristic: Physical harm usually implies safety incidents which are distinct from pure economic/bias
    # If both present, usually the Physical aspect is the 'incident' trigger (e.g., crash)
    if scores['Physical'] > 0:
        return 'Physical'
    elif scores['Non-Physical'] > 0:
        return 'Non-Physical'
    return None

df_aiid['derived_harm'] = df_aiid['text_corpus'].apply(classify_harm)

# --- 3. Analysis ---
# Filter for rows where both Sector and Harm were identified
df_analysis = df_aiid.dropna(subset=['derived_sector', 'derived_harm'])

print(f"Total AIID Incidents: {len(df_aiid)}")
print(f"Incidents with identified Sector & Harm: {len(df_analysis)}")
print("Sector Breakdown in Analysis Set:")
print(df_analysis['derived_sector'].value_counts())

# Contingency Table
contingency = pd.crosstab(df_analysis['derived_sector'], df_analysis['derived_harm'])
print("\nContingency Table (Sector vs Harm Type):")
print(contingency)

# Statistical Test
if contingency.size >= 4:
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"p-value: {p:.4e}")
    
    # Calculate Percentages
    row_props = pd.crosstab(df_analysis['derived_sector'], df_analysis['derived_harm'], normalize='index') * 100
    print("\nHarm Type Distribution by Sector (%):")
    print(row_props.round(2))
    
    # Check Hypothesis
    # Hypothesis: Financial -> Non-Physical (Economic), Transportation -> Physical
    fin_non_phys = row_props.loc['Financial', 'Non-Physical'] if 'Financial' in row_props.index and 'Non-Physical' in row_props.columns else 0
    trans_phys = row_props.loc['Transportation', 'Physical'] if 'Transportation' in row_props.index and 'Physical' in row_props.columns else 0
    
    print(f"\nFinancial Incidents causing Non-Physical Harm: {fin_non_phys:.1f}%")
    print(f"Transportation Incidents causing Physical Harm: {trans_phys:.1f}%")
    
    # Plot
    try:
        row_props.plot(kind='bar', stacked=True, color=['orange', 'red'], alpha=0.7, figsize=(8, 6))
        plt.title('Harm Fingerprints: Physical vs Non-Physical Harm by Sector')
        plt.xlabel('Sector')
        plt.ylabel('Percentage of Incidents')
        plt.legend(title='Harm Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plotting error: {e}")
else:
    print("Insufficient data for statistical test.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Total AIID Incidents: 1362
Incidents with identified Sector & Harm: 207
Sector Breakdown in Analysis Set:
derived_sector
Transportation    109
Financial          98
Name: count, dtype: int64

Contingency Table (Sector vs Harm Type):
derived_harm    Non-Physical  Physical
derived_sector                        
Financial                 92         6
Transportation            30        79

Chi-Square Statistic: 91.1611
p-value: 1.3244e-21

Harm Type Distribution by Sector (%):
derived_harm    Non-Physical  Physical
derived_sector                        
Financial              93.88      6.12
Transportation         27.52     72.48

Financial Incidents causing Non-Physical Harm: 93.9%
Transportation Incidents causing Physical Harm: 72.5%


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Stacked Bar Chart.
*   **Purpose:** To compare the proportional composition of "Harm Type" (Physical vs. Non-Physical) across two distinct sectors (Financial and Transportation). It allows for an easy comparison of the percentage breakdown of incidents within each category.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Sector"
    *   **Labels:** Two categorical variables: "Financial" and "Transportation" (labels are oriented vertically).
*   **Y-Axis:**
    *   **Title:** "Percentage of Incidents"
    *   **Range:** 0 to 100.
    *   **Units:** Percentage (%).
    *   **Increments:** Ticks are marked every 20 units (0, 20, 40, 60, 80, 100).

### 3. Data Trends
*   **Financial Sector:**
    *   **Pattern:** This sector is overwhelmingly dominated by "Non-Physical" harm.
    *   **Values:** The "Non-Physical" segment (yellow/orange) appears to occupy approximately **90-95%** of the bar, while the "Physical" segment (red) represents a very small minority (approx. 5-10%).
*   **Transportation Sector:**
    *   **Pattern:** This sector shows the opposite trend, with "Physical" harm being the dominant type.
    *   **Values:** The "Physical" segment (red) occupies the majority of the bar, approximately **70-75%**. The "Non-Physical" segment (yellow/orange) comprises the remaining ~25-30%.

### 4. Annotations and Legends
*   **Chart Title:** "Harm Fingerprints: Physical vs Non-Physical Harm by Sector" – located at the top center.
*   **Legend:** Located on the right side of the plot.
    *   **Title:** "Harm Type"
    *   **Keys:**
        *   **Non-Physical:** Represented by the Yellow/Orange color.
        *   **Physical:** Represented by the Red color.

### 5. Statistical Insights
*   **Sector Contrast:** There is a stark dichotomy between the two sectors. The **Financial** sector's risk profile is almost entirely non-physical (likely financial loss, fraud, or data theft), whereas the **Transportation** sector carries a high risk of physical harm (likely accidents or injuries).
*   **Risk Profiles:** If one were to assign a "fingerprint" to these industries based on harm, the Financial sector is defined by intangible impact, while the Transportation sector is defined primarily by tangible, bodily impact.
*   **Magnitude:** The dominance of physical harm in transportation is significant (roughly 3 times larger than non-physical harm in that sector), but it is not as absolute as the dominance of non-physical harm in the financial sector (which appears to be nearly 10 times larger than physical harm in that sector).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
