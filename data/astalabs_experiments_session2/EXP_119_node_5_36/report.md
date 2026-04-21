# Experiment 119: node_5_36

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_36` |
| **ID in Run** | 119 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T06:43:04.673771+00:00 |
| **Runtime** | 308.4s |
| **Parent** | `node_4_17` |
| **Children** | `node_6_24`, `node_6_82` |
| **Creation Index** | 120 |

---

## Hypothesis

> The 'Causal Clarity' of Physical Harm: Incidents resulting in 'Physical' harm
are significantly more likely to have a specific, identified 'AI Technical
Failure' type compared to 'Allocative' or 'Intangible' harms, which often lack
technical attribution.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.2637 (Likely False) |
| **Surprise** | -0.5738 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 30.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 4.0 |
| Definitely False | 56.0 |

---

## Experiment Plan

**Objective:** Test if physical accidents lead to clearer technical root-cause identification than sociotechnical failures.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `aiid_incidents`.
- 2. Categorize incidents into 'Physical' (from `Tangible Harm`) and 'Intangible/Allocative' (from `Harm Domain` or `Harm Distribution Basis`).
- 3. Create a binary variable `Has_Technical_Cause` (1 if `Known AI Technical Failure` is not null/empty, 0 otherwise).
- 4. Perform a Chi-Square test to compare the rate of technical attribution between Physical and Intangible harm groups.

### Deliverables
- 1. Attribution rates (percentage with known cause) for Physical vs. Intangible harms.
- 2. Chi-Square test results.
- 3. Bar chart of attribution rates.

---

## Analysis

The experiment tested the 'Causal Clarity' hypothesis, which posited that AI
incidents causing physical harm are more likely to have identified technical
root causes than those causing intangible harms. Using data from the AI Incident
Database (n=180 categorized incidents), the analysis compared attribution rates
between 'Physical' harms (e.g., injury, property damage) and 'Intangible' harms
(e.g., discrimination, financial loss).

**Results:**
- **Attribution Rates:** The proportion of incidents with a known technical
failure was nearly identical for both groups: **65.1%** for Physical harms and
**64.9%** for Intangible harms.
- **Statistical Significance:** A Chi-Square test confirmed the lack of
difference (Chi2=0.0000, p=1.0000).

**Conclusion:**
The hypothesis is **not supported**. Contrary to the expectation that physical
accidents offer clearer causal evidence, the data indicates that technical
attribution rates are uniform across harm types. Both physical and intangible
incidents have a ~65% likelihood of having a documented technical failure type
in the database.

---

## Review

The experiment successfully tested the 'Causal Clarity' hypothesis using the AI
Incident Database (AIID). After an initial debugging phase to correctly map the
'Tangible Harm' column values, the analysis proceeded with a sample of 180
categorized incidents (94 Intangible, 86 Physical).

**Hypothesis**: Incidents resulting in 'Physical' harm are significantly more
likely to have a specific, identified 'AI Technical Failure' compared to
'Intangible' harms.

**Findings**:
1.  **Uniform Attribution Rates**: Contrary to the hypothesis, the rate of
identified technical failures was nearly identical between the two groups:
**65.1%** for Physical harms and **64.9%** for Intangible harms.
2.  **Statistical Significance**: The Chi-Square test yielded a p-value of
**1.0000** (Chi2=0.00), indicating absolutely no statistical difference between
the groups.

**Conclusion**: The hypothesis is **not supported**. The data suggests that
'Causal Clarity' regarding technical failures is not dependent on whether the
harm is physical or intangible; both types of incidents have a ~65% likelihood
of having a documented technical root cause in the database.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

print("Starting experiment...")

# 1. Load Data
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
df_incidents = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Loaded {len(df_incidents)} incidents from AIID.")

# 2. Define Groups based on 'Tangible Harm' strings found in debug
# Physical: Actual harm or imminent risk of harm
# Intangible: Explicitly labeled as 'no tangible harm' (implying other types of harm like allocative/societal)

def categorize_harm(val):
    if pd.isna(val):
        return None
    s = str(val).lower()
    
    # Physical Keywords based on debug output
    if "tangible harm definitively occurred" in s:
        return "Physical"
    if "imminent risk" in s and "did occur" in s:
        return "Physical"
        
    # Intangible Keywords
    if "no tangible harm" in s:
        return "Intangible"
        
    return None

df_incidents['Harm_Group'] = df_incidents['Tangible Harm'].apply(categorize_harm)

# 3. Define Technical Cause
# 1 if 'Known AI Technical Failure' is populated, 0 otherwise
def has_tech_cause(val):
    if pd.isna(val):
        return 0
    s = str(val).strip()
    if s in ["", "nan", "None", "[]"]:
        return 0
    return 1

df_incidents['Has_Technical_Cause'] = df_incidents['Known AI Technical Failure'].apply(has_tech_cause)

# 4. Filter for Analysis
df_analysis = df_incidents.dropna(subset=['Harm_Group']).copy()

print("\n--- Analysis Groups ---@")
print(df_analysis['Harm_Group'].value_counts())

# 5. Statistical Test
contingency = pd.crosstab(df_analysis['Harm_Group'], df_analysis['Has_Technical_Cause'])
print("\n--- Contingency Table (0=No Tech Cause, 1=Has Tech Cause) ---")
print(contingency)

if contingency.shape == (2, 2):
    # Chi-Square Test
    chi2, p, dof, ex = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test Results:\n  Chi2 Statistic: {chi2:.4f}\n  P-value: {p:.5f}")
    
    # Calculate Attribution Rates
    rates = df_analysis.groupby('Harm_Group')['Has_Technical_Cause'].mean()
    print("\n--- Attribution Rates (Proportion with Known Cause) ---")
    print(rates)
    
    # Visualization
    plt.figure(figsize=(8, 6))
    # Color: Red for Physical, Blue for Intangible
    colors = ['#3498db' if x == 'Intangible' else '#e74c3c' for x in rates.index]
    bars = plt.bar(rates.index, rates.values, color=colors, alpha=0.8)
    
    plt.title('"Causal Clarity": Technical Attribution by Harm Type')
    plt.ylabel('Proportion of Incidents with Identified Technical Failure')
    plt.ylim(0, max(rates.values) * 1.2 if max(rates.values) > 0 else 1.0)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, 
                 f'{height:.1%}', ha='center', va='bottom')
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.show()

else:
    print("\nInsufficient data for 2x2 Chi-Square test.")
    print("Rows with defined Harm Group:", len(df_analysis))

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting experiment...
Loaded 1362 incidents from AIID.

--- Analysis Groups ---@
Harm_Group
Intangible    94
Physical      86
Name: count, dtype: int64

--- Contingency Table (0=No Tech Cause, 1=Has Tech Cause) ---
Has_Technical_Cause   0   1
Harm_Group                 
Intangible           33  61
Physical             30  56

Chi-Square Test Results:
  Chi2 Statistic: 0.0000
  P-value: 1.00000

--- Attribution Rates (Proportion with Known Cause) ---
Harm_Group
Intangible    0.648936
Physical      0.651163
Name: Has_Technical_Cause, dtype: float64


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

**1. Plot Type**
*   **Type:** Bar Plot.
*   **Purpose:** The plot compares two categorical variables ("Intangible" and "Physical" harm types) against a quantitative measure (Proportion of incidents where a technical failure was identified). It is designed to show the similarity or disparity in technical attribution between these two categories.

**2. Axes**
*   **Y-Axis:**
    *   **Label:** "Proportion of Incidents with Identified Technical Failure".
    *   **Range:** The axis is scaled from 0.0 to approximately 0.8 (with visible ticks marking intervals of 0.1 up to 0.7).
    *   **Units:** Proportions (0.0 to 1.0), which are also interpreted as percentages in the annotations.
*   **X-Axis:**
    *   **Label:** Implicitly represents "Harm Type".
    *   **Categories:** The axis displays two distinct categories: "Intangible" and "Physical".

**3. Data Trends**
*   **Pattern:** The data shows a remarkable consistency between the two categories. There is almost no significant difference in the height of the bars.
*   **Tallest Bar:** The "Physical" category is marginally higher, representing a proportion of roughly 0.651.
*   **Shortest Bar:** The "Intangible" category is slightly lower, representing a proportion of roughly 0.649.
*   **Comparison:** The visual trend indicates parity; the likelihood of attributing a technical failure is essentially the same regardless of whether the harm type is physical or intangible.

**4. Annotations and Legends**
*   **Chart Title:** "Causal Clarity": Technical Attribution by Harm Type.
*   **Bar Annotations:** Specific percentage values are placed directly above each bar for precision:
    *   Intangible: **64.9%**
    *   Physical: **65.1%**
*   **Gridlines:** Horizontal dashed gridlines are included at 0.1 intervals to assist in visual estimation of the bar heights.

**5. Statistical Insights**
*   **Uniformity of Attribution:** The most significant insight is that the rate of identified technical failure is nearly identical across both harm types (a difference of only 0.2%). This suggests that the nature of the harm (whether it touches the physical world or remains intangible/digital) does not impact the likelihood of a technical root cause being identified.
*   **High Attribution Rate:** For both categories, nearly two-thirds of incidents (approx. 65%) are attributed to identified technical failures. This indicates a relatively high level of "Causal Clarity" in the dataset being analyzed.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
