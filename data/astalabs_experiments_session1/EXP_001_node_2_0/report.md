# Experiment 1: node_2_0

| Property | Value |
|---|---|
| **Experiment ID** | `node_2_0` |
| **ID in Run** | 1 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T21:31:16.718272+00:00 |
| **Runtime** | 186.2s |
| **Parent** | `node_1_0` |
| **Children** | `node_3_3`, `node_3_6`, `node_3_18` |
| **Creation Index** | 2 |

---

## Hypothesis

> Security-related harm types are exclusively associated with 'Integration
Readiness' competency gaps, whereas non-security harms (Privacy, Bias, Safety)
are predominantly associated with 'Trust Readiness' gaps.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.5000 (Uncertain) |
| **Posterior** | 0.1281 (Likely False) |
| **Surprise** | -0.4316 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 15.0 |
| Uncertain | 0.0 |
| Maybe False | 15.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 90.0 |

---

## Experiment Plan

**Objective:** Test the alignment between harm categories and competency bundles.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Categorize 'harm_type' into 'Security' (e.g., security, supply_chain) and 'Non-Security' (privacy, bias_discrimination, reliability).
- 3. Extract the primary 'trust_integration_split' for each incident.
- 4. Construct a confusion matrix or contingency table between Harm Category and Readiness Split.
- 5. Calculate the correlation coefficient (e.g., Phi coefficient) or perform a Fisher's Exact Test to assess the strength of the association.

### Deliverables
- Heatmap of Harm Type vs. Competency Split and statistical test results.

---

## Analysis

The experiment tested the hypothesis that Security harms align with 'Integration
Readiness' gaps and Non-Security harms with 'Trust Readiness' gaps. The results
leads to the rejection of this hypothesis. The contingency table reveals an
overwhelming dominance of the 'Both' classification: 100% of Non-Security
incidents (9/9) and 86% of Security incidents (37/43) are classified as 'Both'.
Consequently, the Chi-Square test yielded a p-value of 0.4917, indicating no
statistically significant association between harm type and competency split.
The data suggests that real-world AI incidents typically involve failures in
both Trust and Integration capabilities simultaneously, rather than separating
cleanly into one domain.

---

## Review

The experiment was successfully executed, and the results provide strong
evidence to reject the hypothesis. The contingency table and Chi-Square test
(p=0.4917) show no significant association between harm types and specific
competency splits. Instead, the data reveals a universal 'trust-integration
integration' pattern, where 88% of all incidents (46/52) involve gaps in both
bundles simultaneously, regardless of whether the harm is security-related or
not. The hypothesis that security harms map to integration gaps and non-security
harms map to trust gaps is unsupported.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import os

# Define file path
filename = 'step3_incident_coding.csv'

# Attempt to load the dataset
try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    # Fallback to parent directory if not found in current
    df = pd.read_csv(os.path.join('..', filename))

# Define the mapping for Harm Types
security_harms = ['security', 'supply_chain', 'intellectual_property', 'autonomy_misuse']
non_security_harms = ['privacy', 'bias_discrimination', 'reliability']

def categorize_harm(harm):
    h = str(harm).strip()
    if h in security_harms:
        return 'Security'
    elif h in non_security_harms:
        return 'Non-Security'
    else:
        return 'Other'

# Apply categorization
df['harm_category'] = df['harm_type'].apply(categorize_harm)

# Filter out 'Other' if any (though metadata suggests all are covered)
df = df[df['harm_category'] != 'Other']

# Generate Contingency Table
# Columns expected in 'trust_integration_split': 'trust-dominant', 'integration-dominant', 'both'
contingency = pd.crosstab(df['harm_category'], df['trust_integration_split'])

print("=== Contingency Table (Harm Category vs. Readiness Split) ===")
print(contingency)

# Perform Chi-Square Test of Independence
stat, p, dof, expected = chi2_contingency(contingency)

print(f"\n=== Statistical Test Results (Chi-Square) ===")
print(f"Chi2 Statistic: {stat:.4f}")
print(f"P-Value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")

# Row Percentages for clearer interpretation
contingency_pct = pd.crosstab(df['harm_category'], df['trust_integration_split'], normalize='index') * 100
print("\n=== Row Percentages ===")
print(contingency_pct.round(2))

# Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(contingency, annot=True, fmt='d', cmap='YlGnBu', cbar_kws={'label': 'Incident Count'})
plt.title('Heatmap: Harm Category vs. Trust/Integration Split')
plt.xlabel('Competency Split')
plt.ylabel('Harm Category')
plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: === Contingency Table (Harm Category vs. Readiness Split) ===
trust_integration_split  both  integration-dominant  trust-dominant
harm_category                                                      
Non-Security                9                     0               0
Security                   37                     4               2

=== Statistical Test Results (Chi-Square) ===
Chi2 Statistic: 1.4196
P-Value: 0.4917
Degrees of Freedom: 2

=== Row Percentages ===
trust_integration_split    both  integration-dominant  trust-dominant
harm_category                                                        
Non-Security             100.00                   0.0            0.00
Security                  86.05                   9.3            4.65


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Annotated Heatmap (representing a contingency table).
*   **Purpose:** To visualize the frequency distribution and relationship between two categorical variables: "Harm Category" and "Competency Split." The color intensity and numerical annotations indicate the count of incidents where these categories intersect.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Competency Split"
    *   **Labels:** "both", "integration-dominant", "trust-dominant"
*   **Y-Axis:**
    *   **Title:** "Harm Category"
    *   **Labels:** "Non-Security", "Security"
*   **Color Scale (Legend):**
    *   **Label:** "Incident Count"
    *   **Range:** 0 to approximately 37 (based on the maximum data point). The scale uses a gradient from pale yellow (0) to deep blue (high values).

### 3. Data Trends
*   **Highest Value Area:** The most significant cluster is found at the intersection of **"Security"** and **"both"**, showing a count of **37**. This cell is colored the darkest blue, indicating it is the dominant category combination by a large margin.
*   **Secondary Cluster:** The intersection of **"Non-Security"** and **"both"** shows a moderate count of **9** (light green).
*   **Low Value Areas:**
    *   The "Security" category has minor counts in "integration-dominant" (4) and "trust-dominant" (2).
    *   The "Non-Security" category has **0** incidents in both "integration-dominant" and "trust-dominant" columns.
*   **Overall Pattern:** The data is heavily skewed towards the "both" column on the x-axis. Regardless of the Harm Category, the vast majority of incidents fall under the "both" competency split.

### 4. Annotations and Legends
*   **Cell Annotations:** Each cell in the grid contains a numerical integer representing the exact count of incidents for that specific intersection (e.g., "9", "0", "37").
*   **Color Bar:** A vertical bar on the right side provides a reference for the color coding, confirming that darker shades represent higher incident counts.
*   **Title:** The chart is clearly titled "Heatmap: Harm Category vs. Trust/Integration Split".

### 5. Statistical Insights
*   **Dominance of Security Incidents:** The total number of incidents shown is 52 ($9+0+0+37+4+2$). Of these, "Security" related incidents account for **43** (approx. 83%), while "Non-Security" accounts for **9** (approx. 17%).
*   **Concentration in "Both" Competency:** The "both" category is overwhelmingly the most common competency split, accounting for **46 out of 52** incidents (approx. 88%).
*   **Exclusive Distribution:** There is a distinct lack of data for "Non-Security" harms outside of the "both" category. This suggests that whenever a Non-Security harm occurs in this dataset, it is exclusively associated with the "both" competency split.
*   **Rare Events:** "Integration-dominant" and "Trust-dominant" splits are rare overall, appearing only in the context of Security harms, and even then, in very low numbers ($<10\%$ of total data combined).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
