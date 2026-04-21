# Experiment 92: node_5_23

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_23` |
| **ID in Run** | 92 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-21T00:29:56.267317+00:00 |
| **Runtime** | 202.2s |
| **Parent** | `node_4_26` |
| **Children** | None |
| **Creation Index** | 93 |

---

## Hypothesis

> Incidents precipitated by external threat actors are significantly more likely
to exploit 'Integration Readiness' gaps, whereas insider or accidental incidents
are more likely to involve 'Trust Readiness' gaps.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.3760 (Maybe False) |
| **Surprise** | -0.4247 |
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
| Maybe False | 90.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Determine if the threat actor type (External vs. Internal) predicts the type of competency failure.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Categorize the 'actor' column into 'External' (e.g., Nation State, Outsider) and 'Internal/Accidental' (e.g., Insider, Researcher, User).
- 3. Extract the 'trust_integration_split' variable.
- 4. Create a contingency table of Actor Type vs. Competency Split.
- 5. Perform a Chi-Square test or Fisher's Exact Test.

### Deliverables
- Heatmap of Actor Type vs. Competency Split and statistical test summary.

---

## Analysis

The experiment tested the hypothesis that incidents precipitated by external
threat actors are more likely to exploit 'Integration Readiness' gaps, while
insider incidents exploit 'Trust Readiness' gaps. The analysis of 52 case
studies categorized actors into 'External' (n=45) and 'Internal/Accidental'
(n=7). The contingency table revealed a heavy skew towards incidents involving
'Both' competency domains (46 out of 52 cases).

Specific findings:
- External Actors: 40 'Both', 4 'Integration-dominant', 1 'Trust-dominant'.
- Internal Actors: 6 'Both', 0 'Integration-dominant', 1 'Trust-dominant'.

The Chi-Square test of independence yielded a statistic of 2.9212 with a p-value
of 0.2321. Since the p-value is greater than 0.05, the null hypothesis cannot be
rejected. There is no statistically significant association between the type of
threat actor and the competency gap domain. The hypothesis is unsupported; the
data suggests that real-world AI incidents overwhelmingly require simultaneous
failures in both governance (Trust) and engineering (Integration) to occur,
regardless of the threat actor's origin.

---

## Review

The experiment successfully tested the hypothesis that incidents precipitated by
external threat actors are more likely to exploit 'Integration Readiness' gaps,
while insider incidents exploit 'Trust Readiness' gaps. The analysis of 52 case
studies categorized actors into 'External' (n=45) and 'Internal/Accidental'
(n=7). The contingency table revealed a heavy skew towards incidents involving
'Both' competency domains (46 out of 52 cases).

Specific findings:
- External Actors: 40 'Both', 4 'Integration-dominant', 1 'Trust-dominant'.
- Internal Actors: 6 'Both', 0 'Integration-dominant', 1 'Trust-dominant'.

The Chi-Square test of independence yielded a statistic of 2.9212 with a p-value
of 0.2321. Since the p-value is greater than 0.05, the null hypothesis cannot be
rejected. There is no statistically significant association between the type of
threat actor and the competency gap domain. The hypothesis is unsupported; the
data suggests that real-world AI incidents overwhelmingly require simultaneous
failures in both governance (Trust) and engineering (Integration) to occur,
regardless of the threat actor's origin.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import numpy as np
import os

# Attempt to locate the file, prioritizing current directory then parent
filename = 'step3_incident_coding.csv'
if os.path.exists(filename):
    file_path = filename
elif os.path.exists(os.path.join('..', filename)):
    file_path = os.path.join('..', filename)
else:
    # Fallback to absolute path check or error
    file_path = filename

print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path)

# Function to categorize actors
def categorize_actor(actor_raw):
    if pd.isna(actor_raw):
        return 'Unknown'
    actor = str(actor_raw).lower()
    
    # Keywords for Internal / Accidental / Research
    # In ATLAS, 'Researcher' is often treated as a proxy for internal/authorized access in proof-of-concepts
    # 'User' implies authorized user.
    internal_keywords = ['insider', 'researcher', 'user', 'employee', 'developer', 'accidental', 'academic', 'student']
    
    for keyword in internal_keywords:
        if keyword in actor:
            return 'Internal/Accidental'
            
    return 'External'

# Apply categorization
df['actor_category'] = df['actor'].apply(categorize_actor)

# Remove 'Unknown' if any, though the logic defaults to External. 
# Let's see if there are empty rows.
df = df[df['actor'].notna()]

# Generate Contingency Table
# Rows: Actor Type, Cols: Competency Split
contingency_table = pd.crosstab(df['actor_category'], df['trust_integration_split'])

print("=== Actor Categorization Samples ===")
print(df[['actor', 'actor_category']].drop_duplicates().head(10))

print("\n=== Contingency Table (Actor vs Competency Split) ===")
print(contingency_table)

# Perform Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\n=== Chi-Square Test Results ===")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"p-value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")

# Calculate Cramer's V
n = contingency_table.sum().sum()
min_dim = min(contingency_table.shape) - 1
cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
print(f"Cramer's V: {cramers_v:.4f}")

# Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues')
plt.title('Threat Actor vs. Competency Gap (Trust/Integration)')
plt.ylabel('Actor Category')
plt.xlabel('Competency Gap Type')
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step3_incident_coding.csv
=== Actor Categorization Samples ===
                                        actor       actor_category
0         Palo Alto Networks AI Research Team             External
2                                     Unknown             External
3                              Skylight Cyber             External
4                             Two individuals             External
5   Berkeley Artificial Intelligence Research             External
6                   Researchers at spiderSilk  Internal/Accidental
7             Researchers at Brown University  Internal/Accidental
8        Researchers at Silent Break Security  Internal/Accidental
9                                 4chan Users  Internal/Accidental
10                      Microsoft AI Red Team             External

=== Contingency Table (Actor vs Competency Split) ===
trust_integration_split  both  integration-dominant  trust-dominant
actor_category                                                     
External                   40                     4               1
Internal/Accidental         6                     0               1

=== Chi-Square Test Results ===
Chi2 Statistic: 2.9212
p-value: 0.2321
Degrees of Freedom: 2
Cramer's V: 0.2370


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis of the plot:

### 1. Plot Type
*   **Type:** Heatmap (specifically, a contingency table visualized as a heatmap).
*   **Purpose:** To visualize the frequency distribution and relationship between two categorical variables: "Actor Category" and "Competency Gap Type." The intensity of the color corresponds to the count (frequency) of occurrences for each combination.

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Title:** Competency Gap Type
    *   **Labels:** "both", "integration-dominant", "trust-dominant"
*   **Y-Axis (Vertical):**
    *   **Title:** Actor Category
    *   **Labels:** "External", "Internal/Accidental"
*   **Color Bar (Legend Axis):**
    *   **Range:** 0 to 40
    *   **Unit:** Frequency count (number of occurrences).
    *   **Gradient:** Light blue (low frequency) to Dark blue (high frequency).

### 3. Data Trends
*   **Highest Concentration:** The intersection of **"External"** actors and **"both"** competency gap types is by far the most significant, with a value of **40**. This area is represented by the darkest blue square.
*   **Secondary Cluster:** The next highest value is **6**, found at the intersection of **"Internal/Accidental"** actors and **"both"** competency gap types.
*   **Low Values/Outliers:**
    *   **"Trust-dominant"** gaps are rare for both actor categories (1 for External, 1 for Internal/Accidental).
    *   **"Integration-dominant"** gaps are infrequent for External actors (4) and non-existent for Internal/Accidental actors (0).
*   **Pattern:** The column "both" dominates the dataset regardless of the actor category, suggesting that when competency gaps occur, they usually involve both trust and integration issues simultaneously.

### 4. Annotations and Legends
*   **Cell Annotations:** Each cell contains a number representing the exact count of observations for that specific category pair (e.g., the top-left cell displays "40").
*   **Color Bar Legend:** Located on the right, it provides a visual reference for the numerical scale, indicating that darker shades of blue represent higher counts and lighter shades represent lower counts.
*   **Title:** "Threat Actor vs. Competency Gap (Trust/Integration)" serves as the main title, summarizing the comparison being made.

### 5. Statistical Insights
*   **Dominance of External Actors:** External actors account for the vast majority of the data points.
    *   External Total: $40 + 4 + 1 = 45$
    *   Internal/Accidental Total: $6 + 0 + 1 = 7$
    *   *Insight:* External actors are involved in approximately **86.5%** of the recorded incidents/cases.
*   **Prevalence of "Both" Gap Types:**
    *   "Both" Total: $40 + 6 = 46$
    *   "Integration-dominant" Total: 4
    *   "Trust-dominant" Total: 2
    *   *Insight:* In roughly **88.5%** of cases, the competency gap involves elements of *both* trust and integration, rather than one dominating the other.
*   **Zero Incidence:** There are **zero** recorded instances where an "Internal/Accidental" actor was associated with a strictly "integration-dominant" competency gap.
*   **Conclusion:** The primary threat model visualized here suggests that "External" actors exploiting "both" trust and integration gaps simultaneously constitutes the primary risk scenario, overshadowing all other combinations significantly.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
