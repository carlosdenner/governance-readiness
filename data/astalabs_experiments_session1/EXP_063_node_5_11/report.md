# Experiment 63: node_5_11

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_11` |
| **ID in Run** | 63 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:31:50.607528+00:00 |
| **Runtime** | 196.0s |
| **Parent** | `node_4_3` |
| **Children** | None |
| **Creation Index** | 64 |

---

## Hypothesis

> The 'RAG Architecture & Data Grounding' control is significantly more likely to
be required by 'Integration Readiness' competencies than 'Trust Readiness'
competencies.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.1613 (Likely False) |
| **Posterior** | 0.0413 (Definitely False) |
| **Surprise** | -0.1392 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 18.0 |
| Definitely False | 12.0 |

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

**Objective:** Validate the architectural specificity of Retrieval-Augmented Generation controls.

### Steps
- 1. Load 'step2_crosswalk_matrix.csv'.
- 2. Identify the column 'RAG Architecture & Data Grounding' and the 'bundle' column.
- 3. Create a contingency table counting the presence ('X') vs. absence of this control for each bundle.
- 4. Perform a Fisher's Exact Test to evaluate the association.
- 5. Calculate the odds ratio.

### Deliverables
- Contingency table, Fisher's Exact Test results (p-value, odds ratio), and a bar chart of control prevalence by bundle.

---

## Analysis

The experiment successfully tested the hypothesis regarding the architectural
specificity of 'RAG Architecture & Data Grounding'. The analysis of 42
governance requirements from the `step2_crosswalk_matrix.csv` dataset revealed
that this control is required by exactly 2 instances in 'Integration Readiness'
(2/23, 8.7%) and 2 instances in 'Trust Readiness' (2/19, 10.5%). Contrary to the
hypothesis that RAG is an Integration-dominant control, the data shows a nearly
identical prevalence across both bundles (with Trust Readiness having a slightly
higher percentage due to a smaller denominator). The Fisher's Exact Test yielded
a p-value of 1.0000 and an Odds Ratio of 0.81, confirming no statistically
significant difference. The hypothesis is therefore rejected; the 'RAG
Architecture & Data Grounding' control appears to be a cross-cutting concern
relevant to both trust (likely for hallucination mitigation and grounding) and
integration (system architecture), rather than being exclusive to one domain.

---

## Review

The experiment was successfully executed and the hypothesis was rigorously
tested. The code correctly loaded the dataset, processed the relevant columns,
and performed the Fisher's Exact Test. The analysis revealed that the 'RAG
Architecture & Data Grounding' control is present in 2 'Integration Readiness'
requirements (8.7%) and 2 'Trust Readiness' requirements (10.5%). The Fisher's
Exact Test resulted in a p-value of 1.0000 and an Odds Ratio of 0.8095,
indicating no statistically significant difference between the two bundles. This
finding leads to the rejection of the hypothesis that RAG is significantly more
prevalent in Integration Readiness. Instead, the data suggests RAG is a cross-
cutting control relevant to both domains (likely addressing
hallucination/grounding in Trust and architectural retrieval in Integration).

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import sys

# [debug]
print("Starting experiment...")

# Robust file loading
filename = 'step2_crosswalk_matrix.csv'
filepath = None

if os.path.exists(filename):
    filepath = filename
elif os.path.exists(os.path.join('..', filename)):
    filepath = os.path.join('..', filename)
else:
    print(f"Error: File {filename} not found in current or parent directory.")
    print(f"Current working directory: {os.getcwd()}")
    try:
        print(f"Listing current dir: {os.listdir('.')}")
        print(f"Listing parent dir: {os.listdir('..')}")
    except Exception as e:
        print(f"Could not list directories: {e}")
    sys.exit(1)

print(f"Loading dataset from: {filepath}")
df = pd.read_csv(filepath)

# Clean column names
df.columns = [c.strip() for c in df.columns]

# Verify columns
rag_col = 'RAG Architecture & Data Grounding'
bundle_col = 'bundle'

if rag_col not in df.columns or bundle_col not in df.columns:
    print(f"Required columns not found. Available: {df.columns.tolist()}")
    sys.exit(1)

# Filter for relevant bundles (just in case there are others/nans)
target_bundles = ['Integration Readiness', 'Trust Readiness']
df = df[df[bundle_col].isin(target_bundles)].copy()

# Create binary variable for RAG control
# Assuming 'X' marks presence, anything else is absence
df['has_rag'] = df[rag_col].apply(lambda x: 1 if str(x).strip().upper() == 'X' else 0)

# Create Contingency Table
# Rows: Bundle, Columns: Has RAG (0, 1)
contingency = pd.crosstab(df[bundle_col], df['has_rag'])

# Ensure columns 0 and 1 exist
for c in [0, 1]:
    if c not in contingency.columns:
        contingency[c] = 0
contingency = contingency[[0, 1]]

print("\n=== Contingency Table (Count) ===")
print(contingency)

# Calculate percentages for reporting
contingency_pct = pd.crosstab(df[bundle_col], df['has_rag'], normalize='index') * 100
print("\n=== Contingency Table (Percentage) ===")
print(contingency_pct)

# Fisher's Exact Test
# We want to test if Integration Readiness is MORE likely to have RAG than Trust Readiness.
# Construct 2x2 matrix: [[Integration_Yes, Integration_No], [Trust_Yes, Trust_No]]

# Get counts
try:
    int_yes = contingency.loc['Integration Readiness', 1]
    int_no = contingency.loc['Integration Readiness', 0]
    trust_yes = contingency.loc['Trust Readiness', 1]
    trust_no = contingency.loc['Trust Readiness', 0]
except KeyError as e:
    print(f"Error accessing bundle keys: {e}")
    sys.exit(1)

# Table for stats: [[Yes, No]] for Group 1, then Group 2
# This aligns with Odds Ratio = (Yes1/No1) / (Yes2/No2)
stats_table = [
    [int_yes, int_no],      # Integration Readiness
    [trust_yes, trust_no]   # Trust Readiness
]

# Perform Fisher's Exact Test (Two-sided to be conservative, check OR for direction)
odds_ratio, p_value = stats.fisher_exact(stats_table, alternative='two-sided')

print("\n=== Fisher's Exact Test Results ===")
print(f"Comparison: Integration Readiness vs. Trust Readiness")
print(f"Integration RAG Rate: {int_yes}/{int_yes+int_no} ({int_yes/(int_yes+int_no)*100:.1f}%)")
print(f"Trust RAG Rate:       {trust_yes}/{trust_yes+trust_no} ({trust_yes/(trust_yes+trust_no)*100:.1f}%)")
print(f"Odds Ratio: {odds_ratio:.4f}")
print(f"P-value:    {p_value:.4f}")

interpretation = ""
if p_value < 0.05:
    interpretation = "Statistically Significant Difference."
else:
    interpretation = "No Statistically Significant Difference."

if odds_ratio > 1:
    direction = "Integration Readiness is more associated with RAG."
elif odds_ratio < 1:
    direction = "Trust Readiness is more associated with RAG."
else:
    direction = "No directional difference."

print(f"Conclusion: {interpretation} {direction}")

# Visualization
plt.figure(figsize=(8, 6))
# Plot percentage of 'Yes' (column 1)
ax = contingency_pct[1].plot(kind='bar', color=['skyblue', 'salmon'], edgecolor='black', rot=0)

plt.title('Prevalence of "RAG Architecture & Data Grounding" Control')
plt.xlabel('Competency Bundle')
plt.ylabel('Percentage of Requirements (%)')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add count labels
for i, p in enumerate(ax.patches):
    bundle_name = contingency_pct.index[i]
    count = contingency.loc[bundle_name, 1]
    total = contingency.loc[bundle_name].sum()
    height = p.get_height()
    ax.annotate(f'{height:.1f}%\n(n={count}/{total})',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting experiment...
Loading dataset from: step2_crosswalk_matrix.csv

=== Contingency Table (Count) ===
has_rag                 0  1
bundle                      
Integration Readiness  21  2
Trust Readiness        17  2

=== Contingency Table (Percentage) ===
has_rag                        0          1
bundle                                     
Integration Readiness  91.304348   8.695652
Trust Readiness        89.473684  10.526316

=== Fisher's Exact Test Results ===
Comparison: Integration Readiness vs. Trust Readiness
Integration RAG Rate: 2/23 (8.7%)
Trust RAG Rate:       2/19 (10.5%)
Odds Ratio: 0.8095
P-value:    1.0000
Conclusion: No Statistically Significant Difference. Trust Readiness is more associated with RAG.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot.
*   **Purpose:** The plot compares the prevalence (expressed as a percentage) of a specific control mechanism—"RAG Architecture & Data Grounding"—across two distinct categories termed "Competency Bundles."

### 2. Axes
*   **X-axis:**
    *   **Title:** "Competency Bundle"
    *   **Categories:** The axis displays two categorical variables: "Integration Readiness" and "Trust Readiness."
*   **Y-axis:**
    *   **Title:** "Percentage of Requirements (%)"
    *   **Value Range:** The scale ranges from **0 to 100**, representing percentage points.
    *   **Ticks:** Major ticks are marked at intervals of 20 (0, 20, 40, 60, 80, 100), with horizontal gridlines provided for readability.

### 3. Data Trends
*   **Tallest Bar:** The **"Trust Readiness"** bar (salmon/red color) is slightly taller, reaching approximately 10.5%.
*   **Shortest Bar:** The **"Integration Readiness"** bar (light blue color) is shorter, reaching approximately 8.7%.
*   **Overall Pattern:** Both values are quite low on the 0-100% scale, indicating that this specific control is a minor component of both competency bundles. The visual difference between the two bars is minimal.

### 4. Annotations and Legends
*   **Bar Annotations:** Each bar is annotated with specific statistical data located directly above it:
    *   **Integration Readiness:** "8.7% (n=2/23)"
    *   **Trust Readiness:** "10.5% (n=2/19)"
*   **Legend:** There is no separate legend box; however, the categories are clearly labeled on the x-axis. Distinct colors (blue and salmon) are used to distinguish the two categories visually.

### 5. Statistical Insights
*   **Low Occurrence Rate:** The control "RAG Architecture & Data Grounding" is relatively rare in both contexts. It appears in roughly **1 in 10** requirements for both categories.
*   **Identical Absolute Frequency:** Interestingly, the absolute count of occurrences is identical for both bundles (**n=2**).
*   **Denominator Effect:** The difference in percentage (10.5% vs. 8.7%) is driven entirely by the sample size (denominator). Since "Trust Readiness" has a smaller total pool of requirements (19) compared to "Integration Readiness" (23), the same number of occurrences (2) results in a slightly higher percentage for Trust Readiness.
*   **Conclusion:** There is no significant difference in the *number* of times this control is required between the two bundles; it is an infrequent requirement for both.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
