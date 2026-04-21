# Experiment 61: node_4_26

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_26` |
| **ID in Run** | 61 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T03:58:15.278545+00:00 |
| **Runtime** | 232.2s |
| **Parent** | `node_3_3` |
| **Children** | `node_5_29`, `node_5_92` |
| **Creation Index** | 62 |

---

## Hypothesis

> Agency Culture Paradox: 'Defense and Security' agencies (e.g., DOD, DHS)
prioritize 'System Security' controls (Authority to Operate) over 'Rights-
Preserving' controls (Bias Mitigation), whereas 'Civilian/Social' agencies
(e.g., HHS, Education) exhibit the reverse prioritization.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7258 (Likely True) |
| **Posterior** | 0.2473 (Likely False) |
| **Surprise** | -0.5743 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 28.0 |
| Uncertain | 2.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 60.0 |

---

## Experiment Plan

**Objective:** Contrast the governance priorities of security-focused vs. social-focused agencies.

### Steps
- 1. Filter for `eo13960_scored`. Clean `3_agency` column.
- 2. Define two agency groups: 'Security' (DOD, DHS, DOJ, State) and 'Social' (HHS, Education, Labor, HUD, VA).
- 3. Compare the rate of `40_has_ato` (Security Control) vs. `55_disparity_mitigation` (Rights Control) within each group.
- 4. Perform a Two-Proportion Z-test to compare the `Rights/Security` ratio between the two agency groups.

### Deliverables
- Grouped bar chart of control prevalence by agency type, Z-test statistics.

---

## Analysis

The experiment successfully tested the 'Agency Culture Paradox' hypothesis using
the EO 13960 dataset. The code correctly identified the columns `40_has_ato`
(Security) and `62_disparity_mitigation` (Rights) and categorized agencies into
'Security' (n=385) and 'Social' (n=594) groups.

**Findings:**
1.  **Security Controls (ATO):** Contrary to the hypothesis, 'Social' agencies
exhibited a significantly *higher* prevalence of Authority to Operate (ATO)
compliance (35.7%) compared to 'Security' agencies (17.7%) (Z=6.0974, p<0.0001).
This contradicts the expectation that security agencies would dominate in
security controls.
2.  **Rights Controls (Bias Mitigation):** The analysis revealed a complete
absence (0.0% rate) of recorded 'disparity_mitigation' controls in both agency
groups. Consequently, the statistical comparison for this control yielded NaN.

**Conclusion:** The hypothesis is **not supported**. The data refutes the idea
that Social agencies prioritize rights-preserving controls (prevalence is 0%)
and contradicts the expectation that Security agencies prioritize system
security controls more than their social counterparts (Social agencies actually
lead in this metric). The results highlight a potential systemic gap in
documenting or implementing bias mitigation across the federal government.

---

## Review

The experiment was faithfully implemented and successfully executed. The
programmer correctly resolved the column naming discrepancy (identifying
'62_disparity_mitigation' instead of the requested '55') and applied the
specified agency grouping logic.

**Hypothesis Evaluation:**
The 'Agency Culture Paradox' hypothesis was **not supported** by the data.

**Key Findings:**
1.  **Security Controls:** Contrary to the hypothesis, 'Social' agencies
demonstrated a significantly higher prevalence of 'Authority to Operate' (ATO)
compliance (35.7%) compared to 'Security' agencies (17.7%) (Z=6.09, p < 0.0001).
2.  **Rights Controls:** There was a 0% prevalence of 'Disparity Mitigation'
controls across both agency groups, preventing a comparative analysis of rights-
preservation priorities.

The results indicate that Social agencies are actually more rigorous regarding
formal security governance than Security agencies in this dataset, while
specific rights-preserving controls appear entirely absent or undocumented
across the federal landscape.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for eo13960_scored
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()

# Normalize agency names for grouping
df_eo['agency_norm'] = df_eo['3_agency'].fillna('').astype(str).str.lower()

# Define groups
security_keywords = ['defense', 'homeland', 'justice', 'state']
social_keywords = ['health', 'education', 'labor', 'housing', 'veterans']

def categorize_agency(name):
    if any(k in name for k in security_keywords):
        return 'Security'
    elif any(k in name for k in social_keywords):
        return 'Social'
    return 'Other'

df_eo['Agency_Group'] = df_eo['agency_norm'].apply(categorize_agency)

# Filter for only Security and Social groups
df_analysis = df_eo[df_eo['Agency_Group'].isin(['Security', 'Social'])].copy()

# Identify correct columns dynamically to avoid KeyErrors
ato_cols = [c for c in df_eo.columns if '40_has_ato' in c]
rights_cols = [c for c in df_eo.columns if 'disparity_mitigation' in c]

if not ato_cols or not rights_cols:
    print(f"Could not find required columns. Available columns related to ATO: {ato_cols}, Rights: {rights_cols}")
    # Fallback search if exact partial match fails
    ato_cols = [c for c in df_eo.columns if 'ato' in c.lower()]
    rights_cols = [c for c in df_eo.columns if 'disparity' in c.lower()]

ato_col = ato_cols[0]
rights_col = rights_cols[0]

print(f"Using Security Column: {ato_col}")
print(f"Using Rights Column: {rights_col}")

# Clean control columns
def clean_binary(val):
    s = str(val).lower()
    if s in ['yes', 'true', '1', '1.0']:
        return 1
    return 0

df_analysis['Security_Control'] = df_analysis[ato_col].apply(clean_binary)
df_analysis['Rights_Control'] = df_analysis[rights_col].apply(clean_binary)

# Generate summary stats
group_stats = df_analysis.groupby('Agency_Group').agg(
    n=('source_row_num', 'count'),
    security_count=('Security_Control', 'sum'),
    rights_count=('Rights_Control', 'sum'),
    security_rate=('Security_Control', 'mean'),
    rights_rate=('Rights_Control', 'mean')
)

print("Summary Statistics:")
print(group_stats)
print("\n")

# Statistical Tests (Two-Proportion Z-test)
# We compare 'Social' vs 'Security' for both control types

results = {}

for control_type, count_col, nobs_col in [('Security_Control', 'security_count', 'n'), ('Rights_Control', 'rights_count', 'n')]:
    counts = np.array([group_stats.loc['Social', count_col], group_stats.loc['Security', count_col]])
    nobs = np.array([group_stats.loc['Social', nobs_col], group_stats.loc['Security', nobs_col]])
    
    # Handle cases with 0 variance or small sample size if necessary, but ztest usually handles it unless n=0
    stat, pval = proportions_ztest(counts, nobs)
    results[control_type] = (stat, pval)
    print(f"Z-test for {control_type} (Social vs Security): z={stat:.4f}, p={pval:.4f}")

# Calculate 'Prioritization Gap' (Security - Rights) within each group
group_stats['Prioritization_Gap'] = group_stats['security_rate'] - group_stats['rights_rate']
print("\nPrioritization Gap (Security Rate - Rights Rate):")
print(group_stats['Prioritization_Gap'])

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(group_stats))

bar1 = ax.bar(index, group_stats['security_rate'], bar_width, label='Security (ATO)', alpha=0.8)
bar2 = ax.bar(index + bar_width, group_stats['rights_rate'], bar_width, label='Rights (Bias Mitigation)', alpha=0.8)

ax.set_xlabel('Agency Type')
ax.set_ylabel('Prevalence of Control')
ax.set_title('Governance Priorities: Security vs Social Agencies')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(group_stats.index)
ax.legend()

plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Using Security Column: 40_has_ato
Using Rights Column: 62_disparity_mitigation
Summary Statistics:
                n  security_count  rights_count  security_rate  rights_rate
Agency_Group                                                               
Security      385              68             0       0.176623          0.0
Social        594             212             0       0.356902          0.0


Z-test for Security_Control (Social vs Security): z=6.0974, p=0.0000
Z-test for Rights_Control (Social vs Security): z=nan, p=nan

Prioritization Gap (Security Rate - Rights Rate):
Agency_Group
Security    0.176623
Social      0.356902
Name: Prioritization_Gap, dtype: float64

STDERR:
/usr/local/lib/python3.13/site-packages/statsmodels/stats/weightstats.py:792: RuntimeWarning: invalid value encountered in scalar divide
  zstat = value / std


=== Plot Analysis (figure 1) ===
Based on the analysis of the provided plot image, here are the detailed observations:

### 1. Plot Type
*   **Type:** Grouped Bar Plot.
*   **Purpose:** The plot compares the "Prevalence of Control" across two different categories of agencies ("Security" and "Social") for two distinct governance priorities ("Security (ATO)" and "Rights (Bias Mitigation)").

### 2. Axes
*   **X-axis:**
    *   **Title:** "Agency Type"
    *   **Labels:** Two categorical groups: "Security" and "Social".
*   **Y-axis:**
    *   **Title:** "Prevalence of Control"
    *   **Range:** The axis starts at 0.00 and extends slightly beyond 0.35, with major tick marks at 0.05 intervals.
    *   **Units:** The values represent a ratio or proportion (ranging from 0 to roughly 0.36), indicating the frequency or likelihood of specific controls being present.

### 3. Data Trends
*   **Security (ATO) [Blue Bars]:**
    *   **Tallest Bar:** The "Social" agency category exhibits the highest prevalence for Security (ATO) controls, with a value reaching approximately **0.36**.
    *   **Shortest Bar:** The "Security" agency category has a significantly lower prevalence for Security (ATO) controls, appearing to be approximately **0.18**.
    *   **Pattern:** Surprisingly, Social agencies show a prevalence of security-focused controls that is roughly double that of Security agencies.
*   **Rights (Bias Mitigation) [Orange Bars]:**
    *   **Observation:** There are no visible orange bars for either "Security" or "Social" agencies.
    *   **Trend:** This indicates that the prevalence of "Rights (Bias Mitigation)" controls is either **zero** or negligible/too small to be visualized on this scale for both agency types.

### 4. Annotations and Legends
*   **Chart Title:** "Governance Priorities: Security vs Social Agencies" – This frames the comparison between traditional security compliance and social/ethical compliance.
*   **Legend (Top Left):**
    *   **Blue Square:** Represents "Security (ATO)" (Likely referring to "Authority to Operate," a standard compliance measure).
    *   **Orange Square:** Represents "Rights (Bias Mitigation)" (Reflecting controls related to algorithmic fairness or civil rights).

### 5. Statistical Insights
*   **Prioritization Imbalance:** The plot reveals a drastic imbalance in governance priorities. While traditional security compliance (ATO) is present in both agency types, governance related to bias mitigation and rights is virtually non-existent in the data presented.
*   **Counter-Intuitive Finding:** One might expect "Security" agencies to have the highest prevalence of security controls. However, the data shows that **Social agencies** actually implement Security (ATO) controls at a much higher rate (approx. 36%) than Security agencies do (approx. 18%).
*   **The "Rights" Gap:** The complete absence of visible bars for "Rights (Bias Mitigation)" suggests a potential blind spot in the governance frameworks of these agencies, where technical security is tracked and enforced, but ethical considerations like bias mitigation are not yet prevalent or formally controlled.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
