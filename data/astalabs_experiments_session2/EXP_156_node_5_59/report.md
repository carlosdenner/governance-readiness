# Experiment 156: node_5_59

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_59` |
| **ID in Run** | 156 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T08:24:11.425906+00:00 |
| **Runtime** | 186.3s |
| **Parent** | `node_4_22` |
| **Children** | None |
| **Creation Index** | 157 |

---

## Hypothesis

> Malice is Intangible: Adversarial AI attacks (from ATLAS) are significantly less
likely to result in 'Physical' harm compared to general AI failures (from AIID),
as adversarial tactics primarily target model integrity or data confidentiality.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.8065 (Likely True) |
| **Posterior** | 0.9341 (Definitely True) |
| **Surprise** | +0.1531 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 8.0 |
| Maybe True | 22.0 |
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

**Objective:** Compare the nature of consequences between adversarial attacks and general accidents.

### Steps
- 1. Load `astalabs_discovery_all_data.csv`.
- 2. Create a combined dataframe containing `atlas_cases` (Adversarial) and `aiid_incidents` (General).
- 3. Create a binary 'Physical Harm' indicator by searching text fields (`summary` for ATLAS, `description` for AIID) for physical safety keywords (death, injury, crash, kill).
- 4. Compare the rate of Physical Harm between the 'Adversarial' and 'General' groups.
- 5. Perform a Fisher's Exact Test (due to the small sample size of ATLAS).

### Deliverables
- Bar chart comparing Physical Harm rates across sources; Statistical test results.

---

## Analysis

The experiment successfully tested the 'Malice is Intangible' hypothesis by
comparing physical harm rates between adversarial AI attacks (ATLAS dataset) and
general AI failures (AIID dataset). Using text mining to identify physical
safety keywords (e.g., 'death', 'injury', 'crash') in incident descriptions, the
analysis found that **Adversarial attacks are significantly less likely to
result in physical harm** compared to general AI accidents.

Specifically, only **3.8%** (2/52) of adversarial cases involved physical harm,
whereas **14.3%** (195/1362) of general AI incidents did. A Fisher's Exact Test
yielded a **p-value of 0.0166**, which is statistically significant at the p <
0.05 level. This supports the hypothesis that malicious actors currently target
digital assets (confidentiality, integrity) rather than physical safety, whereas
accidental AI failures pose a higher relative risk of physical consequences.

---

## Review

The experiment was successfully executed and the hypothesis 'Malice is
Intangible' was supported by the data. The analysis compared 52 adversarial
cases (ATLAS) against 1,362 general AI incidents (AIID) using keyword-based text
mining to identify physical harm outcomes (e.g., death, injury, crash). The
results showed a statistically significant difference (Fisher's Exact Test
p=0.017), with general AI failures being nearly four times more likely to result
in physical harm (14.3%) compared to adversarial attacks (3.8%). This confirms
that within the current landscape of reported AI issues, malicious actors are
primarily targeting digital assets (integrity/confidentiality) rather than
causing physical destruction, whereas accidental failures pose a higher relative
risk to physical safety.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, chi2_contingency

# [debug]
print("Starting experiment: Malice is Intangible")

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    print(f"Dataset loaded. Shape: {df.shape}")
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
        print(f"Dataset loaded (local). Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: Dataset not found.")
        exit(1)

# Filter subsets
df_atlas = df[df['source_table'] == 'atlas_cases'].copy()
df_aiid = df[df['source_table'] == 'aiid_incidents'].copy()

print(f"ATLAS cases: {len(df_atlas)}")
print(f"AIID incidents: {len(df_aiid)}")

# identify text columns for mining
# ATLAS usually has 'summary' or 'description'
# AIID usually has 'description' or 'summary'

# Check available columns for ATLAS
atlas_text_cols = [c for c in df_atlas.columns if df_atlas[c].notna().any()]
# Prefer 'summary', then 'description', then 'name'
atlas_col = 'summary' if 'summary' in atlas_text_cols else 'description' if 'description' in atlas_text_cols else 'name'

# Check available columns for AIID
aiid_text_cols = [c for c in df_aiid.columns if df_aiid[c].notna().any()]
aiid_col = 'description' if 'description' in aiid_text_cols else 'summary' if 'summary' in aiid_text_cols else 'title'

print(f"Using column '{atlas_col}' for ATLAS text mining.")
print(f"Using column '{aiid_col}' for AIID text mining.")

# Keywords for Physical Harm
physical_keywords = [
    'death', 'dead', 'die', 'kill', 'fatality', 'fatal',
    'injury', 'injure', 'hurt', 'wound',
    'crash', 'collision', 'accident',
    'physical harm', 'bodily', 'safety',
    'destroy', 'explosion', 'fire', 'burn'
]

def check_physical(text):
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    for kw in physical_keywords:
        # Simple substring match, could be improved with regex boundary but sufficient for broad classification
        if kw in text_lower:
            return True
    return False

# Apply classification
df_atlas['is_physical'] = df_atlas[atlas_col].apply(check_physical)
df_aiid['is_physical'] = df_aiid[aiid_col].apply(check_physical)

# Aggregation
atlas_physical_count = df_atlas['is_physical'].sum()
atlas_total = len(df_atlas)
atlas_rate = atlas_physical_count / atlas_total if atlas_total > 0 else 0

aiid_physical_count = df_aiid['is_physical'].sum()
aiid_total = len(df_aiid)
aiid_rate = aiid_physical_count / aiid_total if aiid_total > 0 else 0

print(f"\nATLAS (Adversarial): {atlas_physical_count} / {atlas_total} ({atlas_rate:.1%}) physical harm incidents.")
print(f"AIID (General): {aiid_physical_count} / {aiid_total} ({aiid_rate:.1%}) physical harm incidents.")

# Statistical Test
# Contingency Table:
#              Physical | Non-Physical
# Adversarial (ATLAS) |       a  |      b
# General (AIID)      |       c  |      d

a = atlas_physical_count
b = atlas_total - atlas_physical_count
c = aiid_physical_count
d = aiid_total - aiid_physical_count

contingency_table = [[a, b], [c, d]]
print(f"\nContingency Table:\n{contingency_table}")

# Fisher's Exact Test is appropriate for small sample sizes (ATLAS has ~52)
odds_ratio, p_value = fisher_exact(contingency_table, alternative='less') 
# alternative='less' tests if ATLAS is LESS likely to have physical harm than AIID

print(f"Fisher's Exact Test p-value: {p_value:.4f}")
print(f"Odds Ratio: {odds_ratio:.4f}")

if p_value < 0.05:
    print("Result: Statistically Significant. Adversarial attacks are less likely to cause physical harm.")
else:
    print("Result: Not Statistically Significant.")

# Visualization
labels = ['Adversarial (ATLAS)', 'General (AIID)']
physical_rates = [atlas_rate * 100, aiid_rate * 100]
non_physical_rates = [100 - x for x in physical_rates]

fig, ax = plt.subplots(figsize=(8, 6))

width = 0.5
x = np.arange(len(labels))

# Stacked bar chart
p1 = ax.bar(x, physical_rates, width, label='Physical Harm', color='#d62728', alpha=0.8)
p2 = ax.bar(x, non_physical_rates, width, bottom=physical_rates, label='Non-Physical / Other', color='#1f77b4', alpha=0.8)

ax.set_ylabel('Percentage of Incidents')
ax.set_title(f'Physical Harm Rate: Adversarial vs General AI Failures\n(p={p_value:.3f})')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add percentage labels
for i, rect in enumerate(p1):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2., height / 2.,
            f'{height:.1f}%',
            ha='center', va='center', color='white', fontweight='bold')

for i, rect in enumerate(p2):
    height = rect.get_height()
    y_pos = physical_rates[i] + height / 2.
    ax.text(rect.get_x() + rect.get_width() / 2., y_pos,
            f'{height:.1f}%',
            ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting experiment: Malice is Intangible
Dataset loaded (local). Shape: (6705, 196)
ATLAS cases: 52
AIID incidents: 1362
Using column 'summary' for ATLAS text mining.
Using column 'description' for AIID text mining.

ATLAS (Adversarial): 2 / 52 (3.8%) physical harm incidents.
AIID (General): 195 / 1362 (14.3%) physical harm incidents.

Contingency Table:
[[np.int64(2), np.int64(50)], [np.int64(195), np.int64(1167)]]
Fisher's Exact Test p-value: 0.0166
Odds Ratio: 0.2394
Result: Statistically Significant. Adversarial attacks are less likely to cause physical harm.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** 100% Stacked Bar Chart.
*   **Purpose:** The plot compares the distribution of two mutually exclusive outcomes ("Physical Harm" vs. "Non-Physical / Other") across two different categories of AI failures ("Adversarial" vs. "General"). By normalizing the bars to 100%, it facilitates a direct comparison of proportions rather than raw counts.

### 2. Axes
*   **X-Axis:**
    *   **Title/Labels:** Categorical labels representing the data sources/types: "Adversarial (ATLAS)" and "General (AIID)".
*   **Y-Axis:**
    *   **Title:** "Percentage of Incidents".
    *   **Range:** 0 to 100 (representing 0% to 100%).
    *   **Units:** Percentage (%).

### 3. Data Trends
*   **Adversarial (ATLAS) Category:**
    *   This category shows a very low incidence of physical harm.
    *   **Physical Harm (Red):** 3.8%
    *   **Non-Physical / Other (Blue):** 96.2%
*   **General (AIID) Category:**
    *   This category shows a noticeably higher incidence of physical harm compared to the adversarial group.
    *   **Physical Harm (Red):** 14.3%
    *   **Non-Physical / Other (Blue):** 85.7%
*   **Comparison:** The proportion of physical harm in "General" AI failures is nearly four times greater than in "Adversarial" failures (14.3% vs. 3.8%).

### 4. Annotations and Legends
*   **Title:** "Physical Harm Rate: Adversarial vs General AI Failures".
*   **Statistical Annotation:** Included in the title is "(p=0.017)", indicating the statistical significance of the difference between the two groups.
*   **Bar Annotations:** White text overlays on the bars clearly state the exact percentages for each segment (3.8%, 96.2%, 14.3%, 85.7%).
*   **Legend:** Located at the bottom center-right, identifying the color coding:
    *   **Red:** Physical Harm
    *   **Blue:** Non-Physical / Other

### 5. Statistical Insights
*   **Significant Difference:** The p-value of **0.017** is less than the standard significance threshold of 0.05. This suggests that the difference in physical harm rates between Adversarial and General AI failures is statistically significant and unlikely to be a result of random chance.
*   **Nature of Failures:** The data indicates that General AI failures (represented by the AIID dataset) pose a significantly higher risk of physical harm compared to Adversarial attacks (represented by the ATLAS dataset). Adversarial incidents appear to be overwhelmingly non-physical (likely involving data theft, model evasion, or digital manipulation rather than physical impact).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
