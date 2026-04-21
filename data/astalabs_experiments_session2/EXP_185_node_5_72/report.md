# Experiment 185: node_5_72

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_72` |
| **ID in Run** | 185 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T09:47:36.793644+00:00 |
| **Runtime** | 291.4s |
| **Parent** | `node_4_41` |
| **Children** | `node_6_80` |
| **Creation Index** | 186 |

---

## Hypothesis

> Public Service Transparency Gap: AI systems designated as 'Public Services' are
significantly more likely to implement 'AI Notice' mechanisms compared to
internal administrative systems.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.5824 (Maybe True) |
| **Surprise** | -0.1914 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
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
| Uncertain | 60.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Assess if public-facing systems are adhering to higher transparency standards.

### Steps
- 1. Load `eo13960_scored` data.
- 2. Define two groups based on `26_public_service` (Yes vs No).
- 3. Define the outcome based on `59_ai_notice` (Yes vs No).
- 4. Perform a Chi-Square test.
- 5. Calculate the percentage of notice implementation for both groups.

### Deliverables
- Bar chart comparing AI Notice rates for Public vs Internal systems; Chi-Square results.

---

## Analysis

The experiment successfully executed the corrected code, resolving previous type
errors by implementing robust string normalization for the 'Public Service' and
'AI Notice' columns. The analysis identified 36 relevant use cases (35 Public, 1
Internal) out of the dataset. Public-facing AI systems demonstrated a 48.6%
compliance rate for AI Notice (17/35), while the single Internal system
identified had no notice (0/1). Due to the extreme lack of data in the control
group (n=1) resulting from the filtering logic, the Chi-Square test yielded a
p-value of 1.0, failing to reject the null hypothesis statistically despite the
large observed percentage gap. The hypothesis is visually suggested but not
statistically supported due to insufficient sample size.

---

## Review

The experiment was successfully executed. The code corrected the previous data
type errors by implementing robust string normalization and classification logic
for the 'Public Service' and 'AI Notice' columns.

**Hypothesis Verification:**
The hypothesis that **Public Service AI systems are significantly more likely to
implement AI Notice mechanisms than internal systems** was **not statistically
supported**, primarily due to severe data imbalance.

*   **Results:** The analysis identified 35 'Public Service' systems and only 1
'Internal/Other' system.
*   **Adoption Rates:** Public Service systems showed a **48.6%** (17/35)
adoption rate of AI Notices. The single Internal system identified had **0%**
(0/1) adoption.
*   **Significance:** A Chi-Square test yielded a p-value of **1.00**, failing
to reject the null hypothesis.

**Finding:** While the visualization shows a large absolute gap (48.6% vs 0%),
the extremely small sample size for the control group (Internal systems)
prevents any statistical validity. The dataset appears to be heavily skewed
towards describing public-facing use cases, or 'Internal' use cases are not
explicitly labeled as 'No' in the `26_public_service` column but rather
described with text that the current logic classified as 'Public'.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Define file path
file_path = "../astalabs_discovery_all_data.csv"
if not os.path.exists(file_path):
    file_path = "astalabs_discovery_all_data.csv"

print(f"Loading dataset from {file_path}...")

try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit(1)

# Filter for EO 13960 Scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored subset shape: {eo_data.shape}")

# Columns
col_public = '26_public_service'
col_notice = '59_ai_notice'

# Robust normalization function
def normalize_text(val):
    if pd.isna(val):
        return ""
    return str(val).lower().strip()

eo_data[col_public] = eo_data[col_public].apply(normalize_text)
eo_data[col_notice] = eo_data[col_notice].apply(normalize_text)

# --- logic to classify Public Service ---
def classify_public(val):
    if val == "" or val == 'nan':
        return None
    if val == 'no':
        return 'Internal/Other'
    # Any other non-empty string implies a description of a public service
    return 'Public Service'

# --- logic to classify AI Notice ---
def classify_notice(val):
    if val == "" or val == 'nan':
        return None
    
    negatives = [
        'none of the above',
        'n/a - individuals are not interacting with the ai for this use case',
        'ai is not safety or rights-impacting.',
        'agency caio has waived this minimum practice and reported such waiver to omb.'
    ]
    
    if val in negatives:
        return 'No Notice'
    
    # Check for keywords indicating presence of notice
    positives_keywords = ['online', 'in-person', 'email', 'telephone', 'other', 'terms']
    if any(keyword in val for keyword in positives_keywords):
        return 'Has Notice'
    
    # Fallback for anything else that isn't explicitly negative
    return 'No Notice'

# Apply classification
eo_data['service_type'] = eo_data[col_public].apply(classify_public)
eo_data['notice_status'] = eo_data[col_notice].apply(classify_notice)

# Filter out rows where we couldn't classify one or the other
analysis_df = eo_data.dropna(subset=['service_type', 'notice_status']).copy()

print(f"Data shape after classification and filtering: {analysis_df.shape}")
print("Service Type Counts:\n", analysis_df['service_type'].value_counts())
print("Notice Status Counts:\n", analysis_df['notice_status'].value_counts())

# Create contingency table
contingency_table = pd.crosstab(analysis_df['service_type'], analysis_df['notice_status'])
print("\nContingency Table:")
print(contingency_table)

# Calculate percentages
public_rate = 0.0
internal_rate = 0.0
public_n = 0
internal_n = 0
public_has_notice = 0
internal_has_notice = 0

if 'Public Service' in contingency_table.index:
    public_n = contingency_table.loc['Public Service'].sum()
    if 'Has Notice' in contingency_table.columns:
        public_has_notice = contingency_table.loc['Public Service', 'Has Notice']
        if public_n > 0:
            public_rate = (public_has_notice / public_n) * 100

if 'Internal/Other' in contingency_table.index:
    internal_n = contingency_table.loc['Internal/Other'].sum()
    if 'Has Notice' in contingency_table.columns:
        internal_has_notice = contingency_table.loc['Internal/Other', 'Has Notice']
        if internal_n > 0:
            internal_rate = (internal_has_notice / internal_n) * 100

print(f"\nPublic Service Notice Rate: {public_rate:.2f}% ({public_has_notice}/{public_n})")
print(f"Internal/Other Notice Rate: {internal_rate:.2f}% ({internal_has_notice}/{internal_n})")

# Perform Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Test Results:")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

alpha = 0.05
if p < alpha:
    print("Conclusion: REJECT Null Hypothesis. There is a significant difference in AI Notice rates.")
else:
    print("Conclusion: FAIL TO REJECT Null Hypothesis. No significant difference detected.")

# Visualization
plt.figure(figsize=(8, 6))
labels = ['Public Service', 'Internal/Other']
rates = [public_rate, internal_rate]
colors = ['#1f77b4', '#7f7f7f']

bars = plt.bar(labels, rates, color=colors)
plt.ylabel('Percentage with AI Notice (%)')
plt.title('Transparency Gap: AI Notice Implementation\nPublic Service vs. Internal Systems')
plt.ylim(0, max(rates) * 1.2 if max(rates) > 0 else 10)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, 
             f'{height:.1f}%',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
EO 13960 Scored subset shape: (1757, 196)
Data shape after classification and filtering: (36, 198)
Service Type Counts:
 service_type
Public Service    35
Internal/Other     1
Name: count, dtype: int64
Notice Status Counts:
 notice_status
No Notice     19
Has Notice    17
Name: count, dtype: int64

Contingency Table:
notice_status   Has Notice  No Notice
service_type                         
Internal/Other           0          1
Public Service          17         18

Public Service Notice Rate: 48.57% (17/35)
Internal/Other Notice Rate: 0.00% (0/1)

Chi-Square Test Results:
Chi2 Statistic: 0.0000
P-value: 1.0000e+00
Conclusion: FAIL TO REJECT Null Hypothesis. No significant difference detected.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Chart.
*   **Purpose:** The chart is designed to compare the prevalence (percentage) of "AI Notice Implementation" between two distinct categories of systems: Public Service and Internal/Other.

### 2. Axes
*   **X-Axis:**
    *   **Label/Title:** The axis represents categories of AI implementation. The specific labels are "Public Service" and "Internal/Other".
*   **Y-Axis:**
    *   **Label/Title:** "Percentage with AI Notice (%)".
    *   **Value Range:** The axis is marked with intervals of 10, ranging from **0 to 50**.

### 3. Data Trends
*   **Tallest Bar:** The "Public Service" category is the dominant bar, reaching a height representing **48.6%**.
*   **Shortest Bar:** The "Internal/Other" category is non-existent visually, sitting flat on the x-axis with a value of **0.0%**.
*   **Pattern:** The chart displays a binary trend where nearly half of the public-facing services have AI notices, while internal systems in this dataset show a complete absence of such notices.

### 4. Annotations and Legends
*   **Title:** The main title is "Transparency Gap: AI Notice Implementation Public Service vs. Internal Systems," which sets the context for the comparison.
*   **Data Labels:** There are precise numerical annotations placed directly above the bars to indicate exact values:
    *   "48.6%" above the Public Service bar.
    *   "0.0%" above the Internal/Other bar position.
*   **Legend:** There is no separate legend box; the categories are identified directly by the x-axis labels.

### 5. Statistical Insights
*   **The "Transparency Gap":** The plot visualizes the concept mentioned in the title. There is a massive disparity (a gap of 48.6 percentage points) between how AI is disclosed to the public versus how it is disclosed internally.
*   **External vs. Internal Policy:** The data suggests that while organizations (or the entities surveyed) are making an effort to inform the public about AI usage (approaching 50% adoption), there is zero transparency regarding AI usage within internal systems or "other" categories.
*   **Internal Opacity:** The 0.0% statistic is significant. It implies that in the sampled data, not a single internal system included a notice regarding the use of Artificial Intelligence, indicating total opacity in that specific sector.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
