# Experiment 28: node_5_0

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_0` |
| **ID in Run** | 28 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T22:21:29.457299+00:00 |
| **Runtime** | 199.2s |
| **Parent** | `node_4_3` |
| **Children** | `node_6_0`, `node_6_4` |
| **Creation Index** | 29 |

---

## Hypothesis

> The mapping of governance requirements to architecture controls follows a Pareto
distribution, where the top 20% of architecture controls address at least 80% of
the total governance requirements.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.1901 (Likely False) |
| **Surprise** | -0.6405 |
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
| Maybe False | 0.0 |
| Definitely False | 90.0 |

---

## Experiment Plan

**Objective:** Verify if a small subset of 'keystone' architecture controls disproportionately satisfies governance obligations.

### Steps
- 1. Load 'step2_crosswalk_matrix.csv'.
- 2. Identify the 18 architecture control columns.
- 3. Calculate the column sum for each control (frequency of appearance).
- 4. Sort controls by frequency in descending order.
- 5. Calculate the cumulative percentage of total mappings covered by these controls.
- 6. Identify the cut-off for the top 20% of controls (approx 4 controls) and check if their cumulative coverage exceeds 80%.

### Deliverables
- Sorted frequency table with cumulative percentages, Lorenz curve visualization, and a statement confirming or rejecting the 80/20 hypothesis.

---

## Analysis

The experiment successfully tested the Pareto distribution hypothesis on the
mapping between governance requirements and architecture controls. The analysis
identified 18 unique architecture controls. According to the Pareto Principle
(80/20 rule), the top 20% of controls (approximately 4 controls) were expected
to cover 80% of the requirements. The results show that the top 4 controls—'AI
Risk Policy & Accountability Structures', 'Evaluation & Monitoring
Infrastructure', 'GenAIOps / MLOps Lifecycle Governance', and 'Nondeterminism
Controls & Output Validation'—cover only 54.55% of the total mappings (42
mappings out of 77). To reach the 80% coverage threshold, the top 10 controls
(approx. 55% of the total) are required. Consequently, the hypothesis is
rejected. The data suggests that governance compliance requires a broad
implementation of architectural capabilities rather than relying on a small
subset of 'keystone' controls.

---

## Review

The experiment faithfully implemented the Pareto analysis to test whether a
small subset of architecture controls ('keystone' controls) satisfies the
majority of governance requirements. The analysis correctly identified 18
architecture controls from the 'step2_crosswalk_matrix.csv' dataset and
calculated their mapping frequencies. The results demonstrated that the top 20%
of controls (4 out of 18) cover only 54.55% of the total governance mappings,
falling significantly short of the 80% threshold predicted by the Pareto
principle. To achieve 80% coverage, the data shows that the top 10 controls
(approx. 55%) are required. Consequently, the hypothesis is rejected, indicating
that AI governance compliance relies on a broad implementation of diverse
architectural capabilities rather than a narrow set of high-impact controls.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# Load the dataset
# Based on previous successful runs, the file is likely in the current directory.
file_path = 'step2_crosswalk_matrix.csv'

try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    # Fallback to checking parent directory if current fails, though unlikely based on history
    try:
        file_path = '../step2_crosswalk_matrix.csv'
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully from parent directory.")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

# Identify architecture control columns
# The first 6 columns are metadata
metadata_cols = ['req_id', 'source', 'function', 'requirement', 'bundle', 'competency_statement']
control_cols = [col for col in df.columns if col not in metadata_cols]

print(f"Identified {len(control_cols)} architecture controls.")

# Calculate frequency of mappings
# The matrix uses 'X' (or non-null) for presence.
# specific check for 'X' or non-null values
control_counts = df[control_cols].apply(lambda x: x.notna().sum())

# Create a DataFrame for analysis
pareto_df = pd.DataFrame({'Control': control_counts.index, 'Frequency': control_counts.values})

# Sort by Frequency descending
pareto_df = pareto_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)

# Calculate cumulative metrics
total_mappings = pareto_df['Frequency'].sum()
pareto_df['Cumulative_Frequency'] = pareto_df['Frequency'].cumsum()
pareto_df['Cumulative_Percentage'] = (pareto_df['Cumulative_Frequency'] / total_mappings) * 100

# Add Rank
pareto_df['Rank'] = pareto_df.index + 1

# Hypothesis Check: Top 20% of controls
num_controls = len(control_cols)
top_20_cutoff_index = int(np.ceil(num_controls * 0.2)) # 20% of 18 is 3.6 -> 4 controls

# Get coverage at the cutoff
coverage_at_cutoff = pareto_df.loc[top_20_cutoff_index - 1, 'Cumulative_Percentage']
top_controls_names = pareto_df.loc[:top_20_cutoff_index-1, 'Control'].tolist()

print("\n--- Pareto Analysis Table ---")
print(pareto_df[['Rank', 'Control', 'Frequency', 'Cumulative_Percentage']].to_string(index=False))

print(f"\nTotal Mappings: {total_mappings}")
print(f"Top 20% (approx {top_20_cutoff_index} controls): {coverage_at_cutoff:.2f}% coverage")
print(f"Top controls: {top_controls_names}")

hypothesis_result = "CONFIRMED" if coverage_at_cutoff >= 80 else "REJECTED"
print(f"\nHypothesis {hypothesis_result}: The top {top_20_cutoff_index} controls cover {coverage_at_cutoff:.1f}% of mappings (Target: 80%).")

# Visualization
fig, ax1 = plt.subplots(figsize=(14, 8))

# Bar plot for Frequency
color = 'tab:blue'
ax1.set_xlabel('Architecture Controls (Ranked)')
ax1.set_ylabel('Frequency (Mapping Count)', color=color)
bars = ax1.bar(pareto_df['Control'], pareto_df['Frequency'], color=color, alpha=0.7)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticklabels(pareto_df['Control'], rotation=45, ha='right', fontsize=9)

# Line plot for Cumulative Percentage
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Cumulative Percentage (%)', color=color)
ax2.plot(pareto_df['Control'], pareto_df['Cumulative_Percentage'], color=color, marker='o', linestyle='-', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 110)

# Reference lines
ax2.axhline(y=80, color='green', linestyle='--', linewidth=2, label='80% Target')
ax2.axvline(x=top_20_cutoff_index - 0.5, color='orange', linestyle='--', linewidth=2, label='Top 20% Cutoff')

# Adding text labels to the line points
for i, txt in enumerate(pareto_df['Cumulative_Percentage']):
    ax2.annotate(f"{txt:.1f}%", (i, txt), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.title(f'Pareto Analysis: Do the top 20% of controls cover 80% of requirements?\nResult: {hypothesis_result}')
fig.tight_layout()
plt.legend(loc='lower right')
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Dataset loaded successfully.
Identified 18 architecture controls.

--- Pareto Analysis Table ---
 Rank                                      Control  Frequency  Cumulative_Percentage
    1   AI Risk Policy & Accountability Structures         13              16.883117
    2       Evaluation & Monitoring Infrastructure         13              33.766234
    3        GenAIOps / MLOps Lifecycle Governance         10              46.753247
    4  Nondeterminism Controls & Output Validation          6              54.545455
    5            Data Governance & Access Controls          5              61.038961
    6          Regulatory Compliance Documentation          4              66.233766
    7            RAG Architecture & Data Grounding          4              71.428571
    8                Threat Modeling & Red-Teaming          3              75.324675
    9          Supply Chain & Vendor Risk Controls          3              79.220779
   10          Prompt Management & Secret Handling          3              83.116883
   11             Human-in-the-Loop Approval Gates          3              87.012987
   12       Incident Response & Recovery Playbooks          2              89.610390
   13          Human Override & Control Mechanisms          2              92.207792
   14 Tool-Use Boundaries & Least-Privilege Access          2              94.805195
   15            Multi-Agent Orchestration Pattern          1              96.103896
   16           Single-Agent Orchestration Pattern          1              97.402597
   17   Scalable Modular Architecture (Archetypes)          1              98.701299
   18                    Audit Logging & Telemetry          1             100.000000

Total Mappings: 77
Top 20% (approx 4 controls): 54.55% coverage
Top controls: ['AI Risk Policy & Accountability Structures', 'Evaluation & Monitoring Infrastructure', 'GenAIOps / MLOps Lifecycle Governance', 'Nondeterminism Controls & Output Validation']

Hypothesis REJECTED: The top 4 controls cover 54.5% of mappings (Target: 80%).

STDERR:
<ipython-input-1-2cbfbe616e65>:79: UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
  ax1.set_xticklabels(pareto_df['Control'], rotation=45, ha='right', fontsize=9)


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** **Pareto Chart** (a combination of a bar chart and a line graph).
*   **Purpose:** This chart is used to analyze the frequency of specific "Architecture Controls" and their cumulative contribution to requirements. It specifically tests the **Pareto Principle (80/20 rule)** to determine if a small subset (top 20%) of controls accounts for the majority (80%) of the mapped requirements.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Architecture Controls (Ranked)"
    *   **Content:** Categorical variables representing specific control types (e.g., "AI Risk Policy & Accountability Structures," "Evaluation & Monitoring Infrastructure"). The categories are sorted in descending order of frequency.
*   **Primary Y-Axis (Left):**
    *   **Label:** "Frequency (Mapping Count)"
    *   **Range:** 0 to roughly 13 (ticks visible every 2 units).
    *   **Represented by:** Blue bars.
*   **Secondary Y-Axis (Right):**
    *   **Label:** "Cumulative Percentage (%)"
    *   **Range:** 0% to roughly 110% (ticks visible every 20%).
    *   **Represented by:** Red line with markers.

### 3. Data Trends
*   **Bar Plot (Frequency):**
    *   **Tallest Bars:** The top two controls, "AI Risk Policy & Accountability Structures" and "Evaluation & Monitoring Infrastructure," share the highest frequency with a count of roughly **13 mappings** each.
    *   **Shortest Bars:** The final three controls ("Multi-Agent Orchestration Pattern," "Single-Agent Orchestration Pattern," etc.) have the lowest frequency, with a count of **1 mapping** each.
    *   **Pattern:** There is a gradual decline in frequency. While the first three bars are notably higher, the drop-off is not precipitous; the middle section of the graph (bars 4–10) maintains a moderate frequency (ranging from 6 down to 3).
*   **Line Plot (Cumulative Percentage):**
    *   The curve starts at **16.9%** and rises steadily.
    *   It crosses the **50%** mark at the 4th control.
    *   It crosses the **80%** target threshold at the 10th control ("Threat Modeling & Red-Teaming").

### 4. Annotations and Legends
*   **Title/Header:** The chart answers a specific hypothesis: "Pareto Analysis: Do the top 20% of controls cover 80% of requirements?" with the explicit conclusion **"Result: REJECTED"**.
*   **Green Dashed Line ("80% Target"):** A horizontal reference line indicating the 80% cumulative threshold, which is the standard target in Pareto analysis.
*   **Orange Dashed Line ("Top 20% Cutoff"):** A vertical reference line indicating where the top 20% of the total number of control categories ends (occurring after the 4th bar).
*   **Data Labels:**
    *   Red percentage labels above the line markers indicate the exact cumulative percentage at each step (e.g., 16.9%, 33.8%, 46.8%, ... 100.0%).
    *   The final point confirms **100.0%** coverage at the last item.

### 5. Statistical Insights
*   **Hypothesis Rejection:** The analysis explicitly rejects the 80/20 rule for this dataset.
*   **The "Top 20%" Performance:**
    *   There are 18 total control categories. The top 20% constitutes roughly the first 3.6 (rounded to 4) controls.
    *   At the **Top 20% Cutoff** (the orange line), the cumulative percentage is only **54.5%** (indicated by the data point above the 4th bar).
    *   This falls significantly short of the **80% target**.
*   **Effort Distribution:** To achieve 80% coverage of requirements (crossing the green line), one must implement the top **10 controls** (up to "Threat Modeling & Red-Teaming"). This represents roughly **55%** of the available control types, rather than 20%.
*   **Conclusion:** The requirements are more evenly distributed across various controls than a standard Pareto distribution would suggest. Relying solely on the top few controls would leave nearly half of the requirements unaddressed.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
