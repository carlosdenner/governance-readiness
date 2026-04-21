# Experiment 144: node_5_49

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_49` |
| **ID in Run** | 144 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T07:48:42.820997+00:00 |
| **Runtime** | 178.9s |
| **Parent** | `node_4_44` |
| **Children** | `node_6_17` |
| **Creation Index** | 145 |

---

## Hypothesis

> The 'EU-NIST' Implementation Gap: Requirements derived from the 'EU AI Act' map
to a significantly higher number of unique architectural controls per
requirement than those from the 'NIST AI RMF', indicating a higher technical
compliance burden for European regulations.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.2692 (Likely False) |
| **Surprise** | -0.5672 |
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
| Maybe False | 6.0 |
| Definitely False | 54.0 |

---

## Experiment Plan

**Objective:** Compare the density of mapped architectural controls between EU AI Act and NIST AI RMF.

### Steps
- 1. Load 'context_crosswalk_evidence.json'.
- 2. Extract the 'source' framework (EU AI Act vs NIST AI RMF) from the 'req_id' or 'competency_statement'.
- 3. Count the number of unique 'applicable_controls' for each requirement.
- 4. Perform a T-test or Mann-Whitney U test to compare the control counts between the two sources.

### Deliverables
- Box plot of control counts by Source; Statistical test results.

---

## Analysis

The experiment successfully analyzed the 'EU-NIST Implementation Gap' hypothesis
using the crosswalk evidence dataset. By filtering for relevant requirements, 23
NIST AI RMF items and 9 EU AI Act items were identified and compared based on
the count of mapped architectural controls.

1. **Hypothesis Rejection**: The results **reject** the hypothesis that the EU
AI Act imposes a significantly higher technical compliance burden than the NIST
AI RMF. The distributions of mapped controls were nearly identical.
2. **Statistical Equivalence**: A Mann-Whitney U test yielded a p-value of
0.7630, indicating no statistically significant difference between the two
groups.
3. **Control Density**: Both frameworks exhibited a similar density of controls,
with NIST averaging 1.87 controls per requirement and the EU AI Act averaging
1.78. Both had a median of 2.0 controls.
4. **Implication**: In the context of this specific architectural crosswalk, the
technical implementation requirements for complying with EU regulations are not
structurally more complex or denser than those for the US-based NIST framework.

---

## Review

The experiment successfully analyzed the 'EU-NIST Implementation Gap' hypothesis
using the crosswalk evidence dataset. By filtering for relevant requirements, 23
NIST AI RMF items and 9 EU AI Act items were identified and compared based on
the count of mapped architectural controls.

1. **Hypothesis Rejection**: The results **reject** the hypothesis that the EU
AI Act imposes a significantly higher technical compliance burden than the NIST
AI RMF. The distributions of mapped controls were nearly identical.
2. **Statistical Equivalence**: A Mann-Whitney U test yielded a p-value of
0.7630, indicating no statistically significant difference between the two
groups.
3. **Control Density**: Both frameworks exhibited a similar density of controls,
with NIST averaging 1.87 controls per requirement and the EU AI Act averaging
1.78. Both had a median of 2.0 controls.
4. **Implication**: In the context of this specific architectural crosswalk, the
technical implementation requirements for complying with EU regulations are not
structurally more complex or denser than those for the US-based NIST framework.

---

## Code

```python
import json
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import os

# Define file path
file_path = '../context_crosswalk_evidence.json'

# Load dataset
if not os.path.exists(file_path):
    # Fallback for local testing if directory structure differs
    file_path = 'context_crosswalk_evidence.json'

print(f"Loading dataset from {file_path}...")
with open(file_path, 'r') as f:
    data = json.load(f)

# Lists to store control counts
nist_counts = []
eu_counts = []

ignored_count = 0
ignored_ids = []

print(f"Processing {len(data)} requirements...")

for entry in data:
    req_id = str(entry.get('req_id', '')).strip()
    controls = entry.get('applicable_controls', [])
    
    # Determine number of controls
    if isinstance(controls, list):
        count = len(controls)
    elif isinstance(controls, str):
        # Handle string representation if necessary (e.g. "Control1, Control2")
        if controls.strip() == "":
            count = 0
        else:
            count = len([c.strip() for c in controls.split(',') if c.strip()])
    else:
        count = 0

    # Classification Logic
    # Assuming typical identifiers based on dataset description
    # NIST usually: "NIST", "Map", "RMF"
    # EU usually: "EU", "Art", "AI Act"
    
    req_upper = req_id.upper()
    
    if "NIST" in req_upper:
        nist_counts.append(count)
    elif "EU" in req_upper or "ART" in req_upper:
        eu_counts.append(count)
    else:
        # Check reasoning or statement if ID is ambiguous
        statement = str(entry.get('competency_statement', '')).upper()
        if "NIST" in statement:
            nist_counts.append(count)
        elif "EU AI ACT" in statement or "EUROPEAN" in statement:
            eu_counts.append(count)
        else:
            ignored_count += 1
            ignored_ids.append(req_id)

print(f"Found {len(nist_counts)} NIST requirements.")
print(f"Found {len(eu_counts)} EU AI Act requirements.")
if ignored_count > 0:
    print(f"Ignored {ignored_count} requirements from other sources (e.g., ISO, OWASP). Sample ignored: {ignored_ids[:3]}")

# Analysis
if len(nist_counts) < 2 or len(eu_counts) < 2:
    print("Insufficient data points for statistical analysis.")
else:
    # Descriptive Statistics
    nist_mean = np.mean(nist_counts)
    eu_mean = np.mean(eu_counts)
    nist_median = np.median(nist_counts)
    eu_median = np.median(eu_counts)
    
    print("\n--- Descriptive Statistics ---")
    print(f"NIST AI RMF: Mean = {nist_mean:.2f}, Median = {nist_median}, Max = {np.max(nist_counts)}, Min = {np.min(nist_counts)}")
    print(f"EU AI Act:   Mean = {eu_mean:.2f}, Median = {eu_median}, Max = {np.max(eu_counts)}, Min = {np.min(eu_counts)}")
    
    # Mann-Whitney U Test (Non-parametric test for independent samples)
    u_stat, p_val = stats.mannwhitneyu(eu_counts, nist_counts, alternative='two-sided')
    
    print("\n--- Statistical Test Results ---")
    print(f"Test: Mann-Whitney U Test")
    print(f"Hypothesis: EU requirements map to a different number of controls than NIST.")
    print(f"U-Statistic: {u_stat}")
    print(f"P-Value: {p_val:.4f}")
    
    if p_val < 0.05:
        print("Conclusion: Statistically significant difference detected.")
    else:
        print("Conclusion: No statistically significant difference detected.")

    # Visualization
    plt.figure(figsize=(10, 6))
    
    # Create boxplot
    data_to_plot = [nist_counts, eu_counts]
    labels = [f'NIST AI RMF\n(n={len(nist_counts)})', f'EU AI Act\n(n={len(eu_counts)})']
    
    plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                boxprops=dict(facecolor='#ADD8E6', color='blue'),
                medianprops=dict(color='red'))
    
    plt.title('Density of Architectural Controls: NIST AI RMF vs EU AI Act')
    plt.ylabel('Number of Mapped Controls per Requirement')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate means
    plt.text(1, nist_mean + 0.1, f'Mean: {nist_mean:.1f}', ha='center', va='bottom', fontweight='bold')
    plt.text(2, eu_mean + 0.1, f'Mean: {eu_mean:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from context_crosswalk_evidence.json...
Processing 42 requirements...
Found 23 NIST requirements.
Found 9 EU AI Act requirements.
Ignored 10 requirements from other sources (e.g., ISO, OWASP). Sample ignored: ['OWASP-1', 'OWASP-2', 'OWASP-3']

--- Descriptive Statistics ---
NIST AI RMF: Mean = 1.87, Median = 2.0, Max = 3, Min = 1
EU AI Act:   Mean = 1.78, Median = 2.0, Max = 3, Min = 1

--- Statistical Test Results ---
Test: Mann-Whitney U Test
Hypothesis: EU requirements map to a different number of controls than NIST.
U-Statistic: 96.5
P-Value: 0.7630
Conclusion: No statistically significant difference detected.

STDERR:
<ipython-input-1-15ace3087cf3>:109: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(data_to_plot, labels=labels, patch_artist=True,


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** This is a **Box Plot** (also known as a box-and-whisker plot).
*   **Purpose:** It compares the distribution of a quantitative variable ("Number of Mapped Controls per Requirement") across two different categorical groups ("NIST AI RMF" and "EU AI Act"). It allows for a visual comparison of central tendency (median/mean), variability (interquartile range), and overall range (min/max).

### 2. Axes
*   **Y-Axis:**
    *   **Label:** "Number of Mapped Controls per Requirement"
    *   **Range:** The axis is marked from **1.00 to 3.00**, with grid lines at intervals of 0.25.
    *   **Units:** Count (integer values representing controls).
*   **X-Axis:**
    *   **Labels:** Two specific regulatory frameworks are compared:
        1.  **NIST AI RMF** (National Institute of Standards and Technology AI Risk Management Framework)
        2.  **EU AI Act** (European Union Artificial Intelligence Act)
    *   **Context:** The axis represents categorical groups.

### 3. Data Trends
*   **Similar Distributions:** Both frameworks exhibit remarkably similar distributions.
    *   **Range:** Both plots show a total range (whiskers) extending from a minimum of **1** to a maximum of **3**.
    *   **Interquartile Range (IQR):** For both categories, the main "box" (representing the middle 50% of the data) spans from **1.0 to 2.0**.
    *   **Medians:** The red line indicating the median is situated at **2.0** for both groups.
*   **Box Structure:**
    *   The bottom of the box (25th percentile) aligns with the bottom whisker at 1.0, suggesting a concentration of data at the lower end of the scale.
    *   The top of the box (75th percentile) is at 2.0.

### 4. Annotations and Legends
*   **Mean Values:**
    *   **NIST AI RMF:** Annotated with **"Mean: 1.9"**.
    *   **EU AI Act:** Annotated with **"Mean: 1.8"**.
    *   This textual annotation provides the arithmetic average, offering a slightly more precise comparison than the visual median lines, which appear identical.
*   **Sample Size (n):**
    *   **NIST AI RMF:** **n=23**. This indicates that 23 requirements or data points were analyzed for this framework.
    *   **EU AI Act:** **n=9**. This indicates a much smaller sample size of 9 requirements.
*   **Visual Styling:**
    *   **Red Lines:** Represent the median.
    *   **Light Blue Box:** Represents the Interquartile Range (IQR) from the 25th to the 75th percentile.

### 5. Statistical Insights
*   **Uniformity of Density:** The plot suggests that, architecturally, both the NIST AI RMF and the EU AI Act have a very similar "density" of controls. On average, both frameworks map approximately **2 controls per requirement**.
*   **Slight Variation in Averages:** While the medians are identical (2.0), the NIST AI RMF has a marginally higher mean (1.9 vs 1.8). However, this difference (0.1) is negligible, reinforcing the similarity between the two.
*   **Sample Size Disparity:** There is a significant difference in the amount of data analyzed (**n=23** vs **n=9**). While the distributions look similar, the statistical confidence in the EU AI Act distribution is lower due to the small sample size.
*   **Capping at 3:** Neither framework appears to require more than 3 mapped controls for any single requirement in this dataset, indicating a relatively low complexity of control mapping per individual requirement.
*   **Skew:** Since the means (1.9 and 1.8) are slightly lower than the medians (2.0), the data is slightly left-skewed, meaning there is a pull toward the lower bound (1 control per requirement).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
