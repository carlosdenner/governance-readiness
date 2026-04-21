# Experiment 200: node_5_78

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_78` |
| **ID in Run** | 200 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T10:23:59.408837+00:00 |
| **Runtime** | 295.5s |
| **Parent** | `node_4_43` |
| **Children** | `node_6_37` |
| **Creation Index** | 201 |

---

## Hypothesis

> The 'Public Transparency' Paradox: Federal AI systems explicitly designated as
'Public Facing' are no more likely to provide 'AI Notice' to users than
internal/non-public systems, indicating a failure in transparency obligations.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7500 (Likely True) |
| **Posterior** | 0.2555 (Likely False) |
| **Surprise** | -0.5934 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 1.0 |
| Maybe True | 29.0 |
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
| Definitely False | 60.0 |

---

## Experiment Plan

**Objective:** Test if public-facing status correlates with the presence of AI Notice mechanisms.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' (EO13960 subset).
- 2. Create a 'Public_Facing' flag using columns '26_public_service' or '27_public_info' (Yes=1, No=0).
- 3. Create a 'Has_Notice' flag using column '59_ai_notice' (Yes=1, No=0).
- 4. Calculate the rate of 'Has_Notice' for Public vs Non-Public systems.
- 5. Perform a Chi-square test or Z-test to see if Public systems have a statistically higher notification rate.

### Deliverables
- 1. Notification rates for Public vs Private systems.
- 2. Statistical test results.
- 3. Interpretation of the 'Transparency Gap'.

---

## Analysis

The experiment successfully tested the 'Public Transparency Paradox' hypothesis
using the EO 13960 dataset. After implementing robust string matching to handle
categorical text fields (mapping 'Online', 'In-person', etc., to True for
notices, and service descriptions to True for public-facing status), the
analysis yielded the following results:

1.  **Data Segmentation**: 255 systems were identified as 'Public Facing' and
1,502 as 'Internal/Non-Public'.
2.  **Transparency Rates**: Public-facing systems demonstrated a significantly
higher AI Notice rate (10.2%, 26/255) compared to internal systems (2.8%,
42/1502).
3.  **Statistical Outcome**: The Chi-square test (statistic=30.13, p=4.05e-08)
confirmed this difference is highly statistically significant.

**Conclusion**: The hypothesis was **REJECTED**. Contrary to the 'paradox'
prediction, public-facing systems are significantly *more* likely (approx. 3.6x)
to provide AI notices than internal systems. However, the experiment notes that
the absolute transparency rate remains objectively low (<11%) even for public-
facing deployments.

---

## Review

The experiment successfully tested the 'Public Transparency Paradox' hypothesis
using the EO 13960 dataset. After implementing robust string matching to handle
categorical text fields (mapping 'Online', 'In-person', etc., to True for
notices, and service descriptions to True for public-facing status), the
analysis yielded the following results:

1.  **Data Segmentation**: 255 systems were identified as 'Public Facing' and
1,502 as 'Internal/Non-Public'.
2.  **Transparency Rates**: Public-facing systems demonstrated a significantly
higher AI Notice rate (10.2%, 26/255) compared to internal systems (2.8%,
42/1502).
3.  **Statistical Outcome**: The Chi-square test (statistic=30.13, p=4.05e-08)
confirmed this difference is highly statistically significant.

**Conclusion**: The hypothesis was **REJECTED**. Contrary to the 'paradox'
prediction, public-facing systems are significantly *more* likely (approx. 3.6x)
to provide AI notices than internal systems. However, the experiment notes that
the absolute transparency rate remains objectively low (<11%) even for public-
facing deployments.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def run_experiment():
    # Load dataset
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        try:
            df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
        except FileNotFoundError:
            print("Error: Dataset not found.")
            return

    # Filter for EO 13960 source
    df = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"Loaded {len(df)} records from EO 13960.")

    # --- Logic for Public Facing ---
    # 27_public_info: Explicit 'Yes'/'No'.
    def is_public_info(val):
        if pd.isna(val):
            return False
        s = str(val).strip().lower()
        return s == 'yes'

    # 26_public_service: Free text descriptions implies True, unless empty or 'No'.
    def is_public_service(val):
        if pd.isna(val):
            return False
        s = str(val).strip().lower()
        if s == '' or s == 'no' or s == 'nan':
            return False
        return True

    df['public_facing'] = df['27_public_info'].apply(is_public_info) | df['26_public_service'].apply(is_public_service)

    # --- Logic for AI Notice ---
    # 59_ai_notice: Categorical.
    # Positive keywords: 'online', 'in-person', 'email', 'telephone', 'other'
    # Negative/Neutral: 'n/a', 'none', 'waived', 'not safety'
    def has_notice(val):
        if pd.isna(val):
            return False
        s = str(val).strip().lower()
        
        # explicit negatives
        if any(x in s for x in ['n/a', 'none of the above', 'waived', 'not safety']):
            return False
            
        # explicit positives
        if any(x in s for x in ['online', 'in-person', 'email', 'telephone', 'other', 'terms', 'instruction']):
            return True
            
        return False

    df['has_notice'] = df['59_ai_notice'].apply(has_notice)

    # Grouping
    public_group = df[df['public_facing']]
    internal_group = df[~df['public_facing']]

    n_public = len(public_group)
    n_internal = len(internal_group)
    
    print(f"\n--- Categorization ---")
    print(f"Public Facing: {n_public}")
    print(f"Internal/Non-Public: {n_internal}")

    if n_public == 0 or n_internal == 0:
        print("Cannot perform test: One group is empty.")
        return

    # Calculate rates
    n_public_notice = public_group['has_notice'].sum()
    n_internal_notice = internal_group['has_notice'].sum()

    rate_public = n_public_notice / n_public if n_public > 0 else 0
    rate_internal = n_internal_notice / n_internal if n_internal > 0 else 0

    print("\n--- Descriptive Statistics ---")
    print(f"Public-Facing Systems ({n_public}):")
    print(f"  With AI Notice: {n_public_notice} ({rate_public:.2%})")
    print(f"Internal/Other Systems ({n_internal}):")
    print(f"  With AI Notice: {n_internal_notice} ({rate_internal:.2%})")

    # Statistical Test
    observed = np.array([
        [n_public_notice, n_public - n_public_notice],
        [n_internal_notice, n_internal - n_internal_notice]
    ])
    
    # Check for zeroes in rows/cols to avoid error, though chi2_contingency handles 0 observed well, it fails if expected is 0.
    # If row sums are 0, we can't do it.
    if n_public == 0 or n_internal == 0:
         print("Skipping test due to empty group.")
    elif (n_public_notice == 0 and n_internal_notice == 0):
         print("\nResult: No notices found in EITHER group. Rates are identical (0%).")
         print("Hypothesis Status: Technically supported (no difference), but functionally a universal failure of transparency.")
    else:
        chi2, p, dof, expected = stats.chi2_contingency(observed)
        print("\n--- Statistical Test Results (Chi-Square) ---")
        print(f"Chi2 Statistic: {chi2:.4f}")
        print(f"P-value: {p:.4e}")

        alpha = 0.05
        print("\n--- Interpretation ---")
        if p < alpha:
            print("Result: Significant difference detected.")
            if rate_public > rate_internal:
                print("Direction: Public-facing systems are significantly MORE likely to provide notice.")
                print("Hypothesis Status: REJECTED (Transparency works better for public systems).")
            else:
                print("Direction: Public-facing systems are significantly LESS likely to provide notice.")
                print("Hypothesis Status: SUPPORTED (Paradox confirmed).")
        else:
            print("Result: No significant difference detected.")
            print(f"Gap: {(rate_public - rate_internal)*100:.2f} percentage points.")
            print("Hypothesis Status: SUPPORTED (Paradox confirmed - public status does not significantly improve transparency).")

        # Visualization
        plt.figure(figsize=(8, 6))
        categories = ['Public Facing', 'Internal/Non-Public']
        percentages = [rate_public * 100, rate_internal * 100]
        
        bars = plt.bar(categories, percentages, color=['#d62728', '#7f7f7f'], edgecolor='black', alpha=0.8)
        plt.ylabel('Percentage with AI Notice (%)')
        plt.title('Transparency Gap: AI Notice Rates by Deployment Type')
        plt.ylim(0, max(max(percentages)*1.2, 5)) # Ensure at least 0-5 scale
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded 1757 records from EO 13960.

--- Categorization ---
Public Facing: 255
Internal/Non-Public: 1502

--- Descriptive Statistics ---
Public-Facing Systems (255):
  With AI Notice: 26 (10.20%)
Internal/Other Systems (1502):
  With AI Notice: 42 (2.80%)

--- Statistical Test Results (Chi-Square) ---
Chi2 Statistic: 30.1255
P-value: 4.0496e-08

--- Interpretation ---
Result: Significant difference detected.
Direction: Public-facing systems are significantly MORE likely to provide notice.
Hypothesis Status: REJECTED (Transparency works better for public systems).


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot.
*   **Purpose:** The plot is designed to compare the prevalence (rates) of AI notices between two distinct categorical groups: "Public Facing" deployments and "Internal/Non-Public" deployments.

### 2. Axes
*   **Y-Axis:**
    *   **Title:** "Percentage with AI Notice (%)"
    *   **Unit:** Percentage.
    *   **Range:** The axis is scaled from **0 to 12**, with major tick marks at intervals of 2 (0, 2, 4, 6, 8, 10, 12).
*   **X-Axis:**
    *   **Title:** The axis does not have a specific title, but it represents "Deployment Type."
    *   **Labels:** It features two distinct categories: **"Public Facing"** and **"Internal/Non-Public"**.

### 3. Data Trends
*   **Tallest Bar:** The **"Public Facing"** category corresponds to the tallest bar (colored red), indicating the highest value in the dataset.
*   **Shortest Bar:** The **"Internal/Non-Public"** category corresponds to the shortest bar (colored grey), indicating a significantly lower value.
*   **Pattern:** There is a stark contrast between the two categories. The visual disparity highlights that AI notices are much more common in public-facing contexts compared to internal ones.

### 4. Annotations and Legends
*   **Bar Annotations:** The exact values are annotated in bold text directly above each bar for clarity:
    *   Public Facing: **10.2%**
    *   Internal/Non-Public: **2.8%**
*   **Title:** The chart title, **"Transparency Gap: AI Notice Rates by Deployment Type,"** provides context for the data, explicitly framing the difference in values as a "gap" in transparency.
*   **Color Coding:** The use of distinct colors (Red for Public Facing vs. Grey for Internal) reinforces the contrast between the primary focus group and the secondary group.

### 5. Statistical Insights
*   **Significant Disparity:** There is a large "transparency gap" of **7.4 percentage points** between the two groups.
*   **Relative Likelihood:** Public-facing AI deployments are **more than 3.6 times as likely** ($10.2 \div 2.8 \approx 3.64$) to carry an AI notice compared to internal or non-public deployments.
*   **Conclusion:** The data suggests that while AI transparency (in the form of notices) is relatively low overall (peaking at roughly 10%), organizations are significantly more diligent about disclosing AI usage to the public than they are for internal or non-public applications.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
