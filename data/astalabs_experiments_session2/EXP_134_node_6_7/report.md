# Experiment 134: node_6_7

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_7` |
| **ID in Run** | 134 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T07:19:18.324277+00:00 |
| **Runtime** | 295.9s |
| **Parent** | `node_5_41` |
| **Children** | `node_7_1` |
| **Creation Index** | 135 |

---

## Hypothesis

> The 'Forced Participation' Paradox: Public-facing AI services are statistically
less likely to offer 'Opt-Out' mechanisms compared to internal-facing employee
systems.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.3387 (Maybe False) |
| **Posterior** | 0.7747 (Likely True) |
| **Surprise** | +0.5232 |
| **Surprise Interpretation** | Strong Positive (hypothesis strengthened) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 5.0 |
| Uncertain | 0.0 |
| Maybe False | 25.0 |
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

**Objective:** Investigate if citizen-facing AI deployments prioritize service delivery over user agency (opt-out choice).

### Steps
- 1. Load 'eo13960_scored'.
- 2. Use '26_public_service' to distinguish 'Public-Facing' vs 'Internal' systems.
- 3. Clean '67_opt_out' into binary (Yes/No).
- 4. Compare the Opt-Out rates using a Chi-Square test.
- 5. Visualize the proportion of 'Mandatory' AI interaction for the public vs employees.

### Deliverables
- Comparative statistics, Visualization of User Control by Target Audience.

---

## Analysis

The experiment was successfully executed and yielded statistically significant
results supporting the 'Forced Participation' Paradox hypothesis. After
resolving initial data cleaning issues by treating free-text service
descriptions as 'Public-Facing' indicators and interpreting verbose 'No'
responses in the opt-out column, the script analyzed 120 valid use cases from
the EO 13960 inventory.

The Chi-Square Test of Independence resulted in a statistic of 30.12 and a
p-value of 4.07e-08, which is well below the 0.05 threshold. This provides
strong evidence to **reject the null hypothesis** and confirms that target
audience is a significant predictor of opt-out availability.

The data reveals a stark disparity in user agency: Internal systems (likely
employee-facing) offer opt-out mechanisms 83.0% of the time, whereas Public-
Facing systems offer them only 28.1% of the time. This 55-percentage-point gap
confirms the existence of a 'Forced Participation' dynamic, where the general
public is significantly less likely to be afforded the choice to decline AI
interaction compared to federal employees.

---

## Review

The experiment was successfully executed and yielded robust statistical evidence
supporting the 'Forced Participation' Paradox hypothesis. After overcoming
initial data cleaning challenges—specifically identifying that the 'Public
Service' column used free-text descriptions rather than binary flags—the
analysis successfully isolated 120 valid use cases.

The Chi-Square test (p = 4.07e-08) revealed a highly significant disparity:
Internal (employee-facing) AI systems offer opt-out mechanisms 83.0% of the
time, whereas Public-Facing systems offer them only 28.1% of the time. This
55-percentage-point gap confirms that while federal employees often retain the
agency to decline AI interaction, the general public is frequently subjected to
mandatory AI participation in government services, validating the 'Forced
Participation' paradox.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load dataset
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 subset
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

# --- Step 1: Define Audience (Public vs Internal) ---
# Logic: '26_public_service' contains descriptions for public services.
# We assume populated (and meaningful) text implies 'Public-Facing'.
# Null or empty/short artifacts implies 'Internal'.

def categorize_audience(val):
    if pd.isna(val):
        return 'Internal'
    s = str(val).strip()
    if len(s) < 3:  # Filter out artifacts like '.', ' ', or '\n'
        return 'Internal'
    return 'Public-Facing'

eo_df['audience_clean'] = eo_df['26_public_service'].apply(categorize_audience)

# --- Step 2: Define Opt-Out (Yes vs No) ---
# Logic: '67_opt_out' contains 'Yes', verbose 'No...', or 'Waived'.

def categorize_opt_out(val):
    if pd.isna(val):
        return None
    s = str(val).lower().strip()
    
    if s == 'yes' or s.startswith('yes'):
        return 'Yes'
    if s.startswith('no') or 'waived' in s:
        return 'No'
    return None # Exclude 'Other', 'N/A' if ambiguous, or NaN

eo_df['opt_out_clean'] = eo_df['67_opt_out'].apply(categorize_opt_out)

# --- Step 3: Filter Valid Data ---
# We only analyze rows where a definitive Opt-Out status is recorded.
analysis_df = eo_df.dropna(subset=['opt_out_clean']).copy()

print("Data Preparation Complete.")
print(f"Total EO 13960 Rows: {len(eo_df)}")
print(f"Rows with Valid Opt-Out Status: {len(analysis_df)}")
print("\nDistribution of Audience in Analysis Set:")
print(analysis_df['audience_clean'].value_counts())

# --- Step 4: Statistical Analysis ---
if len(analysis_df) > 0:
    # Contingency Table
    crosstab = pd.crosstab(analysis_df['audience_clean'], analysis_df['opt_out_clean'])
    print("\nContingency Table (Count):")
    print(crosstab)
    
    # Check if we have enough data dimensions
    if crosstab.shape == (2, 2):
        # Percentages
        crosstab_pct = pd.crosstab(analysis_df['audience_clean'], analysis_df['opt_out_clean'], normalize='index') * 100
        print("\nOpt-Out Availability Rates (%):")
        print(crosstab_pct)
        
        # Chi-Square
        chi2, p, dof, ex = chi2_contingency(crosstab)
        print(f"\nChi-Square Test Results:")
        print(f"Statistic: {chi2:.4f}")
        print(f"P-Value: {p:.4e}")
        
        # Interpretation
        if p < 0.05:
            print("Result: Statistically Significant Association.")
        else:
            print("Result: No Significant Association.")
            
        # Visualization
        plt.figure(figsize=(8, 6))
        yes_rates = crosstab_pct['Yes']
        colors = ['#2ca02c', '#1f77b4'] # Green vs Blue
        ax = yes_rates.plot(kind='bar', color=colors, alpha=0.9, edgecolor='black', rot=0)
        
        plt.title("The 'Forced Participation' Paradox:\nOpt-Out Availability by Audience")
        plt.ylabel("Use Cases with Opt-Out (%)")
        plt.xlabel("Target Audience")
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        for i, v in enumerate(yes_rates):
            ax.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
            
        plt.tight_layout()
        plt.show()
        
    else:
        print("\nInsufficient dimensions for Chi-Square (need 2x2).")
        # Fallback print if one category is missing (e.g., all Internal are No)
        print("Raw Counts:")
        print(crosstab)
else:
    print("No valid data available for analysis.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Data Preparation Complete.
Total EO 13960 Rows: 1757
Rows with Valid Opt-Out Status: 120

Distribution of Audience in Analysis Set:
audience_clean
Internal         88
Public-Facing    32
Name: count, dtype: int64

Contingency Table (Count):
opt_out_clean   No  Yes
audience_clean         
Internal        15   73
Public-Facing   23    9

Opt-Out Availability Rates (%):
opt_out_clean          No        Yes
audience_clean                      
Internal        17.045455  82.954545
Public-Facing   71.875000  28.125000

Chi-Square Test Results:
Statistic: 30.1175
P-Value: 4.0664e-08
Result: Statistically Significant Association.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

**1. Plot Type**
*   **Type:** Vertical Bar Plot (or Bar Chart).
*   **Purpose:** The plot compares the percentage of use cases that offer an "Opt-Out" mechanism across two distinct categories of target audiences ("Internal" vs. "Public-Facing").

**2. Axes**
*   **X-Axis (Horizontal):**
    *   **Title:** "Target Audience"
    *   **Labels:** The axis categorizes the data into "Internal" and "Public-Facing."
*   **Y-Axis (Vertical):**
    *   **Title:** "Use Cases with Opt-Out (%)"
    *   **Range:** 0 to 100.
    *   **Units:** Percentage (%). The axis is marked with intervals of 20 (0, 20, 40, 60, 80, 100).

**3. Data Trends**
*   **Tallest Bar:** The "Internal" audience bar (colored green) is the tallest.
*   **Shortest Bar:** The "Public-Facing" audience bar (colored blue) is significantly shorter.
*   **Pattern:** There is a drastic disparity between the two categories. Use cases targeted at internal audiences are nearly three times as likely to have opt-out availability compared to those targeted at the public.

**4. Annotations and Legends**
*   **Main Title:** "The 'Forced Participation' Paradox: Opt-Out Availability by Audience". This title frames the data as counter-intuitive (a "paradox"), implying that one might expect different results (perhaps expecting the public to have more choice).
*   **Data Labels:**
    *   **Internal:** Annotated with **"83.0%"** at the top of the green bar.
    *   **Public-Facing:** Annotated with **"28.1%"** at the top of the blue bar.
*   **Gridlines:** Horizontal dashed gridlines appear at 20% intervals to aid in visual estimation.

**5. Statistical Insights**
*   **Significant Discrepancy:** There is a massive 54.9 percentage point gap between the two groups. While the vast majority of internal use cases (83.0%) allow users to opt out, less than a third (28.1%) of public-facing use cases offer the same choice.
*   **The "Paradox":** The chart suggests that internal stakeholders (likely employees or organizational members) are afforded significantly more agency and consent regarding participation than the general public. This is often described as a paradox because public-facing technologies are usually subject to higher scrutiny and broader regulatory privacy requirements, yet here they show much lower voluntary participation rates.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
