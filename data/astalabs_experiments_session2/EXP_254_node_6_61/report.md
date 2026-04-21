# Experiment 254: node_6_61

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_61` |
| **ID in Run** | 254 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T13:00:06.285763+00:00 |
| **Runtime** | 284.4s |
| **Parent** | `node_5_8` |
| **Children** | None |
| **Creation Index** | 255 |

---

## Hypothesis

> The 'Public-Facing' Commercial Gap: AI systems providing a direct 'Public
Service' are significantly less likely to be 'Commercial' (COTS) products
compared to internal administrative systems, reflecting a preference for custom
solutions in citizen-facing roles.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.6374 (Maybe True) |
| **Surprise** | -0.1255 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | No |

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
| Maybe True | 20.0 |
| Uncertain | 40.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Analyze procurement patterns (Build vs Buy) based on the system's service orientation.

### Steps
- 1. Filter 'eo13960_scored'.
- 2. Create binary 'is_public_service' from '26_public_service' (Yes vs No).
- 3. Create binary 'is_commercial' from '10_commercial_ai' (Commercial vs Custom/Other).
- 4. Calculate Odds Ratio of using Commercial AI for Internal vs Public-facing tasks.

### Deliverables
- Odds Ratio and analysis of commercial software usage in public vs internal contexts.

---

## Analysis

The experiment successfully analyzed the procurement patterns of AI systems in
the EO 13960 inventory. By classifying systems based on their service
orientation, the code identified 65 'Public Service' systems and 1,692
'Internal/Admin' systems. The analysis of commercial (COTS) vs.
custom/government-built solutions revealed that public-facing systems have a
lower rate of commercial AI adoption (13.8%, 9/65) compared to internal systems
(20.3%, 344/1692). The calculated Odds Ratio of 0.63 indicates that public-
facing systems are approximately 1.59 times less likely to use commercial AI
than internal systems. However, the Fisher's Exact Test yielded a p-value of
0.268, indicating that this difference is not statistically significant at the p
< 0.05 level. Therefore, while the trend supports the hypothesis of a preference
for custom solutions in public services, the evidence is not strong enough to
rule out random variation.

---

## Review

The experiment was successfully executed. The code correctly parsed the sparse
and messy text fields in the EO 13960 dataset to classify AI systems by service
type (Public vs. Internal) and procurement method (Commercial vs. Custom).

Results:
- **Sample**: 1,757 systems (1,692 Internal/Admin, 65 Public Service).
- **Commercial Adoption**: Internal systems had a 20.3% commercial adoption rate
(344/1692), while Public Service systems had a 13.8% rate (9/65).
- **Statistical Test**: The calculated Odds Ratio was 0.63, suggesting public-
facing systems are roughly 1.6 times *less* likely to use commercial AI.
However, the Fisher's Exact Test yielded a p-value of 0.268.

Conclusion:
While the data exhibits a directional trend supporting the hypothesis that
public-facing systems favor custom solutions over commercial ones, the
difference is not statistically significant (p > 0.05). The null hypothesis
cannot be rejected based on this dataset.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact, chi2_contingency
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 scored data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"EO 13960 Dataset Size: {len(eo_df)}")

# --- 1. Variable Construction ---

# Construct 'is_public_service'
# Logic: If '26_public_service' has content (and isn't 'No'), it's a public service.
# If it's NaN or 'No', it's Internal.
eo_df['26_public_service'] = eo_df['26_public_service'].astype(str).replace('nan', np.nan)

def classify_service(val):
    if pd.isna(val):
        return False
    if val.lower().strip() == 'no':
        return False
    # If it has substantial text, it's a description of the service -> True
    if len(val) > 2:
        return True
    return False

eo_df['is_public_service'] = eo_df['26_public_service'].apply(classify_service)

print("\nConstructed 'is_public_service':")
print(eo_df['is_public_service'].value_counts())

# Construct 'is_commercial'
# Logic: Based on column 10_commercial_ai.
# "None of the above." -> Custom/Other (False)
# Specific use cases -> Commercial (True)
# NaN -> Assume Custom/Other (False) for now, or exclude. Let's assume False to be conservative.

def classify_commercial(val):
    if pd.isna(val):
        return False
    s = str(val).strip()
    if "None of the above" in s:
        return False
    return True

eo_df['is_commercial'] = eo_df['10_commercial_ai'].apply(classify_commercial)

print("\nConstructed 'is_commercial':")
print(eo_df['is_commercial'].value_counts())

# --- 2. Contingency Analysis ---

# Create Crosstab
contingency = pd.crosstab(eo_df['is_public_service'], eo_df['is_commercial'])

# Labeling for clarity (Printing raw first to avoid index errors)
print("\n--- Raw Contingency Table ---")
print(contingency)

# Check if we have a 2x2 matrix
if contingency.shape == (2, 2):
    contingency.index = ['Internal/Admin', 'Public Service']
    contingency.columns = ['Custom/Gov-Built', 'Commercial/COTS']
    print("\n--- Labeled Contingency Table ---")
    print(contingency)
    
    # --- 3. Statistical Testing ---
    
    # Chi-Square
    chi2, p, dof, ex = chi2_contingency(contingency)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Odds Ratio (Fisher Exact)
    # Table: [[Internal_Custom, Internal_Comm], [Public_Custom, Public_Comm]]
    # OR > 1 means Public is MORE likely to be Commercial
    # OR < 1 means Public is LESS likely to be Commercial
    # We want OR of Commercial for Public vs Internal
    # Odds_Public = Comm / Custom
    # Odds_Internal = Comm / Custom
    # OR = Odds_Public / Odds_Internal
    
    # Use Fisher Exact for precision
    # fisher_exact expects [[a, b], [c, d]]
    # We want to check association between Row 2 (Public) and Col 2 (Commercial)
    # Let's align it: 
    #              Comm   Custom
    # Public       a      b
    # Internal     c      d
    
    # Current table:
    #              Custom  Comm
    # Internal     A       B
    # Public       C       D
    
    # Re-arranging for the specific hypothesis test:
    # Rows: Public, Internal
    # Cols: Commercial, Custom
    
    # Public_Comm
    pc = contingency.loc['Public Service', 'Commercial/COTS']
    # Public_Custom
    p_cust = contingency.loc['Public Service', 'Custom/Gov-Built']
    # Internal_Comm
    ic = contingency.loc['Internal/Admin', 'Commercial/COTS']
    # Internal_Custom
    i_cust = contingency.loc['Internal/Admin', 'Custom/Gov-Built']
    
    obs = [[pc, p_cust], [ic, i_cust]]
    
    odds_r, p_val_fisher = fisher_exact(obs)
    
    print(f"\nFisher Exact Odds Ratio: {odds_r:.4f}")
    print(f"Fisher P-value: {p_val_fisher:.4e}")
    
    if odds_r < 1:
        print(f"Result: Public-facing systems are {1/odds_r:.2f}x LESS likely to use Commercial AI.")
    else:
        print(f"Result: Public-facing systems are {odds_r:.2f}x MORE likely to use Commercial AI.")

else:
    print("\nError: Contingency table is not 2x2. Cannot perform Odds Ratio analysis.")
    print("Check variable construction logic.")

# --- 4. Visualization ---
if contingency.shape == (2, 2):
    # Calculate percentages
    props = contingency.div(contingency.sum(axis=1), axis=0)
    
    ax = props.plot(kind='bar', stacked=True, color=['lightgray', 'steelblue'], figsize=(8, 6))
    plt.title('Commercial AI Adoption: Public Service vs Internal')
    plt.ylabel('Proportion')
    plt.xlabel('Service Type')
    plt.legend(title='Procurement', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: EO 13960 Dataset Size: 1757

Constructed 'is_public_service':
is_public_service
False    1692
True       65
Name: count, dtype: int64

Constructed 'is_commercial':
is_commercial
False    1404
True      353
Name: count, dtype: int64

--- Raw Contingency Table ---
is_commercial      False  True 
is_public_service              
False               1348    344
True                  56      9

--- Labeled Contingency Table ---
                Custom/Gov-Built  Commercial/COTS
Internal/Admin              1348              344
Public Service                56                9

Chi-Square Statistic: 1.2606
P-value: 2.6155e-01

Fisher Exact Odds Ratio: 0.6298
Fisher P-value: 2.6824e-01
Result: Public-facing systems are 1.59x LESS likely to use Commercial AI.


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Stacked Bar Chart (specifically a 100% stacked bar chart).
*   **Purpose:** This plot compares the proportional composition of AI procurement methods ("Custom/Gov-Built" vs. "Commercial/COTS") across two different categories of service ("Internal/Admin" and "Public Service").

### 2. Axes
*   **X-axis:**
    *   **Title:** "Service Type"
    *   **Labels:** The axis displays two categorical variables: "Internal/Admin" and "Public Service".
*   **Y-axis:**
    *   **Title:** "Proportion"
    *   **Range:** 0.0 to 1.0 (representing 0% to 100%).
    *   **Units:** The values are decimal proportions (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).

### 3. Data Trends
*   **Dominant Category:** In both service types, the "Custom/Gov-Built" procurement method (represented by the grey bar) makes up the vast majority of the proportion, significantly overshadowing "Commercial/COTS" (blue bar).
*   **Internal/Admin:** The "Custom/Gov-Built" portion appears to sit right at the **0.80** mark, meaning approximately **80%** of AI adoption in this sector is custom or government-built, leaving roughly **20%** for commercial/COTS solutions.
*   **Public Service:** The "Custom/Gov-Built" portion is noticeably taller than in the Internal/Admin column, reaching approximately **0.86** to **0.87**. This indicates a higher reliance (approx. **86-87%**) on custom solutions for public-facing services compared to internal administrative functions. Conversely, the commercial portion is smaller (approx. **13-14%**).

### 4. Annotations and Legends
*   **Title:** "Commercial AI Adoption: Public Service vs Internal" clearly sets the context of the comparison.
*   **Legend:** Located in the upper right corner, titled "Procurement".
    *   **Light Grey:** Represents "Custom/Gov-Built".
    *   **Blue:** Represents "Commercial/COTS" (Commercial Off-The-Shelf).

### 5. Statistical Insights
*   **Preference for Custom Solutions:** Across the board, there is a strong preference for building AI solutions internally or customizing them specifically for government use rather than buying off-the-shelf commercial products.
*   **Service Type Variance:** While both sectors favor custom builds, **Public Service** applications are even less likely to use commercial AI than **Internal/Admin** applications.
    *   *Potential Interpretation:* Internal administrative tasks (like payroll or HR) are often standardized across industries, making Commercial/COTS products more viable (approx. 20% adoption). However, Public Service delivery likely involves unique regulatory requirements, privacy concerns, or specialized workflows that necessitate bespoke, government-built solutions (reducing commercial adoption to roughly 14%).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
