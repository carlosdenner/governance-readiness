# Experiment 97: node_5_21

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_21` |
| **ID in Run** | 97 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T05:45:34.505640+00:00 |
| **Runtime** | 275.2s |
| **Parent** | `node_4_20` |
| **Children** | `node_6_20` |
| **Creation Index** | 98 |

---

## Hypothesis

> Sector-Harm Divergence: Incidents in the 'Public Sector' are significantly more
likely to result in 'Intangible/Risk' harms compared to 'Private Sector'
incidents, which are more prone to 'Tangible' harms.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7258 (Likely True) |
| **Posterior** | 0.2527 (Likely False) |
| **Surprise** | -0.5677 |
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
| Maybe False | 2.0 |
| Definitely False | 58.0 |

---

## Experiment Plan

**Objective:** Analyze the AIID subset to determine if there is a statistically significant association between the deployment sector (Public vs. Private) and the nature of the harm (Tangible vs. Intangible/Risk).

### Steps
- 1. Load the dataset 'astalabs_discovery_all_data.csv' and filter for rows where 'source_table' is 'aiid_incidents'.
- 2. Create a new variable 'Sector_Group' based on the 'Sector of Deployment' column:
    - Categorize as 'Public' if the string contains: 'public administration', 'defense', 'law enforcement', 'education', 'social work', or 'government'.
    - Categorize as 'Private' if it does not match 'Public' but contains: 'finance', 'manufacturing', 'retail', 'entertainment', 'transportation', 'accommodation', 'information and communication', or 'professional'.
    - Drop rows that do not fit these categories.
- 3. Create a new variable 'Harm_Category' based on the 'Tangible Harm' column:
    - Map 'tangible harm definitively occurred' to 'Tangible'.
    - Map 'no tangible harm, near-miss, or issue', 'imminent risk of tangible harm (near miss) did occur', and 'non-imminent risk of tangible harm (an issue) occurred' to 'Intangible/Risk'.
    - Drop rows with null or 'unclear' values.
- 4. Create a contingency table of 'Sector_Group' vs. 'Harm_Category'.
- 5. Perform a Chi-square test of independence to assess statistical significance.
- 6. Calculate the Odds Ratio to quantify the strength of the association (comparing the odds of Intangible harm in Public vs. Private sectors).

### Deliverables
- 1. Contingency Table (Sector vs. Harm Category).
- 2. Chi-square statistic and p-value.
- 3. Odds Ratio calculation.

---

## Analysis

The experiment successfully analyzed 191 AIID incidents to test the 'Sector-Harm
Divergence' hypothesis.

**Data Processing:**
- **Sector Classification:** Incidents were categorized into 'Private' (n=141)
and 'Public' (n=50) based on keyword matching.
- **Harm Classification:** Outcomes were binary coded as 'Tangible'
(physical/financial damage) or 'Intangible/Risk' (near-misses, issues, non-
tangible harms).

**Findings:**
- **Private Sector:** 34.0% of incidents resulted in tangible harm (48 Tangible
vs. 93 Intangible/Risk).
- **Public Sector:** 42.0% of incidents resulted in tangible harm (21 Tangible
vs. 29 Intangible/Risk).

**Statistical Analysis:**
- **Chi-Square Test:** The test yielded a statistic of 0.6974 and a **p-value of
0.4037**, indicating no statistically significant difference between the
sectors.
- **Odds Ratio:** The Odds Ratio of Tangible Harm (Private vs. Public) was 0.71.
This indicates that, in this sample, the Private sector actually had slightly
*lower* odds of tangible harm compared to the Public sector, which contradicts
the directionality of the original hypothesis, though the result is likely due
to chance (p > 0.05).

**Conclusion:**
The hypothesis is **rejected**. The data does not support the claim that Public
Sector incidents are significantly more likely to result in Intangible harms
while Private Sector incidents result in Tangible harms. In fact, the
distribution of harm types was statistically indistinguishable between the two
sectors.

---

## Review

The experiment was successfully executed following the proposed plan. The
programmer correctly implemented the keyword-based categorization for sectors
and the specific value mapping for harm types. The statistical analysis (Chi-
square and Odds Ratio) was performed correctly on the resulting contingency
table.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# 3. Feature Engineering: Sector
def classify_sector(sector_str):
    if pd.isna(sector_str):
        return None
    
    s = sector_str.lower()
    
    # Public Sector keywords
    public_keywords = ['public administration', 'defense', 'law enforcement', 'education', 'social work', 'government']
    if any(k in s for k in public_keywords):
        return 'Public'
    
    # Private Sector keywords (if not public)
    private_keywords = ['financial', 'manufacturing', 'retail', 'entertainment', 'transportation', 
                        'accommodation', 'information', 'communication', 'professional', 'real estate', 'arts']
    if any(k in s for k in private_keywords):
        return 'Private'
    
    return None

aiid_df['Sector_Group'] = aiid_df['Sector of Deployment'].apply(classify_sector)

# 4. Feature Engineering: Harm
def classify_harm(harm_str):
    if pd.isna(harm_str) or harm_str == 'unclear':
        return None
    
    if harm_str == 'tangible harm definitively occurred':
        return 'Tangible'
    elif harm_str in ['no tangible harm, near-miss, or issue', 
                      'imminent risk of tangible harm (near miss) did occur', 
                      'non-imminent risk of tangible harm (an issue) occurred']:
        return 'Intangible/Risk'
    
    return None

aiid_df['Harm_Category'] = aiid_df['Tangible Harm'].apply(classify_harm)

# 5. Drop rows with missing values in relevant columns
analysis_df = aiid_df.dropna(subset=['Sector_Group', 'Harm_Category'])

# 6. Create Contingency Table
contingency_table = pd.crosstab(analysis_df['Sector_Group'], analysis_df['Harm_Category'])

print("--- Contingency Table (Sector vs. Harm) ---")
print(contingency_table)

# 7. Statistical Tests
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-square statistic: {chi2:.4f}")
print(f"p-value: {p:.4f}")

# 8. Odds Ratio Calculation
# OR = (a*d) / (b*c)
# Table structure usually: 
#           Intangible  Tangible
# Private       a          b
# Public        c          d
# We want odds of Intangible in Public vs Private, or Tangible in Private vs Public.
# Let's calculate Odds of Tangible Harm for Private vs. Public.

if 'Tangible' in contingency_table.columns and 'Private' in contingency_table.index:
    # Counts
    private_tangible = contingency_table.loc['Private', 'Tangible']
    private_intangible = contingency_table.loc['Private', 'Intangible/Risk']
    public_tangible = contingency_table.loc['Public', 'Tangible']
    public_intangible = contingency_table.loc['Public', 'Intangible/Risk']
    
    # Odds
    odds_private = private_tangible / private_intangible if private_intangible > 0 else np.nan
    odds_public = public_tangible / public_intangible if public_intangible > 0 else np.nan
    
    odds_ratio = odds_private / odds_public if odds_public > 0 else np.nan
    
    print(f"\nOdds of Tangible Harm (Private): {odds_private:.4f}")
    print(f"Odds of Tangible Harm (Public): {odds_public:.4f}")
    print(f"Odds Ratio (Private vs Public for Tangible Harm): {odds_ratio:.4f}")
else:
    print("\nCould not calculate Odds Ratio due to missing categories in table.")

# 9. Visualization
contingency_table.plot(kind='bar', stacked=True)
plt.title('Harm Category Distribution by Sector')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Contingency Table (Sector vs. Harm) ---
Harm_Category  Intangible/Risk  Tangible
Sector_Group                            
Private                     93        48
Public                      29        21

Chi-square statistic: 0.6974
p-value: 0.4037

Odds of Tangible Harm (Private): 0.5161
Odds of Tangible Harm (Public): 0.7241
Odds Ratio (Private vs Public for Tangible Harm): 0.7127


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Stacked Bar Chart.
*   **Purpose:** This plot is designed to compare the total count of incidents across two different sectors ("Private" and "Public") while simultaneously visualizing the composition of these counts broken down by "Harm_Category" ("Intangible/Risk" vs. "Tangible").

### 2. Axes
*   **X-axis:**
    *   **Title:** "Sector_Group"
    *   **Labels:** Two distinct categories: "Private" and "Public".
*   **Y-axis:**
    *   **Title:** "Count"
    *   **Range:** The axis ranges from 0 to approximately 145 (the highest tick mark is 140, but the data extends slightly beyond).
    *   **Units:** Integer counts representing frequency.

### 3. Data Trends
*   **Private Sector:**
    *   This is the **tallest bar**, indicating a significantly higher total count compared to the Public sector.
    *   The total count appears to be slightly above **140**.
    *   The "Intangible/Risk" (blue) segment is the dominant component, reaching a count of approximately **90-95**.
    *   The "Tangible" (orange) segment sits on top, contributing roughly **45-50** to the count.
*   **Public Sector:**
    *   This is the **shortest bar**, with a total count significantly lower than the Private sector.
    *   The total count sits exactly at the **50** mark.
    *   The "Intangible/Risk" (blue) segment accounts for approximately **30** of the total count.
    *   The "Tangible" (orange) segment accounts for the remaining **20**.

### 4. Annotations and Legends
*   **Chart Title:** "Harm Category Distribution by Sector" is displayed at the top center.
*   **Legend:** Located in the top-right corner, titled "Harm_Category".
    *   **Blue:** Represents "Intangible/Risk".
    *   **Orange:** Represents "Tangible".

### 5. Statistical Insights
*   **Sector Disparity:** There is a substantial disparity in volume between the sectors. The Private sector accounts for nearly three times the number of recorded incidents compared to the Public sector (approx. 140 vs. 50).
*   **Predominance of Intangible Risk:** In both sectors, "Intangible/Risk" harms are more frequent than "Tangible" harms.
    *   In the **Private sector**, Intangible/Risk comprises roughly **65-70%** of the total.
    *   In the **Public sector**, Intangible/Risk comprises roughly **60%** of the total (30 out of 50).
*   **Consistency:** The proportional distribution of harm types is relatively consistent across sectors (with Intangible/Risk being the majority), despite the large difference in absolute volume.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
