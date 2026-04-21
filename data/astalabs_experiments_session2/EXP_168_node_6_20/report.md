# Experiment 168: node_6_20

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_20` |
| **ID in Run** | 168 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T08:56:57.059796+00:00 |
| **Runtime** | 190.8s |
| **Parent** | `node_5_21` |
| **Children** | None |
| **Creation Index** | 169 |

---

## Hypothesis

> Biometric Risk Profile: AI incidents involving 'Facial Recognition' or
'Biometric' technologies are significantly more likely to involve 'Civil Rights'
violations compared to other AI technologies.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9839 (Definitely True) |
| **Posterior** | 0.9945 (Definitely True) |
| **Surprise** | +0.0128 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 30.0 |
| Maybe True | 0.0 |
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

**Objective:** Analyze the specific harm profile of biometric technologies in the AIID dataset.

### Steps
- 1. Filter for `source_table` = 'aiid_incidents'.
- 2. Create a `Tech_Type` variable: 'Biometric' (if `Known AI Technology` or `Description` contains 'face', 'facial', 'biometric', 'surveillance') vs. 'Other'.
- 3. Create a `Harm_Type` variable: 'Civil Rights' (if `Harm Domain` or `Description` contains 'civil rights', 'discrimination', 'privacy', 'due process') vs. 'Other'.
- 4. Create a contingency table and run a Fisher's Exact Test or Chi-square test.

### Deliverables
- Contingency table and statistical significance of the association.

---

## Analysis

The experiment successfully analyzed 1,362 AIID incidents to test the 'Biometric
Risk Profile' hypothesis.

**Findings:**
- **Biometric/Facial Technologies:** 8.1% of incidents (22 out of 271) involved
Civil Rights violations.
- **Other Technologies:** Only 1.7% of incidents (19 out of 1,091) involved
Civil Rights violations.

**Statistical Analysis:**
- The Chi-square test yielded a statistic of 28.09 and a **p-value of <
0.0001**, indicating a highly statistically significant difference.
- The **Odds Ratio of 4.99** suggests that incidents involving biometric or
facial recognition technology are approximately five times more likely to
involve civil rights concerns (such as discrimination, privacy violations, or
due process issues) compared to other AI technologies.

**Conclusion:**
The hypothesis is **strongly supported**. The data validates that biometric and
facial recognition systems carry a distinct risk profile heavily skewed toward
civil rights harms compared to the broader landscape of AI incidents.

---

## Review

The experiment was successfully executed and the code faithfully implemented the
analysis plan. The text-mining approach correctly categorized incidents based on
relevant keywords in both technology and harm descriptions.

**Hypothesis:** Biometric Risk Profile: AI incidents involving 'Facial
Recognition' or 'Biometric' technologies are significantly more likely to
involve 'Civil Rights' violations compared to other AI technologies.

**Findings:**
1.  **Incidence Rates:** Incidents involving Biometric/Facial technologies had
an 8.1% rate of Civil Rights violations (22/271), whereas other AI technologies
had a significantly lower rate of 1.7% (19/1,091).
2.  **Statistical Significance:** The Chi-square test yielded a statistic of
28.09 and a p-value of < 0.0001, confirming the difference is highly
statistically significant.
3.  **Effect Size:** An Odds Ratio of 4.99 indicates that biometric-related
incidents are approximately 5 times more likely to involve civil rights issues
(discrimination, privacy, due process) than other AI incidents.

**Conclusion:** The hypothesis is **strongly supported**. The analysis confirms
that biometric technologies carry a distinct and elevated risk profile regarding
civil rights harms compared to the general population of AI failures.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
import matplotlib.pyplot as plt

# [debug]
print("Loading dataset...")

try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(aiid)}")

# Identify relevant columns dynamically due to potential prefixes (e.g., '84: Known AI Technology')
tech_cols = [c for c in aiid.columns if 'Known AI Technology' in str(c)]
harm_cols = [c for c in aiid.columns if 'Harm Domain' in str(c)]
desc_cols = [c for c in aiid.columns if 'description' in str(c).lower() or 'summary' in str(c).lower()]

tech_col = tech_cols[0] if tech_cols else None
harm_col = harm_cols[0] if harm_cols else None
desc_col = desc_cols[0] if desc_cols else None

print(f"Using columns -> Tech: {tech_col}, Harm: {harm_col}, Description: {desc_col}")

if not tech_col or not harm_col:
    print("Error: Critical columns 'Known AI Technology' or 'Harm Domain' not found.")
else:
    # Define Keywords
    biometric_keywords = ['face', 'facial', 'biometric', 'surveillance']
    civil_rights_keywords = ['civil rights', 'discrimination', 'privacy', 'due process']

    def check_keywords(text, keywords):
        if pd.isna(text):
            return False
        text = str(text).lower()
        return any(k in text for k in keywords)

    # 1. Feature Engineering: Tech_Type
    # Check technology column
    aiid['is_biometric'] = aiid[tech_col].apply(lambda x: check_keywords(x, biometric_keywords))
    # Check description column if it exists
    if desc_col:
        aiid['is_biometric'] = aiid['is_biometric'] | aiid[desc_col].apply(lambda x: check_keywords(x, biometric_keywords))
    
    aiid['Tech_Type'] = np.where(aiid['is_biometric'], 'Biometric/Facial', 'Other')

    # 2. Feature Engineering: Harm_Type
    # Check harm column
    aiid['is_civil_rights'] = aiid[harm_col].apply(lambda x: check_keywords(x, civil_rights_keywords))
    # Check description column if it exists
    if desc_col:
        aiid['is_civil_rights'] = aiid['is_civil_rights'] | aiid[desc_col].apply(lambda x: check_keywords(x, civil_rights_keywords))
    
    aiid['Harm_Type'] = np.where(aiid['is_civil_rights'], 'Civil Rights', 'Other')

    # 3. Statistical Analysis
    # Create Contingency Table
    contingency = pd.crosstab(aiid['Tech_Type'], aiid['Harm_Type'])
    print("\n--- Contingency Table ---")
    print(contingency)

    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")

    # Calculate Odds Ratio
    # Format:
    #               Civil Rights   Other
    # Biometric     a              b
    # Other         c              d
    if 'Civil Rights' in contingency.columns and 'Biometric/Facial' in contingency.index:
        a = contingency.loc['Biometric/Facial', 'Civil Rights']
        b = contingency.loc['Biometric/Facial', 'Other']
        c = contingency.loc['Other', 'Civil Rights']
        d = contingency.loc['Other', 'Other']
        
        try:
            odds_ratio = (a * d) / (b * c)
            print(f"Odds Ratio: {odds_ratio:.4f}")
        except ZeroDivisionError:
            print("Odds Ratio: Undefined (division by zero)")
    else:
        print("Odds Ratio could not be calculated due to missing categories.")

    # Visualizing
    # Normalize to get percentages for better comparison
    contingency_pct = contingency.div(contingency.sum(1), axis=0) * 100
    
    ax = contingency_pct.plot(kind='bar', stacked=True, color=['#d62728', '#1f77b4'], alpha=0.8)
    plt.title('Harm Distribution: Biometric vs Other Technologies')
    plt.ylabel('Percentage')
    plt.xlabel('Technology Type')
    plt.legend(title='Harm Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset...
AIID Incidents loaded: 1362
Using columns -> Tech: Known AI Technology, Harm: Harm Domain, Description: description

--- Contingency Table ---
Harm_Type         Civil Rights  Other
Tech_Type                            
Biometric/Facial            22    249
Other                       19   1072

Chi-Square Statistic: 28.0868
P-value: 0.0000
Odds Ratio: 4.9850

STDERR:
<ipython-input-1-61f6459b5bcf>:95: Pandas4Warning: Starting with pandas version 4.0 all arguments of sum will be keyword-only.
  contingency_pct = contingency.div(contingency.sum(1), axis=0) * 100


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** 100% Stacked Bar Plot.
*   **Purpose:** This plot is designed to compare the relative proportions of different harm types ("Civil Rights" vs. "Other") across two distinct categories of technology ("Biometric/Facial" vs. "Other"). By stacking the bars to reach 100%, it normalizes the data, allowing for a comparison of distribution percentages rather than absolute counts.

### 2. Axes
*   **X-axis:**
    *   **Title:** "Technology Type".
    *   **Labels:** The axis contains two categorical variables: "Biometric/Facial" and "Other". The labels are oriented vertically (rotated 90 degrees).
*   **Y-axis:**
    *   **Title:** "Percentage".
    *   **Units:** Percent (%).
    *   **Range:** The scale runs from 0 to 100, with tick marks at intervals of 20 (0, 20, 40, 60, 80, 100).

### 3. Data Trends
*   **Overall Pattern:** For both technology categories, the vast majority of harms fall under the "Other" harm type (represented by the blue section).
*   **Biometric/Facial Bar:**
    *   The "Civil Rights" harm (red section) comprises a visible minority of the bar. Visually, it appears to represent approximately 8% to 10% of the total harms in this category.
    *   The "Other" harm type (blue section) makes up the remaining ~90-92%.
*   **Other (Technology) Bar:**
    *   The "Civil Rights" harm (red section) is extremely minimal, appearing as a thin sliver at the bottom. It likely represents less than 2% of the total.
    *   The "Other" harm type (blue section) dominates almost the entirely of the bar, representing close to 98-99%.
*   **Comparison:** The proportion of harms related to Civil Rights is noticeably larger in "Biometric/Facial" technologies compared to "Other" technologies.

### 4. Annotations and Legends
*   **Title:** "Harm Distribution: Biometric vs Other Technologies" – This clearly sets the context of the comparison.
*   **Legend:** A box located to the right of the plot titled "Harm Type" defines the color coding:
    *   **Red:** Represents "Civil Rights" harms.
    *   **Blue:** Represents "Other" types of harms.

### 5. Statistical Insights
*   **Disproportionate Impact:** The data suggests that while Civil Rights violations are not the majority of reported harms for Biometric/Facial recognition technologies, they occur at a significantly higher rate (proportionally) within that sector compared to other general technologies.
*   **Category Dominance:** In both technology sectors, non-civil rights harms (grouped as "Other") are the predominant issue reported, suggesting that while civil rights are a specific concern for biometrics, other issues (likely technical errors, privacy, security, etc.) are more frequent overall.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
