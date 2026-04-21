# Experiment 91: node_4_29

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_29` |
| **ID in Run** | 91 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-21T00:29:56.256540+00:00 |
| **Runtime** | 267.5s |
| **Parent** | `node_3_6` |
| **Children** | None |
| **Creation Index** | 92 |

---

## Hypothesis

> Regulatory frameworks (e.g., EU AI Act) generate a significantly higher
proportion of 'Trust Readiness' requirements compared to technical frameworks
(e.g., OWASP Top 10 LLM), which skew towards 'Integration Readiness'.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9435 (Definitely True) |
| **Posterior** | 0.4277 (Maybe False) |
| **Surprise** | -0.5987 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 25.0 |
| Maybe True | 5.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 90.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Determine if the source of a governance mandate predicts its domain classification (Trust vs. Integration).

### Steps
- 1. Load the dataset 'step2_crosswalk_matrix.csv' (ensure to look in the current directory).
- 2. Group the data by the 'source' column (e.g., EU AI Act, NIST AI RMF, OWASP, NIST GenAI).
- 3. For each source, calculate the counts of 'Trust Readiness' vs. 'Integration Readiness' in the 'bundle' column.
- 4. Create a contingency table (Source x Bundle) showing these counts.
- 5. Perform a Chi-Square Test of Independence (or Fisher's Exact Test if counts are low) to assess if the distribution of bundles differs significantly across sources.

### Deliverables
- Contingency table (counts of Trust vs Integration per Source) and Chi-Square test results (statistic and p-value).

---

## Analysis

The experiment successfully tested the hypothesis that regulatory frameworks
generate more 'Trust Readiness' requirements while technical frameworks skew
towards 'Integration Readiness'.

1. **Data Loading**: The dataset `step2_crosswalk_matrix.csv` was successfully
loaded.
2. **Contingency Table**: The analysis mapped 42 requirements across four
sources:
   - **EU AI Act**: 5 Trust, 4 Integration (Balanced, slight Trust lean).
   - **NIST AI RMF 1.0**: 11 Trust, 8 Integration (Broadest coverage, slight
Trust lean).
   - **NIST GenAI Profile**: 2 Trust, 2 Integration (Perfectly balanced, low
count).
   - **OWASP Top 10 LLM**: 1 Trust, 9 Integration (Heavily skewed toward
Integration).
3. **Statistical Test**: A Chi-Square Test of Independence yielded a p-value of
**0.0834**.
   - Since 0.0834 > 0.05, we **fail to reject the null hypothesis** at the
standard 5% significance level.
   - However, the result is marginally significant (p < 0.10), suggesting a
trend.
4. **Conclusion**: While the OWASP Top 10 LLM shows a distinct visual skew
toward Integration Readiness (90%), the overall distribution across all
frameworks does not statistically prove that the *source* dictates the *bundle*
classification with high confidence (p=0.08). The hypothesis is **not
supported** statistically, although the visual trend for OWASP is consistent
with the prediction.

---

## Review

The experiment was successfully executed and the analysis is faithful to the
generated data. The results lead to the rejection of the hypothesis that
regulatory frameworks exclusively drive 'Trust Readiness' while technical
frameworks drive 'Integration Readiness'.

**Findings:**
1. **Technical Frameworks:** The hypothesis held true for the technical
framework (OWASP Top 10 LLM), which showed a strong skew toward Integration
Readiness (90% Integration, 10% Trust).
2. **Regulatory Frameworks:** The hypothesis failed for regulatory frameworks
(EU AI Act, NIST AI RMF), which were unexpectedly balanced (EU AI Act: 44%
Integration / 56% Trust). Rather than being pure 'Trust' mandates, these
frameworks impose significant architectural/integration requirements.
3. **Statistical Significance:** The Chi-Square test (p=0.0834) indicates no
statistically significant association between the source and the bundle at the
p<0.05 level, suggesting that modern AI governance frameworks are 'dual-nature,'
bridging the gap between high-level trust principles and technical
implementation controls.

---

## Code

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import os

def load_data(filename):
    # Try current directory first
    if os.path.exists(filename):
        return pd.read_csv(filename)
    # Try parent directory
    elif os.path.exists(f"../{filename}"):
        return pd.read_csv(f"../{filename}")
    else:
        raise FileNotFoundError(f"{filename} not found in current or parent directory.")

try:
    # 1. Load Data
    df = load_data('step2_crosswalk_matrix.csv')
    print(f"Dataset loaded. Shape: {df.shape}")

    # 2. Group and Create Contingency Table
    # We want to see how 'source' relates to 'bundle'
    contingency = pd.crosstab(df['source'], df['bundle'])
    print("\nContingency Table (Source vs Bundle):")
    print(contingency)

    # 3. Statistical Test (Chi-Square)
    chi2, p, dof, expected = chi2_contingency(contingency)
    print("\n--- Chi-Square Test of Independence ---")
    print(f"Chi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.6f}")
    print(f"Degrees of Freedom: {dof}")
    
    alpha = 0.05
    if p < alpha:
        print("\nResult: Reject Null Hypothesis (Significant Association)")
        print("The source framework significantly influences the distribution of Trust vs Integration requirements.")
    else:
        print("\nResult: Fail to Reject Null Hypothesis (No Significant Association)")

    # 4. Visualizations
    # Heatmap of counts
    plt.figure(figsize=(10, 6))
    sns.heatmap(contingency, annot=True, cmap="YlGnBu", fmt='d')
    plt.title("Heatmap: Governance Source vs. Competency Bundle")
    plt.ylabel("Source Framework")
    plt.xlabel("Readiness Bundle")
    plt.tight_layout()
    plt.show()

    # Normalized Stacked Bar Chart (to see proportions)
    # Normalize by row (Source) to compare proportions
    contingency_norm = contingency.div(contingency.sum(axis=1), axis=0)
    
    ax = contingency_norm.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis', edgecolor='black')
    plt.title("Proportion of Readiness Bundles by Source Framework")
    plt.ylabel("Proportion")
    plt.xlabel("Source Framework")
    plt.legend(title='Bundle', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Dataset loaded. Shape: (42, 24)

Contingency Table (Source vs Bundle):
bundle                 Integration Readiness  Trust Readiness
source                                                       
EU AI Act (2024/1689)                      4                5
NIST AI RMF 1.0                            8               11
NIST GenAI Profile                         2                2
OWASP Top 10 LLM                           9                1

--- Chi-Square Test of Independence ---
Chi-Square Statistic: 6.6643
P-value: 0.083404
Degrees of Freedom: 3

Result: Fail to Reject Null Hypothesis (No Significant Association)


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Heatmap (Annotated).
*   **Purpose:** To visualize the relationship and density of mappings between different "Governance Source Frameworks" and "Readiness Bundles." The color intensity and numerical values indicate the frequency or strength of the association between the row and column variables.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Readiness Bundle"
    *   **Categories:** Two distinct bundles are listed: "Integration Readiness" and "Trust Readiness".
*   **Y-Axis:**
    *   **Label:** "Source Framework"
    *   **Categories:** Four frameworks are listed:
        *   EU AI Act (2024/1689)
        *   NIST AI RMF 1.0
        *   NIST GenAI Profile
        *   OWASP Top 10 LLM
*   **Value Scale (Color Bar):**
    *   Located on the right side of the plot.
    *   **Range:** The color gradient ranges from light yellow (representing low values, ~1) to dark blue (representing high values, ~11).

### 3. Data Trends
*   **Highest Value:** The intersection of **NIST AI RMF 1.0** and **Trust Readiness** has the highest value of **11**, indicated by the darkest blue color.
*   **High Integration Focus:** The **OWASP Top 10 LLM** framework shows a strong skew towards **Integration Readiness** with a value of **9**, while having a very low score for Trust Readiness (1).
*   **Balanced Distribution:** The **EU AI Act (2024/1689)** is relatively balanced across both bundles (4 for Integration, 5 for Trust). Similarly, the **NIST GenAI Profile** is balanced but with lower overall coverage (2 for both).
*   **Dominant Framework:** **NIST AI RMF 1.0** appears to be the most comprehensive source in this context, showing high values for both Integration (8) and Trust (11) readiness.
*   **Lowest Value:** The lowest observed value is **1**, found at the intersection of **OWASP Top 10 LLM** and **Trust Readiness**.

### 4. Annotations and Legends
*   **Title:** "Heatmap: Governance Source vs. Competency Bundle" appears at the top.
*   **Cell Annotations:** Each cell in the grid is annotated with the exact integer value corresponding to that data point (e.g., 4, 5, 8, 11, etc.), providing precise readings alongside the visual color cues.
*   **Color Bar:** A vertical bar on the right serves as a legend for the heat intensity, where yellow represents lower counts and dark blue represents higher counts.

### 5. Statistical Insights
*   **NIST AI RMF 1.0 is the Leading Framework:** With a total count of 19 (8 + 11), the NIST AI RMF 1.0 provides the most significant coverage for the competencies listed, suggesting it is a robust framework for both Integration and Trust readiness.
*   **Specialization of OWASP:** The OWASP Top 10 LLM is highly specialized. Its high score in "Integration Readiness" (9) and minimal score in "Trust Readiness" (1) aligns with OWASP’s traditional focus on technical security implementation and vulnerability management rather than broader governance or trust policies.
*   **Gap Analysis:** The NIST GenAI Profile currently shows the least depth in this specific comparison, with only 4 total points. This might indicate that the profile is either narrower in scope regarding these specific bundles or that the mapping between the profile and these bundles is less direct than the other frameworks.
==================================================

=== Plot Analysis (figure 2) ===
Based on the provided plot image, here is a detailed analysis:

### 1. Plot Type
*   **Type:** 100% Stacked Bar Chart.
*   **Purpose:** The plot is designed to compare the relative distribution (proportion) of two specific categories ("Integration Readiness" and "Trust Readiness") across four different Source Frameworks. Unlike a standard bar chart that compares absolute totals, this chart normalizes all bars to a height of 1.0 (100%) to focus purely on the composition ratio within each framework.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Source Framework"
    *   **Labels:** The axis lists four distinct regulatory or guideline frameworks:
        1.  EU AI Act (2024/1689)
        2.  NIST AI RMF 1.0
        3.  NIST GenAI Profile
        4.  OWASP Top 10 LLM
    *   **Orientation:** The labels are rotated approximately 45 degrees to ensure readability.
*   **Y-Axis:**
    *   **Title:** "Proportion"
    *   **Range:** 0.0 to 1.0 (representing 0% to 100%).
    *   **Ticks:** Intervals of 0.2 (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).

### 3. Data Trends
*   **Comparison of Bundles:**
    *   **EU AI Act (2024/1689):** The split is fairly even but leans slightly towards "Trust Readiness" (Yellow). "Integration Readiness" (Purple) occupies approximately 44% of the bar, while Trust Readiness takes up the remaining 56%.
    *   **NIST AI RMF 1.0:** Similar to the EU AI Act, this framework favors "Trust Readiness." The "Integration Readiness" portion appears slightly smaller here, roughly 42%, leaving about 58% for Trust Readiness.
    *   **NIST GenAI Profile:** This framework demonstrates a perfect or near-perfect balance. The bar is split evenly at the 0.5 mark (50% Integration Readiness, 50% Trust Readiness).
    *   **OWASP Top 10 LLM:** This is the most distinct outlier in the dataset. It is heavily dominated by "Integration Readiness" (Purple), which appears to occupy roughly 90% of the bar, leaving only ~10% for "Trust Readiness."

### 4. Annotations and Legends
*   **Plot Title:** "Proportion of Readiness Bundles by Source Framework"
*   **Legend:** Located in the top right corner, titled **"Bundle"**. It defines the color coding for the bar segments:
    *   **Dark Purple:** Integration Readiness
    *   **Yellow:** Trust Readiness

### 5. Statistical Insights
*   **Divergent Focus:** The plot reveals a significant divergence in focus between the frameworks. The **OWASP Top 10 LLM** is clearly focused on technical implementation/integration (90%), which aligns with OWASP's general focus on specific security vulnerabilities and technical mitigation.
*   **Regulatory Focus on Trust:** The broader regulatory frameworks (**EU AI Act** and **NIST AI RMF**) lean more heavily toward "Trust Readiness." This suggests these frameworks are likely more concerned with governance, ethics, reliability, and policy rather than just technical integration.
*   **Balanced Profile:** The **NIST GenAI Profile** serves as a bridge, giving equal weight to both integration and trust, suggesting it treats technical deployment and safety/trustworthiness as equally critical components of readiness.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
