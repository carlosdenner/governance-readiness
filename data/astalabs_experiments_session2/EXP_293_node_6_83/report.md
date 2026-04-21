# Experiment 293: node_6_83

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_83` |
| **ID in Run** | 293 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T15:04:26.775535+00:00 |
| **Runtime** | 307.7s |
| **Parent** | `node_5_44` |
| **Children** | None |
| **Creation Index** | 294 |

---

## Hypothesis

> Commercial Opacity: In federal AI deployments (EO 13960), systems procured from
commercial vendors are statistically less likely to grant government agencies
access to source code compared to government-developed systems.

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

**Objective:** Compare code access rates between commercial and government-developed AI systems.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'source_table' = 'eo13960_scored'.
- 2. Clean '10_commercial_ai' column into binary categories: 'Commercial' (e.g., 'Yes') and 'Government/Custom' (e.g., 'No').
- 3. Clean '38_code_access' column into binary categories: 'Access' (e.g., 'Yes') and 'No Access' (e.g., 'No').
- 4. Generate a contingency table.
- 5. Perform a Chi-square test of independence.

### Deliverables
- Contingency table, Chi-square statistic, p-value, and interpretation.

---

## Analysis

The experiment was successfully executed and the hypothesis was strongly
supported. After refining the data cleaning logic to correctly identify
'Developed with contracting resources.' as the commercial proxy, the analysis
included 781 federal AI use cases (390 Commercial/Contractor, 391 Government/In-
House). The results revealed a dramatic disparity in transparency: Government-
developed systems granted source code access in 90.5% of cases, whereas
Commercial/Contractor systems granted access in only 33.8% of cases. The Chi-
square test confirmed this difference is highly statistically significant (p <
0.001, Chi2=264.6). This empirical evidence validates the 'Commercial Opacity'
hypothesis, demonstrating that outsourcing AI development significantly reduces
the likelihood of federal agencies retaining access to the underlying model
code.

---

## Review

The experiment was successfully executed and the hypothesis was strongly
supported. After refining the data cleaning logic to correctly identify
'Developed with contracting resources' as the commercial proxy, the analysis
included 781 federal AI use cases (390 Commercial/Contractor, 391 Government/In-
House). The results revealed a dramatic disparity in transparency: Government-
developed systems granted source code access in 90.5% of cases, whereas
Commercial/Contractor systems granted access in only 33.8% of cases. The Chi-
square test confirmed this difference is highly statistically significant (p <
0.001, Chi2=264.6). This empirical evidence validates the 'Commercial Opacity'
hypothesis, demonstrating that outsourcing AI development significantly reduces
the likelihood of federal agencies retaining access to the underlying model
code.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"EO 13960 Data Shape: {eo_data.shape}")

# Target columns
dev_col = '22_dev_method'
access_col = '38_code_access'

# Check if columns exist
if dev_col not in eo_data.columns or access_col not in eo_data.columns:
    print(f"Columns '{dev_col}' or '{access_col}' not found.")
else:
    # 1. Clean Development Method
    def clean_dev_method(val):
        if pd.isna(val):
            return None
        v = str(val).lower()
        # Updated mapping based on dataset values
        if 'contracting resources' in v:
            return 'Commercial/Contractor'
        if 'in-house' in v and 'contracting' not in v: # Strict in-house
            return 'Government/In-House'
        return None

    eo_data['procurement_type'] = eo_data[dev_col].apply(clean_dev_method)

    # 2. Clean Code Access
    def clean_access(val):
        if pd.isna(val):
            return None
        v = str(val).lower().strip()
        if v.startswith('no') or 'not have access' in v:
            return 'No Access'
        if 'yes' in v:
            return 'Access Granted'
        return None

    eo_data['code_access_status'] = eo_data[access_col].apply(clean_access)

    # Filter dataset for analysis
    analysis_df = eo_data.dropna(subset=['procurement_type', 'code_access_status'])

    print(f"\nRows available for analysis: {len(analysis_df)}")
    print(f"Breakdown by Procurement Type:\n{analysis_df['procurement_type'].value_counts()}")

    if len(analysis_df['procurement_type'].unique()) > 1:
        # Contingency Table
        contingency_table = pd.crosstab(analysis_df['procurement_type'], analysis_df['code_access_status'])
        print("\nContingency Table (Count):")
        print(contingency_table)
        
        # Percentage Table
        contingency_pct = pd.crosstab(analysis_df['procurement_type'], analysis_df['code_access_status'], normalize='index') * 100
        print("\nContingency Table (Percentage):")
        print(contingency_pct.round(2))

        # Chi-square test
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"\nChi-Square Test Results:")
        print(f"Chi2 Stat: {chi2:.4f}")
        print(f"P-value: {p:.4e}")
        
        # Interpretation
        alpha = 0.05
        if p < alpha:
            print("\nResult: Statistically significant relationship found.")
            # Check direction
            try:
                comm_access = contingency_pct.loc['Commercial/Contractor', 'Access Granted']
                gov_access = contingency_pct.loc['Government/In-House', 'Access Granted']
                print(f"Commercial/Contractor Access Rate: {comm_access:.2f}%")
                print(f"Government/In-House Access Rate: {gov_access:.2f}%")
                
                if comm_access < gov_access:
                    print(f"Conclusion: Commercial systems are significantly LESS likely to grant access ({comm_access:.1f}% vs {gov_access:.1f}%), supporting the hypothesis.")
                else:
                    print(f"Conclusion: Commercial systems are MORE or EQUALLY likely to grant access, rejecting the hypothesis direction.")
            except KeyError:
                print("Could not determine directionality due to missing keys in pivot.")
        else:
            print("\nResult: No statistically significant relationship found.")
            
        # Visualization
        plt.figure(figsize=(10, 6))
        sns.heatmap(contingency_pct, annot=True, fmt='.1f', cmap='RdBu', cbar_kws={'label': 'Percentage'})
        plt.title('Code Access: Commercial vs. In-House Development')
        plt.ylabel('Procurement Type')
        plt.xlabel('Code Access Status')
        plt.show()
    else:
        print("Insufficient data: Only one procurement type found after filtering.")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: EO 13960 Data Shape: (1757, 196)

Rows available for analysis: 781
Breakdown by Procurement Type:
procurement_type
Government/In-House      391
Commercial/Contractor    390
Name: count, dtype: int64

Contingency Table (Count):
code_access_status     Access Granted  No Access
procurement_type                                
Commercial/Contractor             132        258
Government/In-House               354         37

Contingency Table (Percentage):
code_access_status     Access Granted  No Access
procurement_type                                
Commercial/Contractor           33.85      66.15
Government/In-House             90.54       9.46

Chi-Square Test Results:
Chi2 Stat: 264.5628
P-value: 1.7376e-59

Result: Statistically significant relationship found.
Commercial/Contractor Access Rate: 33.85%
Government/In-House Access Rate: 90.54%
Conclusion: Commercial systems are significantly LESS likely to grant access (33.8% vs 90.5%), supporting the hypothesis.


=== Plot Analysis (figure 1) ===
Based on the analysis of the provided plot image, here are the detailed findings:

### 1. Plot Type
*   **Type:** Heatmap (specifically a contingency table visualized as a heatmap).
*   **Purpose:** The plot visualizes the relationship between "Procurement Type" and "Code Access Status." It uses color intensity and numerical annotations to represent the percentage distribution within these categories, facilitating a quick comparison of code accessibility between commercial and in-house development.

### 2. Axes
*   **Y-Axis (Vertical):**
    *   **Label:** "Procurement Type"
    *   **Categories:** "Commercial/Contractor" (top) and "Government/In-House" (bottom).
*   **X-Axis (Horizontal):**
    *   **Label:** "Code Access Status"
    *   **Categories:** "Access Granted" (left) and "No Access" (right).
*   **Value Ranges:** The axes represent categorical data. However, the color scale (z-axis) represents percentages, ranging from approximately **10% to 90%**.

### 3. Data Trends
*   **High Values:** The highest value is found in the **Government/In-House** row under **Access Granted** at **90.5%**, indicated by a dark blue color.
*   **Low Values:** The lowest value is found in the **Government/In-House** row under **No Access** at **9.5%**, indicated by a dark red color.
*   **Commercial Trends:** For Commercial/Contractor projects, the trend leans toward restriction. **No Access (66.2%)** is nearly double the rate of **Access Granted (33.8%)**.
*   **In-House Trends:** For Government/In-House projects, the trend is overwhelmingly open. **Access Granted (90.5%)** dominates **No Access (9.5%)**.

### 4. Annotations and Legends
*   **Title:** "Code Access: Commercial vs. In-House Development" – clearly defines the subject matter.
*   **Color Bar (Legend):** Located on the right side, labeled "Percentage." It uses a diverging color palette:
    *   **Dark Red:** Low percentage (~10%).
    *   **White/Light Colors:** Mid-range percentages (~50%).
    *   **Dark Blue:** High percentage (~90%).
*   **Cell Annotations:** Each cell contains the specific percentage value (e.g., 33.8, 66.2, 90.5, 9.5), providing precise data points alongside the visual color cues.

### 5. Statistical Insights
*   **Stark Contrast in Transparency:** There is a significant disparity in code availability depending on the procurement method. **Government/In-House** development creates an environment where code is almost always accessible (over 9 out of 10 times).
*   **Barriers in Outsourcing:** **Commercial/Contractor** development presents a major barrier to code access. In approximately two-thirds of these cases (66.2%), the code remains inaccessible.
*   **Inverse Relationship:** The data shows an almost inverse relationship between the two procurement types. While in-house development grants access ~90% of the time, commercial development denies it ~66% of the time.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
