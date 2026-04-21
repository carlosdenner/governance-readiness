# Experiment 275: node_6_74

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_74` |
| **ID in Run** | 275 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T14:06:46.523072+00:00 |
| **Runtime** | 287.6s |
| **Parent** | `node_5_33` |
| **Children** | None |
| **Creation Index** | 276 |

---

## Hypothesis

> Sector-Specific Failure Modes: The 'Healthcare' sector is significantly more
likely to experience 'Unsafe System Behavior' failures compared to the
'Government' sector, which is more prone to 'Human-Computer Interaction'
failures.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7097 (Likely True) |
| **Posterior** | 0.5165 (Uncertain) |
| **Surprise** | -0.2318 |
| **Surprise Interpretation** | Mild Negative (hypothesis weakened) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 26.0 |
| Uncertain | 4.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 40.0 |
| Maybe False | 20.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Analyze the distribution of technical failure types across Healthcare and Government sectors.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'aiid_incidents'.
- 2. Filter for 'Sector of Deployment' = 'Healthcare' and 'Government'.
- 3. Group '85_Known AI Technical Failure' into categories (e.g., 'Unsafe System Behavior', 'HCI', 'Bad Data').
- 4. Perform a Chi-Square test of independence between Sector and Failure Category.

### Deliverables
- Contingency table of Sector vs. Failure Type; Bar chart; Chi-Square statistics.

---

## Analysis

The experiment successfully executed the analysis but failed to find statistical
support for the hypothesis due to data sparsity and mismatching failure
taxonomies. Analyzing the AI Incident Database (AIID) subset, the code
identified 40 relevant incidents (25 Government, 15 Healthcare). However, the
categorization of failure types revealed that 92.5% (37/40) of these incidents
fell into the 'Other' category, indicating that the hypothesized failure modes
('Unsafe System Behavior' and 'Human-Computer Interaction') are not the primary
classifications used in this dataset (which favors technical descriptors like
'Generalization Failure' or 'Bias'). A Chi-Square test on the available
distributions yielded a p-value of 0.6924, confirming no statistically
significant association between the sectors and the specified failure types.
Consequently, the hypothesis that Healthcare and Government sectors experience
distinct, predictable failure modes of this type is not supported by the current
data.

---

## Review

The experiment was faithfully implemented and successfully adapted to the data
quality challenges identified in the previous step. The code correctly
normalized the unstructured text in the 'Sector of Deployment' and 'Known AI
Technical Failure' columns, allowing for a valid test of the hypothesis. The
analysis of 40 relevant incidents (25 Government, 15 Healthcare) revealed that
the hypothesized failure modes ('Unsafe System Behavior' and 'Human-Computer
Interaction') are not the primary classification terms used in the AI Incident
Database, with 92.5% of incidents falling into the 'Other' category.
Consequently, the Chi-Square test (p=0.69) showed no significant association.
The experiment correctly concludes that the hypothesis is not supported by the
current dataset, highlighting a taxonomy mismatch rather than a definitive
disproof of the theoretical relationship.

---

## Code

```python
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# Normalize 'Sector of Deployment' and 'Known AI Technical Failure' columns
sector_col = [c for c in aiid_df.columns if 'Sector of Deployment' in c][0]
failure_col = [c for c in aiid_df.columns if 'Known AI Technical Failure' in c][0]
aiid_df = aiid_df.rename(columns={sector_col: 'Sector', failure_col: 'Failure'})

# --- 1. Define Sectors ---
def categorize_sector(val):
    if pd.isna(val):
        return None
    val = str(val).lower()
    if 'health' in val:
        return 'Healthcare'
    # Public administration, defense, law enforcement map to Government
    # We exclude 'health' to handle mixed cases by prioritizing healthcare or just distinguishing
    if any(x in val for x in ['public administration', 'defense', 'law enforcement', 'government']):
        return 'Government'
    return None

aiid_df['Derived_Sector'] = aiid_df['Sector'].apply(categorize_sector)

# Filter for target sectors
target_df = aiid_df[aiid_df['Derived_Sector'].isin(['Healthcare', 'Government'])].copy()
print(f"Rows found for Healthcare/Government: {len(target_df)}")
print(target_df['Derived_Sector'].value_counts())

# --- 2. Categorize Failures ---
def categorize_failure(val):
    if pd.isna(val):
        return 'Other'
    val = str(val).lower()
    
    # Define keywords
    unsafe_keywords = ['unsafe', 'control', 'robustness', 'reliability', 'system behavior']
    hci_keywords = ['human', 'operator', 'interaction', 'user', 'mistake', 'hci']
    
    is_unsafe = any(k in val for k in unsafe_keywords)
    is_hci = any(k in val for k in hci_keywords)
    
    if is_unsafe and is_hci:
        return 'Both'
    elif is_unsafe:
        return 'Unsafe System Behavior'
    elif is_hci:
        return 'Human-Computer Interaction'
    else:
        return 'Other'

target_df['Failure_Category'] = target_df['Failure'].apply(categorize_failure)

print("\n--- Failure Category Distribution ---")
print(target_df['Failure_Category'].value_counts())

# --- 3. Statistical Analysis ---
# We focus on the hypothesis: Healthcare -> Unsafe, Government -> HCI
# We will filter for just these two failure types to see the direct trade-off, 
# or use the full table to see independence.
# Let's use 'Unsafe System Behavior' and 'Human-Computer Interaction' categories.

analysis_df = target_df[target_df['Failure_Category'].isin(['Unsafe System Behavior', 'Human-Computer Interaction'])].copy()

if len(analysis_df) < 5:
    print("\nInsufficient data for specific failure comparison. showing full contingency.")
    contingency = pd.crosstab(target_df['Derived_Sector'], target_df['Failure_Category'])
else:
    contingency = pd.crosstab(analysis_df['Derived_Sector'], analysis_df['Failure_Category'])

print("\n--- Contingency Table (Target Categories) ---")
print(contingency)

# Chi-Square Test
if contingency.size > 0 and contingency.sum().sum() > 5:
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test Results:")
    print(f"Chi2: {chi2:.4f}, p-value: {p:.4e}")
    
    # Visualization
    plt.figure(figsize=(8, 6))
    # Calculate row-wise percentages for better comparison
    props = contingency.div(contingency.sum(axis=1), axis=0).reset_index()
    props_melted = props.melt(id_vars='Derived_Sector', var_name='Failure Type', value_name='Proportion')
    
    sns.barplot(data=props_melted, x='Derived_Sector', y='Proportion', hue='Failure Type')
    plt.title('Comparison of Failure Types by Sector')
    plt.ylabel('Proportion of Incidents (within filtered types)')
    plt.show()
else:
    print("Not enough data for statistical test.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Rows found for Healthcare/Government: 40
Derived_Sector
Government    25
Healthcare    15
Name: count, dtype: int64

--- Failure Category Distribution ---
Failure_Category
Other                         37
Unsafe System Behavior         2
Human-Computer Interaction     1
Name: count, dtype: int64

Insufficient data for specific failure comparison. showing full contingency.

--- Contingency Table (Target Categories) ---
Failure_Category  Human-Computer Interaction  Other  Unsafe System Behavior
Derived_Sector                                                             
Government                                 1     23                       1
Healthcare                                 0     14                       1

Chi-Square Test Results:
Chi2: 0.7351, p-value: 6.9242e-01


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Grouped Bar Chart (or Clustered Bar Plot).
*   **Purpose:** This chart is designed to compare the relative proportions of different incident failure types across two distinct sectors: Government and Healthcare. It allows for comparison between the sectors as well as the distribution of failure types within each sector.

### 2. Axes
*   **X-axis:**
    *   **Label:** `Derived_Sector`
    *   **Categories:** Represents two distinct sectors: "Government" and "Healthcare".
*   **Y-axis:**
    *   **Label:** `Proportion of Incidents (within filtered types)`
    *   **Range:** The numerical markers range from **0.0 to 0.8**, though the data extends slightly above **0.9**.
    *   **Units:** The values represent a proportion (frequency ratio), where 1.0 would equal 100%.

### 3. Data Trends
*   **Dominant Category:** The **"Other"** failure type (orange bars) is overwhelmingly the most common in both sectors, representing over 90% (approx. 0.92–0.94) of the incidents.
*   **Government Sector:**
    *   The "Other" category is the tallest bar (~0.92).
    *   "Human-Computer Interaction" (blue) and "Unsafe System Behavior" (green) are nearly equal, both showing very low proportions (estimated around 0.04 or 4%).
*   **Healthcare Sector:**
    *   The "Other" category is even more dominant here than in the Government sector.
    *   **"Human-Computer Interaction" is notably absent** (the blue bar is missing or effectively zero).
    *   "Unsafe System Behavior" (green) is present and appears slightly higher than in the Government sector (estimated around 0.06 or 6%).

### 4. Annotations and Legends
*   **Chart Title:** "Comparison of Failure Types by Sector" located at the top center.
*   **Legend:** Located in the upper right corner with the title **"Failure Type"**. It maps colors to categories:
    *   **Blue:** Human-Computer Interaction
    *   **Orange:** Other
    *   **Green:** Unsafe System Behavior

### 5. Statistical Insights
*   **Prevalence of Unspecified Failures:** The vast majority of incidents in this dataset fall into the "Other" category. This suggests that specific technical failure definitions like "Human-Computer Interaction" or "Unsafe System Behavior" account for a very small minority of cases, or that the data has a high rate of unclassifiable incidents.
*   **Sector Differences:**
    *   **Healthcare** appears to have a slightly higher rate of "Unsafe System Behavior" compared to the Government sector.
    *   **Government** faces issues with "Human-Computer Interaction," whereas this failure type is not recorded or is negligible in the Healthcare sector data presented here.
*   **Data Skew:** The distribution is heavily skewed. Any analysis focused specifically on "Human-Computer Interaction" or "Unsafe System Behavior" would be dealing with less than 10% of the total dataset shown in this plot.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
