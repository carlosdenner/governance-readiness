# Experiment 277: node_5_98

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_98` |
| **ID in Run** | 277 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T14:11:36.916795+00:00 |
| **Runtime** | 514.6s |
| **Parent** | `node_4_19` |
| **Children** | None |
| **Creation Index** | 278 |

---

## Hypothesis

> In real-world AI incidents, higher levels of system autonomy (High) are
significantly associated with 'Physical' harm domains, whereas lower autonomy
levels (Low/Medium) are associated with 'Non-Physical' harm domains (e.g.,
Economic, Consumer, Intangible).

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7500 (Likely True) |
| **Posterior** | 0.3159 (Maybe False) |
| **Surprise** | -0.5209 |
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
| Maybe False | 22.0 |
| Definitely False | 38.0 |

---

## Experiment Plan

**Objective:** Investigate the relationship between AI Autonomy Level and the nature of the Harm Domain (Physical vs. Non-Physical) using the AIID dataset.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter rows where 'source_table' is 'aiid_incidents'.
- 2. Clean the 'Autonomy Level' column: Map 'Autonomy1' to 'Low', 'Autonomy2' to 'Medium', and 'Autonomy3' to 'High'. Drop rows with 'unclear', null, or other values.
- 3. Clean the 'Harm Domain' column: Create a new binary category 'Harm_Type'. If 'Harm Domain' contains the string 'Physical' or 'Safety' (case-insensitive), classify as 'Physical'. Classify all other known harm domains as 'Non-Physical'. Drop rows where 'Harm Domain' is null or 'unclear'.
- 4. Generate a Contingency Table: Rows = Autonomy Level (Low, Medium, High), Columns = Harm_Type (Physical, Non-Physical).
- 5. Perform a Chi-Square Test of Independence on the contingency table.
- 6. Calculate Cramer's V to measure the effect size.
- 7. Print the unique values of 'Harm Domain' found, the Contingency Table, the Chi-Square statistic, p-value, and Cramer's V.

### Deliverables
- 1. List of unique 'Harm Domain' values used for categorization.
- 2. Contingency Table (Autonomy vs. Harm Type).
- 3. Chi-Square Test Results (Statistic, p-value).
- 4. Cramer's V Effect Size.

---

## Analysis

The experiment successfully analyzed 185 AIID incidents with valid autonomy
levels to test the hypothesis that higher autonomy correlates with physical
harm. Due to the lack of a structured 'Harm Type' column, a text-mining approach
was successfully implemented to classify incidents as 'Physical' (keywords:
death, injury, safety, etc.) or 'Non-Physical' based on their titles and
descriptions.

1.  **Hypothesis Rejection**: The Chi-Square Test of Independence yielded a
p-value of **0.9390**, which is far above the significance threshold of 0.05.
Consequently, the null hypothesis cannot be rejected.
2.  **Uniform Risk Profile**: The analysis revealed a remarkably consistent rate
of physical harm across all autonomy levels: Low (20%), Medium (22%), and High
(19%). This suggests that in the current dataset, the level of system autonomy
is not a discriminator for the type of harm (physical vs. non-physical) caused.
3.  **Data Distribution**: The dataset is skewed toward 'Low' autonomy systems
(n=105) compared to 'High' (n=53) and 'Medium' (n=27), reflecting the prevalence
of recommender/classifier systems over fully autonomous agents in the incident
database.

---

## Review

The experiment was successfully executed. The initial plan to use the structured
'Harm Domain' column was correctly identified as unviable due to data quality
issues (boolean values instead of categories). The programmer successfully
adapted by implementing a robust text-mining approach using keywords (e.g.,
'death', 'injury', 'safety') on incident titles and descriptions to derive the
'Harm_Type' variable. The statistical analysis (Chi-Square Test and Cramer's V)
was performed correctly on the 185 valid records.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    with np.errstate(divide='ignore', invalid='ignore'):
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        if min((kcorr-1), (rcorr-1)) == 0:
             return 0.0
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

print("Starting analysis...")

# 1. Load Data
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    aiid = df[df['source_table'] == 'aiid_incidents'].copy()
    print(f"Loaded {len(aiid)} AIID incidents.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# 2. Clean Autonomy Level
autonomy_map = {
    'Autonomy1': 'Low',
    'Autonomy2': 'Medium',
    'Autonomy3': 'High'
}
aiid['Autonomy_Clean'] = aiid['Autonomy Level'].map(autonomy_map)

# Drop rows with unknown autonomy
aiid_clean = aiid.dropna(subset=['Autonomy_Clean']).copy()
print(f"Rows after cleaning Autonomy: {len(aiid_clean)}")

# 3. Derive Harm Type from Text (Title + Description)
# Fill NaNs with empty string
aiid_clean['text_corpus'] = aiid_clean['title'].fillna('') + " " + aiid_clean['description'].fillna('')
aiid_clean['text_corpus'] = aiid_clean['text_corpus'].str.lower()

# Define keywords for Physical Harm
physical_keywords = [
    'death', 'dead', 'die', 'kill', 'fatal', 'mortality', 
    'injury', 'injured', 'hurt', 'wound', 'harm', 
    'crash', 'accident', 'collision', 'hit', 'struck',
    'safety', 'physical', 'violence', 'assault', 'attack',
    'medical', 'health', 'patient', 'hospital', 'surgery'
]

# Regex pattern: word boundaries to avoid partial matches (e.g. 'timeline' matching 'die' if not careful, though 'die' is short)
# specific check: using word boundaries for short words
pattern = '|'.join([f'\\b{w}\\b' for w in physical_keywords])

def categorize_harm(text):
    if pd.isna(text) or text.strip() == '':
        return 'Non-Physical' # Default if no info
    # Check for keywords
    if pd.Series(text).str.contains(pattern, regex=True).any():
        return 'Physical'
    return 'Non-Physical'

# Vectorized apply is faster for regex
aiid_clean['Harm_Type'] = np.where(aiid_clean['text_corpus'].str.contains(pattern, regex=True), 'Physical', 'Non-Physical')

# 4. Analysis
# Contingency Table
contingency_table = pd.crosstab(aiid_clean['Autonomy_Clean'], aiid_clean['Harm_Type'])

# Reorder for ordinality
order = ['Low', 'Medium', 'High']
contingency_table = contingency_table.reindex(order)

print("\n--- Contingency Table: Autonomy Level vs. Derived Harm Type ---")
print(contingency_table)

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)
v = cramers_v(contingency_table)

print("\n--- Statistical Results ---")
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")
print(f"Cramer's V: {v:.4f}")

if p < 0.05:
    print("Conclusion: Statistically significant association found.")
else:
    print("Conclusion: No statistically significant association found.")

# Calculate Proportions for better interpretation
props = contingency_table.div(contingency_table.sum(axis=1), axis=0)
print("\n--- Proportions of Physical Harm by Autonomy ---")
print(props)

# Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Heatmap of AI Autonomy Level vs. Physical Harm')
plt.ylabel('Autonomy Level')
plt.xlabel('Harm Type')
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting analysis...
Loaded 1362 AIID incidents.
Rows after cleaning Autonomy: 185

--- Contingency Table: Autonomy Level vs. Derived Harm Type ---
Harm_Type       Non-Physical  Physical
Autonomy_Clean                        
Low                       84        21
Medium                    21         6
High                      43        10

--- Statistical Results ---
Chi-Square Statistic: 0.1258
P-value: 9.3904e-01
Cramer's V: 0.0000
Conclusion: No statistically significant association found.

--- Proportions of Physical Harm by Autonomy ---
Harm_Type       Non-Physical  Physical
Autonomy_Clean                        
Low                 0.800000  0.200000
Medium              0.777778  0.222222
High                0.811321  0.188679


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Heatmap (specifically a contingency table visualization).
*   **Purpose:** To visualize the frequency distribution and relationship between two categorical variables: "Autonomy Level" and "Harm Type." The color intensity helps identify patterns of concentration (high frequency) and scarcity (low frequency) within the data.

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Label:** "Harm Type"
    *   **Categories:** "Non-Physical" and "Physical"
*   **Y-Axis (Vertical):**
    *   **Label:** "Autonomy Level"
    *   **Categories:** "Low", "Medium", and "High"
*   **Color Bar (Z-Axis/Legend):**
    *   **Range:** The scale on the right indicates values ranging roughly from roughly 6 to 84.
    *   **Scale:** A gradient from light yellow (representing low counts) to dark red (representing high counts).

### 3. Data Trends
*   **Highest Value (Hotspot):** The intersection of **Low Autonomy Level** and **Non-Physical Harm** contains the highest count (**84**), indicated by the dark maroon color. This suggests that the vast majority of recorded incidents in this dataset involve low-autonomy AI causing non-physical harm.
*   **Lowest Value:** The intersection of **Medium Autonomy Level** and **Physical Harm** has the lowest count (**6**), indicated by the palest yellow color.
*   **Dominant Column:** The "Non-Physical" column has significantly higher values across all autonomy levels (84, 21, 43) compared to the "Physical" column (21, 6, 10).
*   **Autonomy Pattern:** Regardless of harm type, the "Low" autonomy category has the highest counts, followed by "High", with "Medium" having the lowest representation in the dataset.

### 4. Annotations and Legends
*   **Title:** "Heatmap of AI Autonomy Level vs. Physical Harm" appears at the top.
*   **Cell Annotations:** Each cell contains a specific integer representing the count for that intersection:
    *   Low / Non-Physical: 84
    *   Low / Physical: 21
    *   Medium / Non-Physical: 21
    *   Medium / Physical: 6
    *   High / Non-Physical: 43
    *   High / Physical: 10
*   **Color Bar:** Located on the right side to interpret the color intensity.

### 5. Statistical Insights
*   **Prevalence of Non-Physical Harm:** The data is heavily skewed toward non-physical harm. Summing the columns shows **148** non-physical incidents vs. only **37** physical incidents.
*   **Frequency by Autonomy:** The dataset contains the most records for "Low" autonomy systems (Total: 105), followed by "High" (Total: 53), and lastly "Medium" (Total: 27). This suggests either that low-autonomy systems are more common or that they are reported more frequently in this specific context.
*   **Consistent Proportions:** Interestingly, while the raw numbers vary, the ratio of physical to non-physical harm is relatively consistent across all three autonomy levels (roughly 1 physical incident for every 4 to 5 non-physical incidents).
    *   Low: ~20% Physical (21/105)
    *   Medium: ~22% Physical (6/27)
    *   High: ~19% Physical (10/53)
    *   This implies that, in this dataset, the **likelihood of physical harm occurring does not drastically increase or decrease based on the AI's autonomy level**; rather, the volume of incidents tracks with the total number of systems/reports in each category.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
