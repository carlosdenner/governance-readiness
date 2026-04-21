# Experiment 141: node_6_11

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_11` |
| **ID in Run** | 141 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T07:44:35.198816+00:00 |
| **Runtime** | 245.1s |
| **Parent** | `node_5_16` |
| **Children** | None |
| **Creation Index** | 142 |

---

## Hypothesis

> Generative AI & Reputation Risk: Incidents involving 'Generative AI'
technologies are significantly more likely to result in 'Reputational Harm'
compared to incidents involving 'Discriminative/Predictive AI'.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.6531 (Maybe True) |
| **Surprise** | -0.1066 |
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
| Maybe True | 0.0 |
| Uncertain | 18.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Test if the failure modes of Generative AI (hallucinations, deepfakes) disproportionately damage reputation compared to traditional AI.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' (AIID subset).
- 2. Create a 'Technology Type' variable by searching keywords (e.g., 'LLM', 'Generative', 'Diffusion', 'GPT') in `Known AI Technology` or `description`. Classify others as 'Discriminative/Predictive'.
- 3. Create a 'Reputational Harm' variable by searching keywords (e.g., 'reputation', 'brand', 'embarrassment') in `Harm Domain` or `Tangible Harm`.
- 4. Perform a Chi-square test: Tech Type vs. Reputational Harm.

### Deliverables
- Contingency table, Chi-square statistics, and a heatmap of technology vs. harm type.

---

## Analysis

The experiment was successfully executed but yielded a null result due to data
constraints.

1.  **Classification**: The code successfully segmented the 1,362 incidents into
'Discriminative/Predictive AI' (n=765) and 'Generative AI' (n=597) using the
provided keyword heuristics.
2.  **Zero Matches for Harm**: The keyword search for 'Reputational Harm' (terms
like 'reputation', 'defamation', 'brand damage') returned **zero** matches
across the entire dataset. All 1,362 incidents were classified as 'Other Harm'.
3.  **Statistical Outcome**: Because there was no variance in the dependent
variable (Harm Class), the contingency table was degenerate (2x1), resulting in
a Chi-square statistic of 0.0 and a p-value of 1.0.

**Conclusion**: The hypothesis could not be effectively tested. The data
indicates that either 'Reputational Harm' is not present in this subset of the
AIID, or the terminology used in the 'Harm Domain' and 'Tangible Harm' fields
does not match the keywords provided (e.g., it might be coded under broader
categories like 'Economic' or 'Psychological').

---

## Review

The experiment was faithfully executed according to the plan. The programmer
correctly filtered the AIID dataset, implemented the keyword-based
classification for 'Technology Type' and 'Reputational Harm', and attempted the
statistical analysis.

**Findings:**
1.  **Technology Classification:** The heuristic successfully segmented the data
into **597 Generative AI** and **765 Discriminative/Predictive AI** incidents.
2.  **Harm Classification (Null Result):** The keyword search for 'Reputational
Harm' (e.g., 'reputation', 'brand damage', 'defamation') yielded **zero
matches** across all 1,362 incidents in the target columns (`Harm Domain`,
`Tangible Harm`, `Special Interest Intangible Harm`).
3.  **Hypothesis Test:** Because the dependent variable ('Harm_Class') had no
variance (100% 'Other Harm'), the contingency table was degenerate (2x1),
resulting in a Chi-square statistic of 0.0 and a p-value of 1.0.

**Conclusion:** The hypothesis could not be supported. The result suggests that
'Reputational Harm' is either not a distinct category in the AIID taxonomy or is
not captured by the specified keywords in the available text fields. The
experiment is technically successful in that it correctly identified the absence
of this signal using the defined parameters.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import re

# [debug] Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# --- Feature Engineering: Technology Type ---
# Define keywords for Generative AI
genai_keywords = [
    r'generative', r'llm', r'large language model', r'gpt', r'chatgpt', r'chatbot',
    r'diffusion', r'dall-e', r'midjourney', r'stable diffusion', r'transformer',
    r'bert', r'gan', r'deepfake', r'synthetic', r'text-to-image', r'text-to-video',
    r'bard', r'llama', r'claude', r'copilot', r'gemini'
]

def classify_tech(row):
    # Combine relevant columns for search
    text = str(row.get('Known AI Technology', '')) + " " + str(row.get('Potential AI Technology', '')) + " " + str(row.get('description', ''))
    text = text.lower()
    
    for keyword in genai_keywords:
        if re.search(keyword, text):
            return 'Generative AI'
    return 'Discriminative/Predictive AI'

aiid_df['Technology_Type'] = aiid_df.apply(classify_tech, axis=1)

# --- Feature Engineering: Reputational Harm ---
# Define keywords for Reputational Harm
reputation_keywords = [
    r'reputation', r'reputational', r'defamation', r'libel', r'slander',
    r'brand damage', r'public relation', r'scandal', r'embarrassment', r'discredit'
]

def classify_harm(row):
    # Combine relevant columns for search
    text = str(row.get('Harm Domain', '')) + " " + str(row.get('Tangible Harm', '')) + " " + str(row.get('Special Interest Intangible Harm', ''))
    text = text.lower()
    
    for keyword in reputation_keywords:
        if re.search(keyword, text):
            return 'Reputational'
    return 'Other Harm'

aiid_df['Harm_Class'] = aiid_df.apply(classify_harm, axis=1)

# --- Analysis ---
print(f"Total Incidents Analysis: {len(aiid_df)}")
print("\nDistribution of Technology Type:")
print(aiid_df['Technology_Type'].value_counts())
print("\nDistribution of Harm Class:")
print(aiid_df['Harm_Class'].value_counts())

# Contingency Table
contingency_table = pd.crosstab(aiid_df['Technology_Type'], aiid_df['Harm_Class'])
print("\nContingency Table (Technology vs. Harm):")
print(contingency_table)

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\n--- Chi-Square Test Results ---")
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")
print(f"Degrees of Freedom: {dof}")

# Calculate percentages for interpretation
# Row-wise normalization (Probability of Harm given Tech)
row_props = pd.crosstab(aiid_df['Technology_Type'], aiid_df['Harm_Class'], normalize='index') * 100
print("\nRow Percentages (Propensity for Harm Type by Tech):")
print(row_props)

# --- Visualization ---
plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Heatmap: AI Technology Type vs. Reputational Harm')
plt.xlabel('Harm Category')
plt.ylabel('Technology Type')
plt.tight_layout()
plt.show()

# Stacked Bar Chart for Proportions
ax = row_props.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
plt.title('Proportion of Reputational vs. Other Harm by Technology Type')
plt.ylabel('Percentage')
plt.xlabel('Technology Type')
plt.legend(title='Harm Class', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add labels
for c in ax.containers:
    ax.bar_label(c, fmt='%.1f%%', label_type='center')

plt.tight_layout()
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Total Incidents Analysis: 1362

Distribution of Technology Type:
Technology_Type
Discriminative/Predictive AI    765
Generative AI                   597
Name: count, dtype: int64

Distribution of Harm Class:
Harm_Class
Other Harm    1362
Name: count, dtype: int64

Contingency Table (Technology vs. Harm):
Harm_Class                    Other Harm
Technology_Type                         
Discriminative/Predictive AI         765
Generative AI                        597

--- Chi-Square Test Results ---
Chi-Square Statistic: 0.0000
P-value: 1.0000e+00
Degrees of Freedom: 0

Row Percentages (Propensity for Harm Type by Tech):
Harm_Class                    Other Harm
Technology_Type                         
Discriminative/Predictive AI       100.0
Generative AI                      100.0


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Heatmap.
*   **Purpose:** The plot visualizes the magnitude of values using color intensity. In this specific case, it compares the frequency or count of a specific harm category ("Other Harm") across two different types of AI technologies.

### 2. Axes
*   **Y-Axis (Vertical):**
    *   **Label:** "Technology Type".
    *   **Categories:**
        1.  Discriminative/Predictive AI
        2.  Generative AI
*   **X-Axis (Horizontal):**
    *   **Label:** "Harm Category".
    *   **Categories:** The plot shows a single category labeled "Other Harm".
*   **Color Scale (Z-Axis/Legend):**
    *   The color bar on the right represents the count/value range.
    *   **Range:** Approximately 600 (light yellow) to 760 (dark blue).
    *   **Units:** While not explicitly labeled with a unit (e.g., "incidents," "reports"), the integer values suggest a count of occurrences.

### 3. Data Trends
*   **High Values (Hotspots):** The highest value is found in the top cell corresponding to **Discriminative/Predictive AI**, with a value of **765**. This is represented by a dark blue color.
*   **Low Values:** The lower value is found in the bottom cell corresponding to **Generative AI**, with a value of **597**. This is represented by a light cream/yellow color.
*   **Pattern:** There is a distinct difference in magnitude between the two technology types, with Discriminative/Predictive AI showing a significantly higher association with "Other Harm" than Generative AI.

### 4. Annotations and Legends
*   **Title:** "Heatmap: AI Technology Type vs. Reputational Harm". This sets the context that the data is related to reputational risks or harms.
*   **Cell Annotations:** The exact numerical values are printed within the cells:
    *   Top Cell: **765**
    *   Bottom Cell: **597**
*   **Color Bar:** A gradient bar on the right side indicates that darker blue colors equate to higher numerical values, while lighter yellow colors equate to lower values.

### 5. Statistical Insights
*   **Prevalence:** In the category of "Other Harm" (within the context of Reputational Harm), **Discriminative/Predictive AI** is the dominant category, accounting for **765** instances compared to **597** for Generative AI.
*   **Relative Difference:** Discriminative/Predictive AI has **168 more** recorded instances than Generative AI in this category.
*   **Proportion:** Discriminative/Predictive AI represents approximately **56%** of the total count displayed (Total = 1362), while Generative AI represents roughly **44%**. This suggests that, for this specific dataset and harm category, traditional predictive AI models are currently associated with a higher volume of reported "Other Harm" issues than newer generative models.
==================================================

=== Plot Analysis (figure 2) ===
Based on the provided image, here is the detailed analysis:

**1. Plot Type**
*   **Type:** Vertical Bar Plot.
*   **Purpose:** This plot is designed to compare the percentage distribution of harm classes (specifically "Other Harm" versus "Reputational Harm") across two different categories of Artificial Intelligence technologies.

**2. Axes**
*   **X-Axis:**
    *   **Label:** "Technology Type"
    *   **Categories:** The axis displays two distinct categories: "Discriminative/Predictive AI" and "Generative AI". The labels are oriented vertically for readability.
*   **Y-Axis:**
    *   **Label:** "Percentage"
    *   **Range:** The scale ranges from 0 to 100.
    *   **Units:** Percent (%).

**3. Data Trends**
*   **Pattern:** The data exhibits a completely uniform pattern. There is no variation between the two categories.
*   **Bar Heights:** Both the "Discriminative/Predictive AI" bar and the "Generative AI" bar extend to the maximum value of 100 on the y-axis.
*   **Comparison:** Both technology types show identical results regarding the classification of harm. There are no tallest or shortest bars relative to each other; they are equal.

**4. Annotations and Legends**
*   **Legend:** A legend is positioned to the right of the plot titled "Harm Class." It indicates that the dark purple color represents "Other Harm."
*   **Annotations:** Inside each bar, explicitly centered, is the text annotation "100.0%". This numerical label confirms that the entire proportion of the data for both categories falls into the specific class shown.
*   **Title:** The chart is titled "Proportion of Reputational vs. Other Harm by Technology Type."

**5. Statistical Insights**
*   **Dominance of "Other Harm":** The most significant insight is that for the dataset visualized here, 100% of the recorded harms for both Discriminative/Predictive AI and Generative AI fall into the "Other Harm" category.
*   **Absence of Reputational Harm:** Although the title suggests a comparison between "Reputational" and "Other" harm, the graph indicates that "Reputational Harm" is non-existent (0%) in this specific data slice for these technology types.
*   **Conclusion:** There is no difference in the proportion of harm types between Discriminative/Predictive AI and Generative AI in this context; both exclusively exhibit "Other Harm."
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
