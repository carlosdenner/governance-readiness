# Experiment 183: node_6_25

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_25` |
| **ID in Run** | 183 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T09:44:23.484743+00:00 |
| **Runtime** | 175.6s |
| **Parent** | `node_5_30` |
| **Children** | None |
| **Creation Index** | 184 |

---

## Hypothesis

> The Generative Shift: The proportion of AI incidents attributed to 'Generative'
or 'Language Modeling' technologies has increased significantly in the post-2022
era compared to the pre-2022 era, marking a structural change in the failure
landscape.

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

**Objective:** Quantify the temporal shift in AI failure modes towards Generative AI.

### Steps
- 1. Load 'aiid_incidents' from the main CSV.
- 2. Parse 'date' to extract the year.
- 3. Create two time periods: 'Pre-2023' (Up to Dec 31, 2022) and 'Post-2022' (Jan 1, 2023 onwards).
- 4. Create a binary classifier for 'Generative AI' based on keywords in '84_Known_AI_Technology' and '92_tactics' (e.g., 'LLM', 'Generative', 'Chatbot', 'Transformer', 'Language Model').
- 5. Construct a contingency table: [Period] x [Is_GenAI].
- 6. Perform a Chi-square test for trend/independence.
- 7. Calculate the relative risk or fold-increase.

### Deliverables
- 1. Time-segmented incident counts.
- 2. Proportions of GenAI incidents per period.
- 3. Chi-square test results.
- 4. Trend visualization.

---

## Analysis

The experiment successfully validated the 'Generative Shift' hypothesis with
strong statistical evidence.

1. **Data Segmentation**: The analysis successfully partitioned the AIID dataset
into 'Pre-2023' (n=521) and 'Post-2022' (n=841) cohorts.

2. **Trend Discovery**: The proportion of incidents attributed to Generative AI
technologies (e.g., LLMs, Diffusers, Chatbots) increased dramatically from
**10.6%** in the pre-2023 period to **24.0%** in the post-2022 period.

3. **Statistical Significance**: The Chi-square test yielded a statistic of
**37.21** and a p-value of **1.06e-09** (p < 0.001), confirming that this
distribution shift is not due to chance. The analysis calculated a **2.28x fold
increase** in the relative prevalence of Generative AI failures.

4. **Conclusion**: The results quantitatively confirm a structural change in the
AI incident landscape, verifying that the post-2022 era is distinctively
characterized by a surge in failures related to generative technologies.

---

## Review

The experiment was successfully executed and robustly validated the 'Generative
Shift' hypothesis. The methodology for temporal segmentation and keyword-based
classification was sound, yielding clear and statistically significant results.

1. **Cohort Analysis**: The data was successfully split into 'Pre-2023' (n=521)
and 'Post-2022' (n=841) groups, providing a sufficient sample size for
comparison.
2. **Trend Confirmation**: The analysis revealed a sharp increase in the
proportion of incidents involving Generative AI, rising from **10.6%** in the
pre-2023 period to **24.0%** in the post-2022 period.
3. **Statistical Significance**: The Chi-square test (statistic=37.21,
p=1.06e-09) confirms that this 2.28x fold increase is statistically significant
and not due to random chance.
4. **Conclusion**: The findings provide strong empirical evidence that the
release and adoption of major generative models (like ChatGPT) in late 2022
correlated with a structural shift in the landscape of reported AI failures.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# [debug] Only used to ensure valid execution context
# print("Starting experiment...")

# 1. Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if running in a different environment structure
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# 3. Parse Date and Create Periods
# The 'date' column is expected to be in a parseable format (e.g., YYYY-MM-DD)
aiid_df['date_parsed'] = pd.to_datetime(aiid_df['date'], errors='coerce')

# Drop rows where date could not be parsed
aiid_df = aiid_df.dropna(subset=['date_parsed'])

# Define cut-off date: 2023-01-01
cutoff_date = pd.to_datetime('2023-01-01')

# Create Period Column
aiid_df['period'] = aiid_df['date_parsed'].apply(
    lambda d: 'Post-2022' if d >= cutoff_date else 'Pre-2023'
)

# 4. Classify Generative AI Incidents
# We will search for keywords in '84: Known AI Technology', 'title', and 'description' if available.
# Note: Column names in the CSV often have prefixes like "84: ". We handle this dynamically.

# Identify relevant columns
cols_to_search = []
possible_tech_cols = [c for c in aiid_df.columns if 'Known AI Technology' in c]
possible_desc_cols = [c for c in aiid_df.columns if 'description' in c.lower() or 'summary' in c.lower() or 'title' in c.lower()]

cols_to_search.extend(possible_tech_cols)
cols_to_search.extend(possible_desc_cols)

# Define GenAI keywords
genai_keywords = [
    'generative', 'llm', 'large language model', 'gpt', 'chatbot', 
    'transformer', 'diffusion', 'dalle', 'midjourney', 'stable diffusion', 
    'bard', 'chatgpt', 'llama', 'copilot', 'gemini', 'anthropic', 'claude',
    'foundation model', 'text-to-image', 'genai'
]

def check_genai(row):
    text_blob = ""
    for col in cols_to_search:
        val = row[col]
        if pd.notna(val):
            text_blob += str(val).lower() + " "
    
    for kw in genai_keywords:
        if kw in text_blob:
            return True
    return False

aiid_df['is_genai'] = aiid_df.apply(check_genai, axis=1)

# 5. Summary Statistics and Contingency Table
contingency_table = pd.crosstab(aiid_df['period'], aiid_df['is_genai'])
contingency_table.columns = ['Non-GenAI', 'GenAI']

print("--- Contingency Table: Period vs. Generative AI ---")
print(contingency_table)
print("\n")

# Calculate Proportions
summary = contingency_table.copy()
summary['Total'] = summary['Non-GenAI'] + summary['GenAI']
summary['GenAI_Rate'] = summary['GenAI'] / summary['Total']

print("--- Proportions ---")
print(summary)
print("\n")

# 6. Statistical Test (Chi-Square)
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print(f"--- Chi-Square Test Results ---")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")
print(f"Degrees of Freedom: {dof}")

if p < 0.05:
    print("Result: Statistically Significant Shift detected.")
else:
    print("Result: No Statistically Significant Shift detected.")

# Calculate Fold Increase (Relative Risk)
try:
    rate_pre = summary.loc['Pre-2023', 'GenAI_Rate']
    rate_post = summary.loc['Post-2022', 'GenAI_Rate']
    fold_increase = rate_post / rate_pre if rate_pre > 0 else np.nan
    print(f"Fold Increase (Post / Pre): {fold_increase:.2f}x")
except KeyError:
    print("Could not calculate fold increase due to missing periods.")

# 7. Visualization
plt.figure(figsize=(8, 6))
periods = summary.index
rates = summary['GenAI_Rate'] * 100

bars = plt.bar(periods, rates, color=['skyblue', 'salmon'])
plt.ylabel('Percentage of Incidents involving GenAI (%)')
plt.title('Prevalence of Generative AI in Incidents (Pre-2023 vs Post-2022)')
plt.ylim(0, max(rates) * 1.2)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: --- Contingency Table: Period vs. Generative AI ---
           Non-GenAI  GenAI
period                     
Post-2022        639    202
Pre-2023         466     55


--- Proportions ---
           Non-GenAI  GenAI  Total  GenAI_Rate
period                                        
Post-2022        639    202    841    0.240190
Pre-2023         466     55    521    0.105566


--- Chi-Square Test Results ---
Chi2 Statistic: 37.2113
P-value: 1.0600e-09
Degrees of Freedom: 1
Result: Statistically Significant Shift detected.
Fold Increase (Post / Pre): 2.28x


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Vertical Bar Plot (or Bar Chart).
*   **Purpose:** The plot is designed to compare the prevalence (frequency) of incidents involving Generative AI across two distinct time categories: "Post-2022" and "Pre-2023".

### 2. Axes
*   **X-Axis (Horizontal):**
    *   **Label/Categories:** Represents time periods divided into two categories: "Post-2022" and "Pre-2023".
    *   **Units:** Categorical (Time).
*   **Y-Axis (Vertical):**
    *   **Label:** "Percentage of Incidents involving GenAI (%)".
    *   **Units:** Percentage (%).
    *   **Range:** The axis is marked in increments of 5, ranging from **0 to 25**. The visual space extends slightly above 25, implying a maximum plot range around 30.

### 3. Data Trends
*   **Tallest Bar:** The "Post-2022" bar (colored light blue) is the tallest, reaching a value of **24.0%**.
*   **Shortest Bar:** The "Pre-2023" bar (colored salmon/light red) is the shortest, reaching a value of **10.6%**.
*   **Pattern:** There is a distinct and sharp increase in the percentage of incidents involving Generative AI in the period following 2022 compared to the period prior to 2023.

### 4. Annotations and Legends
*   **Title:** "Prevalence of Generative AI in Incidents (Pre-2023 vs Post-2022)" clearly defines the scope of the comparison.
*   **Data Labels:** The exact percentage values are annotated directly above each bar ("24.0%" and "10.6%"), eliminating the need to estimate values based on the y-axis ticks.
*   **Color Coding:** The bars are distinct in color (Blue for Post-2022, Red for Pre-2023) to visually differentiate the two time periods, though a specific legend box is not present (nor necessary given the x-axis labels).

### 5. Statistical Insights
*   **Significant Increase:** The data indicates a substantial rise in incidents involving Generative AI. The prevalence jumped from **10.6%** in the Pre-2023 era to **24.0%** in the Post-2022 era.
*   **Rate of Growth:** This represents a **13.4 percentage point increase**.
*   **Relative Growth:** In relative terms, the frequency of GenAI incidents has **more than doubled** (approximately a **126% increase**) after 2022.
*   **Context:** This trend likely correlates with the widespread public release and adoption of major Generative AI tools (such as ChatGPT) in late 2022, suggesting that the accessibility of these tools has significantly impacted the landscape of reported incidents.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
