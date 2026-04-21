# Experiment 124: node_4_48

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_48` |
| **ID in Run** | 124 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T06:53:52.435780+00:00 |
| **Runtime** | 216.1s |
| **Parent** | `node_3_8` |
| **Children** | `node_5_58`, `node_5_85` |
| **Creation Index** | 125 |

---

## Hypothesis

> Generative AI's Adversarial Profile: AIID incidents involving 'Generative'
technologies are significantly more likely to match ATLAS-style adversarial
keywords (e.g., 'injection', 'jailbreak', 'extraction') in their descriptions
compared to 'Discriminative' systems.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.8952 (Likely True) |
| **Posterior** | 0.4643 (Uncertain) |
| **Surprise** | -0.5171 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 19.0 |
| Maybe True | 11.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 58.0 |
| Definitely False | 2.0 |

---

## Experiment Plan

**Objective:** Cross-reference AIID incidents with ATLAS adversarial concepts to see if GenAI drives a shift in failure modes.

### Steps
- 1. Load `aiid_incidents`.
- 2. Categorize systems as 'Generative' (search `Known AI Technology` for 'generative', 'LLM', 'diffusion', 'chat') vs 'Discriminative'.
- 3. Define a set of ATLAS adversarial keywords (e.g., 'injection', 'poisoning', 'evasion', 'extraction', 'jailbreak').
- 4. Flag incidents where the `description` or `summary` contains these keywords.
- 5. Compare the proportion of flagged incidents between Generative and Discriminative groups using a Chi-Square test.

### Deliverables
- Comparison Chart of Adversarial Keyword Frequency by Tech Type; Chi-Square statistics.

---

## Analysis

The experiment successfully analyzed 1,362 AIID incidents to determine if
Generative AI systems are more prone to adversarial failure modes (e.g.,
'injection', 'jailbreak') than Discriminative systems. The data was categorized
into Discriminative (n=1,285) and Generative (n=77) groups based on technology
keywords. While Generative AI incidents showed a slightly higher proportion of
adversarial keywords (20.8%) compared to Discriminative AI (17.6%), the
difference was not statistically significant (Chi-Square p-value = 0.577).
Consequently, the hypothesis is rejected; the current dataset does not provide
evidence that Generative AI incidents are significantly more likely to involve
ATLAS-style adversarial attacks than traditional systems, potentially due to the
relatively small sample size of GenAI incidents or the prevalence of non-
adversarial failure modes (e.g., hallucinations) in the GenAI category.

---

## Review

The experiment was successfully executed and faithfully tested the hypothesis
regarding Generative AI's adversarial profile. The code correctly identified
relevant columns, categorized 1,362 incidents into 'Generative' (n=77) and
'Discriminative' (n=1,285) groups, and flagged descriptions containing ATLAS-
style adversarial keywords.

Results showed that Generative AI incidents had a slightly higher proportion of
adversarial keywords (20.8%) compared to Discriminative systems (17.6%).
However, the Chi-Square test yielded a p-value of 0.58, indicating that this
difference is not statistically significant. Therefore, the hypothesis is
rejected based on this dataset. The analysis suggests that while adversarial
attacks like 'jailbreaking' are prominent in GenAI discourse, they do not yet
constitute a significantly larger slice of reported failures compared to the
adversarial modes (or lack thereof) in traditional systems, potentially due to
the high volume of non-adversarial GenAI failures (e.g., hallucinations) or
sample size limitations.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Loaded {len(aiid)} AIID incidents.")

# Dynamic Column Search
all_cols = aiid.columns.tolist()

# Find Technology columns
tech_candidates = [c for c in all_cols if 'technology' in c.lower()]
print(f"Found Technology columns: {tech_candidates}")

# Find Text columns (Description/Summary)
text_candidates = [c for c in all_cols if 'description' in c.lower() or 'summary' in c.lower()]
print(f"Found Text columns: {text_candidates}")

# Combine technology columns
if not tech_candidates:
    print("Error: No technology columns found. checking for 'AI' in columns.")
    tech_candidates = [c for c in all_cols if 'ai' in c.lower() and 'known' in c.lower()]
    print(f"Alternative Tech columns: {tech_candidates}")

aiid['tech_combined'] = aiid[tech_candidates].apply(lambda x: ' '.join(x.dropna().astype(str).str.lower()), axis=1)

# Define Generative AI keywords
genai_keywords = ['generative', 'llm', 'diffusion', 'chat', 'gpt', 'transformer', 'language model', 'genai', 'chatbot', 'foundation model', 'midjourney', 'dall-e', 'stable diffusion', 'bert', 'large language model']

def classify_tech(text):
    if any(k in text for k in genai_keywords):
        return 'Generative'
    return 'Discriminative'

aiid['tech_type'] = aiid['tech_combined'].apply(classify_tech)

# Combine text columns for keyword search
# We prioritize description/summary columns, but if none, we might look at 'title' or just fall back to empty
if text_candidates:
    aiid['text_combined'] = aiid[text_candidates].apply(lambda x: ' '.join(x.dropna().astype(str).str.lower()), axis=1)
else:
    print("Warning: No description/summary found. Trying 'title'.")
    if 'title' in all_cols:
        aiid['text_combined'] = aiid['title'].astype(str).str.lower()
    else:
        aiid['text_combined'] = ""

# Define ATLAS/Adversarial keywords
adversarial_keywords = [
    'injection', 'jailbreak', 'extraction', 'poisoning', 'evasion', 
    'adversarial', 'prompt', 'red team', 'bypass', 'attack', 'manipulat', 
    'inference', 'inversion', 'membership inference', 'model stealing', 'trojan', 'backdoor'
]

def check_adversarial(text):
    return any(k in text for k in adversarial_keywords)

aiid['has_adversarial_keywords'] = aiid['text_combined'].apply(check_adversarial)

# Generate stats
contingency_table = pd.crosstab(aiid['tech_type'], aiid['has_adversarial_keywords'])
print("\nContingency Table (Rows: Tech Type, Cols: Has Adversarial Keywords):")
print(contingency_table)

# Calculate proportions
summary = aiid.groupby('tech_type')['has_adversarial_keywords'].agg(['count', 'sum', 'mean'])
summary.columns = ['Total Incidents', 'Adversarial Matches', 'Proportion']
print("\nSummary Statistics:")
print(summary)

# Chi-Square Test
chi2, p, dof, ex = chi2_contingency(contingency_table)
print(f"\nChi-Square Test Results:")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Visualization
plt.figure(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e']

if not summary.empty:
    prop_plot = summary['Proportion'].plot(kind='bar', color=colors, alpha=0.8)
    plt.title('Proportion of Incidents with Adversarial Keywords by AI Type')
    plt.ylabel('Proportion (0-1)')
    plt.xlabel('AI Technology Type')
    plt.xticks(rotation=0)
    # Set ylim with margin
    top_val = summary['Proportion'].max()
    if top_val > 0:
        plt.ylim(0, top_val * 1.2)
    
    for i, v in enumerate(summary['Proportion']):
        plt.text(i, v + (top_val*0.01), f"{v:.1%}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()
else:
    print("Summary is empty, cannot plot.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded 1362 AIID incidents.
Found Technology columns: ['Known AI Technology', 'Potential AI Technology']
Found Text columns: ['description', 'summary']

Contingency Table (Rows: Tech Type, Cols: Has Adversarial Keywords):
has_adversarial_keywords  False  True 
tech_type                             
Discriminative             1059    226
Generative                   61     16

Summary Statistics:
                Total Incidents  Adversarial Matches  Proportion
tech_type                                                       
Discriminative             1285                  226    0.175875
Generative                   77                   16    0.207792

Chi-Square Test Results:
Chi2 Statistic: 0.3116
P-value: 5.7670e-01


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

**1. Plot Type**
*   **Type:** Vertical Bar Chart.
*   **Purpose:** The plot compares categorical data, specifically looking at the proportion of incidents containing adversarial keywords across two different types of AI technologies: Discriminative and Generative.

**2. Axes**
*   **X-axis:**
    *   **Title:** "AI Technology Type"
    *   **Labels:** The axis represents two distinct categories: "Discriminative" and "Generative".
*   **Y-axis:**
    *   **Title:** "Proportion (0-1)"
    *   **Range:** The axis starts at **0.00** and extends to approximately **0.25** (though the highest marked tick is not explicitly labeled above 0.20, the space suggests the next interval).
    *   **Units:** The axis uses decimal notation (0.00, 0.05, 0.10, 0.15, 0.20) to represent proportions.

**3. Data Trends**
*   **Tallest Bar:** The **Generative** AI category has the tallest bar, indicating a higher proportion of incidents.
*   **Shortest Bar:** The **Discriminative** AI category has the shorter bar.
*   **Comparison:** There is a visible difference between the two categories, with Generative AI showing a higher prevalence of adversarial keywords in incidents compared to Discriminative AI.

**4. Annotations and Legends**
*   **Title:** The chart is titled "**Proportion of Incidents with Adversarial Keywords by AI Type**".
*   **Bar Annotations:** Specific percentage values are annotated in bold directly above each bar to provide precise data points:
    *   Discriminative: **17.6%**
    *   Generative: **20.8%**
*   **Color Coding:** The bars are distinct colors (blue for Discriminative, orange for Generative) to visually separate the categories, although no separate legend box is provided or necessary given the clear x-axis labels.

**5. Statistical Insights**
*   **Higher Incidence in Generative AI:** The data indicates that incidents involving Generative AI are more likely to contain adversarial keywords (20.8%) compared to Discriminative AI incidents (17.6%).
*   **Magnitude of Difference:** The difference between the two AI types is **3.2 percentage points**.
*   **Overall Prevalence:** For both AI types, roughly one-fifth of the recorded incidents involve adversarial keywords, suggesting that this is a significant issue for both technologies, though slightly more pronounced in Generative models.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
