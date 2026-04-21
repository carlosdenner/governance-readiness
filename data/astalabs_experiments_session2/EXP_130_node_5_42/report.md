# Experiment 130: node_5_42

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_42` |
| **ID in Run** | 130 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T07:09:38.776245+00:00 |
| **Runtime** | 206.7s |
| **Parent** | `node_4_10` |
| **Children** | `node_6_40` |
| **Creation Index** | 131 |

---

## Hypothesis

> GenAI Harm Profile: Incidents involving 'Generative AI' are significantly more
likely to result in 'Intangible' harms (e.g., Reputation, Bias) compared to Non-
Generative systems, which are more associated with 'Tangible' harms.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7661 (Likely True) |
| **Posterior** | 0.4258 (Maybe False) |
| **Surprise** | -0.4084 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 3.0 |
| Maybe True | 27.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 60.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Analyze the relationship between AI technology type and the nature of the harm produced.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `source_table` = 'aiid_incidents'.
- 2. Create a binary `is_genai` by searching `Known AI Technology` for keywords ['Generative', 'LLM', 'GPT', 'Diffusion', 'Language Model', 'Chatbot'].
- 3. Create a binary `is_intangible_harm` by mapping `Harm Domain` or `Tangible Harm` columns (if `Tangible Harm` is 'NaN' or explicitly 'Intangible', set to 1; if it lists physical/financial harms, set to 0. Requires data inspection).
- 4. Perform a Chi-square test on the association between Technology Type and Harm Type.

### Deliverables
- Contingency table and statistical test results.

---

## Analysis

The experiment successfully analyzed the relationship between Generative AI and
harm type using the AIID dataset. Correcting for previous data mapping issues,
the code identified 25 Generative AI incidents and 1,337 Non-Generative
incidents.

**Findings:**
1. **Data Distribution:**
   - **Generative AI:** 25 incidents total. 100% were classified as 'Intangible'
(25 incidents), with 0 'Tangible' harms.
   - **Non-Generative AI:** 1,337 incidents total. 94.6% were 'Intangible'
(1,265 incidents), and 5.4% resulted in 'Tangible' harm (72 incidents).

2. **Statistical Test:**
   - A Fisher's Exact Test was performed due to the zero cell count.
   - **P-value:** 0.6388
   - **Odds Ratio:** 0.0

3. **Conclusion:**
   The difference is **not statistically significant** (p > 0.05). Although
GenAI incidents had a 0% rate of tangible harm compared to 5.4% for non-GenAI,
the small sample size of GenAI incidents (n=25) combined with the overall rarity
of 'definitively occurred' tangible harm in the dataset means this result could
easily happen by chance. The hypothesis that GenAI is *significantly* more
associated with intangible harm is **rejected** based on this evidence.

---

## Review

The experiment was successfully executed and the code faithfully implemented the
plan, incorporating the necessary corrections from previous iterations regarding
categorical string matching for 'Tangible Harm'.

**Hypothesis:** GenAI Harm Profile (GenAI incidents are significantly more
likely to cause Intangible harm than Tangible harm compared to Non-GenAI).

**Findings:**
1.  **Data Distribution:**
    -   **Generative AI:** 25 incidents identified. 100% (25) were associated
with Intangible harm; 0 were associated with definitively Tangible harm.
    -   **Non-Generative AI:** 1,337 incidents identified. 94.6% (1,265) were
Intangible; 5.4% (72) were Tangible.

2.  **Statistical Test:**
    -   A Fisher's Exact Test was performed due to the zero cell count in the
GenAI/Tangible category.
    -   **P-value:** 0.6388
    -   **Odds Ratio:** 0.0 (indicating infinite odds in the other direction or
undefined, but practically meaning 0 tangible cases).

3.  **Conclusion:**
    The difference is **not statistically significant** (p > 0.05). While GenAI
showed a 0% rate of tangible harm compared to 5.4% for Non-GenAI, the small
sample size of GenAI incidents (n=25) means this difference is indistinguishable
from random chance. The hypothesis is **rejected** based on this dataset; there
is insufficient evidence to claim GenAI harm profiles are distinct regarding
tangibility.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import numpy as np

# Load dataset
file_path = 'astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
df_aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents loaded: {len(df_aiid)} rows")

# --- Step 1: define GenAI ---
# Keywords for Generative AI
genai_keywords = ['generative', 'llm', 'gpt', 'diffusion', 'language model', 'chatbot', 'deepfake', 'dall-e', 'midjourney']

def is_genai(text):
    if pd.isna(text):
        return False
    text_lower = str(text).lower()
    return any(keyword in text_lower for keyword in genai_keywords)

df_aiid['is_genai'] = df_aiid['Known AI Technology'].apply(is_genai)

# --- Step 2: Define Harm Type ---
# Based on previous debug, 'Tangible Harm' column contains specific strings.
# We define 'Tangible' as cases where harm definitively occurred.
# We define 'Intangible' as cases where it did not (near-misses, issues, or explicitly no tangible harm).

tangible_marker = 'tangible harm definitively occurred'

def classify_harm(val):
    if pd.isna(val):
        return 'Intangible' # Treat missing as Intangible/Unknown for this binary split or drop? 
        # Safer to treat as Intangible if we assume Tangible is the exception, 
        # but let's check if 'nan' means no info. 
        # For this experiment, let's map strictly based on the string.
    
    val_lower = str(val).lower()
    if tangible_marker in val_lower:
        return 'Tangible'
    else:
        return 'Intangible'

df_aiid['harm_type'] = df_aiid['Tangible Harm'].apply(classify_harm)

# --- Step 3: Analysis ---

# Create Contingency Table
contingency_table = pd.crosstab(df_aiid['is_genai'], df_aiid['harm_type'])

# Rename indices for clarity
contingency_table.index = ['Non-GenAI', 'GenAI']
print("\n--- Contingency Table (GenAI vs Harm Type) ---")
print(contingency_table)

# Check if we have data in both columns
if 'Tangible' not in contingency_table.columns:
    contingency_table['Tangible'] = 0
if 'Intangible' not in contingency_table.columns:
    contingency_table['Intangible'] = 0

# Calculate percentages for context
contingency_pct = pd.crosstab(df_aiid['is_genai'], df_aiid['harm_type'], normalize='index') * 100
print("\n--- Percentages (Row-wise) ---")
print(contingency_pct)

# Statistical Test
# Using Fisher's Exact Test if sample size is small, otherwise Chi2.
# Given the likely imbalance, Fisher's is safer or Chi2 with Yates correction.
# We will use Fisher's Exact Test for 2x2.

odds_ratio, p_value = stats.fisher_exact(contingency_table.loc[['Non-GenAI', 'GenAI'], ['Intangible', 'Tangible']])

print(f"\nFisher's Exact Test Results:")
print(f"Odds Ratio: {odds_ratio:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
print("\n--- Interpretation ---")
if p_value < 0.05:
    print("Result: Statistically Significant.")
    if odds_ratio > 1:
        print("GenAI incidents are significantly more likely to be associated with Intangible harm (vs Tangible) compared to Non-GenAI.")
    else:
        print("GenAI incidents are significantly LESS likely to be associated with Intangible harm compared to Non-GenAI.")
else:
    print("Result: Not Statistically Significant. No evidence that GenAI harm profiles differ from Non-GenAI in this dataset.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: AIID Incidents loaded: 1362 rows

--- Contingency Table (GenAI vs Harm Type) ---
harm_type  Intangible  Tangible
Non-GenAI        1265        72
GenAI              25         0

--- Percentages (Row-wise) ---
harm_type  Intangible  Tangible
is_genai                       
False       94.614809  5.385191
True       100.000000  0.000000

Fisher's Exact Test Results:
Odds Ratio: 0.0000
P-value: 0.6388

--- Interpretation ---
Result: Not Statistically Significant. No evidence that GenAI harm profiles differ from Non-GenAI in this dataset.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
