# Experiment 13: node_3_4

| Property | Value |
|---|---|
| **Experiment ID** | `node_3_4` |
| **ID in Run** | 13 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T01:37:18.318882+00:00 |
| **Runtime** | 231.3s |
| **Parent** | `node_2_7` |
| **Children** | `node_4_6`, `node_4_42`, `node_4_52` |
| **Creation Index** | 14 |

---

## Hypothesis

> Generative AI Vulnerability Profile: In ATLAS cases, systems identifiable as
'Generative AI' (via keywords) are significantly more prone to 'Defense Evasion'
tactics (e.g., jailbreaking) compared to Non-GenAI systems, which cluster around
'Exfiltration' or 'Collection' tactics.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.8145 (Likely True) |
| **Posterior** | 0.4423 (Maybe False) |
| **Surprise** | -0.4467 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 9.0 |
| Maybe True | 21.0 |
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

**Objective:** Distinguish the adversarial threat landscape of Generative vs Predictive AI.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for 'atlas_cases'.
- 2. Text-mine the 'summary' or 'name' columns to flag 'Generative AI' (keywords: LLM, GPT, chat, gen, diffusion) vs 'Other'.
- 3. Extract lists of tactics from the 'tactics' column.
- 4. Create a binary flag for the presence of 'Defense Evasion' (or specific tactic IDs like AML.T00XX).
- 5. Compare the proportion of Defense Evasion incidents between GenAI and Other groups using a Fisher's Exact Test (due to small sample size).

### Deliverables
- Frequency table of Tactics by AI Type; Fisher's Exact Test p-value.

---

## Analysis

The experiment successfully analyzed the relationship between Generative AI
systems and 'Defense Evasion' tactics within the ATLAS dataset (n=52).

1. **Data Classification**:
   - **Generative AI**: 13 cases identified.
   - **Non-Generative AI**: 39 cases identified.
   - **Defense Evasion**: Successfully flagged using the corrected identifier
(`defense_evasion`), fixing the issue from the previous attempt.

2. **Statistical Results**:
   - **Contingency Table**:
     - Non-GenAI: 18 with Defense Evasion (46.1%), 21 without.
     - GenAI: 5 with Defense Evasion (38.5%), 8 without.
   - **Fisher's Exact Test**: p-value = 0.7519 (> 0.05).
   - **Odds Ratio**: 0.7292.

3. **Conclusion**: The analysis yielded no statistically significant difference
in the prevalence of 'Defense Evasion' tactics between Generative AI and Non-
Generative AI systems. The hypothesis that Generative AI is significantly more
prone to Defense Evasion is **not supported** by this dataset; in fact, the
observed proportion was slightly lower for GenAI, though statistically
indistinguishable from random chance.

---

## Review

The experiment was faithfully implemented and successfully executed. The code
correctly loaded the ATLAS dataset, classified cases into Generative AI (n=13)
and Non-Generative AI (n=39) using text-mining keywords, and accurately
identified 'Defense Evasion' tactics after correcting for data formatting issues
(handling the underscore in 'defense_evasion').

**Hypothesis Test Results**:
The hypothesis that Generative AI is significantly more prone to 'Defense
Evasion' was **not supported**.
- **Descriptive Statistics**: Defense Evasion tactics were observed in 38.5%
(5/13) of Generative AI cases versus 46.2% (18/39) of Non-Generative AI cases.
- **Statistical Test**: Fisher's Exact Test yielded a p-value of 0.7519 and an
Odds Ratio of 0.7292, indicating no statistically significant difference between
the two groups.

The analysis sufficiently tested the proposed relationship and found no evidence
to support the claim that Defense Evasion is a unique or elevated vulnerability
profile for Generative AI in the current ATLAS dataset.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import numpy as np

# Load dataset
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

# Filter for ATLAS cases
atlas_df = df[df['source_table'] == 'atlas_cases'].copy()
print(f"Loaded {len(atlas_df)} ATLAS cases.")

# 1. Flag Generative AI systems
# Keywords: LLM, GPT, chat, gen, diffusion.
genai_keywords = ['llm', 'gpt', 'chat', 'generative', 'diffusion', 'foundation model', 'transformer', 'bert']

def check_genai(row):
    text = (str(row['name']) + ' ' + str(row['summary'])).lower()
    for kw in genai_keywords:
        if kw in text:
            return True
    return False

atlas_df['is_genai'] = atlas_df.apply(check_genai, axis=1)

# 2. Flag 'Defense Evasion' tactics
# The tactics column contains strings like '{{defense_evasion.id}}'.
# We look for 'defense_evasion' (case insensitive).
def check_defense_evasion(val):
    if pd.isna(val):
        return False
    return 'defense_evasion' in str(val).lower()

atlas_df['has_defense_evasion'] = atlas_df['tactics'].apply(check_defense_evasion)

# 3. Generate Contingency Table
# Rows: GenAI vs Non-GenAI
# Cols: Defense Evasion vs No Defense Evasion
contingency_table = pd.crosstab(atlas_df['is_genai'], atlas_df['has_defense_evasion'])

# Ensure the table is 2x2 even if some categories are missing
# We expect index [False, True] and columns [False, True]
contingency_table = contingency_table.reindex(index=[False, True], columns=[False, True], fill_value=0)

# Rename for clarity
contingency_table.index = ['Non-GenAI', 'GenAI']
contingency_table.columns = ['No Defense Evasion', 'Has Defense Evasion']

print("\nContingency Table (Frequency of Defense Evasion Tactics):")
print(contingency_table)

# 4. Fisher's Exact Test
oddsratio, pvalue = stats.fisher_exact(contingency_table)
print(f"\nFisher's Exact Test p-value: {pvalue:.4f}")
print(f"Odds Ratio: {oddsratio:.4f}")

# Interpretation
alpha = 0.05
if pvalue < alpha:
    print("Result: Statistically significant difference in Defense Evasion prevalence.")
else:
    print("Result: No statistically significant difference found.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loaded 52 ATLAS cases.

Contingency Table (Frequency of Defense Evasion Tactics):
           No Defense Evasion  Has Defense Evasion
Non-GenAI                  21                   18
GenAI                       8                    5

Fisher's Exact Test p-value: 0.7519
Odds Ratio: 0.7292
Result: No statistically significant difference found.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
