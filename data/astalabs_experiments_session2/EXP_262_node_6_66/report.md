# Experiment 262: node_6_66

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_66` |
| **ID in Run** | 262 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T13:29:24.675741+00:00 |
| **Runtime** | 161.1s |
| **Parent** | `node_5_87` |
| **Children** | None |
| **Creation Index** | 263 |

---

## Hypothesis

> Generative Sector Bias: 'Generative AI' incidents in the AIID dataset are
disproportionately concentrated in the 'Technology' and 'Information' sectors
compared to 'Healthcare' or 'Transportation', where discriminative models
prevail.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7823 (Likely True) |
| **Posterior** | 0.7610 (Likely True) |
| **Surprise** | -0.0255 |
| **Surprise Interpretation** | Neutral (no significant belief shift) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 5.0 |
| Maybe True | 25.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 60.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Compare the sector distribution of Generative AI incidents versus other AI technologies.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `source_table` = 'aiid_incidents'.
- 2. Define `is_genai`: Search `84_known_ai_technology` (or similar) for 'Generative', 'Language Model', 'LLM', 'GPT', 'Diffusion'.
- 3. Define `sector_type`: Map `78_sector_of_deployment` to 'Tech/Info' (e.g., Technology, Media, Information) vs 'Safety-Critical' (e.g., Healthcare, Transportation, Energy).
- 4. Create a contingency table of GenAI Status vs. Sector Type.
- 5. Perform a Chi-square test.
- 6. Compare the proportion of GenAI cases in Tech/Info vs. Safety-Critical sectors.

### Deliverables
- Contingency table, Chi-square statistics, and sectoral proportions for Generative AI.

---

## Analysis

The experiment was successfully executed, though the analysis was heavily
constrained by data quality issues within the AIID dataset. Specifically, 1,161
of the 1,362 incidents (85%) lacked a defined 'sector_of_deployment',
drastically reducing the effective sample size to 122 incidents for this
comparison.

The observed data showed a strong directional trend supporting the hypothesis:
15.96% (15/94) of incidents in 'Tech/Info' sectors involved Generative AI,
whereas 0% (0/28) of incidents in 'Safety-Critical' sectors did.

However, the Chi-square test resulted in a p-value of 0.0537. Since this is
slightly above the standard alpha of 0.05, the difference is not statistically
significant at the 95% confidence level. Therefore, while there is a visible
concentration of Generative AI incidents in the technology sector compared to
safety-critical ones (where no GenAI incidents were recorded in this subset),
the null hypothesis cannot be strictly rejected due to the insufficient sample
size caused by missing metadata.

---

## Review

The experiment was successfully executed and the code faithfully implemented the
analysis plan. The analysis faced a significant data quality limitation: 85% of
AIID incidents (1,161/1,362) lacked sector metadata, drastically reducing the
effective sample size to 122 cases. Despite this, the analysis revealed a strong
directional trend: 15.96% of incidents in 'Tech/Info' sectors involved
Generative AI, compared to 0.00% in 'Safety-Critical' sectors. The Chi-square
test yielded a p-value of 0.0537, which is marginally above the standard 0.05
significance threshold. Consequently, while the data strongly suggests
Generative AI is concentrated in information sectors rather than safety-critical
ones, the result is not statistically significant at the 95% confidence level
due to the low sample count.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load the dataset
file_path = '../astalabs_discovery_all_data.csv'
try:
    # Reading with low_memory=False to handle mixed types warning from previous steps
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    # Fallback if file is in current directory
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

print(f"Total AIID incidents: {len(aiid_df)}")

# Identify relevant columns based on previous metadata
# 84_known_ai_technology and 78_sector_of_deployment
# Normalize column names to be safe
aiid_df.columns = [c.strip().lower().replace(' ', '_').replace(':', '') for c in aiid_df.columns]

# Find the specific columns
tech_col = next((c for c in aiid_df.columns if 'known_ai_technology' in c), None)
sector_col = next((c for c in aiid_df.columns if 'sector_of_deployment' in c), None)

if not tech_col or not sector_col:
    print("Could not identify required columns. Available columns:")
    print(aiid_df.columns.tolist())
    exit()

print(f"Using technology column: {tech_col}")
print(f"Using sector column: {sector_col}")

# Fill NaNs
aiid_df[tech_col] = aiid_df[tech_col].fillna('')
aiid_df[sector_col] = aiid_df[sector_col].fillna('')

# Define GenAI keywords
genai_keywords = ['generative', 'language model', 'llm', 'gpt', 'diffusion', 'chatbot', 'transformer', 'foundation model']

# Create is_genai flag
aiid_df['is_genai'] = aiid_df[tech_col].apply(lambda x: any(k in str(x).lower() for k in genai_keywords))

# Define Sector Groups
# Tech/Info vs Safety-Critical
# Let's inspect unique sectors first to ensure correct mapping
unique_sectors = aiid_df[sector_col].unique()
print(f"\nTop 10 Sectors found:\n{aiid_df[sector_col].value_counts().head(10)}")

def map_sector(sector_str):
    s = str(sector_str).lower()
    if any(x in s for x in ['technology', 'media', 'information', 'internet', 'software', 'telecom', 'entertainment']):
        return 'Tech/Info'
    elif any(x in s for x in ['healthcare', 'transportation', 'energy', 'automotive', 'aviation', 'medical', 'hospital', 'utility', 'defense', 'military']):
        return 'Safety-Critical'
    return 'Other'

aiid_df['sector_group'] = aiid_df[sector_col].apply(map_sector)

# Filter for only the two groups of interest
analysis_df = aiid_df[aiid_df['sector_group'].isin(['Tech/Info', 'Safety-Critical'])].copy()

print(f"\nAnalysis set size (filtered for relevant sectors): {len(analysis_df)}")

# Contingency Table
contingency_table = pd.crosstab(analysis_df['is_genai'], analysis_df['sector_group'])
print("\nContingency Table (Count):")
print(contingency_table)

# Calculate Proportions
prop_table = pd.crosstab(analysis_df['is_genai'], analysis_df['sector_group'], normalize='columns')
print("\nContingency Table (Proportions):")
print(prop_table)

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"p-value: {p:.4e}")

# Interpretation
genai_tech_rate = prop_table.loc[True, 'Tech/Info'] if True in prop_table.index else 0
genai_safety_rate = prop_table.loc[True, 'Safety-Critical'] if True in prop_table.index else 0

print(f"\nGenAI Incidence Rate in Tech/Info: {genai_tech_rate:.2%}")
print(f"GenAI Incidence Rate in Safety-Critical: {genai_safety_rate:.2%}")

if p < 0.05:
    print("Result: Statistically Significant Difference.")
    if genai_tech_rate > genai_safety_rate:
        print("Hypothesis Supported: GenAI is more concentrated in Tech/Info sectors.")
    else:
        print("Hypothesis Refuted: GenAI is more concentrated in Safety-Critical sectors.")
else:
    print("Result: No Statistically Significant Difference found.")
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Total AIID incidents: 1362
Using technology column: known_ai_technology
Using sector column: sector_of_deployment

Top 10 Sectors found:
sector_of_deployment
                                                                     1161
information and communication                                          44
transportation and storage                                             21
Arts, entertainment and recreation, information and communication      14
wholesale and retail trade                                             11
human health and social work activities                                10
Arts, entertainment and recreation                                      9
law enforcement                                                         9
information and communication, Arts, entertainment and recreation       9
Education                                                               7
Name: count, dtype: int64

Analysis set size (filtered for relevant sectors): 122

Contingency Table (Count):
sector_group  Safety-Critical  Tech/Info
is_genai                                
False                      28         79
True                        0         15

Contingency Table (Proportions):
sector_group  Safety-Critical  Tech/Info
is_genai                                
False                     1.0   0.840426
True                      0.0   0.159574

Chi-Square Statistic: 3.7221
p-value: 5.3697e-02

GenAI Incidence Rate in Tech/Info: 15.96%
GenAI Incidence Rate in Safety-Critical: 0.00%
Result: No Statistically Significant Difference found.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
