# Experiment 298: node_6_85

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_85` |
| **ID in Run** | 298 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T15:20:34.175568+00:00 |
| **Runtime** | 283.1s |
| **Parent** | `node_5_62` |
| **Children** | None |
| **Creation Index** | 299 |

---

## Hypothesis

> The 'Physical-Impact' Link (ATLAS - Revised): Adversarial attacks in 'Physical'
sectors (inferred from case descriptions) are significantly more likely to
involve the 'Impact' tactic (destructive payload) compared to attacks in
'Digital' sectors.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.4066 (Maybe False) |
| **Surprise** | -0.4024 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

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
| Uncertain | 0.0 |
| Maybe False | 56.0 |
| Definitely False | 4.0 |

---

## Experiment Plan

**Objective:** Investigate if physical industries face more destructive adversarial AI tactics by inferring sectors from case summaries due to missing metadata.

### Steps
- 1. Load 'astalabs_discovery_all_data.csv' and filter for rows where 'source_table' is 'atlas_cases'.
- 2. FIll missing 'sector' values by analyzing 'name' and 'summary' columns using keyword matching:
   - Physical: ['automotive', 'car', 'vehicle', 'drive', 'energy', 'power', 'grid', 'defense', 'military', 'weapon', 'healthcare', 'medical', 'hospital', 'surveillance', 'camera', 'drone', 'robot', 'biometric', 'physical']
   - Digital: ['finance', 'bank', 'trading', 'software', 'malware', 'phishing', 'email', 'internet', 'web', 'cloud', 'network', 'chatbot', 'language model', 'llm', 'text', 'image', 'spam', 'bypass']
3. Create a 'has_impact' boolean column by checking if the 'tactics' column contains the substring 'impact' (case-insensitive).
- 4. Filter the dataset to include only rows classified as 'Physical' or 'Digital'.
- 5. Generate a contingency table of Sector vs. Impact Tactic Presence.
- 6. Perform a Fisher's Exact Test (due to small sample size n=52) to test for independence.

### Deliverables
- 1. Contingency table showing counts of Impact/No-Impact for Physical vs Digital sectors.
- 2. Fisher's Exact Test statistic and p-value.
- 3. Conclusion on whether the hypothesis is supported.

---

## Analysis

The experiment successfully tested the revised 'Physical-Impact Link' hypothesis
by inferring sectors from the names and summaries of 52 ATLAS cases. Using
keyword matching, 38 cases were classified (26 Digital, 12 Physical), while 14
remained unclassified and were excluded.

Contrary to the hypothesis that Physical sectors face more destructive 'Impact'
tactics, the data showed a higher prevalence of Impact tactics in Digital
sectors (73.08%, 19/26) compared to Physical sectors (58.33%, 7/12). However,
the Fisher's Exact Test yielded a p-value of 0.4605, indicating that this
difference is not statistically significant.

Therefore, the hypothesis is rejected. There is no evidence in the ATLAS dataset
to support the claim that adversarial attacks in physical industries are more
likely to involve destructive payloads than those in digital industries; in
fact, the trend in this limited sample leaned slightly in the opposite
direction.

---

## Review

The experiment successfully tested the revised 'Physical-Impact Link' hypothesis
using the ATLAS dataset. The implementation effectively handled the missing
'sector' metadata by inferring categories from case summaries, allowing for a
valid statistical comparison.

Analysis of the 38 classified cases (26 Digital, 12 Physical) revealed that
Digital sectors actually had a higher rate of 'Impact' tactics (73.08%) compared
to Physical sectors (58.33%), which contradicts the hypothesis direction.
However, the Fisher's Exact Test yielded a p-value of 0.4605, indicating that
this difference is not statistically significant.

Therefore, the hypothesis is rejected. The data suggests that 'Impact' tactics
(destructive payloads) are a common feature across both physical and digital
adversarial AI scenarios in the ATLAS framework, with no significant skew toward
physical industries.

---

## Code

```python
import pandas as pd
import scipy.stats as stats
import sys

# Load dataset
print("Loading dataset...")
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Dataset not found.")
        sys.exit(1)

# Filter for ATLAS cases
atlas_df = df[df['source_table'] == 'atlas_cases'].copy()
print(f"ATLAS cases loaded: {len(atlas_df)}")

# Define keywords
physical_keywords = [
    'automotive', 'car', 'vehicle', 'drive', 'driving', 'energy', 'power', 'grid', 
    'defense', 'military', 'weapon', 'healthcare', 'medical', 'hospital', 
    'surveillance', 'camera', 'cctv', 'drone', 'robot', 'biometric', 'physical', 
    'traffic', 'face', 'facial', 'recognition', 'sensor', 'gps'
]

digital_keywords = [
    'finance', 'bank', 'trading', 'software', 'malware', 'phishing', 'email', 
    'internet', 'web', 'cloud', 'network', 'chatbot', 'language model', 'llm', 
    'text', 'spam', 'bypass', 'antivirus', 'translation', 'bot', 'algorithm', 
    'filter', 'online', 'media'
]

# Function to categorize sector
def categorize_sector(row):
    # Combine name and summary, handle NaN
    text = str(row.get('name', '')) + " " + str(row.get('summary', ''))
    text = text.lower()
    
    # Check Physical first (prioritizing domain of application)
    for kw in physical_keywords:
        if kw in text:
            return 'Physical'
            
    # Check Digital
    for kw in digital_keywords:
        if kw in text:
            return 'Digital'
            
    return 'Unclassified'

# Apply categorization
atlas_df['inferred_sector'] = atlas_df.apply(categorize_sector, axis=1)

# Identify Impact tactic
# Tactics seem to be strings like "{{impact.id}}|..."
atlas_df['has_impact'] = atlas_df['tactics'].astype(str).str.contains('impact', case=False, na=False)

# Filter out Unclassified
analysis_df = atlas_df[atlas_df['inferred_sector'] != 'Unclassified'].copy()

print("\nSector Inference Results:")
print(analysis_df['inferred_sector'].value_counts())
print(f"Unclassified cases dropped: {len(atlas_df) - len(analysis_df)}")

# Create Contingency Table
contingency_table = pd.crosstab(analysis_df['inferred_sector'], analysis_df['has_impact'])
print("\nContingency Table (Sector vs Has Impact):")
print(contingency_table)

# Check if we have enough data for 2x2
if contingency_table.shape == (2, 2):
    # Perform Fisher's Exact Test
    # Table structure:
    #              False  True
    # inferred_sector
    # Digital       A      B
    # Physical      C      D
    odds_ratio, p_value = stats.fisher_exact(contingency_table)
    
    print(f"\nFisher's Exact Test Results:")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        print("Result: Statistically significant association found.")
    else:
        print("Result: No statistically significant association found.")
        
    # Calculate percentages for clarity
    physical_total = contingency_table.loc['Physical'].sum()
    physical_impact = contingency_table.loc['Physical', True]
    digital_total = contingency_table.loc['Digital'].sum()
    digital_impact = contingency_table.loc['Digital', True]
    
    print(f"\nPhysical Sector Impact Rate: {physical_impact}/{physical_total} ({physical_impact/physical_total:.2%})")
    print(f"Digital Sector Impact Rate: {digital_impact}/{digital_total} ({digital_impact/digital_total:.2%})")

else:
    print("\nInsufficient data dimensions for Fisher's Exact Test (need 2x2 table).")
    print("Observed shape:", contingency_table.shape)

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset...
ATLAS cases loaded: 52

Sector Inference Results:
inferred_sector
Digital     26
Physical    12
Name: count, dtype: int64
Unclassified cases dropped: 14

Contingency Table (Sector vs Has Impact):
has_impact       False  True 
inferred_sector              
Digital              7     19
Physical             5      7

Fisher's Exact Test Results:
Odds Ratio: 0.5158
P-value: 0.4605
Result: No statistically significant association found.

Physical Sector Impact Rate: 7/12 (58.33%)
Digital Sector Impact Rate: 19/26 (73.08%)

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
