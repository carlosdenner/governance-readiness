# Experiment 60: node_4_25

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_25` |
| **ID in Run** | 60 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T03:52:08.100346+00:00 |
| **Runtime** | 365.1s |
| **Parent** | `node_3_13` |
| **Children** | `node_5_26`, `node_5_62`, `node_5_100` |
| **Creation Index** | 61 |

---

## Hypothesis

> The Generative-Malice Link: Incidents involving 'Generative' AI technologies
(e.g., LLMs, GANs) are significantly more likely to be caused by 'Intentional'
actors compared to 'Discriminative' AI technologies, which are more prone to
unintentional errors.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7419 (Likely True) |
| **Posterior** | 0.3516 (Maybe False) |
| **Surprise** | -0.4683 |
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
| Maybe False | 36.0 |
| Definitely False | 24.0 |

---

## Experiment Plan

**Objective:** Investigate if Generative AI is disproportionately weaponized compared to traditional AI.

### Steps
- 1. Filter 'aiid_incidents'.
- 2. Categorize '84_Known AI Technology' into 'Generative' (keywords: Language Model, GAN, Generative, Diffusion) vs 'Discriminative' (Classifier, Regression, Decision Tree).
- 3. Analyze '82_Intentional Harm' (Yes vs No).
- 4. Compare the proportion of Intentional Harm in Generative vs Discriminative groups using a Z-test or Chi-square.

### Deliverables
- Proportion of Intentional Harm by Technology Class; Statistical test results.

---

## Analysis

The experiment successfully tested the 'Generative-Malice Link' hypothesis using
the AIID dataset. After parsing the text-based 'Intentional Harm' column
(mapping descriptions starting with 'Yes' to Intentional and 'No' to
Unintentional) and classifying 'Known AI Technology' into Generative vs.
Discriminative categories, 95 incidents were available for the final analysis.
The results indicated a distinct lack of intentional malice recorded in this
subset for both groups. Discriminative AI systems (n=80) showed a 2.5% rate of
intentional harm (2 incidents), while Generative AI systems (n=15) showed a 0%
rate (0 incidents). The Chi-square test yielded a p-value of 1.00, indicating no
statistically significant difference. Consequently, the hypothesis is rejected;
within this dataset, there is no evidence to suggest that Generative AI
technologies are more frequently associated with intentional harm than
Discriminative technologies.

---

## Review

The experiment successfully tested the 'Generative-Malice Link' hypothesis using
the AIID dataset. The implementation demonstrated strong adaptability by
identifying and resolving data quality issues: specifically, parsing the 'Known
AI Technology' column (which contained comma-separated lists) and the
'Intentional Harm' column (which contained descriptive sentences rather than
boolean values).

Analysis of 95 valid incidents revealed no evidence to support the hypothesis
that Generative AI is disproportionately weaponized.
- **Discriminative AI** (n=80): 2.5% of incidents were intentional (2/80).
- **Generative AI** (n=15): 0% of incidents were intentional (0/15).

The Chi-square test resulted in a p-value of 1.00, statistically confirming no
significant association between the technology type and the intent of the harm
in this dataset. The hypothesis is therefore rejected.

---

## Code

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

print("Starting experiment: The Generative-Malice Link")

# Load dataset
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Total AIID incidents: {len(aiid)}")

# Relevant columns
tech_col = 'Known AI Technology'
intent_col = 'Intentional Harm'

# Drop rows where technology or intent is missing
aiid_clean = aiid.dropna(subset=[tech_col, intent_col])
print(f"Incidents with known technology and intent: {len(aiid_clean)}")

# Define classification logic for Technology
def classify_tech(text):
    text = str(text).lower()
    # Generative keywords
    gen_keywords = [
        'generative', 'gan', 'language model', 'llm', 'gpt', 'diffusion', 'transformer', 
        'text-to-image', 'chatbot', 'deepfake', 'image generator', 'voice cloning', 
        'synthesizer', 'stylegan', 'midjourney', 'dall-e', 'stable diffusion', 'bert', 
        'chatgpt', 'creative', 'writing', 'art'
    ]
    # Discriminative keywords
    disc_keywords = [
        'classifier', 'classification', 'regression', 'decision tree', 'svm', 'support vector', 
        'recommendation', 'ranking', 'detection', 'recognition', 'predictive', 'scoring', 
        'computer vision', 'object detection', 'face recognition', 'neural network', 
        'deep learning', 'distributional learning', 'content-based filtering', 
        'collaborative filtering', 'segmentation', 'clustering', 'reinforcement learning', 
        'supervised', 'unsupervised', 'monitoring'
    ]
    
    # Check for generative first (hypothesis interest)
    if any(k in text for k in gen_keywords):
        return 'Generative'
    # Check for discriminative
    elif any(k in text for k in disc_keywords):
        return 'Discriminative'
    else:
        return 'Other'

# Apply classification
aiid_clean['Tech_Class'] = aiid_clean[tech_col].apply(classify_tech)

# Clean Intent column based on previous debug findings
def clean_intent(val):
    val_str = str(val).lower().strip()
    # Based on debug output: 'No. Not intentionally...', 'Yes. Intentionally...'
    if val_str.startswith('yes'):
        return 'Intentional'
    elif val_str.startswith('no'):
        return 'Unintentional'
    return 'Unknown'

aiid_clean['Intent_Class'] = aiid_clean[intent_col].apply(clean_intent)

# Filter for analysis
analysis_df = aiid_clean[
    (aiid_clean['Tech_Class'].isin(['Generative', 'Discriminative'])) & 
    (aiid_clean['Intent_Class'].isin(['Intentional', 'Unintentional']))
]

print(f"Rows used for analysis: {len(analysis_df)}")

# Generate contingency table
contingency = pd.crosstab(analysis_df['Tech_Class'], analysis_df['Intent_Class'])

print("\n--- Contingency Table ---")
print(contingency)

# Calculate proportions and run statistics
if not contingency.empty and contingency.shape == (2, 2):
    # Add Total column for rate calculation
    contingency['Total'] = contingency['Intentional'] + contingency['Unintentional']
    contingency['Intentional_Rate'] = contingency['Intentional'] / contingency['Total']
    
    print("\n--- Intentional Harm Rates ---")
    print(contingency[['Intentional', 'Total', 'Intentional_Rate']])

    # Statistical Test: Chi-square
    # Extract only the count data
    obs = contingency[['Intentional', 'Unintentional']].values
    chi2, p, dof, expected = chi2_contingency(obs)

    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    alpha = 0.05
    if p < alpha:
        print("Result: Statistically significant difference found.")
        gen_rate = contingency.loc['Generative', 'Intentional_Rate']
        disc_rate = contingency.loc['Discriminative', 'Intentional_Rate']
        if gen_rate > disc_rate:
            print(f"Generative AI has a HIGHER rate of intentional harm ({gen_rate:.2%} vs {disc_rate:.2%}).")
        else:
            print(f"Generative AI has a LOWER rate of intentional harm ({gen_rate:.2%} vs {disc_rate:.2%}).")
    else:
        print("Result: No statistically significant difference found.")
else:
    print("Insufficient data for full 2x2 analysis (one or more categories might be empty).")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting experiment: The Generative-Malice Link
Total AIID incidents: 1362
Incidents with known technology and intent: 115
Rows used for analysis: 95

--- Contingency Table ---
Intent_Class    Intentional  Unintentional
Tech_Class                                
Discriminative            2             78
Generative                0             15

--- Intentional Harm Rates ---
Intent_Class    Intentional  Total  Intentional_Rate
Tech_Class                                          
Discriminative            2     80             0.025
Generative                0     15             0.000

Chi-square Statistic: 0.0000
P-value: 1.0000e+00
Result: No statistically significant difference found.

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
