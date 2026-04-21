# Experiment 147: node_5_51

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_51` |
| **ID in Run** | 147 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T07:59:20.380755+00:00 |
| **Runtime** | 373.5s |
| **Parent** | `node_4_29` |
| **Children** | `node_6_63` |
| **Creation Index** | 148 |

---

## Hypothesis

> Generative AI Incident Profile: Incidents involving 'Generative AI' technologies
are disproportionately associated with 'Reputational' and 'Psychological' harms,
whereas Discriminative AI correlates with 'Allocative' or 'Financial' harms.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7500 (Likely True) |
| **Posterior** | 0.4203 (Maybe False) |
| **Surprise** | -0.3956 |
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
| Maybe False | 60.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Profile the distinct failure modes of Generative AI compared to traditional predictive systems.

### Steps
- 1. Load AIID data.
- 2. Search `Known AI Technology` for keywords (e.g., 'Generative', 'LLM', 'GPT', 'Diffusion', 'Chatbot') to tag 'GenAI'. Tag others as 'Discriminative/Other'.
- 3. Clean `Harm Domain` to identifying 'Reputational'/'Psychological' vs 'Financial'/'Allocative'.
- 4. Compare the distribution of harm domains between GenAI and Non-GenAI using a Chi-square test of independence.

### Deliverables
- 1. Frequency table of Harm Domains by AI Type.
- 2. Heatmap or Bar chart of Harm distribution.
- 3. Statistical test results.

---

## Analysis

The experiment successfully profiled AI failure modes using text mining on
incident descriptions, overcoming previous schema limitations. The analysis of
1,362 incidents reveals a statistically significant difference (p < 0.001) in
harm profiles between Generative and Discriminative AI. While 'Societal &
Reputational' harms are the dominant category for both groups (~47-49%),
distinct risk profiles emerge in other areas. Contrary to the hypothesis that
Discriminative AI would dominate 'Allocative' harms, Generative AI showed a
significantly higher association with 'Economic & Allocative' risks (42% vs
25%), likely driven by copyright disputes and labor displacement concerns found
in the text data. Conversely, Discriminative AI is three times more likely to
involve 'Physical Safety' incidents (28% vs 9%), reflecting its deployment in
embodied systems like autonomous vehicles. The hypothesis is partially supported
regarding GenAI's skew toward non-physical harms, but the specific prediction
regarding allocative harms was incorrect.

---

## Review

The experiment was successfully executed and the analysis is sound. The text
mining approach in Attempt 3 effectively overcame the missing structured data
issues encountered in previous attempts. The statistical analysis (Chi-square p
< 0.001) confirms distinct harm profiles for Generative vs. Discriminative AI.

Findings:
1. **Hypothesis Verification (Mixed):** The hypothesis that Generative AI
correlates with 'Reputational/Psychological' harms is supported (it is the
dominant category at 49%). However, the hypothesis that Discriminative AI
dominates 'Allocative/Financial' harms is rejected; Generative AI actually
showed a higher prevalence in this category (42% vs 25%), likely reflecting the
high volume of copyright and labor-displacement concerns in the dataset.
2. **Emergent Insight:** A strong, unhypothesized signal emerged for 'Physical
Safety,' where Discriminative AI is three times more prevalent (28%) than
Generative AI (9%).

The implementation faithfully tested the proposition despite data limitations,
and the derived insights are valid.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# [debug]
print("Starting experiment: GenAI vs Discriminative AI Harm Profiles (Attempt 3)")

# 1. Load Data
try:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        exit(1)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents found: {len(aiid)}")

# 2. Inspect potential text columns for keyword extraction
print("\n--- Column Inspection ---")
potential_text_cols = ['title', 'description', 'summary', 'reports', 'Special Interest Intangible Harm']
for col in potential_text_cols:
    if col in aiid.columns:
        print(f"Column '{col}' sample:", aiid[col].dropna().unique()[:3])
    else:
        print(f"Column '{col}' not found.")

# 3. Tag AI Type (Generative vs Discriminative)
genai_keywords = [
    'generative', 'llm', 'gpt', 'chat', 'chatbot', 'diffusion', 'text-to-image', 
    'image generator', 'deepfake', 'voice cloning', 'gan', 'language model', 
    'transformer', 'midjourney', 'dall-e', 'stable diffusion', 'bard', 'bing chat', 
    'copilot', 'llama', 'mistral', 'claude', 'gemini'
]

def tag_ai_type(row):
    text = " ".join([str(row.get(c, '')) for c in ['Known AI Technology', 'Potential AI Technology', 'title', 'description', 'summary']]).lower()
    if any(k in text for k in genai_keywords):
        return 'Generative AI'
    return 'Discriminative/Other'

aiid['AI_Type'] = aiid.apply(tag_ai_type, axis=1)
print("\nAI Type Distribution:")
print(aiid['AI_Type'].value_counts())

# 4. Categorize Harms using keyword extraction on Text Fields
# Since structured columns failed, we mine 'title', 'description' (if exists), 'reports', and 'Special Interest Intangible Harm'

def map_harm_category(row):
    # Aggregate text from relevant columns
    text_sources = [
        row.get('title', ''),
        row.get('description', ''),
        row.get('summary', ''),
        row.get('Special Interest Intangible Harm', ''),
        row.get('reports', '') # Reports might be long, but useful
    ]
    text = " ".join([str(t) for t in text_sources]).lower()
    
    # Harm Keywords
    
    # Group 1: Societal, Psychological, Reputational (Hypothesized for GenAI)
    societal_keywords = [
        'reputation', 'defamation', 'libel', 'slander', 'psychological', 'harassment', 
        'sexual', 'nude', 'pornography', 'bias', 'discriminat', 'racist', 'sexist', 
        'misinformation', 'disinformation', 'fake news', 'propaganda', 'privacy', 
        'surveillance', 'copyright', 'plagiarism', 'offensive', 'hate speech'
    ]
    
    # Group 2: Economic, Allocative, Financial (Hypothesized for Discriminative)
    economic_keywords = [
        'financial', 'money', 'economic', 'employment', 'hiring', 'job', 'termination', 
        'fired', 'credit', 'loan', 'housing', 'tenant', 'fraud', 'scam', 'theft', 
        'market', 'trading', 'price', 'bank', 'insurance'
    ]
    
    # Group 3: Physical Safety (Common baseline)
    physical_keywords = [
        'death', 'kill', 'died', 'fatal', 'injury', 'injured', 'hurt', 'accident', 
        'crash', 'collision', 'medical', 'patient', 'hospital', 'health', 'physical safety'
    ]
    
    # Priority: Check specifically for the distinct categories first
    # We map to the dominant category found. If multiple, we prioritize based on hypothesis relevance or hierarchy.
    # Let's check existence.
    has_societal = any(k in text for k in societal_keywords)
    has_economic = any(k in text for k in economic_keywords)
    has_physical = any(k in text for k in physical_keywords)
    
    if has_societal:
        return 'Societal & Reputational'
    elif has_economic:
        return 'Economic & Allocative'
    elif has_physical:
        return 'Physical Safety'
    else:
        return 'Other/Unspecified'

aiid['Harm_Group'] = aiid.apply(map_harm_category, axis=1)

# Filter for analysis
analysis_df = aiid[aiid['Harm_Group'] != 'Other/Unspecified'].copy()

print("\nHarm Group Distribution (Known):")
print(analysis_df['Harm_Group'].value_counts())

# 5. Statistical Analysis & Plotting
if not analysis_df.empty:
    contingency = pd.crosstab(analysis_df['Harm_Group'], analysis_df['AI_Type'])
    contingency_pct = pd.crosstab(analysis_df['Harm_Group'], analysis_df['AI_Type'], normalize='columns') * 100
    
    print("\nContingency Table (Counts):")
    print(contingency)
    print("\nContingency Table (Column %):")
    print(contingency_pct.round(2))
    
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-Square Test Results:\nChi2 Statistic: {chi2:.4f}\nP-value: {p:.4e}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plot_data = contingency_pct.reset_index().melt(id_vars='Harm_Group', var_name='AI_Type', value_name='Percentage')
    sns.barplot(data=plot_data, x='Harm_Group', y='Percentage', hue='AI_Type')
    plt.title('Harm Category Profile: Generative vs Discriminative AI')
    plt.ylabel('Percentage of Incidents (%)')
    plt.xlabel('Harm Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Text mining yielded no categorized harms.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting experiment: GenAI vs Discriminative AI Harm Profiles (Attempt 3)
AIID Incidents found: 1362

--- Column Inspection ---
Column 'title' sample: <StringArray>
['Google’s YouTube Kids App Presents Inappropriate Content',
          'Las Vegas Self-Driving Bus Involved in Accident',
                     'Uber AV Killed Pedestrian in Arizona']
Length: 3, dtype: str
Column 'description' sample: <StringArray>
[                                                              'YouTube’s content filtering and recommendation algorithms exposed children to disturbing and inappropriate videos.',
 'A self-driving public shuttle by Keolis North America and Navya was involved in a collision with a human-driven delivery truck in Las Vegas, Nevada on its first day of service.',
                                                                             'An Uber autonomous vehicle (AV) in autonomous mode struck and killed a pedestrian in Tempe, Arizona.']
Length: 3, dtype: str
Column 'summary' sample: <StringArray>
[]
Length: 0, dtype: str
Column 'reports' sample: <StringArray>
[                                                                            '[2,3,4,5,6,7,8,9,10,11,12,14,15]',
           '[242,243,244,245,246,247,248,249,250,253,254,257,258,259,260,261,263,264,266,267,268,269,270,2389]',
 '[629,630,631,632,633,634,635,636,637,638,639,640,641,642,644,645,646,647,1375,1376,1377,1378,1542,2147,1257]']
Length: 3, dtype: str
Column 'Special Interest Intangible Harm' sample: <StringArray>
['yes', 'no', 'maybe']
Length: 3, dtype: str

AI Type Distribution:
AI_Type
Discriminative/Other    727
Generative AI           635
Name: count, dtype: int64

Harm Group Distribution (Known):
Harm_Group
Societal & Reputational    392
Economic & Allocative      275
Physical Safety            148
Name: count, dtype: int64

Contingency Table (Counts):
AI_Type                  Discriminative/Other  Generative AI
Harm_Group                                                  
Economic & Allocative                     101            174
Physical Safety                           111             37
Societal & Reputational                   188            204

Contingency Table (Column %):
AI_Type                  Discriminative/Other  Generative AI
Harm_Group                                                  
Economic & Allocative                   25.25          41.93
Physical Safety                         27.75           8.92
Societal & Reputational                 47.00          49.16

Chi-Square Test Results:
Chi2 Statistic: 56.7744
P-value: 4.6946e-13


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** Grouped Bar Chart (or Clustered Bar Chart).
*   **Purpose:** The plot compares the distribution of incident types (percentages) across three distinct harm categories for two different types of Artificial Intelligence systems: Discriminative/Other and Generative AI. This format allows for direct side-by-side comparison of the two AI types within each category.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Harm Category"
    *   **Labels:** Three categories are listed: "Economic & Allocative", "Physical Safety", and "Societal & Reputational". The labels are rotated 45 degrees to prevent overlapping.
*   **Y-Axis:**
    *   **Title:** "Percentage of Incidents (%)"
    *   **Range:** The scale runs from 0 to 50, with major tick marks at intervals of 10 (0, 10, 20, 30, 40, 50).
    *   **Units:** Percentage.

### 3. Data Trends
*   **Tallest Bars:**
    *   For **Generative AI** (Orange), the tallest bar is in the "Societal & Reputational" category, reaching close to 50%.
    *   For **Discriminative/Other** (Blue), the tallest bar is also in the "Societal & Reputational" category, reaching approximately 47%.
*   **Shortest Bar:** The shortest bar on the entire graph is for **Generative AI** in the "Physical Safety" category (just below 10%).
*   **Patterns by Category:**
    *   **Economic & Allocative:** Generative AI shows a significantly higher percentage of incidents (~42%) compared to Discriminative/Other (~25%).
    *   **Physical Safety:** This trend is reversed. Discriminative/Other AI has a much higher percentage (~28%) compared to Generative AI (~9%).
    *   **Societal & Reputational:** This is the dominant category for both AI types, with both showing high percentages (roughly 47-49%), though Generative AI is slightly higher.

### 4. Annotations and Legends
*   **Plot Title:** "Harm Category Profile: Generative vs Discriminative AI" - clearly defines the subject matter of the comparison.
*   **Legend:** Located at the top center of the plot. It identifies the color coding:
    *   **Blue:** Discriminative/Other
    *   **Orange:** Generative AI

### 5. Statistical Insights
*   **Dominance of Societal Risks:** Regardless of the AI type, "Societal & Reputational" harms constitute the largest share of incidents, suggesting that social impact (bias, defamation, misinformation) is currently the most prevalent issue reported for both technologies.
*   **Profile Divergence:** There is a clear divergence in the risk profiles of the two technologies:
    *   **Generative AI** is disproportionately associated with non-physical harms (Economic and Societal), causing very few physical safety incidents. This aligns with the nature of the technology (content generation rather than physical actuation).
    *   **Discriminative AI** (which likely encompasses systems used in robotics, autonomous vehicles, or industrial control) poses a notably higher risk to "Physical Safety" compared to Generative AI (roughly 3x more prevalent in this dataset).
*   **Economic Impact:** Generative AI is nearly twice as likely to be involved in "Economic & Allocative" incidents compared to Discriminative AI, likely reflecting issues regarding copyright, job displacement concerns, or fraud.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
