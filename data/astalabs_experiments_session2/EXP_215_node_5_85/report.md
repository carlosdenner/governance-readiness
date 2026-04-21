# Experiment 215: node_5_85

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_85` |
| **ID in Run** | 215 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T11:05:29.418213+00:00 |
| **Runtime** | 223.6s |
| **Parent** | `node_4_48` |
| **Children** | None |
| **Creation Index** | 216 |

---

## Hypothesis

> The Reality Gap: Sectors with the highest frequency of reported AI incidents
(from AIID) show the lowest average Governance Readiness scores (from EO13960),
suggesting a mismatch between risk prevalence and regulatory preparedness.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6371 (Maybe True) |
| **Posterior** | 0.3049 (Maybe False) |
| **Surprise** | -0.3986 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 22.0 |
| Uncertain | 3.0 |
| Maybe False | 5.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 32.0 |
| Definitely False | 28.0 |

---

## Experiment Plan

**Objective:** Correlate historical failure rates with current governance maturity across sectors.

### Steps
- 1. From `aiid_incidents`, aggregate incident counts by `Sector of Deployment` (normalized to EO13960 topic areas).
- 2. From `eo13960_scored`, calculate a 'Governance Score' for each `8_topic_area` (average compliance across key controls like Impact Assessment, Bias Mitigation, Monitoring).
- 3. Inner join the two aggregates on the Sector/Topic name.
- 4. Perform a Pearson or Spearman correlation analysis between 'Incident Count' and 'Governance Score'.

### Deliverables
- Scatter Plot of Incident Volume vs. Governance Score with regression line; Correlation coefficient.

---

## Analysis

The experiment successfully correlated AI incident frequency with governance
readiness across 6 mapped sectors. Contrary to the hypothesis that high-incident
sectors would have lower governance scores (a 'Reality Gap'), the analysis
revealed a positive correlation (Pearson r=0.76, p=0.08; Spearman r=0.26,
p=0.62). The 'Transportation' sector emerged as a high-leverage outlier,
exhibiting both the highest incident count (~28) and the highest governance
readiness, suggesting that critical infrastructure sectors may be maturing their
governance in response to risk. However, the 'Government Services' sector
supported the original premise, showing high incident volume with near-zero
governance readiness. Overall, the hypothesis is rejected: there is no
statistically significant evidence of a systemic negative correlation between
risk prevalence and regulatory preparedness; if anything, the trend is
positively driven by safety-critical industries.

---

## Review

The experiment was successfully executed. The code correctly mapped AIID
incident counts to EO13960 governance scores for 6 overlapping sectors and
performed the required correlation analysis.

**Hypothesis Test:** The hypothesis that high-incident sectors would show lower
governance readiness (a negative correlation) was **rejected**.

**Findings:**
1.  **Positive Correlation:** The analysis revealed a strong positive Pearson
correlation (r=0.76, p=0.08) and a weak positive Spearman correlation (r=0.26,
p=0.62), contradicting the 'Reality Gap' hypothesis.
2.  **Sector Dynamics:** The 'Transportation' sector acted as a significant
high-leverage point, exhibiting both the highest incident volume (~28) and the
highest governance score, suggesting that safety-critical industries are
maturing their governance in response to risk.
3.  **Partial Support:** 'Government Services' was the primary outlier
supporting the original premise, showing high incident frequency with near-zero
governance readiness, but this was insufficient to drive a negative trend
overall.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os
import sys

# Check for dataset in current or parent directory
filename = 'astalabs_discovery_all_data.csv'
if os.path.exists(filename):
    filepath = filename
elif os.path.exists(f'../{filename}'):
    filepath = f'../{filename}'
else:
    print(f"Error: {filename} not found.")
    print(f"CWD: {os.getcwd()}")
    print(f"Files in CWD: {os.listdir('.')}")
    try:
        print(f"Files in Parent: {os.listdir('..')}")
    except: 
        pass
    sys.exit(1)

print(f"Loading dataset from {filepath}...")
df = pd.read_csv(filepath, low_memory=False)

# --- 1. Prepare AIID Data ---
aiid = df[df['source_table'] == 'aiid_incidents'].copy()

# Identify Sector column (Metadata says '78: Sector of Deployment')
# We'll look for it by name similarity to be safe
sector_cols = [c for c in aiid.columns if 'Sector' in c and 'Deployment' in c]
sector_col = sector_cols[0] if sector_cols else '78: Sector of Deployment'

# Clean and Aggregate AIID
# Remove NaN sectors
aiid = aiid.dropna(subset=[sector_col])
# Standardize names (lowercase, strip)
aiid['clean_sector'] = aiid[sector_col].astype(str).str.lower().str.strip()

sector_counts = aiid['clean_sector'].value_counts().reset_index()
sector_counts.columns = ['sector_term', 'incident_count']

print("Top AIID Sectors (cleaned):")
print(sector_counts.head())

# --- 2. Prepare EO13960 Data ---
eo = df[df['source_table'] == 'eo13960_scored'].copy()

# Identify Topic Area (Metadata: '8: 8_topic_area')
topic_cols = [c for c in eo.columns if 'topic_area' in c.lower()]
topic_col = topic_cols[0] if topic_cols else '8: 8_topic_area'

# Governance Columns to score
gov_cols_map = {
    'Impact Assessment': [c for c in eo.columns if '52_impact_assessment' in c],
    'Bias Mitigation': [c for c in eo.columns if '62_disparity_mitigation' in c],
    'Independent Eval': [c for c in eo.columns if '55_independent_eval' in c],
    'Monitoring': [c for c in eo.columns if '56_monitor_postdeploy' in c],
    'Notice': [c for c in eo.columns if '59_ai_notice' in c],
    'Opt Out': [c for c in eo.columns if '67_opt_out' in c]
}

# Flatten list of columns found
found_gov_cols = []
for k, v in gov_cols_map.items():
    if v: found_gov_cols.append(v[0])

print(f"Governance columns used: {found_gov_cols}")

# Calculate Score
# Convert values to binary. Assume 'yes', 'true', '1' are positive.
def parse_gov_bool(val):
    s = str(val).lower().strip()
    return 1 if s in ['yes', 'true', '1', '1.0'] else 0

for col in found_gov_cols:
    eo[col + '_score'] = eo[col].apply(parse_gov_bool)

score_cols = [c + '_score' for c in found_gov_cols]
eo['gov_score'] = eo[score_cols].mean(axis=1)

# Aggregate by Topic
eo['clean_topic'] = eo[topic_col].astype(str).str.lower().str.strip()
topic_scores = eo.groupby('clean_topic')['gov_score'].mean().reset_index()

print("Top EO Topics:")
print(topic_scores.head())

# --- 3. Match Sectors and Topics ---
# Strategy: Fuzzy match or explicit mapping.
# EO Topics are US Federal specific. AIID are general.
# We will try to map common ones manually to ensure accuracy, then fallback to keyword match.

mapping_pairs = []

# Simple keyword mapping logic
for topic in topic_scores['clean_topic'].unique():
    # Keywords for this topic
    topic_words = set(w for w in topic.split() if len(w) > 3)
    
    matched_incidents = 0
    matched_sectors_list = []
    
    for _, row in sector_counts.iterrows():
        sect = row['sector_term']
        count = row['incident_count']
        
        # Check for intersection of significant words
        sect_words = set(w for w in sect.split() if len(w) > 3)
        
        # Custom overrides/synonyms
        synonyms = {
            'health': ['healthcare', 'medicine', 'medical', 'hospital'],
            'transportation': ['automotive', 'vehicle', 'driving', 'airplane', 'aviation'],
            'law enforcement': ['police', 'surveillance', 'crime', 'criminal', 'justice'],
            'finance': ['financial', 'banking', 'trading', 'credit'],
            'education': ['school', 'university', 'student', 'teaching'],
            'energy': ['power', 'grid', 'electricity', 'utility']
        }
        
        is_match = False
        if topic == sect:
            is_match = True
        elif topic_words & sect_words: # Overlap
            is_match = True
        else:
            # Check synonyms
            for k, vals in synonyms.items():
                if k in topic:
                    if any(v in sect for v in vals):
                        is_match = True
        
        if is_match:
            matched_incidents += count
            matched_sectors_list.append(sect)
            
    if matched_incidents > 0:
        mapping_pairs.append({
            'Topic': topic,
            'Governance_Score': topic_scores[topic_scores['clean_topic'] == topic]['gov_score'].values[0],
            'Incident_Count': matched_incidents,
            'Matched_Sectors': ', '.join(matched_sectors_list[:3]) # Show first few
        })

result_df = pd.DataFrame(mapping_pairs)
print("\n--- Merged Data Analysis Frame ---")
print(result_df.sort_values('Incident_Count', ascending=False))

# --- 4. Correlation Analysis ---
if len(result_df) > 3:
    corr, p_val = spearmanr(result_df['Incident_Count'], result_df['Governance_Score'])
    print(f"\nSpearman Correlation: {corr:.4f} (p-value: {p_val:.4f})")
    
    corr_p, p_val_p = pearsonr(result_df['Incident_Count'], result_df['Governance_Score'])
    print(f"Pearson Correlation: {corr_p:.4f} (p-value: {p_val_p:.4f})")

    # Plot
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=result_df, x='Incident_Count', y='Governance_Score', s=100)
    
    # Regression line
    sns.regplot(data=result_df, x='Incident_Count', y='Governance_Score', scatter=False, color='red', ci=None)

    # Labels
    for i, row in result_df.iterrows():
        plt.text(row['Incident_Count']+1, row['Governance_Score'], 
                 row['Topic'].title(), fontsize=9)

    plt.title(f'The Reality Gap: AI Incidents vs. Governance Readiness\n(Spearman r={corr:.2f}, p={p_val:.2f})')
    plt.xlabel('Reported AI Incidents (AIID)')
    plt.ylabel('Governance Readiness Score (EO 13960)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
else:
    print("Insufficient data points for correlation.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
Top AIID Sectors (cleaned):
                                         sector_term  incident_count
0                      information and communication              44
1                         transportation and storage              21
2  arts, entertainment and recreation, informatio...              14
3                         wholesale and retail trade              11
4            human health and social work activities              10
Governance columns used: ['52_impact_assessment', '62_disparity_mitigation', '55_independent_eval', '56_monitor_postdeploy', '59_ai_notice', '67_opt_out']
Top EO Topics:
                                         clean_topic  gov_score
0                          aiml platform/environment        0.0
1                                     classification        0.0
2                                      deep learning        0.0
3                                  diplomacy & trade        0.0
4  diplomacy & trade; mission-enabling (internal ...        0.0

--- Merged Data Analysis Frame ---
                                               Topic  ...                                    Matched_Sectors
5                                     transportation  ...  transportation and storage, wholesale and reta...
2  government services (includes benefits and ser...  ...  administrative and support service activities,...
3                                   health & medical  ...  human health and social work activities, educa...
0                                  diplomacy & trade  ...  wholesale and retail trade, transportation and...
1                              education & workforce  ...  education, information and communication, educ...
4                                              other  ...  other, administrative and support service acti...

[6 rows x 4 columns]

Spearman Correlation: 0.2571 (p-value: 0.6228)
Pearson Correlation: 0.7556 (p-value: 0.0823)


=== Plot Analysis (figure 1) ===
Based on the analysis of the provided image, here is the detailed breakdown:

### 1. Plot Type
*   **Type:** Scatter plot with a linear regression trend line.
*   **Purpose:** The plot aims to visualize the correlation between the frequency of reported AI incidents within specific sectors and the "Governance Readiness Score" of those same sectors. It seeks to determine if sectors experiencing more AI incidents are better prepared in terms of governance.

### 2. Axes
*   **X-Axis:**
    *   **Title:** "Reported AI Incidents (AIID)"
    *   **Range:** The axis is graduated from roughly 4 to 28 on the visual scale (ticks marked at intervals of 5, from 5 to 25).
    *   **Meaning:** Represents the count of AI incidents reported in the AI Incident Database (AIID).
*   **Y-Axis:**
    *   **Title:** "Governance Readiness Score (EO 13960)"
    *   **Range:** The axis ranges from -0.005 to 0.08, with tick marks every 0.01 unit.
    *   **Meaning:** Represents a metric for preparedness or governance capability, likely derived from Executive Order 13960 compliance or metrics.

### 3. Data Trends
*   **Overall Trend:** The red regression line slopes upward from left to right, suggesting a visual positive correlation: as reported incidents increase, the governance readiness score tends to increase. However, the data points are sparse (N=6), making the trend rely heavily on specific points.
*   **Outliers/Key Clusters:**
    *   **Transportation:** This is a distinct outlier and the most influential data point. It has the highest number of reported incidents (approx. 28) and the highest governance readiness score (approx. 0.077), sitting far apart from the other clusters in the top-right corner.
    *   **Diplomacy & Trade:** Despite having a moderate number of incidents (approx. 13), it has a readiness score of essentially zero.
    *   **Government Services:** This sector has a relatively high number of incidents (~17) but a low readiness score (< 0.01), falling well below the trend line.
    *   **Low Incident Cluster:** Sectors like "Other" and "Education & Workforce" have fewer incidents (< 10) and low-to-moderate readiness scores.

### 4. Annotations and Legends
*   **Main Title:** "The Reality Gap: AI Incidents vs. Governance Readiness". This title implies a potential mismatch or an investigation into whether reality (incidents) aligns with policy (readiness).
*   **Statistical Annotation:** "(Spearman r=0.26, p=0.62)". Located in the subtitle, this provides the statistical context for the relationship shown.
*   **Data Labels:** Each blue data point is labeled with its corresponding sector (e.g., "Transportation," "Health & Medical," "Government Services").
*   **Red Line:** A linear fit line representing the general direction of the relationship between the two variables.

### 5. Statistical Insights
*   **Weak Correlation:** The Spearman correlation coefficient (**r=0.26**) indicates a very weak positive monotonic relationship between reported incidents and governance readiness. While the red line suggests an increase, the strength of this association is negligible.
*   **Lack of Statistical Significance:** The p-value (**p=0.62**) is significantly higher than the standard threshold for statistical significance (typically p < 0.05). This means the observed relationship is likely due to chance. We cannot reject the null hypothesis; statistically, there is no proven link between the number of AI incidents a sector faces and its governance readiness score based on this dataset.
*   **Interpretation of the "Reality Gap":** The plot highlights a disconnect. One might expect that sectors with more incidents (like Government Services) would have developed higher governance readiness in response. However, aside from "Transportation," most high-incident sectors do not show correspondingly high readiness scores. The high p-value confirms that incident frequency is not currently a reliable predictor of governance readiness.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
