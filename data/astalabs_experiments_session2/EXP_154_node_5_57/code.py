import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load dataset
print("Loading dataset...")
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Check unified_evidence_base for comparison
unified_df = df[df['source_table'] == 'unified_evidence_base'].copy()
print(f"Unified rows: {len(unified_df)}")

# Check AIID incidents again
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID rows: {len(aiid_df)}")

# Determine which dataframe has better data for 'Harm Domain' or similar
# We will prioritize AIID but fallback to search if needed.

# Define categorization function based on keywords in text if structured column fails
def categorize_from_text(row):
    # Combine relevant text columns
    text = str(row.get('title', '')) + " " + str(row.get('description', '')) + " " + str(row.get('summary', '')) + " " + str(row.get('Harm Domain', ''))
    text = text.lower()
    
    # Classification logic
    is_physical = any(w in text for w in ['death', 'injury', 'kill', 'physical', 'safety', 'accident', 'collision', 'robot', 'autonomous vehicle', 'drone'])
    is_bias = any(w in text for w in ['bias', 'discrimination', 'racist', 'sexist', 'gender', 'race', 'unfair', 'stereotype', 'facial recognition', 'demographic'])
    
    if is_physical and not is_bias:
        return 'Physical Safety'
    if is_bias and not is_physical:
        return 'Bias & Discrimination'
    return 'Other/Mixed'

# Define Severity Mapping
# Based on previous debug: 'AI tangible harm event', 'AI tangible harm near-miss', 'AI tangible harm issue'
severity_map = {
    'ai tangible harm event': 3,
    'ai tangible harm near-miss': 2,
    'ai tangible harm issue': 1,
    'none': 0,
    'unclear': 0
}

def map_severity(val):
    if pd.isna(val): return None
    s = str(val).lower().strip()
    return severity_map.get(s, None)

# Apply logic to AIID incidents
# We use the text-based classification because 'Harm Domain' column was shown to be sparse/boolean
print("Classifying domains based on text analysis...")
aiid_df['domain_group'] = aiid_df.apply(categorize_from_text, axis=1)

# Map Severity
# We assume 'AI Harm Level' is the column name based on previous debug output
print("Mapping severity scores...")
# Ensure we use the correct column name. Previous debug showed 'AI Harm Level'.
aiid_df['severity_score'] = aiid_df['AI Harm Level'].apply(map_severity)

# Filter for analysis
analysis_df = aiid_df.dropna(subset=['severity_score', 'domain_group'])
physical_group = analysis_df[analysis_df['domain_group'] == 'Physical Safety']['severity_score']
bias_group = analysis_df[analysis_df['domain_group'] == 'Bias & Discrimination']['severity_score']

print(f"Physical Safety N={len(physical_group)}")
print(f"Bias & Discrimination N={len(bias_group)}")

# Perform Statistics
if len(physical_group) > 5 and len(bias_group) > 5:
    print("Performing Mann-Whitney U Test...")
    u_stat, p_val = stats.mannwhitneyu(physical_group, bias_group, alternative='greater')
    
    print(f"Mann-Whitney U statistic: {u_stat}")
    print(f"P-value (Physical > Bias): {p_val:.5f}")
    
    print(f"Median Severity (Physical): {physical_group.median()}")
    print(f"Median Severity (Bias): {bias_group.median()}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    data = [physical_group, bias_group]
    plt.boxplot(data, labels=['Physical Safety', 'Bias & Discrimination'])
    plt.title('AI Harm Severity: Physical vs Bias Incidents')
    plt.ylabel('Severity Level (0=None, 1=Issue, 2=Near-Miss, 3=Event)')
    plt.yticks([0, 1, 2, 3], ['None/Unclear', 'Issue', 'Near-Miss', 'Event'])
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()
else:
    print("Insufficient data for statistical analysis after text-based classification.")
    print("Sample of Domain Grouping:", analysis_df['domain_group'].value_counts())
    print("Sample of Severity Mapping:", analysis_df['severity_score'].value_counts())
