import pandas as pd
import scipy.stats as stats
import sys
import os

# Experiment: Autonomy-Harm Escalation (Text Analysis Approach)
# Objective: Infer 'Physical Harm' from incident descriptions since structured labels are missing.

# 1. Load Dataset
filenames = ['astalabs_discovery_all_data.csv', '../astalabs_discovery_all_data.csv']
file_path = None
for fn in filenames:
    if os.path.exists(fn):
        file_path = fn
        break

if not file_path:
    print("Error: Dataset not found.")
    sys.exit(1)

try:
    df = pd.read_csv(file_path, low_memory=False)
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# 2. Filter for AIID Incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Total AIID Incidents: {len(aiid_df)}")

# 3. Identify Columns
# Autonomy
autonomy_col = next((c for c in aiid_df.columns if 'autonomy' in c.lower() and 'level' in c.lower()), None)

# Text Column (Description/Summary)
# We prefer 'description' or 'summary'
text_col = next((c for c in aiid_df.columns if c.lower() in ['description', 'summary', 'text', 'abstract']), None)

if not autonomy_col or not text_col:
    print(f"Critical columns missing. Autonomy: {autonomy_col}, Text: {text_col}")
    # Fallback search for text column
    candidates = [c for c in aiid_df.columns if aiid_df[c].dtype == 'object' and aiid_df[c].str.len().mean() > 50]
    if candidates:
        text_col = candidates[0]
        print(f"Fallback: Using '{text_col}' as text column.")
    else:
        print("No suitable text column found.")
        sys.exit(1)

print(f"Using columns: Autonomy='{autonomy_col}', Text='{text_col}'")

# 4. Mappings

def map_autonomy(val):
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    # High: Autonomy3
    if 'Autonomy3' in val_str:
        return 'High Autonomy'
    # Low/Medium: Autonomy1, Autonomy2
    elif 'Autonomy1' in val_str or 'Autonomy2' in val_str:
        return 'Low/Medium Autonomy'
    return None

def map_harm_from_text(val):
    if pd.isna(val):
        return 'Non-Physical Harm'
    text = str(val).lower()
    # Keywords for Physical Harm / Safety
    physical_keywords = [
        'killed', 'death', 'died', 'injury', 'injured', 'hurt', 'collision', 
        'crash', 'hit', 'accident', 'safety', 'physical', 'bodily', 'fatal', 
        'wound', 'burn', 'assault', 'violence', 'attack', 'robot'
    ]
    if any(k in text for k in physical_keywords):
        return 'Physical Harm'
    return 'Non-Physical Harm'

aiid_df['Autonomy_Category'] = aiid_df[autonomy_col].apply(map_autonomy)
aiid_df['Harm_Category'] = aiid_df[text_col].apply(map_harm_from_text)

# 5. Analysis
analysis_df = aiid_df.dropna(subset=['Autonomy_Category', 'Harm_Category'])
print(f"Rows used for analysis: {len(analysis_df)}")

# Check distribution
print("Harm Category Distribution:")
print(analysis_df['Harm_Category'].value_counts())

contingency_table = pd.crosstab(analysis_df['Autonomy_Category'], analysis_df['Harm_Category'])
print("\n--- Contingency Table (Counts) ---")
print(contingency_table)

if contingency_table.size > 0:
    # Percentages
    prop_table = pd.crosstab(analysis_df['Autonomy_Category'], analysis_df['Harm_Category'], normalize='index') * 100
    print("\n--- Contingency Table (Percentages) ---")
    print(prop_table.round(2))
    
    # Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print("\n--- Chi-Square Test Results ---")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"p-value: {p:.4e}")
    
    if p < 0.05:
        print("Result: Statistically significant relationship found.")
    else:
        print("Result: No statistically significant relationship found.")
else:
    print("Insufficient data for analysis.")
