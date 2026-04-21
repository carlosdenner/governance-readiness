import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys
import os

# 1. Load Dataset
filename = 'astalabs_discovery_all_data.csv'
if os.path.exists(filename):
    filepath = filename
elif os.path.exists(os.path.join('..', filename)):
    filepath = os.path.join('..', filename)
else:
    print("Dataset not found.")
    sys.exit(1)

print(f"Loading {filepath}...")
df = pd.read_csv(filepath, low_memory=False)

# 2. Filter for AIID Incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents subset size: {len(aiid)}")

# 3. Column Selection
# Priority lists based on metadata
autonomy_priorities = ['Autonomy Level', 'autonomy', '81: Autonomy Level']
harm_priorities = ['Harm Domain', 'harm_type', 'Tangible Harm', '73: Harm Domain']

def find_col(df, priorities):
    for col in priorities:
        if col in df.columns:
            # Check if it has data
            if df[col].notna().sum() > 0:
                return col
    return None

autonomy_col = find_col(aiid, autonomy_priorities)
harm_col = find_col(aiid, harm_priorities)

print(f"Selected Autonomy Column: '{autonomy_col}'")
print(f"Selected Harm Column: '{harm_col}'")

if not autonomy_col or not harm_col:
    print("CRITICAL: Could not identify suitable columns. Exiting.")
    print("Available columns:", aiid.columns.tolist())
    sys.exit(1)

# 4. Inspect Data for Mapping
print("\n--- Autonomy Value Counts ---")
print(aiid[autonomy_col].value_counts().head(10))
print("\n--- Harm Value Counts ---")
print(aiid[harm_col].value_counts().head(10))

# 5. Define Mappings
def map_autonomy(val):
    if pd.isna(val): return None
    s = str(val).lower().strip()
    
    # Mapping based on observed values 'Autonomy1', 'Autonomy2', 'Autonomy3'
    # Assuming 1-2 = Low, 3+ = High
    if 'autonomy1' in s or 'autonomy2' in s or 'level 1' in s or 'level 2' in s or 'low' in s:
        return 'Low Autonomy'
    if 'autonomy3' in s or 'autonomy4' in s or 'autonomy5' in s or 'level 3' in s or 'level 4' in s or 'high' in s:
        return 'High Autonomy'
    return None

def map_harm(val):
    if pd.isna(val): return None
    s = str(val).lower().strip()
    
    # Physical/Safety vs Non-Physical
    physical_keys = ['physical', 'safety', 'death', 'kill', 'injur', 'life', 'violence']
    if any(k in s for k in physical_keys):
        return 'Physical/Safety'
    
    # Explicit Non-Physical keys to avoid mapping garbage/unrelated text
    non_physical_keys = ['economic', 'financial', 'rights', 'civil', 'bias', 'discrimination', 'reputation', 'psychological', 'performance', 'near miss']
    if any(k in s for k in non_physical_keys):
        return 'Non-Physical'
        
    return None

# Apply Mappings
aiid['Autonomy_Bin'] = aiid[autonomy_col].apply(map_autonomy)
aiid['Harm_Bin'] = aiid[harm_col].apply(map_harm)

# Filter valid rows
analysis_df = aiid.dropna(subset=['Autonomy_Bin', 'Harm_Bin'])
print(f"\nRows available for analysis: {len(analysis_df)}")

if len(analysis_df) == 0:
    print("No data after mapping. Please check the value counts above and adjust mapping logic.")
    sys.exit(0)

# 6. Statistical Analysis
contingency = pd.crosstab(analysis_df['Autonomy_Bin'], analysis_df['Harm_Bin'])
print("\n--- Contingency Table (Counts) ---")
print(contingency)

contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100
print("\n--- Contingency Table (Percentages) ---")
print(contingency_pct.round(2))

chi2, p, dof, ex = stats.chi2_contingency(contingency)
print(f"\nChi-Square Test Results:")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"p-value: {p:.4e}")

if p < 0.05:
    print("Result: Statistically Significant (Reject Null)")
else:
    print("Result: Not Significant (Fail to Reject Null)")

# 7. Visualization
plt.figure(figsize=(10, 6))
# Use simple colors: Red for Physical, Blue for Non-Physical if possible, but auto-assignment is fine
ax = contingency_pct.plot(kind='bar', stacked=True, color=['#1f77b4', '#d62728'], figsize=(8, 6))
plt.title('Harm Domain Distribution by System Autonomy')
plt.xlabel('Autonomy Level')
plt.ylabel('Percentage of Incidents')
plt.legend(title='Harm Domain', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()