import pandas as pd
import scipy.stats as stats
import os

# 1. Load Data
paths = ['astalabs_discovery_all_data.csv', '../astalabs_discovery_all_data.csv']
df = None
for p in paths:
    if os.path.exists(p):
        print(f"Loading {p}...")
        df = pd.read_csv(p, low_memory=False)
        break

if df is None:
    print("Dataset not found.")
    exit(1)

# 2. Select Subset
subset = df[df['source_table'] == 'step3_incident_coding'].copy()
if subset.empty:
    subset = df[df['source_table'] == 'atlas_cases'].copy()

print(f"Analyzing {len(subset)} rows.")

# 3. Identify Columns
tactic_col = next((c for c in ['tactics_used', 'tactics'] if c in subset.columns), None)
gap_col = next((c for c in ['missing_controls', 'competency_gaps'] if c in subset.columns), None)

if not tactic_col or not gap_col:
    print("Missing columns.")
    exit(0)

# 4. Parsing Logic
# Initialize counts
# Structure: {Tactic_Type: {Gap_Type: Count}}
matrix = {
    'Exfiltration': {'Access Control': 0, 'Robustness': 0},
    'Evasion': {'Access Control': 0, 'Robustness': 0}
}

# Definitions
exfil_code = 'AML.TA0011' # Exfiltration
evasion_code = 'AML.TA0007' # Defense Evasion

# Keywords for Gap Classification
access_keywords = ['access', 'limit', 'encrypt', 'auth', 'privilege', 'permission', 'identity', 'api key']
robust_keywords = ['hardening', 'input', 'detection', 'sanitize', 'robustness', 'ensemble', 'restoration', 'adversarial']

for idx, row in subset.iterrows():
    t_str = str(row[tactic_col])
    g_str = str(row[gap_col])
    
    # Determine Tactics present
    has_exfil = exfil_code in t_str
    has_evasion = evasion_code in t_str
    
    if not (has_exfil or has_evasion):
        continue
        
    # Determine Gaps present
    g_lower = g_str.lower()
    has_access = any(k in g_lower for k in access_keywords)
    has_robust = any(k in g_lower for k in robust_keywords)
    
    # Increment Counts (Independent scenarios)
    if has_exfil:
        if has_access:
            matrix['Exfiltration']['Access Control'] += 1
        if has_robust:
            matrix['Exfiltration']['Robustness'] += 1
            
    if has_evasion:
        if has_access:
            matrix['Evasion']['Access Control'] += 1
        if has_robust:
            matrix['Evasion']['Robustness'] += 1

# 5. Build Contingency DataFrame
contingency = pd.DataFrame(matrix).T
print("\n--- Contingency Table (Incidents with Gap Type) ---")
print(contingency)

# 6. Statistical Test
if contingency.sum().sum() > 0:
    chi2, p, dof, ex = stats.chi2_contingency(contingency)
    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    
    # Calculate Percentages
    for tactic in contingency.index:
        total_hits = contingency.loc[tactic].sum()
        if total_hits > 0:
            acc_p = contingency.loc[tactic, 'Access Control'] / total_hits
            rob_p = contingency.loc[tactic, 'Robustness'] / total_hits
            print(f"{tactic}: Access Gap={acc_p:.1%}, Robustness Gap={rob_p:.1%}")
            
    if p < 0.05:
        print("\nResult: Statistically significant difference in gap distribution.")
    else:
        print("\nResult: No significant difference found.")
else:
    print("Not enough data matched the codes/keywords.")