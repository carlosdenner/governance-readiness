import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import subprocess
import re

# Helper to install seaborn if missing
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

try:
    import seaborn as sns
except ImportError:
    install("seaborn")
    import seaborn as sns

print("Starting Refined Adversarial Tactic-Gap Analysis...")

# 1. Load the dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    try:
        df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

# 2. Filter for 'step3_incident_coding' subset
subset = df[df['source_table'] == 'step3_incident_coding'].copy()
print(f"Subset 'step3_incident_coding' shape: {subset.shape}")

# 3. Define parsing logic (handles both ; and ,)
def parse_terms(val):
    if pd.isna(val):
        return []
    val = str(val)
    # Split by semicolon OR comma
    tokens = re.split(r'[;,]', val)
    # Clean whitespace and empty strings
    return [t.strip() for t in tokens if t.strip()]

# 4. Extract atomic values
# We will inspect multiple columns for gaps to find the most granular one
gap_candidates = ['competency_gaps', 'competency_domains', 'missing_controls']
tactic_col = 'tactics_used'

# Check which gap column has the most variance/content
best_gap_col = None
max_unique = 0

for col in gap_candidates:
    if col in subset.columns:
        unique_terms = set()
        for val in subset[col]:
            unique_terms.update(parse_terms(val))
        print(f"Column '{col}' has {len(unique_terms)} unique atomic values.")
        if len(unique_terms) > max_unique:
            max_unique = len(unique_terms)
            best_gap_col = col

print(f"\nSelected Gap Column: {best_gap_col}")
if not best_gap_col:
    print("No valid gap column found.")
    sys.exit(0)

# Process rows into binary vectors
records = []
all_tactics = set()
all_gaps = set()

for idx, row in subset.iterrows():
    t_list = parse_terms(row[tactic_col])
    g_list = parse_terms(row[best_gap_col])
    
    if not t_list and not g_list:
        continue
        
    all_tactics.update(t_list)
    all_gaps.update(g_list)
    
    records.append({
        'tactics': set(t_list),
        'gaps': set(g_list)
    })

print(f"Processed {len(records)} incidents.")
print(f"Unique Atomic Tactics: {len(all_tactics)}")
print(f"Unique Atomic Gaps: {len(all_gaps)}")

if len(all_tactics) < 2 or len(all_gaps) < 1:
    print("Not enough data variation for correlation.")
    sys.exit(0)

# 5. Build Correlation Matrix
sorted_tactics = sorted(list(all_tactics))
sorted_gaps = sorted(list(all_gaps))

data_rows = []
for r in records:
    row_vec = {}
    for t in sorted_tactics:
        row_vec[f"T:{t}"] = 1 if t in r['tactics'] else 0
    for g in sorted_gaps:
        row_vec[f"G:{g}"] = 1 if g in r['gaps'] else 0
    data_rows.append(row_vec)

matrix_df = pd.DataFrame(data_rows)

# Filter out constant columns (variance=0) to avoid NaN correlations
matrix_df = matrix_df.loc[:, matrix_df.var() > 0]

corr_matrix = matrix_df.corr(method='pearson')

# Extract Tactic vs Gap submatrix
t_cols = [c for c in matrix_df.columns if c.startswith("T:")]
g_cols = [c for c in matrix_df.columns if c.startswith("G:")]

if not t_cols or not g_cols:
    print("No valid variance in tactics or gaps to correlate.")
    sys.exit(0)

sub_corr = corr_matrix.loc[t_cols, g_cols]

# 6. Top Associations
pairs = []
for t in t_cols:
    for g in g_cols:
        val = sub_corr.loc[t, g]
        if not np.isnan(val):
            pairs.append((t, g, val))

pairs.sort(key=lambda x: abs(x[2]), reverse=True)

print("\n--- Top 20 Strongest Correlations (Absolute Value) ---")
print(f"{'Tactic':<40} | {'Gap':<40} | {'Phi':<6}")
print("-" * 90)
for t, g, v in pairs[:20]:
    t_name = t.replace("T:", "")
    g_name = g.replace("G:", "")
    print(f"{t_name:<40} | {g_name:<40} | {v:.3f}")

# Check Hypothesis Specifics (Evasion vs Robustness, Exfiltration vs Access)
print("\n--- Hypothesis Check ---")
hypothesis_keywords = {
    'Evasion': ['AML.TA0007', 'Evasion'],
    'Robustness': ['Robustness', 'Reliability'],
    'Exfiltration': ['AML.TA0012', 'Exfiltration'],
    'Access': ['Access', 'Privilege', 'Permission']
}

found_hyp_corrs = []
for t, g, v in pairs:
    t_clean = t.replace("T:", "")
    g_clean = g.replace("G:", "")
    
    # Check Evasion-Robustness
    is_evasion = any(k in t_clean for k in hypothesis_keywords['Evasion'])
    is_robust = any(k in g_clean for k in hypothesis_keywords['Robustness'])
    
    # Check Exfiltration-Access
    is_exfil = any(k in t_clean for k in hypothesis_keywords['Exfiltration'])
    is_access = any(k in g_clean for k in hypothesis_keywords['Access'])
    
    if (is_evasion and is_robust) or (is_exfil and is_access):
        found_hyp_corrs.append((t_clean, g_clean, v))

if found_hyp_corrs:
    print(f"{'Hypothesis Tactic':<40} | {'Hypothesis Gap':<40} | {'Phi':<6}")
    for t, g, v in found_hyp_corrs:
        print(f"{t:<40} | {g:<40} | {v:.3f}")
else:
    print("No direct correlations found matching specific hypothesis keywords.")

# 7. Visualization
plt.figure(figsize=(14, 10))
# If matrix is too large, take top interactions
if sub_corr.shape[0] * sub_corr.shape[1] > 400:
    # Filter for rows/cols with at least one significant correlation
    significant_mask = (sub_corr.abs() > 0.2).any(axis=1)
    plot_corr = sub_corr.loc[significant_mask]
else:
    plot_corr = sub_corr

sns.heatmap(plot_corr, cmap='RdBu_r', center=0, annot=False)
plt.title('Correlation: Atomic Adversarial Tactics vs Governance Gaps')
plt.tight_layout()
plt.show()
