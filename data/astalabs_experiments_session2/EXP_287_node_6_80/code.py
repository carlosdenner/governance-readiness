import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
import os

def run_experiment():
    # Load Dataset
    filename = 'astalabs_discovery_all_data.csv'
    # Check parent dir first as per instructions
    path_parent = os.path.join('..', filename)
    path_current = filename
    
    fpath = None
    if os.path.exists(path_parent):
        fpath = path_parent
    elif os.path.exists(path_current):
        fpath = path_current
    else:
        print(f"Error: {filename} not found in current or parent directory.")
        return

    print(f"Loading data from: {fpath}")
    df = pd.read_csv(fpath, low_memory=False)

    # 1. Prepare ID Mapping from step1_sub_competencies
    # This helps if incidents use IDs (e.g., TR-1) instead of names
    sub_comp_df = df[df['source_table'] == 'step1_sub_competencies']
    id_map = {}
    if not sub_comp_df.empty:
        if 'id' in sub_comp_df.columns and 'name' in sub_comp_df.columns:
            for _, row in sub_comp_df.iterrows():
                if pd.notna(row['id']) and pd.notna(row['name']):
                    # map both "TR-1" and "tr-1" just in case
                    key = str(row['id']).strip()
                    val = str(row['name']).strip()
                    id_map[key] = val
                    id_map[key.lower()] = val

    # 2. Extract Incident Gaps
    # source_table: step3_incident_coding
    incidents = df[df['source_table'] == 'step3_incident_coding'].copy()
    
    if incidents.empty:
        print("No incident data found in step3_incident_coding.")
        return

    # Identify column
    gap_col = 'competency_gaps'
    if gap_col not in incidents.columns:
        if 'competency_gap' in incidents.columns:
            gap_col = 'competency_gap'
        else:
            print(f"Columns available: {incidents.columns.tolist()}")
            print("Could not find competency_gaps column.")
            return

    # 3. Process Gaps
    # Explode comma-separated strings
    raw_gaps = incidents[gap_col].dropna().astype(str)
    all_gaps = []
    
    for entry in raw_gaps:
        # Split
        tokens = [t.strip() for t in entry.split(',')]
        # Resolve IDs
        resolved_tokens = [id_map.get(t, id_map.get(t.lower(), t)) for t in tokens if t]
        all_gaps.extend(resolved_tokens)

    # 4. Categorize
    # Hypothesis: Safety/Robustness > Privacy
    safety_kw = ['safety', 'robust', 'reliab', 'secur', 'resilien', 'integr']
    privacy_kw = ['priva', 'confiden', 'data protect', 'anonym']
    
    counts = {'Safety': 0, 'Privacy': 0, 'Other': 0}
    
    # Debugging lists
    cat_debug = {'Safety': set(), 'Privacy': set(), 'Other': set()}

    for gap in all_gaps:
        txt = gap.lower()
        if any(k in txt for k in safety_kw):
            counts['Safety'] += 1
            cat_debug['Safety'].add(gap)
        elif any(k in txt for k in privacy_kw):
            counts['Privacy'] += 1
            cat_debug['Privacy'].add(gap)
        else:
            counts['Other'] += 1
            cat_debug['Other'].add(gap)

    # 5. Output Results
    print(f"Total Gaps Analyzed: {len(all_gaps)}")
    print(f"Safety/Robustness Count: {counts['Safety']}")
    print(f"Privacy Count: {counts['Privacy']}")
    print(f"Other Count: {counts['Other']}")
    
    print("\n--- Category Samples ---")
    print(f"Safety: {list(cat_debug['Safety'])[:5]}")
    print(f"Privacy: {list(cat_debug['Privacy'])[:5]}")
    print(f"Other: {list(cat_debug['Other'])[:5]}")

    # 6. Chi-Square Test
    # Compare Safety vs Privacy
    obs = [counts['Safety'], counts['Privacy']]
    if sum(obs) > 0:
        exp = [sum(obs)/2, sum(obs)/2]
        chi2, p = chisquare(obs, f_exp=exp)
        
        print(f"\nChi-Square Goodness of Fit (Safety vs Privacy):")
        print(f"Observed: {obs}")
        print(f"Expected: {exp}")
        print(f"Statistic: {chi2:.4f}, p-value: {p:.5f}")
        
        if p < 0.05:
            direction = "Safety > Privacy" if counts['Safety'] > counts['Privacy'] else "Privacy > Safety"
            print(f"Result: Significant difference ({direction}). Hypothesis supported.")
        else:
            print("Result: No significant difference. Hypothesis rejected.")
    else:
        print("No relevant gaps to test.")

    # 7. Plot
    labels = list(counts.keys())
    values = list(counts.values())
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=['#d62728', '#1f77b4', '#7f7f7f'])
    plt.title('Competency Gaps in Adversarial Incidents (ATLAS)')
    plt.ylabel('Frequency')
    plt.bar_label(bars)
    plt.show()

if __name__ == "__main__":
    run_experiment()