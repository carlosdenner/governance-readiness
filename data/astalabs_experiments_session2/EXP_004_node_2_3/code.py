import json
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Define file path: trying parent directory first as per instructions
filename = 'context_crosswalk_evidence.json'
filepath = f'../{filename}'
if not os.path.exists(filepath):
    filepath = filename # Fallback to current directory

print(f"Attempting to load data from: {filepath}")

try:
    # Load JSON data
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Process data into DataFrame
    records = []
    for item in data:
        # Extract fields
        bundle = item.get('bundle', 'Unknown')
        controls = item.get('applicable_controls', [])
        req_id = item.get('req_id', 'Unknown')
        statement = item.get('competency_statement', '')
        
        # Count controls (handle None or non-list types safely)
        if isinstance(controls, list):
            count = len(controls)
        else:
            count = 0
            
        records.append({
            'bundle': bundle,
            'control_count': count,
            'req_id': req_id,
            'statement': statement
        })
    
    df = pd.DataFrame(records)
    print(f"Data loaded. Shape: {df.shape}")
    
    # Check unique bundles
    unique_bundles = df['bundle'].unique()
    print(f"Unique Bundles found: {unique_bundles}")
    
    # If bundle names are generic or just one group, try to derive specific domains
    # The hypothesis compares Security vs Fairness/Explainability
    if len(unique_bundles) <= 1 or 'Trust Readiness' in unique_bundles[0]:
        print("Refining domains based on statement content...")
        def refine_domain(text):
            text = text.lower()
            if 'security' in text or 'attack' in text or 'adversar' in text: return 'Security'
            if 'fairness' in text or 'bias' in text: return 'Fairness'
            if 'explain' in text or 'interpret' in text or 'transparen' in text: return 'Explainability'
            if 'privacy' in text: return 'Privacy'
            return 'Other'
        
        df['analysis_group'] = df['statement'].apply(refine_domain)
        # Filter out 'Other' if we want cleaner plots, or keep them
        # For now, keep them but print counts
    else:
        df['analysis_group'] = df['bundle']

    print("\nGroup counts for analysis:")
    print(df['analysis_group'].value_counts())
    
    # Statistical Analysis
    print("\n--- Statistical Analysis ---")
    groups = [df[df['analysis_group'] == g]['control_count'].values for g in df['analysis_group'].unique()]
    
    if len(groups) > 1:
        stat, p = stats.kruskal(*groups)
        print(f"Kruskal-Wallis Test across all groups: H={stat:.4f}, p-value={p:.4f}")
    
    # Pairwise Comparisons for Hypothesis
    # Identify specific groups dynamically
    domain_groups = df['analysis_group'].unique()
    sec_group = next((g for g in domain_groups if 'Security' in g), None)
    fair_group = next((g for g in domain_groups if 'Fairness' in g), None)
    exp_group = next((g for g in domain_groups if 'Explainability' in g), None)
    
    def run_mann_whitney(g1_name, g2_name):
        if g1_name and g2_name:
            d1 = df[df['analysis_group'] == g1_name]['control_count']
            d2 = df[df['analysis_group'] == g2_name]['control_count']
            u_stat, p_val = stats.mannwhitneyu(d1, d2)
            print(f"\nMann-Whitney U Test: {g1_name} vs {g2_name}")
            print(f"  U-statistic: {u_stat:.2f}, p-value: {p_val:.4f}")
            print(f"  Mean Controls: {g1_name}={d1.mean():.2f}, {g2_name}={d2.mean():.2f}")
    
    run_mann_whitney(sec_group, fair_group)
    run_mann_whitney(sec_group, exp_group)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    df.boxplot(column='control_count', by='analysis_group', rot=45, grid=True)
    plt.title('Architectural Control Density by Trust Domain')
    plt.suptitle('') # Suppress default pandas title
    plt.ylabel('Number of Mapped Controls')
    plt.xlabel('Trust Domain')
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: File {filepath} not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
