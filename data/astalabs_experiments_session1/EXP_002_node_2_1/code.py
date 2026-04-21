import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import os
import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

try:
    from scipy.stats import mannwhitneyu
except ImportError:
    install('scipy')
    from scipy.stats import mannwhitneyu

# Handle file loading based on location note
filename = 'step2_competency_statements.csv'
possible_paths = [f"../{filename}", filename]
file_path = None

for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        break

if not file_path:
    print(f"Error: Could not find {filename} in {possible_paths}")
else:
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)

    # Map categorical confidence to numerical scores
    confidence_map = {'high': 3, 'medium': 2, 'low': 1}
    
    # Normalize string to handle case/whitespace
    df['confidence_mapped'] = df['confidence'].astype(str).str.lower().str.strip().map(confidence_map)

    # Validation
    if df['confidence_mapped'].isnull().any():
        print("Warning: Some confidence values could not be mapped. Dropping NaNs.")
        print("Unique raw values:", df['confidence'].unique())
        df = df.dropna(subset=['confidence_mapped'])

    # Group by bundle
    trust_bundle = 'Trust Readiness'
    integration_bundle = 'Integration Readiness'
    
    group_stats = df.groupby('bundle')['confidence_mapped'].agg(['count', 'mean', 'std'])
    print("\n=== Descriptive Statistics (Confidence Score: High=3, Medium=2, Low=1) ===")
    print(group_stats)

    # Prepare vectors for statistical test
    trust_scores = df[df['bundle'] == trust_bundle]['confidence_mapped']
    integration_scores = df[df['bundle'] == integration_bundle]['confidence_mapped']

    if len(trust_scores) == 0 or len(integration_scores) == 0:
        print("\nError: One of the bundles has no data. Cannot perform statistical test.")
    else:
        # Mann-Whitney U Test (Two-sided)
        # Using two-sided to detect any difference, though hypothesis suggests Integration > Trust
        u_stat, p_val = mannwhitneyu(integration_scores, trust_scores, alternative='two-sided')

        print("\n=== Mann-Whitney U Test Results ===")
        print(f"Comparison: {integration_bundle} vs {trust_bundle}")
        print(f"U-statistic: {u_stat}")
        print(f"P-value: {p_val:.5f}")
        
        alpha = 0.05
        if p_val < alpha:
            direction = integration_bundle if integration_scores.mean() > trust_scores.mean() else trust_bundle
            print(f"Conclusion: Statistically significant difference detected (p < {alpha}).")
            print(f"Direction: {direction} has higher confidence scores.")
        else:
            print(f"Conclusion: No statistically significant difference detected (p >= {alpha}).")