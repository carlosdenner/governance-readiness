import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import re
import math
import os

def parse_harm_types(harm_str):
    """
    Parses a string like 'security(16); privacy(2)' into a dictionary.
    """
    if pd.isna(harm_str) or str(harm_str).strip() == "":
        return {}
    # Regex to capture name (allowing letters, numbers, underscores, spaces) and count
    pattern = r"([a-zA-Z0-9_\s]+)\((\d+)\)"
    matches = re.findall(pattern, str(harm_str))
    return {name.strip(): int(count) for name, count in matches}

def calculate_entropy(counts):
    """
    Calculates Shannon Entropy based on count dictionary.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log(p)
    return entropy

def run_experiment():
    # 1. Load Dataset
    file_path = '../step3_coverage_map.csv'
    if not os.path.exists(file_path):
        # Fallback to local if parent directory check fails (e.g., if env structure differs)
        file_path = 'step3_coverage_map.csv'
        if not os.path.exists(file_path):
            print("Error: step3_coverage_map.csv not found.")
            return

    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)

    # 2. Parse Harm Types
    df['harm_counts'] = df['primary_harm_types'].apply(parse_harm_types)

    # 3. Calculate Entropy
    df['entropy'] = df['harm_counts'].apply(calculate_entropy)

    # 4. Split by Bundle
    trust_mask = df['bundle'].str.contains('Trust', case=False, na=False)
    integration_mask = df['bundle'].str.contains('Integration', case=False, na=False)

    trust_df = df[trust_mask].copy()
    integration_df = df[integration_mask].copy()

    # 5. Statistical Test
    t_stat, p_val = stats.ttest_ind(trust_df['entropy'], integration_df['entropy'], equal_var=False)
    
    print("\n=== Harm Entropy Statistics ===")
    print(f"Trust Readiness (n={len(trust_df)}):")
    print(f"  Mean Entropy: {trust_df['entropy'].mean():.4f}")
    print(f"  Std Dev:      {trust_df['entropy'].std():.4f}")
    print(f"Integration Readiness (n={len(integration_df)}):")
    print(f"  Mean Entropy: {integration_df['entropy'].mean():.4f}")
    print(f"  Std Dev:      {integration_df['entropy'].std():.4f}")
    
    print("\n=== Hypothesis Test ===")
    print(f"Independent t-test results: t={t_stat:.4f}, p={p_val:.4f}")
    if p_val < 0.05:
        print("Conclusion: Significant difference in Harm Entropy between bundles.")
    else:
        print("Conclusion: No significant difference in Harm Entropy between bundles.")

    # 6. Visualization
    plt.figure(figsize=(14, 7))
    
    # Concatenate for plotting, Trust first then Integration
    plot_df = pd.concat([trust_df, integration_df])
    
    # Color mapping: Trust = SkyBlue, Integration = Salmon
    colors = ['skyblue' if 'Trust' in b else 'salmon' for b in plot_df['bundle']]
    
    # Create Bar Chart
    bars = plt.bar(plot_df['sub_competency_name'], plot_df['entropy'], color=colors, edgecolor='grey')
    
    plt.title('Harm Diversity (Shannon Entropy) by Sub-Competency')
    plt.ylabel('Shannon Entropy (nats)')
    plt.xlabel('Sub-Competency')
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', edgecolor='grey', label='Trust Readiness'),
        Patch(facecolor='salmon', edgecolor='grey', label='Integration Readiness')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.show()

    # Debug output of the dataframe to verify parsing if needed
    # print(df[['sub_competency_id', 'primary_harm_types', 'entropy']])

if __name__ == "__main__":
    run_experiment()
