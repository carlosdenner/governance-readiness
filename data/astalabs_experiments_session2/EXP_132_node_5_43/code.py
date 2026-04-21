import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# [debug]
print("Starting corrected analysis...")

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

# Inspect columns
stage_col = '16_dev_stage'
test_col = '53_real_world_testing'

# --- REFINED MAPPING LOGIC ---

def map_stage(val):
    if pd.isna(val):
        return np.nan
    val_lower = str(val).lower()
    
    # Post-Production keywords
    if any(x in val_lower for x in ['operation', 'maintenance', 'in production', 'in mission', 'implementation', 'deployed']):
        return 'Post-Production'
    # Pre-Production keywords
    elif any(x in val_lower for x in ['acquisition', 'development', 'initiated', 'planned', 'design']):
        return 'Pre-Production'
    else:
        return np.nan

def map_testing_strict(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    
    # Strict prefix/keyword matching based on known unique values
    if val_str.startswith('Performance evaluation') or val_str.startswith('Impact evaluation') or val_str.lower() == 'yes':
        return 'Yes'
    elif val_str.startswith('No testing') or val_str.startswith('Benchmark evaluation') or val_str.startswith('Agency CAIO'):
        return 'No'
    else:
        # Fallback for unexpected strings, treat as NaN to be safe
        return np.nan

# Apply mappings
eo_data['stage_group'] = eo_data[stage_col].apply(map_stage)
eo_data['testing_binary'] = eo_data[test_col].apply(map_testing_strict)

# Filter for analysis (drop NaNs in relevant cols)
analysis_df = eo_data.dropna(subset=['stage_group', 'testing_binary']).copy()

# Debug print to verify fix
print(f"Data points for analysis: {len(analysis_df)}")
print("\nDistribution of Testing Status (Corrected):")
print(analysis_df['testing_binary'].value_counts())
print("\nDistribution of Stage Group:")
print(analysis_df['stage_group'].value_counts())

# Create Contingency Table
contingency_table = pd.crosstab(analysis_df['stage_group'], analysis_df['testing_binary'])
print("\nContingency Table (Count):")
print(contingency_table)

# Calculate Percentages
contingency_pct = pd.crosstab(analysis_df['stage_group'], analysis_df['testing_binary'], normalize='index') * 100
print("\nContingency Table (Percentage within Stage):")
print(contingency_pct)

# Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print(f"\nChi-Square Test Results:")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"p-value: {p:.4f}")

# Visualization
plt.figure(figsize=(8, 6))

# Plot 'Yes' rates if available
if 'Yes' in contingency_pct.columns:
    ax = contingency_pct['Yes'].plot(kind='bar', color=['skyblue', 'salmon'], edgecolor='black')
    plt.title('Rate of Real-World Operational Testing by Stage')
    plt.ylabel('Percentage of Systems Tested (%)')
    plt.xlabel('Lifecycle Stage')
    plt.ylim(0, 100)
    plt.xticks(rotation=0)

    # Annotate
    for p_rect in ax.patches:
        h = p_rect.get_height()
        ax.annotate(f"{h:.1f}%", (p_rect.get_x() + p_rect.get_width() / 2., h),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
else:
    print("No 'Yes' data to plot.")

plt.tight_layout()
plt.show()

# Conclusion generator
alpha = 0.05
print("\n--- Conclusion ---")
if p < alpha:
    print("Result: Statistically Significant Difference.")
    pre_rate = contingency_pct.loc['Pre-Production', 'Yes']
    post_rate = contingency_pct.loc['Post-Production', 'Yes']
    print(f"Pre-Production Testing Rate: {pre_rate:.1f}%")
    print(f"Post-Production Testing Rate: {post_rate:.1f}%")
    if post_rate > pre_rate:
        print("Observation: Testing increases significantly as systems move to production (Gate functioning).")
    else:
        print("Observation: Testing decreases significantly in production (Potential issue).")
else:
    print("Result: No Statistically Significant Difference.")
    print("Observation: The rate of real-world testing does not statistically differ between development and production stages.")
