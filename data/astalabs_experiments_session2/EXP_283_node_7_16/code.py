import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# --- 1. Load Dataset ---
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"Total AIID Incidents loaded: {len(aiid_df)}")

# --- 2. Data Cleaning & Mapping ---

# Map Autonomy Level
# Scheme: Autonomy3 -> High; Autonomy1/2 -> Low/Moderate
def map_autonomy(val):
    if pd.isna(val) or val == 'unclear':
        return np.nan
    if str(val) == 'Autonomy3':
        return 'High Autonomy'
    elif str(val) in ['Autonomy1', 'Autonomy2']:
        return 'Low/Moderate Autonomy'
    return np.nan

# Map Harm Type
# Scheme: 'tangible harm definitively occurred' -> Physical
#         'no tangible harm, near-miss, or issue' -> Non-Physical (Intangible)
#         Exclude risks/near-misses to strictly compare actual harm types.
def map_harm(val):
    if pd.isna(val) or val == 'unclear':
        return np.nan
    val_str = str(val).lower()
    if 'tangible harm definitively occurred' in val_str:
        return 'Physical Harm'
    elif 'no tangible harm' in val_str:
        return 'Non-Physical Harm'
    else:
        # Exclude 'imminent risk' and 'non-imminent risk' to focus on actual outcomes
        return np.nan

# Apply mappings
aiid_df['Autonomy_Bin'] = aiid_df['Autonomy Level'].apply(map_autonomy)
aiid_df['Harm_Bin'] = aiid_df['Tangible Harm'].apply(map_harm)

# Filter clean data
analysis_df = aiid_df.dropna(subset=['Autonomy_Bin', 'Harm_Bin'])
print(f"Data points after filtering for valid Autonomy & Harm outcomes: {len(analysis_df)}")

# --- 3. Statistical Analysis ---

# Contingency Table
contingency = pd.crosstab(analysis_df['Autonomy_Bin'], analysis_df['Harm_Bin'])
print("\n--- Contingency Table (Counts) ---")
print(contingency)

# Check assumptions (expected frequencies > 5)
chi2, p, dof, expected = stats.chi2_contingency(contingency)

# Proportions for interpretation
props = pd.crosstab(analysis_df['Autonomy_Bin'], analysis_df['Harm_Bin'], normalize='index') * 100
print("\n--- Proportions (%) ---")
print(props)

print(f"\n--- Chi-Square Test Results ---")
print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# Interpretation
alpha = 0.05
if p < alpha:
    print("\nResult: Statistically significant relationship found.")
    # Check direction
    high_phys = props.loc['High Autonomy', 'Physical Harm']
    low_phys = props.loc['Low/Moderate Autonomy', 'Physical Harm']
    
    print(f"Physical Harm Rate (High Autonomy): {high_phys:.2f}%")
    print(f"Physical Harm Rate (Low/Mod Autonomy): {low_phys:.2f}%")
    
    if high_phys > low_phys:
        print("Conclusion: Hypothesis SUPPORTED. High autonomy systems are significantly more likely to result in physical harm.")
    else:
        print("Conclusion: Hypothesis REFUTED. Relationship exists, but High autonomy systems have lower physical harm rates.")
else:
    print("\nResult: No statistically significant relationship found.")
    print("Conclusion: Hypothesis NOT SUPPORTED.")

# --- 4. Visualization ---
try:
    # Pivot for stacked bar chart
    ax = props.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'], figsize=(10, 6))
    
    plt.title('Proportion of Physical vs. Non-Physical Harm by Autonomy Level')
    plt.xlabel('Autonomy Level')
    plt.ylabel('Percentage of Incidents')
    plt.legend(title='Harm Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    
    # Annotate bars
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center')
        
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Visualization failed: {e}")
