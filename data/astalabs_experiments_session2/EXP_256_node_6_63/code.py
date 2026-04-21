import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load dataset
print("Loading dataset...")
file_path = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 subset
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 subset shape: {df_eo.shape}")

# 1. Map Impact Type
# Logic: 'Neither' -> Low/Moderate; 'Both', 'Rights-Impacting', 'Safety-Impacting' -> High Impact
def map_impact(val):
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    if val_str == 'Neither':
        return 'Low/Moderate Impact'
    # Check for keywords indicating high impact
    if val_str in ['Both', 'Rights-Impacting', 'Safety-Impacting', 'Safety-impacting']:
        return 'High Impact'
    return None

df_eo['impact_group'] = df_eo['17_impact_type'].apply(map_impact)

# Drop rows where impact is undefined
df_analysis = df_eo.dropna(subset=['impact_group']).copy()

# 2. Map Independent Eval
# Logic: Starts with 'Yes' or is 'TRUE' -> Yes; else -> No
def map_eval(val):
    if pd.isna(val):
        return 'No'
    val_str = str(val).strip().lower()
    if val_str.startswith('yes') or val_str == 'true':
        return 'Yes'
    return 'No'

df_analysis['has_indep_eval'] = df_analysis['55_independent_eval'].apply(map_eval)

# 3. Generate Contingency Table
contingency = pd.crosstab(df_analysis['impact_group'], df_analysis['has_indep_eval'])
print("\nContingency Table (Impact vs Independent Eval):")
print(contingency)

# Calculate percentages for display
print("\nPercentages:")
print(pd.crosstab(df_analysis['impact_group'], df_analysis['has_indep_eval'], normalize='index') * 100)

# 4. Statistical Test
if contingency.size >= 4:
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # 5. Visualization
    # Calculate positive rates
    rates = df_analysis.groupby('impact_group')['has_indep_eval'].apply(lambda x: (x=='Yes').mean())
    
    plt.figure(figsize=(8, 6))
    # Ensure consistent order
    order = ['High Impact', 'Low/Moderate Impact']
    try:
        rates = rates.reindex(order)
    except:
        pass
        
    ax = rates.plot(kind='bar', color=['#d62728', '#1f77b4'], rot=0)
    
    plt.title('Independent Evaluation Rate by AI System Impact Level')
    plt.ylabel('Proportion with Independent Evaluation')
    plt.xlabel('Impact Level')
    plt.ylim(0, max(rates.max() * 1.2, 0.1))  # Dynamic ylim
    
    # Add labels
    for p_rect in ax.patches:
        width = p_rect.get_width()
        height = p_rect.get_height()
        x, y = p_rect.get_xy() 
        ax.text(x + width/2, 
                y + height + 0.005, 
                f'{height:.1%}', 
                ha='center', 
                va='bottom',
                fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
else:
    print("Insufficient data dimensions for Chi-square test.")