import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

def run_experiment():
    # 1. Load Dataset
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        try:
            df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
        except FileNotFoundError:
            print("Dataset not found.")
            return

    # 2. Filter for AIID incidents
    df_aiid = df[df['source_table'] == 'aiid_incidents'].copy()
    
    # 3. Define Sector Mapping Logic (using 'Infrastructure Sectors')
    # Note: 'Infrastructure Sectors' appeared cleaner in debug (e.g., 'financial services', 'healthcare...')
    def map_domain(val):
        s = str(val).lower()
        # Physical Domain keywords
        if any(x in s for x in ['transportation', 'health', 'manufacturing', 'energy', 'defense', 'emergency', 'water', 'nuclear']):
            return 'Physical-Domain'
        # Digital Domain keywords (priority to Physical if overlap, but usually distinct)
        elif any(x in s for x in ['financial', 'information technology', 'communications', 'government', 'commercial']):
            return 'Digital-Domain'
        return None

    # 4. Define Harm Mapping Logic (using 'Tangible Harm')
    # Hypothesis: Physical sectors -> Tangible Harm; Digital sectors -> Intangible (Non-Tangible) Harm
    def map_harm(val):
        s = str(val).lower()
        if 'tangible harm definitively occurred' in s:
            return 'Tangible Harm (Physical)'
        else:
            # Includes near-misses, issues, and explicitly 'no tangible harm' (which implies intangible harm in valid incidents)
            return 'Intangible / Other'

    # Apply mappings
    # Use 'Infrastructure Sectors' primarily, fallback to 'Sector of Deployment' if null
    df_aiid['combined_sector'] = df_aiid['Infrastructure Sectors'].fillna(df_aiid['Sector of Deployment'])
    
    df_aiid['Domain'] = df_aiid['combined_sector'].apply(map_domain)
    df_aiid['Harm_Category'] = df_aiid['Tangible Harm'].apply(map_harm)

    # Filter out unmapped domains
    df_analysis = df_aiid.dropna(subset=['Domain'])

    # 5. Generate Statistics
    print(f"Incidents analyzed: {len(df_analysis)}")
    
    # Contingency Table
    contingency = pd.crosstab(df_analysis['Domain'], df_analysis['Harm_Category'])
    print("\nContingency Table (Domain vs Harm Nature):")
    print(contingency)
    
    # Calculate Percentages for clarity
    contingency_pct = pd.crosstab(df_analysis['Domain'], df_analysis['Harm_Category'], normalize='index') * 100
    print("\nRow Percentages:")
    print(contingency_pct.round(2))

    # 6. Statistical Test
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-Square Test Results:")
    print(f"Chi2 Statistic: {chi2:.4f}")
    print(f"p-value: {p:.4e}")
    
    if p < 0.05:
        print("Result: Statistically significant relationship between Sector Domain and Harm Nature.")
    else:
        print("Result: No significant relationship found.")

    # 7. Visualization
    # Plotting the percentages to visualize the "Divide"
    ax = contingency_pct.plot(kind='bar', stacked=True, color=['lightgray', 'salmon'], figsize=(8, 6))
    plt.title('The Physical-Digital Harm Divide: Tangible Harm Rates by Sector')
    plt.xlabel('Sector Domain')
    plt.ylabel('Percentage of Incidents')
    plt.legend(title='Harm Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()