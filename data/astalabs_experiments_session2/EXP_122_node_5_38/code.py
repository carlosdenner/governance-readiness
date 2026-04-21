import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def main():
    # 1. Load dataset
    file_path = 'astalabs_discovery_all_data.csv'
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # 2. Filter for 'eo13960_scored'
    eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"Loaded 'eo13960_scored' with {len(eo_df)} rows.")

    # 3. Identify and Map Impact Type
    # Finding the exact column name for '17_impact_type'
    impact_cols = [c for c in eo_df.columns if '17_impact_type' in str(c)]
    if not impact_cols:
        print("Column '17_impact_type' not found.")
        return
    impact_col = impact_cols[0]

    # Finding the exact column name for '63_stakeholder_consult'
    consult_cols = [c for c in eo_df.columns if '63_stakeholder_consult' in str(c)]
    if not consult_cols:
        print("Column '63_stakeholder_consult' not found.")
        return
    consult_col = consult_cols[0]

    print(f"Using columns: '{impact_col}' and '{consult_col}'")

    # Drop NaNs for analysis in these specific columns
    # We only care about rows where we have impact info.
    analysis_df = eo_df[[impact_col, consult_col]].dropna(subset=[impact_col]).copy()
    
    print("\n--- Value Analysis ---")
    print(f"Unique values in {impact_col}:\n{analysis_df[impact_col].value_counts()}")
    
    # Mapping Functions
    def map_impact(val):
        s = str(val).lower().strip()
        # Based on EO 13960 categories usually being 'Rights-Impacting', 'Safety-Impacting', etc.
        if 'rights' in s and 'safety' in s:
            return 'Both' # Exclude to isolate the specific effects
        elif 'rights' in s:
            return 'Rights-Impacting'
        elif 'safety' in s:
            return 'Safety-Impacting'
        else:
            return 'Other'

    def map_consult(val):
        # Check for non-null and affirmative content
        if pd.isna(val):
            return 'No'
        s = str(val).lower().strip()
        # If the field is descriptive, we look for keywords indicating consultation happened
        # Common affirmative terms in this dataset context:
        # 'yes', 'consulted', 'engaged', 'completed', 'stakeholders were involved'
        # Negative terms: 'no', 'none', 'n/a', 'not applicable'
        
        negative_keywords = ['no', 'none', 'n/a', 'not applicable', 'not consulted']
        if s in negative_keywords or s == 'nan':
            return 'No'
        
        # If it's a short string like 'Yes' or 'No'
        if s == 'yes':
            return 'Yes'
        if s == 'no':
            return 'No'
            
        # If it's a longer description, we assume existence of text implies some activity
        # UNLESS it explicitly says 'no consultation'
        if any(neg in s for neg in ['no consultation', 'not conducted']):
            return 'No'
            
        return 'Yes'

    analysis_df['Group'] = analysis_df[impact_col].apply(map_impact)
    analysis_df['Consulted'] = analysis_df[consult_col].apply(map_consult)

    print("\n--- Consultation Mapping Check ---")
    print(analysis_df[[consult_col, 'Consulted']].head(10))
    print(analysis_df['Consulted'].value_counts())

    # Filter for Rights vs Safety (excluding Both/Other for clean comparison)
    final_df = analysis_df[analysis_df['Group'].isin(['Rights-Impacting', 'Safety-Impacting'])].copy()
    
    print(f"\nFinal dataset size for analysis (Rights vs Safety): {len(final_df)}")
    print("Distribution by Group:")
    print(final_df['Group'].value_counts())

    # 4. Contingency Table & Stats
    contingency = pd.crosstab(final_df['Group'], final_df['Consulted'])
    
    if contingency.empty or contingency.shape[0] < 2:
        print("Contingency table is insufficient. Cannot perform test.")
        print(contingency)
        return

    print("\n--- Contingency Table (Group x Consulted) ---")
    print(contingency)

    # Proportions
    props = pd.crosstab(final_df['Group'], final_df['Consulted'], normalize='index')
    print("\n--- Proportions ---")
    print(props)

    # Chi-square test
    chi2, p, dof, ex = stats.chi2_contingency(contingency)
    print("\n--- Statistical Test Results ---")
    print(f"Chi-square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.6f}")
    
    significant = p < 0.05
    if significant:
        print("Result: Statistically significant difference found.")
    else:
        print("Result: No statistically significant difference found.")

    # Visualization
    try:
        # Ensure we have Yes/No columns for plotting
        if 'Yes' not in props.columns:
            props['Yes'] = 0
        if 'No' not in props.columns:
            props['No'] = 0
            
        # Sort columns to ensure Yes is usually the focus color (often 2nd in stacked)
        plot_data = props[['No', 'Yes']]
        
        ax = plot_data.plot(kind='bar', stacked=True, color=['#d62728', '#2ca02c'], alpha=0.7, figsize=(8, 6))
        plt.title('Stakeholder Consultation by Impact Type')
        plt.ylabel('Proportion')
        plt.xlabel('Impact Category')
        plt.xticks(rotation=0)
        plt.legend(title='Consulted', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()