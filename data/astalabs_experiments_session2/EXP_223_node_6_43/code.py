import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def run_experiment():
    try:
        # Load dataset
        filename = 'astalabs_discovery_all_data.csv'
        if os.path.exists(filename):
            file_path = filename
        elif os.path.exists(f'../{filename}'):
            file_path = f'../{filename}'
        else:
            print(f"Error: {filename} not found.")
            return

        print(f"Loading dataset from {file_path}...")
        df = pd.read_csv(file_path, low_memory=False)
        
        # Filter for AIID incidents
        aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
        print(f"AIID Incidents loaded: {len(aiid_df)} rows")
        
        # Identify useful columns
        # Autonomy
        autonomy_cols = [c for c in aiid_df.columns if 'autonomy' in str(c).lower() and 'level' in str(c).lower()]
        aut_col = autonomy_cols[0] if autonomy_cols else None
        
        # Text for keyword search (Description/Summary/Title)
        text_cols = [c for c in aiid_df.columns if c.lower() in ['description', 'summary', 'title', 'text', 'incident_description']]
        # Also include 'Tangible Harm' and 'Harm Distribution Basis' for context
        context_cols = [c for c in aiid_df.columns if 'harm' in str(c).lower()]
        
        search_cols = text_cols + context_cols
        print(f"Using Autonomy Column: {aut_col}")
        print(f"Using Text/Context Columns for Harm classification: {search_cols}")
        
        if not aut_col:
            print("Autonomy column not found.")
            return

        # --- MAPPING FUNCTIONS ---
        
        def get_autonomy(row):
            val = str(row[aut_col]).lower()
            if 'autonomy3' in val or 'autonomous' in val:
                return 'High'
            if 'autonomy1' in val or 'autonomy2' in val or 'assist' in val or 'augment' in val:
                return 'Low'
            return None

        def get_harm_type(row):
            # Combine text from all relevant columns
            text_content = " "
            for c in search_cols:
                if c in row and pd.notna(row[c]):
                    text_content += str(row[c]).lower() + " "
            
            # Keywords
            physical_keys = ['death', 'dead', 'kill', 'inju', 'hurt', 'crash', 'accident', 'collision', 'physical safety', 'burned', 'broke', 'fracture']
            intangible_keys = ['bias', 'discriminat', 'racis', 'sexi', 'gender', 'fairness', 'civil right', 'privacy', 'surveillance', 'reputation', 'economic', 'financial', 'credit', 'loan', 'arrest']
            
            # Classification Logic
            has_physical = any(k in text_content for k in physical_keys)
            has_intangible = any(k in text_content for k in intangible_keys)
            
            if has_physical:
                return 'Physical'
            elif has_intangible:
                return 'Intangible'
            else:
                return None

        # Apply mappings
        aiid_df['Autonomy_Bin'] = aiid_df.apply(get_autonomy, axis=1)
        aiid_df['Harm_Bin'] = aiid_df.apply(get_harm_type, axis=1)
        
        # Filter valid rows
        analysis_df = aiid_df.dropna(subset=['Autonomy_Bin', 'Harm_Bin'])
        print(f"\nRows mapped and ready for analysis: {len(analysis_df)}")
        print("Sample of mapped data:")
        print(analysis_df[['Autonomy_Bin', 'Harm_Bin']].head())
        
        if len(analysis_df) < 5:
            print("Insufficient data.")
            return

        # --- STATISTICAL ANALYSIS ---
        
        contingency = pd.crosstab(analysis_df['Autonomy_Bin'], analysis_df['Harm_Bin'])
        print("\n--- Contingency Table ---")
        print(contingency)
        
        # Chi-square
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        print(f"\nChi-Square Statistic: {chi2:.4f}")
        print(f"P-value: {p:.4f}")
        
        if p < 0.05:
            print("Result: Statistically SIGNIFICANT association (p < 0.05).")
        else:
            print("Result: NOT statistically significant (p >= 0.05).")
            
        # Visualization
        contingency.plot(kind='bar', stacked=False, figsize=(8, 6))
        plt.title('Harm Type by Autonomy Level')
        plt.xlabel('Autonomy Level')
        plt.ylabel('Incident Count')
        plt.legend(title='Harm Type')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_experiment()