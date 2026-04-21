import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import os

def run_experiment():
    print("Starting Generative AI Incident Surge analysis...")
    
    # Load dataset
    file_path = '../astalabs_discovery_all_data.csv'
    if not os.path.exists(file_path):
        # Fallback for local testing if needed
        file_path = 'astalabs_discovery_all_data.csv'
    
    try:
        # Low memory=False to handle mixed types warning from previous steps
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Filter for AIID incidents
    aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
    print(f"AIID Incidents loaded (raw): {len(aiid_df)}")

    # Clean column names
    aiid_df.columns = [c.strip() for c in aiid_df.columns]
    
    # Check for required columns
    # Based on previous exploration, 'date' and 'Known AI Technology' should exist
    required_cols = ['date', 'Known AI Technology']
    missing_cols = [c for c in required_cols if c not in aiid_df.columns]
    if missing_cols:
        print(f"Error: Missing columns {missing_cols}. Available: {aiid_df.columns.tolist()}")
        return

    # 1. Parse Date and Extract Year
    aiid_df['date_parsed'] = pd.to_datetime(aiid_df['date'], errors='coerce')
    # Drop rows without valid dates
    aiid_df = aiid_df.dropna(subset=['date_parsed'])
    aiid_df['year'] = aiid_df['date_parsed'].dt.year
    
    # 2. Filter for Known Technology (Drop Nulls to analyze only identified tech)
    # This ensures we are comparing 'Generative' vs 'Other Known Tech'
    aiid_clean = aiid_df.dropna(subset=['Known AI Technology']).copy()
    print(f"Incidents with valid Date and Known Technology: {len(aiid_clean)}")

    # 3. Classify Technology
    # Keywords provided in prompt
    gen_keywords = ['LLM', 'GPT', 'GenAI', 'Diffusion', 'Chatbot', 'Generative']
    
    def is_generative(val):
        val_str = str(val).lower()
        return any(kw.lower() in val_str for kw in gen_keywords)

    aiid_clean['Is_Generative'] = aiid_clean['Known AI Technology'].apply(is_generative)

    # 4. Create Period Variable
    # Post-2022 (Year >= 2023) vs Pre-2022 (Year <= 2022)
    aiid_clean['Period'] = np.where(aiid_clean['year'] >= 2023, 'Post-2022', 'Pre-2022')
    
    # 5. Contingency Table
    contingency = pd.crosstab(aiid_clean['Period'], aiid_clean['Is_Generative'])
    
    # Rename columns/index for clarity
    if True in contingency.columns and False in contingency.columns:
        contingency = contingency.rename(columns={False: 'Other', True: 'Generative'})
    elif True in contingency.columns:
        contingency = contingency.rename(columns={True: 'Generative'})
        contingency['Other'] = 0
    else:
        contingency = contingency.rename(columns={False: 'Other'})
        contingency['Generative'] = 0
        
    # Ensure row order
    desired_order = ['Pre-2022', 'Post-2022']
    contingency = contingency.reindex(desired_order).fillna(0)

    print("\n--- Contingency Table (Period vs Technology) ---")
    print(contingency)

    # 6. Statistical Test (Chi-Square)
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")

    # Calculate Proportions and Odds Ratio
    try:
        pre_gen = contingency.loc['Pre-2022', 'Generative']
        pre_tot = contingency.loc['Pre-2022'].sum()
        post_gen = contingency.loc['Post-2022', 'Generative']
        post_tot = contingency.loc['Post-2022'].sum()
        
        prop_pre = pre_gen / pre_tot if pre_tot > 0 else 0
        prop_post = post_gen / post_tot if post_tot > 0 else 0
        
        print(f"\nProportion Generative (Pre-2022): {prop_pre:.2%}")
        print(f"Proportion Generative (Post-2022): {prop_post:.2%}")
        
        # Odds Ratio
        # (Gen_Post / Other_Post) / (Gen_Pre / Other_Pre)
        odds_post = post_gen / contingency.loc['Post-2022', 'Other'] if contingency.loc['Post-2022', 'Other'] > 0 else np.nan
        odds_pre = pre_gen / contingency.loc['Pre-2022', 'Other'] if contingency.loc['Pre-2022', 'Other'] > 0 else np.nan
        
        if odds_pre > 0:
            print(f"Odds Ratio: {odds_post / odds_pre:.4f}")
        else:
            print("Odds Ratio: Undefined (Zero denominator in Pre-2022)")
            
    except Exception as e:
        print(f"Error calculating stats: {e}")

    # 7. Visualization
    # Group by year to show trend
    yearly = aiid_clean.groupby('year')['Is_Generative'].agg(['sum', 'count'])
    yearly['proportion'] = yearly['sum'] / yearly['count']
    
    # Filter to relevant timeline (e.g. 2015-2024) to avoid noisy early years
    plot_data = yearly[yearly.index >= 2015]
    
    if not plot_data.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(plot_data.index, plot_data['proportion'], marker='o', linestyle='-', linewidth=2, color='darkblue')
        plt.title('Proportion of AI Incidents Involving Generative Technologies (2015-2024)')
        plt.xlabel('Year')
        plt.ylabel('Proportion (Generative / Total Known)')
        plt.axvline(x=2022.5, color='red', linestyle='--', label='End of 2022')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough data to generate plot.")

if __name__ == "__main__":
    run_experiment()