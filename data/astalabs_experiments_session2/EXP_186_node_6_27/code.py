import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact
import numpy as np

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID Incidents
df_incidents = df[df['source_table'] == 'aiid_incidents'].copy()

# Identify relevant columns
tech_col = next((c for c in df_incidents.columns if 'Known AI Technology' in c), None)
intent_col = next((c for c in df_incidents.columns if 'Intentional Harm' in c), None)

if not tech_col or not intent_col:
    print(f"Critical columns not found. Available: {list(df_incidents.columns)}")
else:
    print(f"Using columns: Technology='{tech_col}', Intent='{intent_col}'")

    # 1. Classify Modality
    def classify_modality(val):
        if not isinstance(val, str):
            return 'Other'
        val_lower = val.lower()
        # Vision keywords
        if any(x in val_lower for x in ['vision', 'image', 'face', 'facial', 'video', 'surveillance', 'recognition']):
            return 'Vision'
        # Language keywords
        if any(x in val_lower for x in ['nlp', 'text', 'language', 'chatbot', 'llm', 'translation', 'generative text', 'gpt', 'bert']):
            return 'Language'
        return 'Other'

    df_incidents['modality'] = df_incidents[tech_col].apply(classify_modality)
    
    # Filter for only Vision and Language
    df_analysis = df_incidents[df_incidents['modality'].isin(['Vision', 'Language'])].copy()
    
    print("\n--- Modality Counts ---")
    print(df_analysis['modality'].value_counts())

    # 2. Clean Intentional Harm (Fixed Logic)
    def is_intentional(val):
        if pd.isna(val):
            return False
        val_str = str(val).lower().strip()
        # Check for presence of 'yes' or 'intentionally'
        if val_str.startswith('yes'):
            return True
        if 'intentionally designed' in val_str and 'not intentionally' not in val_str:
             # Handle cases like "Yes. Intentionally..."
             return True
        return False

    df_analysis['is_intentional'] = df_analysis[intent_col].apply(is_intentional)
    
    print("\n--- Intentional vs Unintentional Counts (Processed) ---")
    print(df_analysis['is_intentional'].value_counts())

    # 3. Contingency Table
    contingency = pd.crosstab(df_analysis['modality'], df_analysis['is_intentional'])
    
    # Reindex to ensure both False and True columns exist
    contingency = contingency.reindex(columns=[False, True], fill_value=0)
    contingency.columns = ['Unintentional', 'Intentional']
    
    print("\n--- Contingency Table (Final) ---")
    print(contingency)

    # Check if we have any intentional cases at all
    if contingency['Intentional'].sum() == 0:
        print("\nNo intentional cases found in the filtered subset. Statistical test cannot be performed.")
    else:
        # 4. Statistical Test
        # Use Fisher's Exact Test if sample sizes are small (< 5 in any cell of expected), otherwise Chi2
        # Given the previous debug output, counts might be very low.
        # Let's try Chi2 first, catch error, fallback to Fisher if needed or if appropriate.
        
        try:
            chi2, p, dof, ex = chi2_contingency(contingency)
            print(f"\nChi-Square Statistic: {chi2:.4f}")
            print(f"P-Value: {p:.4e}")
            
            # Check expected frequencies for validity of Chi2
            if (ex < 5).any():
                print("Warning: Some expected frequencies are < 5. Fisher's Exact Test is recommended.")
                res = fisher_exact(contingency)
                print(f"Fisher's Exact Test P-Value: {res[1]:.4e}")
                p = res[1] # Update p for conclusion

            if p < 0.05:
                print("Result: Statistically Significant difference found.")
            else:
                print("Result: No statistically significant difference found.")
        except Exception as e:
            print(f"Statistical test failed: {e}")

        # 5. Visualization
        # Calculate rates
        total = contingency.sum(axis=1)
        # Avoid division by zero
        rates = (contingency['Intentional'] / total.replace(0, 1)) * 100
        
        print("\n--- Intentional Harm Rates ---")
        print(rates)

        plt.figure(figsize=(8, 6))
        bars = plt.bar(rates.index, rates.values, color=['skyblue', 'salmon'])
        plt.title('Rate of Intentional Harm by AI Modality')
        plt.ylabel('Percentage of Intentional Incidents (%)')
        plt.xlabel('AI Modality')
        # Adjust ylim slightly above max value or default to 10 if 0
        top_val = max(rates.values) if len(rates) > 0 else 0
        plt.ylim(0, top_val * 1.2 if top_val > 0 else 10)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, 
                     f'{height:.1f}%',
                     ha='center', va='bottom')
        
        plt.show()
