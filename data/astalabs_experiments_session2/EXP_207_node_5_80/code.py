import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import os

# --- 1. Load Data ---
possible_paths = ['../astalabs_discovery_all_data.csv', 'astalabs_discovery_all_data.csv']
path = next((p for p in possible_paths if os.path.exists(p)), None)

if not path:
    print("Error: Dataset not found in expected locations.")
else:
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path, low_memory=False)

    # Filter for EO 13960 data
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO 13960 records loaded: {len(df_eo)}")

    # --- 2. Keyword Definitions ---
    # Expanded lists to capture nuance
    rights_keywords = [
        'surveillance', 'police', 'eligibility', 'fraud', 'benefits', 'sentencing', 
        'adjudication', 'hiring', 'housing', 'law enforcement', 'asylum', 'border', 
        'biometric', 'facial recognition', 'screening', 'loan', 'credit', 'insurance', 
        'healthcare decisions', 'parole', 'investigation', 'threat', 'security'
    ]
    
    admin_keywords = [
        'process', 'sort', 'email', 'route', 'workflow', 'scheduling', 
        'translation', 'transcription', 'categorize', 'inventory', 'logistics', 
        'search', 'summarize', 'digitize', 'form filling', 'administrative'
    ]

    # --- 3. Categorization Logic ---
    def classify_risk(text):
        if not isinstance(text, str):
            return 'Unclassified'
        text_lower = text.lower()
        
        # Priority: Rights-Impacting > Administrative
        if any(kw in text_lower for kw in rights_keywords):
            return 'Rights-Impacting'
        elif any(kw in text_lower for kw in admin_keywords):
            return 'Administrative'
        else:
            return 'Other'

    df_eo['risk_category'] = df_eo['11_purpose_benefits'].apply(classify_risk)

    # --- 4. Clean Compliance Column ---
    # Column: '52_impact_assessment'. Inspecting values to ensure correct binary mapping.
    # Assuming standard Yes/No or variations.
    def parse_compliance(val):
        s = str(val).lower().strip()
        return 1 if s in ['yes', 'true', '1', 'y'] else 0

    df_eo['has_impact_assessment'] = df_eo['52_impact_assessment'].apply(parse_compliance)

    # --- 5. Analysis ---
    # Group by Risk Category
    summary = df_eo.groupby('risk_category')['has_impact_assessment'].agg(['count', 'sum', 'mean'])
    summary.columns = ['Total Cases', 'Compliant Cases', 'Compliance Rate']
    
    print("\n--- Compliance Summary by Category ---")
    print(summary)

    # Extract groups for statistical testing
    rights_group = df_eo[df_eo['risk_category'] == 'Rights-Impacting']
    admin_group = df_eo[df_eo['risk_category'] == 'Administrative']

    n_rights = len(rights_group)
    n_admin = len(admin_group)

    if n_rights > 0 and n_admin > 0:
        rights_compliant = rights_group['has_impact_assessment'].sum()
        admin_compliant = admin_group['has_impact_assessment'].sum()
        
        # Contingency Table
        #              Compliant | Non-Compliant
        # Rights     |    A      |      B
        # Admin      |    C      |      D
        
        contingency = [
            [rights_compliant, n_rights - rights_compliant],
            [admin_compliant, n_admin - admin_compliant]
        ]
        
        chi2, p, dof, ex = chi2_contingency(contingency)
        
        print("\n--- Statistical Test Results (Chi-Square) ---")
        print(f"Comparison: Rights-Impacting (n={n_rights}) vs Administrative (n={n_admin})")
        print(f"Rights Compliance Rate: {rights_compliant/n_rights:.2%}")
        print(f"Admin Compliance Rate:  {admin_compliant/n_admin:.2%}")
        print(f"Chi2 Statistic: {chi2:.4f}")
        print(f"P-Value: {p:.4f}")
        
        if p < 0.05:
            print("Result: Statistically Significant Difference")
        else:
            print("Result: No Statistically Significant Difference (Paradox Supported)")
            
        # --- 6. Visualization ---
        plt.figure(figsize=(10, 6))
        categories = ['Rights-Impacting', 'Administrative']
        rates = [rights_compliant/n_rights, admin_compliant/n_admin]
        
        # Create bar chart
        bars = plt.bar(categories, rates, color=['#d62728', '#1f77b4'], alpha=0.7)
        
        # Add labels
        plt.ylabel('Impact Assessment Compliance Rate')
        plt.title('Impact Assessment Compliance: Rights-Impacting vs. Administrative Use Cases')
        plt.ylim(0, 1.0)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, 
                     f'{height:.1%}', ha='center', va='bottom')
            
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.show()
        
    else:
        print("\nInsufficient data in one or both categories to perform statistical test.")