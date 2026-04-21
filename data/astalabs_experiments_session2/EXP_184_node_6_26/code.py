import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys

def run_experiment():
    print("Starting experiment: Rights vs. Safety Consultation Split...")
    
    # 1. Load dataset
    # Try loading from parent directory first as per instructions, then current
    file_name = 'astalabs_discovery_all_data.csv'
    paths = [f'../{file_name}', file_name]
    df_all = None
    
    for p in paths:
        try:
            df_all = pd.read_csv(p, low_memory=False)
            print(f"Successfully loaded {p}")
            break
        except FileNotFoundError:
            continue
            
    if df_all is None:
        print("Error: Dataset not found in ../ or current directory.")
        return

    # 2. Filter for EO 13960 Scored data
    df = df_all[df_all['source_table'] == 'eo13960_scored'].copy()
    print(f"Filtered eo13960_scored subset: {len(df)} rows")

    # 3. Define Columns
    col_impact = '17_impact_type'
    col_consult = '63_stakeholder_consult'
    
    # Validate columns exist
    if col_impact not in df.columns or col_consult not in df.columns:
        print(f"Error: Missing columns. Available: {df.columns.tolist()}")
        return

    # 4. Filter Groups (Rights vs Safety)
    # Normalize text to handle potential inconsistencies
    df['impact_norm'] = df[col_impact].fillna('').astype(str).str.lower().str.strip()
    
    # Define masks
    # Rights-Impacting: Contains 'rights', does NOT contain 'safety'
    mask_rights = df['impact_norm'].str.contains('rights') & ~df['impact_norm'].str.contains('safety')
    
    # Safety-Impacting: Contains 'safety', does NOT contain 'rights'
    mask_safety = df['impact_norm'].str.contains('safety') & ~df['impact_norm'].str.contains('rights')
    
    rights_df = df[mask_rights]
    safety_df = df[mask_safety]
    
    print(f"Groups Identified:")
    print(f"  - Rights-Impacting (Exclusive): {len(rights_df)}")
    print(f"  - Safety-Impacting (Exclusive): {len(safety_df)}")
    
    if len(rights_df) == 0 or len(safety_df) == 0:
        print("Error: One or both groups have 0 samples. Cannot perform test.")
        print("Sample Impact Types:", df['impact_norm'].unique()[:10])
        return

    # 5. Calculate Consultation Rates
    # We assume 'Yes' indicates consultation. 
    def parse_consultation(val):
        if pd.isna(val):
            return 0
        val_str = str(val).lower().strip()
        return 1 if val_str == 'yes' else 0

    rights_consulted = rights_df[col_consult].apply(parse_consultation)
    safety_consulted = safety_df[col_consult].apply(parse_consultation)

    k1 = rights_consulted.sum()
    n1 = len(rights_consulted)
    p1 = k1 / n1 if n1 > 0 else 0

    k2 = safety_consulted.sum()
    n2 = len(safety_consulted)
    p2 = k2 / n2 if n2 > 0 else 0

    print(f"\nConsultation Statistics:")
    print(f"  - Rights-Impacting: {k1}/{n1} consulted ({p1:.2%})")
    print(f"  - Safety-Impacting: {k2}/{n2} consulted ({p2:.2%})")

    # 6. Statistical Test (Two-Proportion Z-Test)
    # Pooled probability
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    
    if se == 0:
        print("Standard Error is 0 (identical proportions or zero variance).")
        z_score = 0
        p_value = 1.0
    else:
        z_score = (p1 - p2) / se
        p_value = 2 * (1 - norm.cdf(abs(z_score)))  # Two-sided p-value

    print(f"\nZ-Test Results:")
    print(f"  - Z-Score: {z_score:.4f}")
    print(f"  - P-Value: {p_value:.4e}")
    
    significance = "Significant" if p_value < 0.05 else "Not Significant"
    print(f"  - Conclusion: {significance} difference at alpha=0.05")

    # 7. Visualization
    plt.figure(figsize=(8, 6))
    categories = ['Rights-Impacting', 'Safety-Impacting']
    values = [p1, p2]
    colors = ['#3498db', '#e74c3c']
    
    bars = plt.bar(categories, values, color=colors, alpha=0.8)
    plt.ylabel('Proportion of Stakeholder Consultation')
    plt.title('Stakeholder Consultation: Rights vs. Safety')
    plt.ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
    
    # Add counts and percentages on bars
    for idx, rect in enumerate(bars):
        height = rect.get_height()
        count_text = f"{height:.1%}\n(n={k1 if idx==0 else k2})"
        plt.text(rect.get_x() + rect.get_width()/2., height, count_text,
                 ha='center', va='bottom')

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()