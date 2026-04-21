import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import re
import sys

def load_data():
    try:
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
    except FileNotFoundError:
        try:
            df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)
        except FileNotFoundError:
            print("Dataset not found.")
            sys.exit(1)
    return df

def classify_code_access(val):
    if not isinstance(val, str):
        return np.nan
    val_lower = val.lower()
    
    # Negative indicators
    # Use word boundaries for 'no' to avoid matching inside words like 'innovation'
    # 'proprietary' and 'restricted' are distinctive enough
    is_negative = False
    if re.search(r'\bno\b', val_lower) or 'restricted' in val_lower or 'proprietary' in val_lower:
        is_negative = True
        
    # Positive indicators
    is_positive = False
    if re.search(r'\byes\b', val_lower) or 'open' in val_lower:
        is_positive = True
        
    # Classification Logic
    if is_positive and not is_negative:
        return 1
    elif is_negative and not is_positive:
        return 0
    elif is_positive and is_negative:
        # Conflict: usually 'No, but...' or 'Yes, however restricted...'
        # In context of 'Code Sovereignty', restrictions usually mean lack of full sovereignty.
        return 0
    else:
        return np.nan

def main():
    print("Loading dataset...")
    df = load_data()
    
    # Filter for EO 13960 Scored subset
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO 13960 subset size: {len(df_eo)}")
    
    # Map Development Method
    if '22_dev_method' not in df_eo.columns:
        print("Error: Column '22_dev_method' not found.")
        return
        
    dev_map = {
        'Developed with contracting resources.': 'Contracted',
        'Developed in-house.': 'In-House'
    }
    df_eo['dev_model'] = df_eo['22_dev_method'].map(dev_map)
    
    # Filter valid development models
    df_analysis = df_eo.dropna(subset=['dev_model']).copy()
    print(f"Records with valid Development Method: {len(df_analysis)}")
    
    # Process Code Access
    col_access = '38_code_access'
    if col_access not in df_analysis.columns:
        print(f"Error: Column '{col_access}' not found.")
        return
        
    df_analysis['access_binary'] = df_analysis[col_access].apply(classify_code_access)
    
    # Filter valid access scores
    df_clean = df_analysis.dropna(subset=['access_binary'])
    print(f"Records with valid Code Access data: {len(df_clean)}")
    
    # Calculate Statistics
    summary = df_clean.groupby('dev_model')['access_binary'].agg(['count', 'mean'])
    summary['percent'] = summary['mean'] * 100
    
    print("\nSummary Statistics (Code Access Rates):")
    print(summary)
    
    # Chi-Square Test
    contingency = pd.crosstab(df_clean['dev_model'], df_clean['access_binary'])
    print("\nContingency Table (0=No Access, 1=Access):")
    print(contingency)
    
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi-Square Test Results:")
    print(f"Chi-Square Statistic: {chi2:.4f}")
    print(f"P-Value: {p:.4e}")
    
    if p < 0.05:
        print("Conclusion: Significant relationship between development model and code access.")
    else:
        print("Conclusion: No significant relationship detected.")
        
    # Visualization
    plt.figure(figsize=(8, 6))
    colors = ['#ff9999', '#66b3ff']
    ax = summary['percent'].plot(kind='bar', color=colors, edgecolor='black', rot=0)
    
    plt.title('Code Access Rates by Development Model')
    plt.ylabel('Percentage with Code Access (%)')
    plt.xlabel('Development Model')
    plt.ylim(0, 105)
    
    # Annotate bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
                    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()