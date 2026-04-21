import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load the dataset
filepath = '../astalabs_discovery_all_data.csv'
try:
    df = pd.read_csv(filepath, low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for EO 13960 scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"Total EO 13960 records: {len(eo_data)}")

# --- MAPPING LOGIC ---

def map_internal_review(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    
    if s == '' or s == 'nan':
        return np.nan
        
    # Explicit negatives
    if 'no documentation' in s:
        return False
    
    # Positives (Documentation exists in some form or explicit Yes)
    # Fixed syntax error here: added 'x' before 'in'
    if any(x in s for x in ['limited', 'developed', 'published', 'partially', 'yes']):
        return True
        
    return np.nan

def map_ato(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    
    if s == '' or s == 'nan':
        return np.nan
        
    # Explicit negatives
    if s.startswith('no'):
        return False
        
    # Explicit positives
    if 'yes' in s or 'operated' in s:
        return True
        
    # Ambiguous cases treated as NaN
    return np.nan

# Apply mappings
eo_data['has_review'] = eo_data['50_internal_review'].apply(map_internal_review)
eo_data['has_ato'] = eo_data['40_has_ato'].apply(map_ato)

# Filter valid data
clean_data = eo_data.dropna(subset=['has_review', 'has_ato'])

print(f"\nRecords after robust mapping and cleaning: {len(clean_data)}")
print(f"Dropped {len(eo_data) - len(clean_data)} records.")

# --- ANALYSIS ---

if len(clean_data) == 0:
    print("Insufficient data for analysis.")
else:
    # Contingency Table
    contingency_table = pd.crosstab(clean_data['has_review'], clean_data['has_ato'])
    # Check shape to assign correct index/column names
    if contingency_table.shape == (2, 2):
        contingency_table.index = ['No Review', 'Has Review']
        contingency_table.columns = ['No ATO', 'Has ATO']
    
    print("\nContingency Table (Count):")
    print(contingency_table)

    # Calculate Probabilities
    # P(ATO | Review)
    review_yes = clean_data[clean_data['has_review'] == True]
    p_ato_given_review = review_yes['has_ato'].mean() if len(review_yes) > 0 else 0
    
    # P(ATO | No Review)
    review_no = clean_data[clean_data['has_review'] == False]
    p_ato_given_no_review = review_no['has_ato'].mean() if len(review_no) > 0 else 0
    
    print(f"\nProbability of ATO given Internal Review: {p_ato_given_review:.2%}")
    print(f"Probability of ATO given NO Internal Review: {p_ato_given_no_review:.2%}")
    
    # Chi-Square Test
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")
    
    # Visualization
    labels = ['No Internal Review', 'Internal Review']
    probs = [p_ato_given_no_review, p_ato_given_review]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, probs, color=['#d9534f', '#5bc0de'])
    plt.ylabel('Probability of Having ATO')
    plt.title('The Gatekeeper Effect: Internal Review vs ATO Status')
    plt.ylim(0, 1.0)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.1%}',
                 ha='center', va='bottom')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Conclusion
    alpha = 0.05
    if p < alpha:
        print("\nResult: Statistically Significant. The hypothesis is supported.")
    else:
        print("\nResult: Not Statistically Significant. The hypothesis is not supported.")
