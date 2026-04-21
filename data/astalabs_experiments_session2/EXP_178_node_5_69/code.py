import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest

# Define dataset path - trying current directory based on previous successful runs in context
file_path = 'astalabs_discovery_all_data.csv'

try:
    # Load dataset
    # Using low_memory=False to avoid dtype warnings on mixed columns
    df = pd.read_csv(file_path, low_memory=False)
    
    # Filter for AIID incidents
    aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
    
    # Define Adversarial Keywords
    adversarial_keywords = [
        'attack', 'adversarial', 'poison', 'evasion', 'extraction', 'hack', 
        'backdoor', 'trojan', 'inference', 'inversion', 'manipulation', 'security',
        'exploit', 'breach'
    ]
    
    # Identify potential text columns
    # Check for columns that might contain text data describing the incident
    potential_cols = ['title', 'description', 'summary', 'Known AI Technical Failure', 'incident_title', 'incident_description']
    available_cols = [c for c in potential_cols if c in aiid_df.columns]
    
    print(f"Analyzing {len(aiid_df)} AIID incidents using columns: {available_cols}")
    
    if not available_cols:
        # Fallback: if no specific text columns found, try to use all object columns (risky but better than nothing)
        print("Warning: Expected text columns not found. Searching all object columns.")
        obj_cols = aiid_df.select_dtypes(include=['object']).columns
        available_cols = obj_cols

    # Combine text for search
    # creating a temporary column for searching
    aiid_df['combined_text'] = aiid_df[available_cols].fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1).str.lower()
    
    # Flag Adversarial Incidents
    pattern = '|'.join(adversarial_keywords)
    aiid_df['is_adversarial'] = aiid_df['combined_text'].str.contains(pattern, case=False, regex=True)
    
    # Calculate Statistics
    total_incidents = len(aiid_df)
    adversarial_count = aiid_df['is_adversarial'].sum()
    non_adversarial_count = total_incidents - adversarial_count
    adversarial_prop = adversarial_count / total_incidents
    
    print(f"\n--- Results ---")
    print(f"Total Incidents: {total_incidents}")
    print(f"Adversarial Incidents: {adversarial_count}")
    print(f"Non-Adversarial Incidents: {non_adversarial_count}")
    print(f"Adversarial Proportion: {adversarial_prop:.4%}")
    
    # One-sample Z-test
    # Null Hypothesis (H0): p = 0.05
    # Alternative Hypothesis (H1): p < 0.05
    # We want to see if the real proportion is significantly LESS than 5%
    
    if adversarial_count < total_incidents and total_incidents > 0:
        stat, p_value = proportions_ztest(count=adversarial_count, nobs=total_incidents, value=0.05, alternative='smaller')
        print(f"\nOne-sample Z-test (Test Value=0.05, Alternative='smaller'):")
        print(f"Z-statistic: {stat:.4f}")
        print(f"P-value: {p_value:.4e}")
        
        if p_value < 0.05:
            print("Conclusion: REJECT H0. The proportion of adversarial incidents is significantly less than 5%.")
        else:
            print("Conclusion: FAIL TO REJECT H0. Evidence does not support that the proportion is less than 5%.")
    else:
        print("Cannot perform Z-test due to data constraints (e.g., 0 incidents or 100% match).")

    # Visualization
    labels = ['Non-Adversarial', 'Adversarial']
    sizes = [non_adversarial_count, adversarial_count]
    colors = ['lightgray', 'red']
    explode = (0, 0.1) 

    plt.figure(figsize=(10, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=45)
    plt.title(f'Prevalence of Adversarial Incidents in AIID (n={total_incidents})')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: File '{file_path}' not found in current directory.")
except Exception as e:
    print(f"An error occurred: {e}")
