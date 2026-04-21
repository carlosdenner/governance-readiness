import json
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import sys
import os

# Try to locate the file
filename = 'step3_enrichments.json'
if os.path.exists(filename):
    filepath = filename
elif os.path.exists(os.path.join('..', filename)):
    filepath = os.path.join('..', filename)
else:
    # Last ditch effort: search recursively or just fail
    print(f"Error: {filename} not found in current ({os.getcwd()}) or parent directory.")
    sys.exit(1)

try:
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Ensure required columns exist
    if 'incident_date' not in df.columns or 'technique_count' not in df.columns:
        print("Error: Required columns 'incident_date' or 'technique_count' missing.")
        sys.exit(1)

    # Convert incident_date to datetime
    # Coerce errors to NaT to handle potential malformed dates, then drop them
    df['incident_date_dt'] = pd.to_datetime(df['incident_date'], errors='coerce')
    
    # Filter out rows with invalid dates or technique counts
    df_clean = df.dropna(subset=['incident_date_dt', 'technique_count']).copy()
    
    if len(df_clean) < 2:
        print("Insufficient data points for correlation analysis.")
        sys.exit(0)

    # Convert date to ordinal number for correlation calculation
    df_clean['date_numeric'] = df_clean['incident_date_dt'].apply(lambda x: x.toordinal())

    # Perform Correlation Analysis
    pearson_r, pearson_p = stats.pearsonr(df_clean['date_numeric'], df_clean['technique_count'])
    spearman_r, spearman_p = stats.spearmanr(df_clean['date_numeric'], df_clean['technique_count'])

    print("=== Temporal Analysis of Attack Complexity ===")
    print(f"File loaded: {filepath}")
    print(f"Number of incidents analyzed: {len(df_clean)}")
    print(f"Date Range: {df_clean['incident_date_dt'].min().date()} to {df_clean['incident_date_dt'].max().date()}")
    print("\n--- Correlation Results ---")
    print(f"Pearson Correlation (r): {pearson_r:.4f} (p-value: {pearson_p:.4f})")
    print(f"Spearman Correlation (rho): {spearman_r:.4f} (p-value: {spearman_p:.4f})")

    # Interpretation
    alpha = 0.05
    significance = "statistically significant" if pearson_p < alpha else "not statistically significant"
    direction = "positive" if pearson_r > 0 else "negative"
        
    print(f"\nConclusion: There is a weak {direction} correlation which is {significance}.")

    # Visualization
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(df_clean['incident_date_dt'], df_clean['technique_count'], 
                color='#4c72b0', alpha=0.7, label='Incident', edgecolors='w', s=60)
    
    # Trend line
    z = np.polyfit(df_clean['date_numeric'], df_clean['technique_count'], 1)
    p = np.poly1d(z)
    
    # Plot trend line
    plt.plot(df_clean['incident_date_dt'], p(df_clean['date_numeric']), 
             color='#c44e52', linestyle='--', linewidth=2, 
             label=f'Trend (slope={z[0]:.2e})')
    
    plt.title('Temporal Trend of AI Incident Complexity (Technique Count)')
    plt.xlabel('Incident Date')
    plt.ylabel('Technique Count (Complexity)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
