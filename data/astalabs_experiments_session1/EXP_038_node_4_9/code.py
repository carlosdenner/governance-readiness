import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Define the file path
filename = 'step3_incident_coding.csv'

# Check if file exists in current directory or up one level
if os.path.exists(filename):
    filepath = filename
elif os.path.exists(os.path.join('..', filename)):
    filepath = os.path.join('..', filename)
else:
    # Fallback to the list of files provided in the prompt context to find where they might be
    # Assuming they are in the current working directory based on previous turns usually
    filepath = filename

print(f"Loading dataset from: {filepath}")

try:
    df = pd.read_csv(filepath)
    
    # Verify required columns exist
    required_columns = ['technique_count', 'sub_competency_ids']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns {missing_cols}")
        # Attempt to calculate technique_count if missing but techniques_used exists
        if 'technique_count' in missing_cols and 'techniques_used' in df.columns:
            print("Calculated technique_count from techniques_used.")
            df['technique_count'] = df['techniques_used'].fillna('').apply(lambda x: len(x.split(';')) if x else 0)
        else:
            raise ValueError(f"Cannot proceed without columns: {missing_cols}")

    # Calculate Gap Breadth (count of sub_competency_ids)
    # Handle NaNs and empty strings
    def count_gaps(val):
        if pd.isna(val) or val == '':
            return 0
        # Split by semicolon, strip whitespace, remove empty strings
        items = [x.strip() for x in str(val).split(';') if x.strip()]
        return len(items)

    df['gap_breadth'] = df['sub_competency_ids'].apply(count_gaps)

    # Extract vectors
    x = df['technique_count']
    y = df['gap_breadth']

    # Statistical Analysis
    pearson_corr, pearson_p = stats.pearsonr(x, y)
    spearman_corr, spearman_p = stats.spearmanr(x, y)

    print("\n--- Statistical Analysis ---")
    print(f"N = {len(df)}")
    print(f"Technique Count: Mean={x.mean():.2f}, Std={x.std():.2f}, Min={x.min()}, Max={x.max()}")
    print(f"Gap Breadth:     Mean={y.mean():.2f}, Std={y.std():.2f}, Min={y.min()}, Max={y.max()}")
    print("\nCorrelation Results:")
    print(f"Pearson correlation:  r={pearson_corr:.4f}, p={pearson_p:.4f}")
    print(f"Spearman correlation: rho={spearman_corr:.4f}, p={spearman_p:.4f}")

    # Visualization
    plt.figure(figsize=(10, 6))
    
    # Scatter plot with jitter to handle overlapping points (since data is discrete counts)
    x_jitter = x + np.random.normal(0, 0.1, size=len(x))
    y_jitter = y + np.random.normal(0, 0.1, size=len(y))
    
    plt.scatter(x_jitter, y_jitter, alpha=0.6, label='Incident Data (Jittered)')

    # Regression Line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line_x = np.array([x.min(), x.max()])
    line_y = slope * line_x + intercept
    plt.plot(line_x, line_y, color='red', label=f'Regression (r={r_value:.2f})')

    plt.title('Attack Complexity vs. Competency Gap Breadth')
    plt.xlabel('Technique Count (Complexity)')
    plt.ylabel('Gap Breadth (Missing Sub-competencies)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
