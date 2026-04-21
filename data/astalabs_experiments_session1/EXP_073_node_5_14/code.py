import pandas as pd
import scipy.stats as stats
import numpy as np

# Attempt to load the dataset using the current directory
filename = 'step2_competency_statements.csv'
try:
    df = pd.read_csv(filename)
    print(f"Successfully loaded {filename}")
except FileNotFoundError:
    # Fallback to checking one level up if current dir fails, though previous error suggests current dir is correct context
    try:
        df = pd.read_csv('../' + filename)
        print(f"Successfully loaded ../{filename}")
    except FileNotFoundError:
        print(f"Error: Could not find {filename} in current or parent directory.")
        raise

# Helper function to count controls in the 'applicable_controls' column
# The column contains semicolon-separated values
def count_controls(val):
    if pd.isna(val) or str(val).strip() == '':
        return 0
    # Split by semicolon, strip whitespace, and filter out empty strings
    items = [x.strip() for x in str(val).split(';') if x.strip()]
    return len(items)

# Apply the counting function
df['control_count'] = df['applicable_controls'].apply(count_controls)

# Normalize the 'confidence' column to handle potential case inconsistencies
df['confidence_norm'] = df['confidence'].astype(str).str.lower().str.strip()

# Create the two groups: High vs Medium/Low
group_high = df[df['confidence_norm'] == 'high']['control_count']
group_others = df[df['confidence_norm'].isin(['medium', 'low'])]['control_count']

# Calculate Descriptive Statistics
mean_high = group_high.mean()
std_high = group_high.std()
n_high = len(group_high)

mean_others = group_others.mean()
std_others = group_others.std()
n_others = len(group_others)

print("\n--- Descriptive Statistics ---")
print(f"High Confidence (N={n_high}): Mean = {mean_high:.4f}, Std Dev = {std_high:.4f}")
print(f"Medium/Low Confidence (N={n_others}): Mean = {mean_others:.4f}, Std Dev = {std_others:.4f}")

# Perform Welch's Independent Samples T-test (does not assume equal variance)
t_stat, p_val = stats.ttest_ind(group_high, group_others, equal_var=False)

print("\n--- Hypothesis Test Results ---")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.4f}")

# Interpretation
alpha = 0.05
if p_val < alpha:
    print("Conclusion: The difference is statistically significant (Reject H0).")
else:
    print("Conclusion: The difference is NOT statistically significant (Fail to reject H0).")
