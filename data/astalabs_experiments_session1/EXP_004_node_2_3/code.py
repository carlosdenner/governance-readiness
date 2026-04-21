import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('step3_incident_coding.csv')
except FileNotFoundError:
    # Fallback to parent directory just in case
    try:
        df = pd.read_csv('../step3_incident_coding.csv')
    except FileNotFoundError:
        raise FileNotFoundError("Dataset 'step3_incident_coding.csv' not found in current or parent directory.")

# Convert incident_date to datetime
df['incident_date_dt'] = pd.to_datetime(df['incident_date'], errors='coerce')

# Drop rows with invalid dates
df = df.dropna(subset=['incident_date_dt'])

# Define Period
# Post-2023 includes 2023-01-01 onwards
cutoff_date = pd.Timestamp('2023-01-01')
df['Period'] = df['incident_date_dt'].apply(lambda x: 'Post-2023' if x >= cutoff_date else 'Pre-2023')

# Clean and categorize Gap Type
# Metadata says values are: trust-dominant, integration-dominant, both
df['Gap_Type_Raw'] = df['trust_integration_split'].astype(str).str.strip().str.lower()

mapping = {
    'trust-dominant': 'Trust-Dominant',
    'integration-dominant': 'Integration-Dominant',
    'both': 'Both'
}
df['Gap_Type'] = df['Gap_Type_Raw'].map(mapping)

# Drop rows where Gap_Type mapping failed (if any)
df = df.dropna(subset=['Gap_Type'])

# Generate Contingency Table
contingency_table = pd.crosstab(df['Gap_Type'], df['Period'])

# Ensure column order for logical flow
expected_periods = ['Pre-2023', 'Post-2023']
# Filter to only existing columns in case data is missing for one period
existing_periods = [p for p in expected_periods if p in contingency_table.columns]
contingency_table = contingency_table[existing_periods]

print("=== Contingency Table (Gap Type vs Period) ===")
print(contingency_table)

# Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("\n=== Chi-Square Test Results ===")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")

alpha = 0.05
if p < alpha:
    print("Result: Significant shift in distribution (Reject H0)")
else:
    print("Result: No significant shift detected (Fail to reject H0)")

# Calculate Column Percentages (Distribution within each Period)
col_percentages = contingency_table.div(contingency_table.sum(axis=0), axis=1) * 100
print("\n=== Column Percentages (Distribution per Period) ===")
print(col_percentages.round(2))

# Visualization: Stacked 100% Bar Chart to visualize proportions
# Transpose so X-axis is Period
plot_data = col_percentages.T
ax = plot_data.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')

plt.title('Proportional Distribution of Competency Gap Types (Pre vs Post 2023)')
plt.xlabel('Period')
plt.ylabel('Percentage of Incidents (%)')
plt.legend(title='Gap Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()