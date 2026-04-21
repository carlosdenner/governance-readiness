import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import os
import sys

# Define file path based on instructions (one level above)
file_path = '../step3_incident_coding.csv'

# Fallback to current directory if not found (handling environment variations)
if not os.path.exists(file_path):
    file_path = 'step3_incident_coding.csv'

# Load the dataset
try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded dataset from: {file_path}")
except FileNotFoundError:
    print(f"Error: Could not find dataset at {file_path}")
    sys.exit(1)

# Preprocessing Dates
# 'incident_date' in ATLAS can be Year, Year-Month, or Full Date. 
# pd.to_datetime handles these, defaulting to Jan 1st for Year-only.
df['date_obj'] = pd.to_datetime(df['incident_date'], errors='coerce')

# Check for invalid dates
initial_count = len(df)
df = df.dropna(subset=['date_obj'])
dropped_count = initial_count - len(df)
if dropped_count > 0:
    print(f"Dropped {dropped_count} rows with invalid dates.")

# Ensure technique_count is numeric
df['technique_count'] = pd.to_numeric(df['technique_count'], errors='coerce')
df = df.dropna(subset=['technique_count'])

# Sort by date
df = df.sort_values('date_obj')

# --- Statistical Analysis ---

# Convert date to ordinal (integer) for correlation analysis
df['date_ordinal'] = df['date_obj'].apply(lambda x: x.toordinal())

x = df['date_ordinal']
y = df['technique_count']

if len(df) < 2:
    print("Insufficient data points for analysis.")
    sys.exit(0)

# Pearson Correlation (Linear)
pearson_r, pearson_p = stats.pearsonr(x, y)

# Spearman Correlation (Monotonic - better if trend is non-linear or data is ordinal-like)
spearman_r, spearman_p = stats.spearmanr(x, y)

# Linear Regression for the trend line
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Interpret trend
trend_direction = "Positive" if slope > 0 else "Negative"
significance = "Significant" if p_value < 0.05 else "Not Significant"

print("\n=== Temporal Trend Analysis of Incident Complexity ===")
print(f"Incidents Analyzed: {len(df)}")
print(f"Date Range: {df['date_obj'].min().date()} to {df['date_obj'].max().date()}")
print(f"Correlation (Pearson): r={pearson_r:.4f}, p={pearson_p:.4f}")
print(f"Correlation (Spearman): rho={spearman_r:.4f}, p={spearman_p:.4f}")
print(f"Linear Trend Slope: {slope:.5f} techniques/day")
print(f"Conclusion: {trend_direction} trend ({significance})")

# --- Visualization ---
plt.figure(figsize=(10, 6))

# Scatter plot of incidents
plt.scatter(df['date_obj'], df['technique_count'], color='blue', alpha=0.6, label='Incidents')

# Plot Regression Line
# Create a sequence of dates for the line to look smooth or just use min/max
x_line_dates = df['date_obj']
y_line_pred = slope * df['date_ordinal'] + intercept

plt.plot(x_line_dates, y_line_pred, color='red', linewidth=2, label=f'Trend (r={pearson_r:.2f})')

plt.title('Temporal Trend of AI Incident Complexity (Technique Count)')
plt.xlabel('Incident Date')
plt.ylabel('Complexity (Number of Techniques Used)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Improve date formatting on x-axis
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.show()