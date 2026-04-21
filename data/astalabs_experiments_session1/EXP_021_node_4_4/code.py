import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

print("=== Starting Experiment: Audit Logging & Telemetry Centrality ===")

# 1. Load Data with robust path checking
filename = 'step2_crosswalk_matrix.csv'
filepath = None

if os.path.exists(filename):
    filepath = filename
elif os.path.exists(os.path.join('..', filename)):
    filepath = os.path.join('..', filename)
else:
    print(f"Error: {filename} not found in current ({os.getcwd()}) or parent directory.")
    sys.exit(1)

print(f"Loading dataset from: {filepath}")
df = pd.read_csv(filepath)

# 2. Identify Control Columns
# Metadata columns as per dataset description
metadata_cols = ['req_id', 'source', 'function', 'requirement', 'bundle', 'competency_statement']
control_cols = [c for c in df.columns if c not in metadata_cols]

print(f"Number of control columns identified: {len(control_cols)}")

target_control = 'Audit Logging & Telemetry'
if target_control not in control_cols:
    print(f"Error: '{target_control}' column not found.")
    print("Available columns:", control_cols)
    sys.exit(1)

# 3. Calculate Counts
# The dataset has 'X' for mappings, empty otherwise.
control_counts = {}

for col in control_cols:
    # Fill NA with empty string, convert to string, strip whitespace, uppercase, compare to 'X'
    count = (df[col].fillna('').astype(str).str.strip().str.upper() == 'X').sum()
    control_counts[col] = count

# Convert to Series for easier handling
counts_series = pd.Series(control_counts).sort_values(ascending=False)

# 4. Isolate Target and Others
target_count = counts_series[target_control]
other_counts = counts_series.drop(target_control)

# 5. Calculate Statistics
mean_others = other_counts.mean()
std_others = other_counts.std()

# 6. Calculate Z-Score
# Z = (x - mean) / std
if std_others == 0:
    z_score = 0
    print("Warning: Standard deviation of other controls is 0.")
else:
    z_score = (target_count - mean_others) / std_others

# 7. Output Results
print("\n--- Control Frequency Analysis ---")
print(f"Target Control: '{target_control}'")
print(f"Target Count: {target_count}")
print(f"Mean of Other Controls: {mean_others:.4f}")
print(f"Std Dev of Other Controls: {std_others:.4f}")
print(f"Z-Score: {z_score:.4f}")

if abs(z_score) > 1.96:
    print("Interpretation: The control is a statistical outlier (p < 0.05).")
else:
    print("Interpretation: The control is NOT a statistical outlier.")

print("\nTop 5 Controls:")
print(counts_series.head(5))

# 8. Visualization
plt.figure(figsize=(12, 8))
colors = ['red' if x == target_control else 'skyblue' for x in counts_series.index]
counts_series.plot(kind='bar', color=colors)
plt.title(f'Requirement Mapping Frequency per Architecture Control (Z={z_score:.2f})')
plt.xlabel('Architecture Control')
plt.ylabel('Number of Mapped Requirements')
plt.axhline(y=mean_others, color='green', linestyle='--', label=f'Mean (Others): {mean_others:.1f}')
plt.legend()
plt.tight_layout()
plt.show()