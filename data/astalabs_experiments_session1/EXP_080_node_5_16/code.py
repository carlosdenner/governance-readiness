import pandas as pd
import matplotlib.pyplot as plt
import sys

# [debug] Print python version and current working directory to ensure environment consistency
# print(sys.version)
# import os
# print(os.getcwd())

# Load the dataset
file_path = '../step2_crosswalk_matrix.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    # Fallback if running in a different context where files might be in current dir
    file_path = 'step2_crosswalk_matrix.csv'
    df = pd.read_csv(file_path)

# Identify architecture control columns
# Based on metadata, the first 6 columns are metadata (req_id to competency_statement)
# The rest are architecture controls.
metadata_cols_count = 6
control_cols = df.columns[metadata_cols_count:]

# Calculate frequency of mappings for each control
# Cells contain "X" if mapped, otherwise NaN/empty.
control_counts = {}
for col in control_cols:
    # Count non-null values. 
    # The previous exploration showed count < 42 for sparse columns, implying NaNs for empty.
    control_counts[col] = df[col].count()

# Create a DataFrame for Pareto analysis
pareto_df = pd.DataFrame(list(control_counts.items()), columns=['Control', 'Frequency'])

# Sort by frequency descending
pareto_df = pareto_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)

# Calculate cumulative metrics
total_mappings = pareto_df['Frequency'].sum()
pareto_df['Cumulative Frequency'] = pareto_df['Frequency'].cumsum()
pareto_df['Cumulative Percentage'] = (pareto_df['Cumulative Frequency'] / total_mappings) * 100

# Identify the "Hub" controls (covering 80% of mappings)
threshold = 80.0
# Find the index where cumulative percentage first exceeds or equals the threshold
hubs_df = pareto_df[pareto_df['Cumulative Percentage'] <= threshold].copy()
# If the first item crossing the threshold isn't included (because it jumps from <80 to >80),
# we need to include the first one that puts it over the top.
first_over_idx = pareto_df[pareto_df['Cumulative Percentage'] >= threshold].index.min()
if pd.notna(first_over_idx):
    # Include all up to this index
    hubs_df = pareto_df.iloc[:int(first_over_idx)+1]

num_hubs = len(hubs_df)
total_controls = len(control_cols)
percent_controls_needed = (num_hubs / total_controls) * 100

print("=== Pareto Analysis Results ===")
print(f"Total Mappings (X): {total_mappings}")
print(f"Total Architecture Controls: {total_controls}")
print(f"Controls needed to cover >= 80% of mappings: {num_hubs} ({percent_controls_needed:.1f}% of controls)")

print("\nTop 'Compliance Hub' Controls:")
print(hubs_df[['Control', 'Frequency', 'Cumulative Percentage']].to_string(index=False))

print("\nFull Pareto Table:")
print(pareto_df[['Control', 'Frequency', 'Cumulative Percentage']].to_string())

# Visualisation
fig, ax1 = plt.subplots(figsize=(14, 8))

# Bar Chart for Frequency
ax1.bar(pareto_df['Control'], pareto_df['Frequency'], color='skyblue', label='Frequency')
ax1.set_xlabel('Architecture Controls', fontsize=10)
ax1.set_ylabel('Frequency of Mappings', color='blue', fontsize=10)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticklabels(pareto_df['Control'], rotation=45, ha='right', fontsize=8)

# Line Chart for Cumulative Percentage
ax2 = ax1.twinx()
ax2.plot(pareto_df['Control'], pareto_df['Cumulative Percentage'], color='red', marker='o', linestyle='-', label='Cumulative %')
ax2.set_ylabel('Cumulative Percentage (%)', color='red', fontsize=10)
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(0, 110)

# 80% Threshold Line
ax2.axhline(y=80, color='green', linestyle='--', linewidth=2, label='80% Threshold')

plt.title('Pareto Chart: Governance Requirements vs. Architecture Controls')
fig.tight_layout()
plt.show()
