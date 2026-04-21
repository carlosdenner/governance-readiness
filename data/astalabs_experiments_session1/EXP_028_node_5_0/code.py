import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# Load the dataset
# Based on previous successful runs, the file is likely in the current directory.
file_path = 'step2_crosswalk_matrix.csv'

try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    # Fallback to checking parent directory if current fails, though unlikely based on history
    try:
        file_path = '../step2_crosswalk_matrix.csv'
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully from parent directory.")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

# Identify architecture control columns
# The first 6 columns are metadata
metadata_cols = ['req_id', 'source', 'function', 'requirement', 'bundle', 'competency_statement']
control_cols = [col for col in df.columns if col not in metadata_cols]

print(f"Identified {len(control_cols)} architecture controls.")

# Calculate frequency of mappings
# The matrix uses 'X' (or non-null) for presence.
# specific check for 'X' or non-null values
control_counts = df[control_cols].apply(lambda x: x.notna().sum())

# Create a DataFrame for analysis
pareto_df = pd.DataFrame({'Control': control_counts.index, 'Frequency': control_counts.values})

# Sort by Frequency descending
pareto_df = pareto_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)

# Calculate cumulative metrics
total_mappings = pareto_df['Frequency'].sum()
pareto_df['Cumulative_Frequency'] = pareto_df['Frequency'].cumsum()
pareto_df['Cumulative_Percentage'] = (pareto_df['Cumulative_Frequency'] / total_mappings) * 100

# Add Rank
pareto_df['Rank'] = pareto_df.index + 1

# Hypothesis Check: Top 20% of controls
num_controls = len(control_cols)
top_20_cutoff_index = int(np.ceil(num_controls * 0.2)) # 20% of 18 is 3.6 -> 4 controls

# Get coverage at the cutoff
coverage_at_cutoff = pareto_df.loc[top_20_cutoff_index - 1, 'Cumulative_Percentage']
top_controls_names = pareto_df.loc[:top_20_cutoff_index-1, 'Control'].tolist()

print("\n--- Pareto Analysis Table ---")
print(pareto_df[['Rank', 'Control', 'Frequency', 'Cumulative_Percentage']].to_string(index=False))

print(f"\nTotal Mappings: {total_mappings}")
print(f"Top 20% (approx {top_20_cutoff_index} controls): {coverage_at_cutoff:.2f}% coverage")
print(f"Top controls: {top_controls_names}")

hypothesis_result = "CONFIRMED" if coverage_at_cutoff >= 80 else "REJECTED"
print(f"\nHypothesis {hypothesis_result}: The top {top_20_cutoff_index} controls cover {coverage_at_cutoff:.1f}% of mappings (Target: 80%).")

# Visualization
fig, ax1 = plt.subplots(figsize=(14, 8))

# Bar plot for Frequency
color = 'tab:blue'
ax1.set_xlabel('Architecture Controls (Ranked)')
ax1.set_ylabel('Frequency (Mapping Count)', color=color)
bars = ax1.bar(pareto_df['Control'], pareto_df['Frequency'], color=color, alpha=0.7)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticklabels(pareto_df['Control'], rotation=45, ha='right', fontsize=9)

# Line plot for Cumulative Percentage
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Cumulative Percentage (%)', color=color)
ax2.plot(pareto_df['Control'], pareto_df['Cumulative_Percentage'], color=color, marker='o', linestyle='-', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 110)

# Reference lines
ax2.axhline(y=80, color='green', linestyle='--', linewidth=2, label='80% Target')
ax2.axvline(x=top_20_cutoff_index - 0.5, color='orange', linestyle='--', linewidth=2, label='Top 20% Cutoff')

# Adding text labels to the line points
for i, txt in enumerate(pareto_df['Cumulative_Percentage']):
    ax2.annotate(f"{txt:.1f}%", (i, txt), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.title(f'Pareto Analysis: Do the top 20% of controls cover 80% of requirements?\nResult: {hypothesis_result}')
fig.tight_layout()
plt.legend(loc='lower right')
plt.show()