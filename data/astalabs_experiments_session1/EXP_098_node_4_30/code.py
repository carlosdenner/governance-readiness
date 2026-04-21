import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
import sys

# Robust file loading
possible_paths = ['step3_enrichments.json', '../step3_enrichments.json']
file_path = None

for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        break

if file_path is None:
    print("Error: Could not locate 'step3_enrichments.json' in current or parent directory.")
    # List current directory for debugging purposes if needed, but for now just exit gracefully
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in current dir: {os.listdir('.')}")
    sys.exit(1)

print(f"=== Loading Dataset from {file_path} ===")

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Successfully loaded {len(data)} records.")
except Exception as e:
    print(f"Error loading JSON: {e}")
    sys.exit(1)

# Extract metrics
cot_lengths = []
gap_counts = []

valid_entries = 0
for i, entry in enumerate(data):
    # 1. Chain of Thought Length (Word Count)
    cot = entry.get('chain_of_thought', '')
    if not isinstance(cot, str):
        cot = str(cot) if cot is not None else ''
    word_count = len(cot.split())
    
    # 2. Gap Count (Number of sub-competencies)
    sub_comps = entry.get('sub_competency_ids', [])
    
    count = 0
    if isinstance(sub_comps, list):
        count = len(sub_comps)
    elif isinstance(sub_comps, str):
        # Handle case where it might be a semicolon separated string
        cleaned = sub_comps.strip()
        if cleaned:
            count = len(cleaned.split(';'))
    
    # We only care if there is valid data (optional: filter out 0 gaps if deemed noise, but 0 is a valid data point)
    cot_lengths.append(word_count)
    gap_counts.append(count)
    valid_entries += 1

# Convert to numpy arrays for analysis
x = np.array(gap_counts)
y = np.array(cot_lengths)

# Statistical Analysis
if len(x) > 1:
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    print("\n=== Statistical Analysis ===")
    print(f"Number of Data Points: {len(x)}")
    print(f"Gap Count (Min/Max/Mean): {x.min()}/{x.max()}/{x.mean():.2f}")
    print(f"CoT Word Count (Min/Max/Mean): {y.min()}/{y.max()}/{y.mean():.2f}")
    print(f"Pearson Correlation (r): {r_value:.4f}")
    print(f"R-squared: {r_value**2:.4f}")
    print(f"P-value: {p_value:.4e}")

    # Visualization
    plt.figure(figsize=(10, 6))
    # Scatter plot
    plt.scatter(x, y, alpha=0.7, c='teal', edgecolors='k', label='Incidents')

    # Regression line
    if len(np.unique(x)) > 1: # Only plot line if x varies
        line_x = np.linspace(min(x), max(x), 100)
        line_y = slope * line_x + intercept
        plt.plot(line_x, line_y, color='red', linestyle='--', linewidth=2, label=f'Fit: r={r_value:.2f}, p={p_value:.3f}')

    plt.title('Analytical Effort vs. Incident Severity')
    plt.xlabel('Incident Severity (Number of Competency Gaps)')
    plt.ylabel('Analytical Effort (Chain of Thought Word Count)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
else:
    print("Not enough data points for analysis.")