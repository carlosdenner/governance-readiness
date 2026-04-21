import json
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import os

# Define file path
file_path = '../context_crosswalk_evidence.json'

# Load dataset
if not os.path.exists(file_path):
    # Fallback for local testing if directory structure differs
    file_path = 'context_crosswalk_evidence.json'

print(f"Loading dataset from {file_path}...")
with open(file_path, 'r') as f:
    data = json.load(f)

# Lists to store control counts
nist_counts = []
eu_counts = []

ignored_count = 0
ignored_ids = []

print(f"Processing {len(data)} requirements...")

for entry in data:
    req_id = str(entry.get('req_id', '')).strip()
    controls = entry.get('applicable_controls', [])
    
    # Determine number of controls
    if isinstance(controls, list):
        count = len(controls)
    elif isinstance(controls, str):
        # Handle string representation if necessary (e.g. "Control1, Control2")
        if controls.strip() == "":
            count = 0
        else:
            count = len([c.strip() for c in controls.split(',') if c.strip()])
    else:
        count = 0

    # Classification Logic
    # Assuming typical identifiers based on dataset description
    # NIST usually: "NIST", "Map", "RMF"
    # EU usually: "EU", "Art", "AI Act"
    
    req_upper = req_id.upper()
    
    if "NIST" in req_upper:
        nist_counts.append(count)
    elif "EU" in req_upper or "ART" in req_upper:
        eu_counts.append(count)
    else:
        # Check reasoning or statement if ID is ambiguous
        statement = str(entry.get('competency_statement', '')).upper()
        if "NIST" in statement:
            nist_counts.append(count)
        elif "EU AI ACT" in statement or "EUROPEAN" in statement:
            eu_counts.append(count)
        else:
            ignored_count += 1
            ignored_ids.append(req_id)

print(f"Found {len(nist_counts)} NIST requirements.")
print(f"Found {len(eu_counts)} EU AI Act requirements.")
if ignored_count > 0:
    print(f"Ignored {ignored_count} requirements from other sources (e.g., ISO, OWASP). Sample ignored: {ignored_ids[:3]}")

# Analysis
if len(nist_counts) < 2 or len(eu_counts) < 2:
    print("Insufficient data points for statistical analysis.")
else:
    # Descriptive Statistics
    nist_mean = np.mean(nist_counts)
    eu_mean = np.mean(eu_counts)
    nist_median = np.median(nist_counts)
    eu_median = np.median(eu_counts)
    
    print("\n--- Descriptive Statistics ---")
    print(f"NIST AI RMF: Mean = {nist_mean:.2f}, Median = {nist_median}, Max = {np.max(nist_counts)}, Min = {np.min(nist_counts)}")
    print(f"EU AI Act:   Mean = {eu_mean:.2f}, Median = {eu_median}, Max = {np.max(eu_counts)}, Min = {np.min(eu_counts)}")
    
    # Mann-Whitney U Test (Non-parametric test for independent samples)
    u_stat, p_val = stats.mannwhitneyu(eu_counts, nist_counts, alternative='two-sided')
    
    print("\n--- Statistical Test Results ---")
    print(f"Test: Mann-Whitney U Test")
    print(f"Hypothesis: EU requirements map to a different number of controls than NIST.")
    print(f"U-Statistic: {u_stat}")
    print(f"P-Value: {p_val:.4f}")
    
    if p_val < 0.05:
        print("Conclusion: Statistically significant difference detected.")
    else:
        print("Conclusion: No statistically significant difference detected.")

    # Visualization
    plt.figure(figsize=(10, 6))
    
    # Create boxplot
    data_to_plot = [nist_counts, eu_counts]
    labels = [f'NIST AI RMF\n(n={len(nist_counts)})', f'EU AI Act\n(n={len(eu_counts)})']
    
    plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                boxprops=dict(facecolor='#ADD8E6', color='blue'),
                medianprops=dict(color='red'))
    
    plt.title('Density of Architectural Controls: NIST AI RMF vs EU AI Act')
    plt.ylabel('Number of Mapped Controls per Requirement')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Annotate means
    plt.text(1, nist_mean + 0.1, f'Mean: {nist_mean:.1f}', ha='center', va='bottom', fontweight='bold')
    plt.text(2, eu_mean + 0.1, f'Mean: {eu_mean:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.show()
