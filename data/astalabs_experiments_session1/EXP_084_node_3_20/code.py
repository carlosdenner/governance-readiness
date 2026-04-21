import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import os

# --- 1. Load Dataset ---
file_name = 'step2_crosswalk_matrix.csv'
possible_paths = [file_name, f'../{file_name}']

df = None
for path in possible_paths:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            print(f"Successfully loaded {path}")
            break
        except Exception as e:
            print(f"Failed to read {path}: {e}")

if df is None:
    print(f"Error: Could not find {file_name} in {possible_paths}")
    # Stop execution if dataset not found
    exit(1)

# --- 2. Define Control Groups ---
# Based on hypothesis: GenAI specific vs Traditional Governance

genai_controls = [
    'Single-Agent Orchestration Pattern',
    'Multi-Agent Orchestration Pattern',
    'Tool-Use Boundaries & Least-Privilege Access',
    'Nondeterminism Controls & Output Validation',
    'RAG Architecture & Data Grounding',
    'Prompt Management & Secret Handling',
    'GenAIOps / MLOps Lifecycle Governance'
]

traditional_controls = [
    'AI Risk Policy & Accountability Structures',
    'Threat Modeling & Red-Teaming',
    'Incident Response & Recovery Playbooks',
    'Audit Logging & Telemetry',
    'Regulatory Compliance Documentation',
    'Supply Chain & Vendor Risk Controls',
    'Data Governance & Access Controls',
    'Evaluation & Monitoring Infrastructure',
    'Human Override & Control Mechanisms'
]

# Validate columns exist
available_cols = df.columns.tolist()
genai_controls = [c for c in genai_controls if c in available_cols]
traditional_controls = [c for c in traditional_controls if c in available_cols]

print(f"\nGenAI Controls identified: {len(genai_controls)}")
print(f"Traditional Controls identified: {len(traditional_controls)}")

# --- 3. Calculate Mapping Counts ---
# The matrix contains 'X' or NaN/empty. We count non-null entries.

genai_counts = []
for col in genai_controls:
    # Count non-null and non-empty strings
    count = df[col].apply(lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0).sum()
    genai_counts.append(count)

trad_counts = []
for col in traditional_controls:
    count = df[col].apply(lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0).sum()
    trad_counts.append(count)

# --- 4. Statistical Analysis ---
genai_mean = np.mean(genai_counts)
genai_std = np.std(genai_counts, ddof=1)
trad_mean = np.mean(trad_counts)
trad_std = np.std(trad_counts, ddof=1)

print("\n--- Descriptive Statistics ---")
print(f"GenAI Controls (n={len(genai_counts)}): Mean = {genai_mean:.2f} (SD={genai_std:.2f})")
print(f"Traditional Controls (n={len(trad_counts)}): Mean = {trad_mean:.2f} (SD={trad_std:.2f})")

# Independent T-Test (assuming unequal variance aka Welch's t-test)
t_stat, p_val = stats.ttest_ind(genai_counts, trad_counts, equal_var=False)

print("\n--- Hypothesis Test Results ---")
print(f"T-Statistic: {t_stat:.4f}")
print(f"P-Value: {p_val:.4f}")
if p_val < 0.05:
    print("Conclusion: Significant difference found (Reject H0)")
else:
    print("Conclusion: No significant difference found (Fail to reject H0)")

# --- 5. Visualization ---
plt.figure(figsize=(8, 6))

means = [genai_mean, trad_mean]
# Standard Error for error bars
sem = [genai_std / np.sqrt(len(genai_counts)), trad_std / np.sqrt(len(trad_counts))]
labels = ['GenAI-Native Controls', 'Traditional Controls']
colors = ['#FF9999', '#66B2FF']

bars = plt.bar(labels, means, yerr=sem, capsize=10, color=colors, alpha=0.9, edgecolor='grey')

# Annotate bars
for bar, v in zip(bars, means):
    plt.text(bar.get_x() + bar.get_width()/2, v + 0.2, f"{v:.1f}", ha='center', fontweight='bold')

# Scatter plot of individual points to show distribution
# Add jitter to x-axis for visibility
x_genai = np.random.normal(0, 0.05, size=len(genai_counts))
x_trad = np.random.normal(1, 0.05, size=len(trad_counts))

plt.scatter(x_genai, genai_counts, color='darkred', alpha=0.6, zorder=3, label='GenAI Control Counts')
plt.scatter(x_trad, trad_counts, color='darkblue', alpha=0.6, zorder=3, label='Traditional Control Counts')

plt.ylabel('Avg. Mappings per Control')
plt.title('Governance Gap: Mapping Frequency of GenAI vs Traditional Controls')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
