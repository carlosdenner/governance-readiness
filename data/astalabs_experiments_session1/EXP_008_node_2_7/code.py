import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import sys

# --- Helper Functions ---
def load_dataset(filename):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    elif os.path.exists(os.path.join("..", filename)):
        return pd.read_csv(os.path.join("..", filename))
    else:
        raise FileNotFoundError(f"{filename} not found")

print("=== Loading Datasets ===")
try:
    df_matrix = load_dataset("step2_crosswalk_matrix.csv")
    df_incidents = load_dataset("step3_incident_coding.csv")
    print("Datasets loaded successfully.")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# --- Step 1: Calculate Theoretical Centrality (Mapping Frequency) ---
# Metadata columns to exclude
metadata_cols = ['req_id', 'source', 'function', 'requirement', 'bundle', 'competency_statement']
control_cols = [c for c in df_matrix.columns if c not in metadata_cols]

# Calculate mapping frequency
mapping_counts = []
for col in control_cols:
    # Count non-empty, non-null cells
    count = df_matrix[col].apply(lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0).sum()
    mapping_counts.append(count)

df_analysis = pd.DataFrame({
    'Control_Name': control_cols,
    'Mapping_Frequency': mapping_counts
})

# Define Hub vs Spoke
median_freq = df_analysis['Mapping_Frequency'].median()
df_analysis['Type'] = df_analysis['Mapping_Frequency'].apply(lambda x: 'Hub' if x > median_freq else 'Spoke')

# --- Step 2: Calculate Empirical Failure Rate (Keyword Search) ---
# Define keywords for each control to search in incident narratives
control_keywords = {
    'Single-Agent Orchestration Pattern': ['single-agent', 'orchestration'],
    'Multi-Agent Orchestration Pattern': ['multi-agent'],
    'Tool-Use Boundaries & Least-Privilege Access': ['tool-use', 'least-privilege', 'privilege', 'permissions'],
    'Human-in-the-Loop Approval Gates': ['human-in-the-loop', 'approval', 'gate', 'authorization'],
    'Nondeterminism Controls & Output Validation': ['nondeterminism', 'output validation', 'input validation', 'verification'],
    'RAG Architecture & Data Grounding': ['rag', 'grounding', 'retrieval', 'hallucination', 'context'],
    'GenAIOps / MLOps Lifecycle Governance': ['mlops', 'genaiops', 'lifecycle', 'model hardening', 'versioning'],
    'Evaluation & Monitoring Infrastructure': ['evaluation', 'monitoring', 'observability', 'drift'],
    'Prompt Management & Secret Handling': ['prompt', 'secret', 'injection', 'credential', 'key'],
    'Scalable Modular Architecture (Archetypes)': ['modular', 'archetype', 'scalable', 'architecture'],
    'AI Risk Policy & Accountability Structures': ['risk policy', 'accountability', 'governance', 'policy'],
    'Threat Modeling & Red-Teaming': ['threat', 'red-team', 'adversarial', 'attack simulation'],
    'Incident Response & Recovery Playbooks': ['incident response', 'recovery', 'playbook', 'mitigation'],
    'Audit Logging & Telemetry': ['audit', 'logging', 'telemetry', 'trace', 'logs'],
    'Regulatory Compliance Documentation': ['compliance', 'documentation', 'regulatory', 'legal'],
    'Supply Chain & Vendor Risk Controls': ['supply chain', 'vendor', 'third-party', 'dependency'],
    'Data Governance & Access Controls': ['data governance', 'access control', 'rbac', 'data protection'],
    'Human Override & Control Mechanisms': ['override', 'stop button', 'intervention', 'human control']
}

# Prepare corpus for search
# Combine relevant text fields: summary, llm_gap_description, missing_controls (names)
df_incidents['search_text'] = (
    df_incidents['summary'].fillna('') + " " + 
    df_incidents['llm_gap_description'].fillna('') + " " + 
    df_incidents['missing_controls'].fillna('')
).str.lower()

incident_counts = []
for control in control_cols:
    keywords = control_keywords.get(control, [])
    # Count incidents where ANY keyword is present
    count = df_incidents['search_text'].apply(lambda text: 1 if any(k in text for k in keywords) else 0).sum()
    incident_counts.append(count)

df_analysis['Incident_Frequency'] = incident_counts

# --- Step 3: Statistics ---
print("\n--- Analysis Summary ---")
print(df_analysis[['Control_Name', 'Type', 'Mapping_Frequency', 'Incident_Frequency']].sort_values('Incident_Frequency', ascending=False))

# Pearson Correlation
corr_coef, p_value_corr = stats.pearsonr(df_analysis['Mapping_Frequency'], df_analysis['Incident_Frequency'])
print(f"\nPearson Correlation: r={corr_coef:.4f}, p={p_value_corr:.4f}")

# T-Test
group_hub = df_analysis[df_analysis['Type'] == 'Hub']['Incident_Frequency']
group_spoke = df_analysis[df_analysis['Type'] == 'Spoke']['Incident_Frequency']
t_stat, p_value_ttest = stats.ttest_ind(group_hub, group_spoke, equal_var=False)
print(f"T-Test (Hub vs Spoke): t={t_stat:.4f}, p={p_value_ttest:.4f}")

# --- Step 4: Visualization ---
plt.figure(figsize=(12, 7))

colors = {'Hub': '#d62728', 'Spoke': '#1f77b4'} # Red for Hub, Blue for Spoke

# Scatter Plot
for c_type in ['Hub', 'Spoke']:
    subset = df_analysis[df_analysis['Type'] == c_type]
    plt.scatter(subset['Mapping_Frequency'], 
                subset['Incident_Frequency'], 
                c=colors[c_type], 
                label=c_type, 
                s=150, 
                alpha=0.8, 
                edgecolors='white')

# Annotations
for i, row in df_analysis.iterrows():
    plt.text(row['Mapping_Frequency'], row['Incident_Frequency'] + 0.5, 
             str(i+1), # Using index as ID to avoid clutter, or short name
             fontsize=9, ha='center', fontweight='bold')

# Legend for IDs
print("\n--- Control Legend ---")
for i, row in df_analysis.iterrows():
    print(f"{i+1}: {row['Control_Name']} ({row['Type']})")

plt.title(f'Control Centrality vs. Incident Frequency\n(Correlation r={corr_coef:.2f})')
plt.xlabel('Theoretical Centrality (Governance Mapping Count)')
plt.ylabel('Empirical Failure Rate (Incident Count)')
plt.axvline(x=median_freq, color='gray', linestyle='--', alpha=0.5, label='Median Centrality')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()