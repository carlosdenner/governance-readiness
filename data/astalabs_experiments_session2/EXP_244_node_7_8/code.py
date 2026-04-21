import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for the specific source table
subset = df[df['source_table'] == 'step3_incident_coding'].copy()

# Drop rows with NaN in critical columns
subset = subset.dropna(subset=['tactics_used', 'competency_domains'])

# --- 1. Map Tactics (using MITRE ATLAS IDs) ---
# AML.TA0006: Defense Evasion
# AML.TA0010: Exfiltration
subset['has_evasion_tactic'] = subset['tactics_used'].astype(str).str.contains('AML.TA0006')
subset['has_exfiltration_tactic'] = subset['tactics_used'].astype(str).str.contains('AML.TA0010')

# --- 2. Map Competency Gaps (using text keywords in competency_domains) ---
# Robustness Gap: Looking for 'Robustness' or 'Evasion Detection'
subset['has_robustness_gap'] = subset['competency_domains'].astype(str).str.contains('Robustness', case=False) | \
                               subset['competency_domains'].astype(str).str.contains('Evasion Detection', case=False)

# Access/Logging Gap: Looking for 'Access', 'Logging', 'Audit'
subset['has_access_gap'] = subset['competency_domains'].astype(str).str.contains('Access', case=False) | \
                           subset['competency_domains'].astype(str).str.contains('Logging', case=False) | \
                           subset['competency_domains'].astype(str).str.contains('Audit', case=False)

# --- 3. Filter for relevant rows ---
# We only care about rows that have (Evasion OR Exfiltration) AND (RobustnessGap OR AccessGap)
relevant = subset[
    (subset['has_evasion_tactic'] | subset['has_exfiltration_tactic']) &
    (subset['has_robustness_gap'] | subset['has_access_gap'])
].copy()

print(f"Total rows in subset: {len(subset)}")
print(f"Relevant rows for hypothesis: {len(relevant)}")

# --- 4. Construct Contingency Table ---
# We prioritize categorization. If a row has both, it counts for both in a generalized sense, 
# but for Fisher's test we need a 2x2 matrix of counts. 
# We will count "Incidents with Evasion Tactic" vs "Incidents with Exfiltration Tactic"
# against "Incidents with Robustness Gap" vs "Incidents with Access Gap".
# Note: A single incident could theoretically be in multiple cells if it has multiple tactics/gaps.
# To strictly test the hypothesis "Evasion -> Robustness" vs "Exfil -> Access", we can define the events:
# A: Tactic is Evasion (and not Exfil)
# B: Tactic is Exfil (and not Evasion)
# Outcome 1: Gap is Robustness
# Outcome 2: Gap is Access

# Let's filter for the disjoint sets of Tactics to make the groups independent
evasion_group = relevant[relevant['has_evasion_tactic'] & ~relevant['has_exfiltration_tactic']]
exfil_group = relevant[relevant['has_exfiltration_tactic'] & ~relevant['has_evasion_tactic']]

# Counts
# Group 1: Evasion Tactic
# We check how many have Robustness Gap vs Access Gap
# (Note: An incident can have both gaps, but usually we test the "primary" association.
# If we treat it as binary features, we can just sum them up, but Fisher expects a contingency table of mutually exclusive outcomes usually.
# However, we can test: "Given Tactic X, is Gap A more likely than Gap B?")

# Let's count occurrences. 
# Evasion Tactic -> Robustness Gap
count_evasion_robustness = evasion_group['has_robustness_gap'].sum()
# Evasion Tactic -> Access Gap
count_evasion_access = evasion_group['has_access_gap'].sum()

# Exfiltration Tactic -> Robustness Gap
count_exfil_robustness = exfil_group['has_robustness_gap'].sum()
# Exfiltration Tactic -> Access Gap
count_exfil_access = exfil_group['has_access_gap'].sum()

contingency_table = [
    [count_evasion_robustness, count_evasion_access],
    [count_exfil_robustness, count_exfil_access]
]

print("\nContingency Table (Tactic -> Gap Presence):")
print(f"                  Robustness Gap | Access/Logging Gap")
print(f"Evasion Only      {count_evasion_robustness:<14} | {count_evasion_access:<18}")
print(f"Exfiltration Only {count_exfil_robustness:<14} | {count_exfil_access:<18}")

# Fisher's Exact Test
if (count_evasion_robustness + count_evasion_access + count_exfil_robustness + count_exfil_access) == 0:
    print("Insufficient data for statistical test.")
else:
    oddsratio, pvalue = stats.fisher_exact(contingency_table)
    print(f"\nFisher's Exact Test p-value: {pvalue:.5f}")
    print(f"Odds Ratio: {oddsratio:.5f}")
    
    # Visualization
    labels = ['Evasion Tactic', 'Exfiltration Tactic']
    r_counts = [count_evasion_robustness, count_exfil_robustness]
    a_counts = [count_evasion_access, count_exfil_access]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, r_counts, width, label='Robustness Gap')
    rects2 = ax.bar(x + width/2, a_counts, width, label='Access/Logging Gap')

    ax.set_ylabel('Count of Incidents')
    ax.set_title('Association: Tactic Type vs Governance Gap')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()
