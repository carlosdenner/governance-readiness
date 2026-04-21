import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Define file path
file_path = "../astalabs_discovery_all_data.csv"
if not os.path.exists(file_path):
    file_path = "astalabs_discovery_all_data.csv"

print(f"Loading dataset from {file_path}...")

try:
    df = pd.read_csv(file_path, low_memory=False)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit(1)

# Filter for EO 13960 Scored data
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored subset shape: {eo_data.shape}")

# Columns
col_public = '26_public_service'
col_notice = '59_ai_notice'

# Robust normalization function
def normalize_text(val):
    if pd.isna(val):
        return ""
    return str(val).lower().strip()

eo_data[col_public] = eo_data[col_public].apply(normalize_text)
eo_data[col_notice] = eo_data[col_notice].apply(normalize_text)

# --- logic to classify Public Service ---
def classify_public(val):
    if val == "" or val == 'nan':
        return None
    if val == 'no':
        return 'Internal/Other'
    # Any other non-empty string implies a description of a public service
    return 'Public Service'

# --- logic to classify AI Notice ---
def classify_notice(val):
    if val == "" or val == 'nan':
        return None
    
    negatives = [
        'none of the above',
        'n/a - individuals are not interacting with the ai for this use case',
        'ai is not safety or rights-impacting.',
        'agency caio has waived this minimum practice and reported such waiver to omb.'
    ]
    
    if val in negatives:
        return 'No Notice'
    
    # Check for keywords indicating presence of notice
    positives_keywords = ['online', 'in-person', 'email', 'telephone', 'other', 'terms']
    if any(keyword in val for keyword in positives_keywords):
        return 'Has Notice'
    
    # Fallback for anything else that isn't explicitly negative
    return 'No Notice'

# Apply classification
eo_data['service_type'] = eo_data[col_public].apply(classify_public)
eo_data['notice_status'] = eo_data[col_notice].apply(classify_notice)

# Filter out rows where we couldn't classify one or the other
analysis_df = eo_data.dropna(subset=['service_type', 'notice_status']).copy()

print(f"Data shape after classification and filtering: {analysis_df.shape}")
print("Service Type Counts:\n", analysis_df['service_type'].value_counts())
print("Notice Status Counts:\n", analysis_df['notice_status'].value_counts())

# Create contingency table
contingency_table = pd.crosstab(analysis_df['service_type'], analysis_df['notice_status'])
print("\nContingency Table:")
print(contingency_table)

# Calculate percentages
public_rate = 0.0
internal_rate = 0.0
public_n = 0
internal_n = 0
public_has_notice = 0
internal_has_notice = 0

if 'Public Service' in contingency_table.index:
    public_n = contingency_table.loc['Public Service'].sum()
    if 'Has Notice' in contingency_table.columns:
        public_has_notice = contingency_table.loc['Public Service', 'Has Notice']
        if public_n > 0:
            public_rate = (public_has_notice / public_n) * 100

if 'Internal/Other' in contingency_table.index:
    internal_n = contingency_table.loc['Internal/Other'].sum()
    if 'Has Notice' in contingency_table.columns:
        internal_has_notice = contingency_table.loc['Internal/Other', 'Has Notice']
        if internal_n > 0:
            internal_rate = (internal_has_notice / internal_n) * 100

print(f"\nPublic Service Notice Rate: {public_rate:.2f}% ({public_has_notice}/{public_n})")
print(f"Internal/Other Notice Rate: {internal_rate:.2f}% ({internal_has_notice}/{internal_n})")

# Perform Chi-Square Test
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-Square Test Results:")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

alpha = 0.05
if p < alpha:
    print("Conclusion: REJECT Null Hypothesis. There is a significant difference in AI Notice rates.")
else:
    print("Conclusion: FAIL TO REJECT Null Hypothesis. No significant difference detected.")

# Visualization
plt.figure(figsize=(8, 6))
labels = ['Public Service', 'Internal/Other']
rates = [public_rate, internal_rate]
colors = ['#1f77b4', '#7f7f7f']

bars = plt.bar(labels, rates, color=colors)
plt.ylabel('Percentage with AI Notice (%)')
plt.title('Transparency Gap: AI Notice Implementation\nPublic Service vs. Internal Systems')
plt.ylim(0, max(rates) * 1.2 if max(rates) > 0 else 10)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, 
             f'{height:.1f}%',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()
