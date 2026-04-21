import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'astalabs_discovery_all_data.csv'
# Use low_memory=False to avoid DtypeWarning, or specify dtype if known. 
# Given the sparse nature, low_memory=False is safer for a quick script.
df = pd.read_csv(file_path, low_memory=False)

# Filter for EO 13960 scored data
eo_df = df[df['source_table'] == 'eo13960_scored'].copy()

print(f"Total EO 13960 records: {len(eo_df)}")

# --- Step 1: Parse Dates ---
# Column: '18_date_initiated'
# Attempt to convert to datetime. This handles various formats.
eo_df['date_parsed'] = pd.to_datetime(eo_df['18_date_initiated'], errors='coerce')

# Extract year
eo_df['initiation_year'] = eo_df['date_parsed'].dt.year

# Filter out rows where date could not be parsed
valid_date_df = eo_df.dropna(subset=['initiation_year']).copy()
print(f"Records with valid dates: {len(valid_date_df)} ({(len(valid_date_df)/len(eo_df))*100:.1f}%)")

# Define Legacy vs Modern
# Legacy: < 2020
# Modern: >= 2020
valid_date_df['is_legacy'] = valid_date_df['initiation_year'] < 2020
valid_date_df['cohort'] = valid_date_df['is_legacy'].map({True: 'Legacy (<2020)', False: 'Modern (2020+)'})

# --- Step 2: Analyze Impact Assessment ---
# Column: '52_impact_assessment'
# Check unique values to determine binary mapping
print("\nUnique values in '52_impact_assessment':")
print(valid_date_df['52_impact_assessment'].value_counts(dropna=False))

# Binarize: 'Yes' vs Others
# We define 'Has Assessment' as explicitly 'Yes'. 
# 'No', 'N/A', and specific reasons for No are treated as 'No Assessment'.
valid_date_df['has_assessment'] = valid_date_df['52_impact_assessment'].astype(str).str.strip().str.lower() == 'yes'

# --- Step 3: Statistical Analysis ---

# Group by Cohort
summary = valid_date_df.groupby('cohort')['has_assessment'].agg(['count', 'sum', 'mean'])
summary.columns = ['Total Systems', 'With Assessment', 'Assessment Rate']
summary['Assessment Rate %'] = (summary['Assessment Rate'] * 100).round(2)

print("\n--- Governance Gap Analysis: Impact Assessments ---")
print(summary[['Total Systems', 'With Assessment', 'Assessment Rate %']])

# Contingency Table for Chi-Square
contingency_table = pd.crosstab(valid_date_df['cohort'], valid_date_df['has_assessment'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print(f"\nChi-Square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

alpha = 0.05
if p < alpha:
    print("Result: Statistically Significant difference in governance rates.")
else:
    print("Result: No statistically significant difference found.")

# --- Step 4: Visualization ---
plt.figure(figsize=(8, 6))
colors = ['#ff9999', '#66b3ff']
ax = summary['Assessment Rate %'].plot(kind='bar', color=colors, edgecolor='black')
plt.title('Impact Assessment Rate: Legacy vs. Modern AI Systems')
plt.ylabel('Percentage with Impact Assessment (%)')
plt.xlabel('Initiation Cohort')
plt.ylim(0, 100)

# Add value labels
for i, v in enumerate(summary['Assessment Rate %']):
    ax.text(i, v + 2, f"{v}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()