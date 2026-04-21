import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import sys

# Load dataset
file_name = 'astalabs_discovery_all_data.csv'
# Try checking one level above first as per instructions
file_path = f'../{file_name}'
if not os.path.exists(file_path):
    file_path = file_name
    if not os.path.exists(file_path):
        print(f"Error: Dataset {file_name} not found in ../ or current directory.")
        sys.exit(1)

print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path, low_memory=False)

# Filter for EO 13960 Scored data
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO 13960 Scored rows: {len(df_eo)}")

# --- Step 1: Process Development Method ---
col_dev = '22_dev_method'

# Normalize and inspect
df_eo[col_dev] = df_eo[col_dev].astype(str).fillna('')

print("\n--- Unique values in '22_dev_method' (Top 10) ---")
print(df_eo[col_dev].value_counts().head(10))

def classify_dev_method(val):
    val_lower = val.lower()
    # Priority to Contracted if mixed (assuming external involvement adds the barrier)
    if any(x in val_lower for x in ['contract', 'vendor', 'commercial', 'external', 'private']):
        return 'Contracted'
    elif any(x in val_lower for x in ['agency', 'government', 'in-house', 'federal', 'staff']):
        return 'In-house'
    else:
        return 'Unknown'

df_eo['dev_category'] = df_eo[col_dev].apply(classify_dev_method)

# Filter out Unknowns
df_analysis = df_eo[df_eo['dev_category'] != 'Unknown'].copy()
print(f"\nRows after filtering Dev Method: {len(df_analysis)}")
print(df_analysis['dev_category'].value_counts())

# --- Step 2: Process Code Access ---
col_access = '38_code_access'

# Normalize and inspect
df_analysis[col_access] = df_analysis[col_access].astype(str).fillna('')

print("\n--- Unique values in '38_code_access' (Top 10) ---")
print(df_analysis[col_access].value_counts().head(10))

def classify_access(val):
    val_lower = val.lower()
    # Look for affirmative keywords indicating availability
    if any(x in val_lower for x in ['yes', 'public', 'open', 'available', 'github', 'repo']):
        return 1
    # Treat 'No', 'N/A', 'Restricted', nan as 0
    return 0

df_analysis['is_accessible'] = df_analysis[col_access].apply(classify_access)

# --- Step 3: Statistical Analysis ---
contingency = pd.crosstab(df_analysis['dev_category'], df_analysis['is_accessible'])
print("\n--- Contingency Table (Code Accessibility) ---")
print(contingency)

# Calculate rates
rates = df_analysis.groupby('dev_category')['is_accessible'].mean()
counts = df_analysis['dev_category'].value_counts()
print("\n--- Accessibility Rates ---")
print(rates)

# Chi-square test
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")

# --- Step 4: Visualization ---
plt.figure(figsize=(8, 6))
bar_colors = ['skyblue', 'lightcoral']
ax = rates.plot(kind='bar', color=bar_colors, edgecolor='black')
plt.title('Code/Technical Documentation Accessibility by Development Method')
plt.xlabel('Development Method')
plt.ylabel('Proportion with Accessible Code/Docs')
plt.ylim(0, max(rates.max() * 1.2, 0.1))  # Ensure some headroom

# Add labels
for i, v in enumerate(rates):
    count = counts[rates.index[i]]
    plt.text(i, v + 0.005, f"{v:.1%}\n(n={count})", ha='center', va='bottom')

plt.axhline(0, color='black', linewidth=1)
plt.tight_layout()
plt.show()
