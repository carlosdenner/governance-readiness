import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    # Fallback if running in same dir
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

print("Dataset loaded. Filtering for EO13960...")

# Filter for EO13960
df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO13960 rows: {len(df_eo)}")

# Define columns of interest
# 38_code_access -> Transparency
# 65_appeal_process -> Accountability
# 8_topic_area -> Control Variable
raw_cols = {
    '38_code_access': 'code_access_raw',
    '65_appeal_process': 'appeal_process_raw',
    '8_topic_area': 'topic_area_raw'
}

df_analysis = df_eo[list(raw_cols.keys())].rename(columns=raw_cols).copy()

# --- Data Cleaning & Inspection ---

# Function to binarize Yes/No/Other text fields
def clean_binary(val):
    if pd.isna(val):
        return 0
    s = str(val).strip().lower()
    # Check for affirmative starts
    if s.startswith('yes'):
        return 1
    return 0

# Inspect raw values first
print("\n--- Raw Value Inspection ---")
print("Code Access sample values:", df_analysis['code_access_raw'].unique()[:10])
print("Appeal Process sample values:", df_analysis['appeal_process_raw'].unique()[:10])

# Apply cleaning
df_analysis['code_access'] = df_analysis['code_access_raw'].apply(clean_binary)
df_analysis['appeal_process'] = df_analysis['appeal_process_raw'].apply(clean_binary)

# Clean Topic Area (Keep Top N, label others as 'Other')
df_analysis['topic_area_raw'] = df_analysis['topic_area_raw'].fillna('Unknown')
top_n = 5
top_topics = df_analysis['topic_area_raw'].value_counts().nlargest(top_n).index.tolist()
df_analysis['topic'] = df_analysis['topic_area_raw'].apply(lambda x: x if x in top_topics else 'Other')

# --- Descriptive Statistics ---

print("\n--- Descriptive Statistics ---")
print("Code Access Distribution:\n", df_analysis['code_access'].value_counts())
print("Appeal Process Distribution:\n", df_analysis['appeal_process'].value_counts())

ct = pd.crosstab(df_analysis['code_access'], df_analysis['appeal_process'], normalize='index')
print("\nContingency Table (Row Normalized - Probability of Appeal given Code Access):")
print(ct)

# --- Logistic Regression ---
# Model: appeal_process ~ code_access + topic

print("\n--- Logistic Regression Analysis ---")
formula = 'appeal_process ~ code_access + C(topic)'

try:
    model = smf.logit(formula, data=df_analysis).fit(disp=0)
    print(model.summary())
    
    print("\n--- Odds Ratios (Exp(Coef)) ---")
    params = model.params
    conf = model.conf_int()
    conf['OR'] = params
    conf.columns = ['2.5%', '97.5%', 'OR']
    odds_ratios = np.exp(conf)
    print(odds_ratios)
    
    or_val = np.exp(model.params['code_access'])
    print(f"\nKey Finding: Odds Ratio for Code Access = {or_val:.4f}")
    
except Exception as e:
    print(f"Regression failed: {e}")

# --- Visualization ---
plt.figure(figsize=(10, 6))
# Group by Code Access and calculate mean of Appeal Process (proportion)
summary_stats = df_analysis.groupby('code_access')['appeal_process'].mean()
ax = summary_stats.plot(kind='bar', color=['#d9534f', '#5bc0de'], alpha=0.8)

plt.title('Link Between Transparency (Code Access) and Accountability (Appeal Process)')
plt.xlabel('Has Code Access?')
plt.ylabel('Proportion with Appeal Process')
plt.xticks([0, 1], ['No (Black Box)', 'Yes (Transparent)'], rotation=0)
plt.ylim(0, 1)

# Add value labels
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1%}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.tight_layout()
plt.show()
