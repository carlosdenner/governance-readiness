import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import numpy as np
import os

# Attempt to locate the file, prioritizing current directory then parent
filename = 'step3_incident_coding.csv'
if os.path.exists(filename):
    file_path = filename
elif os.path.exists(os.path.join('..', filename)):
    file_path = os.path.join('..', filename)
else:
    # Fallback to absolute path check or error
    file_path = filename

print(f"Loading dataset from: {file_path}")
df = pd.read_csv(file_path)

# Function to categorize actors
def categorize_actor(actor_raw):
    if pd.isna(actor_raw):
        return 'Unknown'
    actor = str(actor_raw).lower()
    
    # Keywords for Internal / Accidental / Research
    # In ATLAS, 'Researcher' is often treated as a proxy for internal/authorized access in proof-of-concepts
    # 'User' implies authorized user.
    internal_keywords = ['insider', 'researcher', 'user', 'employee', 'developer', 'accidental', 'academic', 'student']
    
    for keyword in internal_keywords:
        if keyword in actor:
            return 'Internal/Accidental'
            
    return 'External'

# Apply categorization
df['actor_category'] = df['actor'].apply(categorize_actor)

# Remove 'Unknown' if any, though the logic defaults to External. 
# Let's see if there are empty rows.
df = df[df['actor'].notna()]

# Generate Contingency Table
# Rows: Actor Type, Cols: Competency Split
contingency_table = pd.crosstab(df['actor_category'], df['trust_integration_split'])

print("=== Actor Categorization Samples ===")
print(df[['actor', 'actor_category']].drop_duplicates().head(10))

print("\n=== Contingency Table (Actor vs Competency Split) ===")
print(contingency_table)

# Perform Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\n=== Chi-Square Test Results ===")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"p-value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")

# Calculate Cramer's V
n = contingency_table.sum().sum()
min_dim = min(contingency_table.shape) - 1
cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
print(f"Cramer's V: {cramers_v:.4f}")

# Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues')
plt.title('Threat Actor vs. Competency Gap (Trust/Integration)')
plt.ylabel('Actor Category')
plt.xlabel('Competency Gap Type')
plt.show()
