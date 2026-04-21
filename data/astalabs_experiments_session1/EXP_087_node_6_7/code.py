import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import sys
import subprocess

# Function to install statsmodels if not present, as mosaic plot is requested
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])

try:
    from statsmodels.graphics.mosaicplot import mosaic
except ImportError:
    install('statsmodels')
    from statsmodels.graphics.mosaicplot import mosaic

# Load dataset
file_path = '../step2_crosswalk_matrix.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {file_path}")
except FileNotFoundError:
    # Fallback for local testing if directory structure differs
    file_path = 'step2_crosswalk_matrix.csv'
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}")
    except FileNotFoundError:
        print("Error: Dataset not found.")
        sys.exit(1)

# Inspect unique sources to ensure correct mapping
print("\nUnique sources in dataset:")
print(df['source'].unique())

# 2. Create 'source_type' variable
# Normative: EU AI Act, NIST AI RMF 1.0
# Technical: OWASP Top 10 LLM, NIST GenAI Profile
def classify_source(source):
    source = str(source).strip()
    if 'EU AI Act' in source or 'NIST AI RMF 1.0' in source:
        return 'Normative'
    elif 'OWASP' in source or 'NIST GenAI Profile' in source:
        return 'Technical'
    else:
        return 'Other'

df['source_type'] = df['source'].apply(classify_source)

# Check for unclassified sources
if 'Other' in df['source_type'].values:
    print("\nWarning: Some sources were classified as 'Other':")
    print(df[df['source_type'] == 'Other']['source'].unique())

# Filter to only Normative and Technical for the test (though 'Other' shouldn't exist based on metadata)
df_analysis = df[df['source_type'] != 'Other'].copy()

# 3. Create Contingency Table
contingency_table = pd.crosstab(df_analysis['source_type'], df_analysis['bundle'])
print("\nContingency Table (Source Type vs Bundle):")
print(contingency_table)

# 4. Chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\nChi-Square Test Results:")
print(f"Chi-square Statistic: {chi2:.4f}")
print(f"P-value: {p:.4f}")
print(f"Degrees of Freedom: {dof}")
print("\nExpected Frequencies:")
print(expected)

# Interpret results
alpha = 0.05
if p < alpha:
    print("\nConclusion: Reject the null hypothesis. There is a statistically significant association between source type and competency bundle.")
else:
    print("\nConclusion: Fail to reject the null hypothesis. No statistically significant association found.")

# 5. Visualization: Mosaic Plot
plt.figure(figsize=(10, 6))
mosaic(df_analysis, ['source_type', 'bundle'], 
       title='Mosaic Plot: Source Type vs Competency Bundle',
       labelizer=lambda k: '',  # Remove internal labels if too cluttered, or keep default
       gap=0.02)
plt.show()
