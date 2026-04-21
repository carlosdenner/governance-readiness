import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# 1. Load Dataset
filename = 'astalabs_discovery_all_data.csv'
file_path = filename if os.path.exists(filename) else os.path.join('..', filename)

if not os.path.exists(file_path):
    print("Dataset not found.")
    exit(1)

print(f"Loading dataset from {file_path}...")
df = pd.read_csv(file_path, low_memory=False)

# 2. Filter for aiid_incidents
df_incidents = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents subset shape: {df_incidents.shape}")

# 3. Identify Columns
tech_col = '84_Known AI Technology' if '84_Known AI Technology' in df_incidents.columns else 'Known AI Technology'
tangible_harm_col = '74_Tangible Harm' if '74_Tangible Harm' in df_incidents.columns else 'Tangible Harm'
bias_col = '76_Harm Distribution Basis' if '76_Harm Distribution Basis' in df_incidents.columns else 'Harm Distribution Basis'
intangible_col = '77_Special Interest Intangible Harm' if '77_Special Interest Intangible Harm' in df_incidents.columns else 'Special Interest Intangible Harm'

print(f"Columns: Tech='{tech_col}', Tangible='{tangible_harm_col}', Bias='{bias_col}'")

# 4. Define Expanded Mapping Logic
def get_tech_category(val):
    if pd.isna(val): return None
    s = str(val).lower()
    # NLP/Text (Expanded based on 'Transformer' seen in data)
    if any(x in s for x in ['transformer', 'language', 'text', 'nlp', 'translation', 'chatbot', 'speech', 'llm', 'generative', 'gpt', 'bert', 'dialogue', 'word', 'sentiment']):
        return 'NLP/Text'
    # Robotics/Physical/Vision (Expanded based on 'Visual', 'Face' seen in data)
    if any(x in s for x in ['robot', 'autonomous', 'drone', 'vehicle', 'car', 'driving', 'physical', 'vision', 'visual', 'image', 'face', 'camera', 'detection', 'segmentation', 'convolutional']):
        return 'Robotics/Vision'
    return None

def get_harm_category(row):
    # Check for Bias/Civil Rights first
    bias_val = str(row.get(bias_col, '')).lower()
    intangible_val = str(row.get(intangible_col, '')).lower()
    
    is_bias = False
    # If distribution basis is specific (not none/unclear/nan)
    if bias_val not in ['nan', 'none', 'unclear', '']:
        is_bias = True
    # If special interest intangible harm is 'yes'
    if intangible_val == 'yes':
        is_bias = True
        
    if is_bias:
        return 'Fairness/Bias'

    # Check for Safety/Tangible Harm
    tangible_val = str(row.get(tangible_harm_col, '')).lower()
    if 'tangible harm definitively occurred' in tangible_val or 'risk of tangible harm' in tangible_val:
        return 'Physical Safety'
        
    return None

# 5. Apply Mapping
df_incidents['Tech_Category'] = df_incidents[tech_col].apply(get_tech_category)
df_incidents['Harm_Category'] = df_incidents.apply(get_harm_category, axis=1)

# 6. Filter and Analyze
df_analysis = df_incidents.dropna(subset=['Tech_Category', 'Harm_Category'])
print(f"\nRows suitable for analysis: {len(df_analysis)}")
print("Category Counts:")
print(df_analysis['Tech_Category'].value_counts())
print(df_analysis['Harm_Category'].value_counts())

if len(df_analysis) > 5:
    # Contingency Table
    ct = pd.crosstab(df_analysis['Tech_Category'], df_analysis['Harm_Category'])
    print("\nContingency Table:")
    print(ct)

    # Stats
    chi2, p, dof, expected = chi2_contingency(ct)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")

    residuals = (ct - expected) / np.sqrt(expected)
    print("\nStandardized Residuals:")
    print(residuals)

    # Hypothesis Evaluation
    print("\n--- Hypothesis Evaluation ---")
    # Hypothesis: NLP -> Bias, Robotics -> Safety
    # Check NLP -> Bias
    try:
        nlp_bias_resid = residuals.loc['NLP/Text', 'Fairness/Bias']
        print(f"NLP/Text association with Fairness/Bias: Residual = {nlp_bias_resid:.2f} (Expected > 1.96)")
    except KeyError:
        print("NLP/Text or Fairness/Bias missing from table.")

    # Check Robotics -> Safety
    try:
        robot_safety_resid = residuals.loc['Robotics/Vision', 'Physical Safety']
        print(f"Robotics/Vision association with Physical Safety: Residual = {robot_safety_resid:.2f} (Expected > 1.96)")
    except KeyError:
        print("Robotics/Vision or Physical Safety missing from table.")
        
    # Visualization
    plt.figure(figsize=(10, 6))
    sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Contingency Table: AI Technology vs Harm Type')
    plt.ylabel('Technology')
    plt.xlabel('Harm Domain')
    plt.tight_layout()
    plt.show()
else:
    print("Insufficient data.")
    print("Debug - Tech Column Sample:", df_incidents[tech_col].dropna().head().values)
    print("Debug - Bias Column Sample:", df_incidents[bias_col].dropna().head().values)
    print("Debug - Tangible Column Sample:", df_incidents[tangible_harm_col].dropna().head().values)