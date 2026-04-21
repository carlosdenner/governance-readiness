import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import os

# 1. Load Dataset
filename = 'astalabs_discovery_all_data.csv'
possible_paths = [filename, f'../{filename}']
file_path = None

for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        break

if file_path is None:
    # Fallback default
    file_path = filename

df = pd.read_csv(file_path, low_memory=False)

# 2. Filter for EO13960
eo_data = df[df['source_table'] == 'eo13960_scored'].copy()
print(f"EO13960 records loaded: {len(eo_data)}")

# 3. Text Processing & Modality Classification
# Combine columns for keyword search
eo_data['combined_text'] = (eo_data['2_use_case_name'].fillna('') + ' ' + eo_data['11_purpose_benefits'].fillna('')).str.lower()

# Define keywords
text_keywords = ['chat', 'language', 'nlp', 'text', 'summariz', 'translat']
vision_keywords = ['vision', 'image', 'video', 'camera', 'detect', 'surveil']

def classify_modality(text):
    if not isinstance(text, str):
        return 'Other'
    has_text = any(k in text for k in text_keywords)
    has_vision = any(k in text for k in vision_keywords)
    
    if has_text and not has_vision:
        return 'Text'
    elif has_vision and not has_text:
        return 'Vision'
    elif has_text and has_vision:
        return 'Mixed'
    else:
        return 'Other'

eo_data['modality'] = eo_data['combined_text'].apply(classify_modality)

# 4. Filter for specific modalities
analysis_df = eo_data[eo_data['modality'].isin(['Text', 'Vision'])].copy()
print(f"\nModality Counts:\n{analysis_df['modality'].value_counts()}")

# 5. Target Variable Creation
# Check raw values first
print(f"\nUnique values in '52_impact_assessment': {analysis_df['52_impact_assessment'].unique()}")

# Map to Boolean: True if contains 'yes', else False
analysis_df['has_assessment'] = analysis_df['52_impact_assessment'].astype(str).str.lower().str.contains('yes')

# 6. Generate Contingency Table
# Explicitly ensure both rows (Text, Vision) and columns (False, True) exist
# We use pd.Categorical to enforce structure even if counts are zero
analysis_df['modality'] = pd.Categorical(analysis_df['modality'], categories=['Text', 'Vision'])
analysis_df['has_assessment'] = pd.Categorical(analysis_df['has_assessment'], categories=[False, True])

contingency_table = pd.crosstab(analysis_df['modality'], analysis_df['has_assessment'], dropna=False)

print("\nContingency Table (Modality x Has Assessment):")
print(contingency_table)

# 7. Statistical Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"\nChi-Square Results:")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value: {p:.4e}")
print(f"Degrees of Freedom: {dof}")

# 8. Rate Calculation & Conclusion
rates = analysis_df.groupby('modality', observed=False)['has_assessment'].apply(lambda x: (x == True).mean())
text_rate = rates['Text']
vision_rate = rates['Vision']

print(f"\nAssessment Rates:")
print(f"Text:   {text_rate:.2%}")
print(f"Vision: {vision_rate:.2%}")

alpha = 0.05
if p < alpha:
    print("\nConclusion: Statistically Significant Difference.")
    if text_rate < vision_rate:
        print("Hypothesis Supported: Text systems have significantly lower assessment rates than Vision systems.")
    else:
        print("Hypothesis Rejected (Direction): Text systems have higher assessment rates.")
else:
    print("\nConclusion: No Statistically Significant Difference found.")
