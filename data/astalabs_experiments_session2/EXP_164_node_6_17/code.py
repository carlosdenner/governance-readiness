import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

print("Starting Threat-Reality Mismatch Experiment...")

# 1. Load Dataset
df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# 2. Prepare AIID (Real-world Incidents)
aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()

# Identify 'Sector' column - heuristic from debug
sector_cols = [c for c in aiid_df.columns if 'Sector of Deployment' in c]
if not sector_cols:
    print("Error: Could not find AIID Sector column.")
    exit(1)
sector_col = sector_cols[0]

# Filter for non-null sectors
aiid_df = aiid_df[aiid_df[sector_col].notna()].copy()
print(f"AIID Records with Sector info: {len(aiid_df)}")

# 3. Prepare ATLAS (Adversarial Cases)
# Using 'step3_incident_coding' as it contains 'target' and 'summary' for better context
atlas_df = df[df['source_table'] == 'step3_incident_coding'].copy()
print(f"ATLAS Records: {len(atlas_df)}")

# 4. Define Categorization Logic
# We define 5 broad categories to allow meaningful statistical comparison

def categorize_sector(text):
    if not isinstance(text, str):
        return 'Other'
    text = text.lower()
    
    # Security / Defense / Cybersecurity
    if any(x in text for x in ['defense', 'military', 'police', 'security', 'surveillance', 'malware', 'virus', 'intrusion', 'cyber', 'attack', 'weapon', 'facial recognition', 'biometric', 'cctv', 'public safety']):
        return 'Security/Defense'
    
    # Transportation / Physical Safety
    if any(x in text for x in ['transport', 'vehicle', 'car', 'automotive', 'driving', 'autopilot', 'tesla', 'aviation', 'drone', 'robot']):
        return 'Transportation/Robotics'
    
    # Healthcare
    if any(x in text for x in ['health', 'medic', 'hospital', 'patient', 'diagnosis', 'cancer', 'disease']):
        return 'Healthcare'
    
    # Finance
    if any(x in text for x in ['financ', 'bank', 'trading', 'market', 'stock', 'credit', 'fraud', 'money']):
        return 'Finance'
        
    # Info / Tech / Consumer
    if any(x in text for x in ['info', 'communicat', 'tech', 'software', 'app', 'media', 'entertainment', 'social', 'content', 'chatbot', 'translation', 'recommend', 'search', 'email', 'spam', 'language model']):
        return 'Info/Tech/Content'
        
    return 'Other'

# Apply categorization
aiid_df['Clean_Sector'] = aiid_df[sector_col].apply(categorize_sector)

# For ATLAS, we concat fields to text context
atlas_df['context'] = atlas_df['name'].fillna('') + " " + atlas_df['summary'].fillna('') + " " + atlas_df.get('target', pd.Series(['']*len(atlas_df))).fillna('')
atlas_df['Clean_Sector'] = atlas_df['context'].apply(categorize_sector)

# 5. Calculate Distributions
aiid_counts = aiid_df['Clean_Sector'].value_counts()
atlas_counts = atlas_df['Clean_Sector'].value_counts()

# Merge into a comparison dataframe
categories = ['Security/Defense', 'Transportation/Robotics', 'Healthcare', 'Finance', 'Info/Tech/Content', 'Other']
comp_df = pd.DataFrame(index=categories)
comp_df['AIID_Count'] = aiid_counts
comp_df['ATLAS_Count'] = atlas_counts
comp_df = comp_df.fillna(0)

# 6. Statistical Test (Chi-Square Goodness of Fit)
# Hypothesis: ATLAS (Observed) follows the distribution of AIID (Expected)

# Calculate Proportions from AIID (Population Baseline)
aiid_total = comp_df['AIID_Count'].sum()
comp_df['AIID_Prop'] = comp_df['AIID_Count'] / aiid_total

# Calculate Expected ATLAS counts based on AIID proportions
atlas_total = comp_df['ATLAS_Count'].sum()
comp_df['Expected_ATLAS'] = comp_df['AIID_Prop'] * atlas_total

# Print Table
print("\n--- Sector Distribution Comparison ---")
print(comp_df[['AIID_Count', 'ATLAS_Count', 'Expected_ATLAS']].round(1))

# Check assumptions: frequencies > 5? 
# If not, we might need to aggregate, but for this experiment we show the raw result
obs = comp_df['ATLAS_Count']
exp = comp_df['Expected_ATLAS']

# Add small epsilon to exp to avoid div by zero if any category is 0 in AIID
exp = exp + 1e-9 

chi2_stat, p_val = stats.chisquare(f_obs=obs, f_exp=exp)

print(f"\nChi-Square Statistic: {chi2_stat:.4f}")
print(f"P-Value: {p_val:.4e}")

result_text = "REJECT" if p_val < 0.05 else "FAIL TO REJECT"
print(f"Result: We {result_text} the null hypothesis that ATLAS follows the same sector distribution as AIID.")

# 7. Visualization
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(categories))
width = 0.35

# Plot percentages for better visual comparison
aiid_pct = comp_df['AIID_Count'] / aiid_total * 100
atlas_pct = comp_df['ATLAS_Count'] / atlas_total * 100

rects1 = ax.bar(x - width/2, aiid_pct, width, label='Real-World (AIID)', color='skyblue')
rects2 = ax.bar(x + width/2, atlas_pct, width, label='Adversarial Research (ATLAS)', color='salmon')

ax.set_ylabel('Percentage of Cases')
ax.set_title('Threat-Reality Mismatch: Sector Distribution')
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.show()
