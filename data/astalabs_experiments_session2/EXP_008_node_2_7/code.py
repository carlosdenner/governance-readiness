import pandas as pd
import numpy as np
import sys
import subprocess
from scipy.stats import spearmanr

# Try importing statsmodels, install if missing
try:
    from statsmodels.miscmodels.ordinal_model import OrderedModel
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "statsmodels"])
    from statsmodels.miscmodels.ordinal_model import OrderedModel

def run_experiment():
    # 1. Load Data
    file_path = 'astalabs_discovery_all_data.csv'
    print(f"Loading dataset from {file_path}...")
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        # Fallback for different directory structure if needed
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

    # 2. Filter for AIID Incidents
    df_aiid = df[df['source_table'] == 'aiid_incidents'].copy()
    print(f"AIID Incidents subset size: {len(df_aiid)}")

    # 3. Data Cleaning and Mapping
    col_autonomy = 'Autonomy Level'
    col_harm = 'AI Harm Level'
    col_sector = 'Sector of Deployment'

    # Check unique values for debugging if needed
    print(f"Unique Autonomy Values: {df_aiid[col_autonomy].unique()}")
    print(f"Unique Harm Values: {df_aiid[col_harm].unique()}")

    # Mapping Logic
    # Autonomy: Autonomy1 (Low), Autonomy2 (Medium), Autonomy3 (High)
    def map_autonomy(val):
        val = str(val).strip()
        if val == 'Autonomy1': return 1
        if val == 'Autonomy2': return 2
        if val == 'Autonomy3': return 3
        return np.nan

    # Harm: none (0), near-miss (1), issue (2), event (3)
    def map_harm(val):
        val = str(val).strip()
        if val == 'none': return 0
        if val == 'AI tangible harm near-miss': return 1
        if val == 'AI tangible harm issue': return 2
        if val == 'AI tangible harm event': return 3
        return np.nan

    df_aiid['autonomy_ord'] = df_aiid[col_autonomy].apply(map_autonomy)
    df_aiid['harm_ord'] = df_aiid[col_harm].apply(map_harm)

    # Clean Sector
    df_aiid['sector_clean'] = df_aiid[col_sector].fillna('Unknown')

    # Drop rows with NaN in key ordinal columns
    df_clean = df_aiid.dropna(subset=['autonomy_ord', 'harm_ord']).copy()
    print(f"Data points after cleaning mappings: {len(df_clean)}")

    if len(df_clean) < 10:
        print("Insufficient data points for regression.")
        return

    # 4. Analysis
    
    # Spearman Correlation
    corr, p = spearmanr(df_clean['autonomy_ord'], df_clean['harm_ord'])
    print(f"\n--- Spearman Correlation ---")
    print(f"Correlation: {corr:.4f}, p-value: {p:.4f}")

    # Ordinal Logistic Regression
    print(f"\n--- Ordinal Logistic Regression ---")
    
    # Prepare Sector Dummies (Top 5 sectors, others grouped)
    top_sectors = df_clean['sector_clean'].value_counts().nlargest(5).index
    df_clean['sector_group'] = df_clean['sector_clean'].apply(lambda x: x if x in top_sectors else 'Other')
    
    print(f"Using sectors: {list(top_sectors)} and 'Other'")

    # Independent variables: Autonomy (treated as continuous/ordinal trend) + Sector dummies
    # We convert autonomy to numeric to see the trend effect.
    # Using drop_first=True to avoid multicollinearity
    exog = pd.get_dummies(df_clean[['autonomy_ord', 'sector_group']], columns=['sector_group'], drop_first=True, dtype=float)
    
    # Dependent variable
    endog = df_clean['harm_ord'].astype(int)

    try:
        # Fit Ordered Logit Model
        # distribution 'logit' is standard for ordered logistic regression
        model = OrderedModel(endog, exog, distr='logit')
        res = model.fit(method='bfgs', disp=False)
        
        print(res.summary())
        
        print("\n--- Odds Ratios (Effect Size) ---")
        params = res.params
        conf = res.conf_int()
        conf['OR'] = params
        conf.columns = ['2.5%', '97.5%', 'OR']
        print(np.exp(conf))

    except Exception as e:
        print(f"Regression failed: {e}")

if __name__ == "__main__":
    run_experiment()