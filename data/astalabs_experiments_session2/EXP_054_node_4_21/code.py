import pandas as pd
import scipy.stats as stats
import sys

# [debug] Print library versions if needed
# print(f"Pandas version: {pd.__version__}")

try:
    # Load dataset
    file_path = 'astalabs_discovery_all_data.csv'
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)

    # Filter for EO 13960 data
    df_eo = df[df['source_table'] == 'eo13960_scored'].copy()
    print(f"EO 13960 records loaded: {len(df_eo)}")

    # Columns
    col_stage = '16_dev_stage'
    col_testing = '53_real_world_testing'

    # --- CLEANING STAGE ---
    # Ensure string type and handle NaNs
    df_eo[col_stage] = df_eo[col_stage].fillna('nan').astype(str).str.lower().str.strip()

    # Logic for is_operational (Independent Variable)
    def map_stage(val):
        # Exclude retired or missing
        if 'retire' in val or 'nan' == val or val == '':
            return None
        
        # Operational keywords
        # 'operation', 'production', 'implementation', 'use', 'maintain'
        if any(x in val for x in ['oper', 'prod', 'impl', 'use', 'maintain', 'sustain']):
            return 'Operational'
        
        # Development keywords
        # 'development', 'initiated', 'planned', 'acquisition'
        elif any(x in val for x in ['dev', 'init', 'plan', 'acq', 'test', 'pilot', 'experiment']):
            return 'Development'
        
        return None # Unclassified

    df_eo['stage_group'] = df_eo[col_stage].apply(map_stage)
    
    # Drop unclassified rows
    df_clean = df_eo.dropna(subset=['stage_group']).copy()
    print(f"\nRecords after stage classification: {len(df_clean)}")
    print("Stage Distribution:")
    print(df_clean['stage_group'].value_counts())

    # --- CLEANING TESTING ---
    # Ensure string type
    df_clean[col_testing] = df_clean[col_testing].fillna('no').astype(str).str.lower().str.strip()

    # Logic for has_testing (Dependent Variable)
    # We look for explicit mentions of 'operational environment' or 'yes'.
    # 'benchmark' explicitly states 'has not been tested in an operational environment', so it is 0.
    def map_testing(val):
        if 'performance evaluation' in val: return 1
        if 'impact evaluation' in val: return 1
        if val == 'yes': return 1
        return 0

    df_clean['has_testing'] = df_clean[col_testing].apply(map_testing)
    
    print("\nTesting Logic Check (sample of raw vs mapped):")
    print(df_clean[[col_testing, 'has_testing']].drop_duplicates().head(10))

    # --- ANALYSIS ---

    # Contingency Table
    contingency_table = pd.crosstab(df_clean['stage_group'], df_clean['has_testing'])
    # Check if we have both 0 and 1 columns. If not, reindex to ensure shape.
    if 0 not in contingency_table.columns: contingency_table[0] = 0
    if 1 not in contingency_table.columns: contingency_table[1] = 0
    contingency_table = contingency_table[[0, 1]]
    contingency_table.columns = ['No Real-World Testing', 'Has Real-World Testing']
    
    print("\nContingency Table (Real World Testing by Stage):")
    print(contingency_table)

    # Calculate Percentages
    summary = df_clean.groupby('stage_group')['has_testing'].agg(['count', 'sum', 'mean'])
    summary['pct_compliant'] = summary['mean'] * 100
    print("\nCompliance Rates (% with Real World Testing):")
    print(summary[['count', 'sum', 'pct_compliant']])

    # Statistical Test
    # Chi-square test of independence
    if contingency_table.sum().sum() > 0:
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"\nChi-Square Test Results:")
        print(f"Chi2 Statistic: {chi2:.4f}")
        print(f"p-value: {p:.4f}")
        
        alpha = 0.05
        if p < alpha:
            print("Result: Statistically Significant (Reject Null Hypothesis)")
            op_rate = summary.loc['Operational', 'pct_compliant']
            dev_rate = summary.loc['Development', 'pct_compliant']
            print(f"Operational Rate: {op_rate:.2f}%")
            print(f"Development Rate: {dev_rate:.2f}%")
            
            if op_rate < dev_rate:
                print("Conclusion: Operational systems have LOWER testing compliance. (SUPPORTS Hypothesis)")
            else:
                print("Conclusion: Operational systems have HIGHER testing compliance. (REFUTES Hypothesis)")
        else:
            print("Result: Not Statistically Significant (Fail to Reject Null Hypothesis)")
    else:
        print("\nWarning: Insufficient data for Chi-square test.")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
