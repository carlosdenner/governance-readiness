import pandas as pd
import numpy as np
import scipy.stats as stats
import os

def run_experiment():
    # 1. Load Dataset
    filename = 'astalabs_discovery_all_data.csv'
    paths = [f'../{filename}', filename]
    df = None
    for p in paths:
        if os.path.exists(p):
            print(f"Found dataset at: {p}")
            df = pd.read_csv(p, low_memory=False)
            break
    
    if df is None:
        print("Error: Dataset not found in . or ..")
        # List files for debugging
        print("Current dir files:", os.listdir('.'))
        if os.path.exists('../'):
            print("Parent dir files:", os.listdir('../'))
        return

    # 2. Segment Data
    aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
    eo_df = df[df['source_table'] == 'eo13960_scored'].copy()
    
    print(f"Loaded {len(aiid_df)} AIID incidents and {len(eo_df)} EO 13960 systems.")

    # 3. Analyze AIID Sectors
    # Find Sector column
    sector_cols = [c for c in aiid_df.columns if 'Sector' in str(c) and 'Public' not in str(c)]
    if not sector_cols:
        # Fallback to probable index or name if search fails
        # Metadata said '78: Sector of Deployment'
        sector_col = '78: Sector of Deployment'
        if sector_col not in aiid_df.columns:
            # Try finding column by index if header names are stripped of numbers
            if 'Sector of Deployment' in aiid_df.columns:
                sector_col = 'Sector of Deployment'
            else:
                print("Could not find Sector column. Columns:", aiid_df.columns[:10])
                return
    else:
        sector_col = sector_cols[0]
        
    print(f"Using Sector column: {sector_col}")
    
    # Clean and count sectors
    # AIID sectors can be comma-separated strings
    sectors_series = aiid_df[sector_col].dropna().astype(str)
    all_sectors = []
    for s in sectors_series:
        # Split by comma
        parts = [x.strip() for x in s.split(',')]
        all_sectors.extend(parts)
        
    sector_counts = pd.Series(all_sectors).value_counts()
    print("\nTop 10 Incident Sectors:")
    print(sector_counts.head(10))
    
    # 4. Map Sectors to Agencies
    # Define explicit mapping for major sectors to Federal Agencies (Abbr)
    # Agencies in EO dataset: HHS, VA, DHS, DOJ, DOD, TREAS, DOC, DOT, DOE, ED, etc.
    
    sector_agency_map = {
        'Healthcare': ['HHS', 'VA', 'CMS', 'CDC', 'FDA', 'NIH'],
        'Medicine': ['HHS', 'VA', 'CMS', 'CDC', 'FDA', 'NIH'],
        'Finance': ['TREAS', 'SEC', 'SBA', 'SSA', 'FDIC'],
        'Financial': ['TREAS', 'SEC', 'SBA', 'SSA', 'FDIC'],
        'Transportation': ['DOT', 'FAA'],
        'Automotive': ['DOT'], # Autonomous vehicles often fall under DOT reg
        'Law Enforcement': ['DOJ', 'DHS', 'FBI'],
        'Government': [], # Too broad, ignore
        'Education': ['ED'],
        'Social Services': ['SSA', 'HHS'],
        'Defense': ['DOD', 'NAVY', 'ARMY', 'USAF'],
        'Military': ['DOD', 'NAVY', 'ARMY', 'USAF']
    }
    
    # Determine High Risk Agencies (Top 3 mappable sectors)
    high_risk_agencies = set()
    mapped_sectors_count = 0
    
    print("\nMapping Top Sectors to Agencies:")
    for sector, count in sector_counts.items():
        # Fuzzy match key
        matched = False
        for key, agencies in sector_agency_map.items():
            if key.lower() in sector.lower():
                if agencies: # If we have agencies mapped
                    high_risk_agencies.update(agencies)
                    print(f"  Sector '{sector}' ({count}) -> {agencies}")
                    matched = True
        if matched:
            mapped_sectors_count += 1
            if mapped_sectors_count >= 3:
                break
                
    print(f"High Risk Agency Pool: {sorted(list(high_risk_agencies))}")
    
    # 5. Determine Low Risk Agencies
    # Strategy: All other agencies present in EO dataset not in High Risk
    all_eo_agencies = eo_df['3_abr'].dropna().unique()
    low_risk_agencies = [a for a in all_eo_agencies if a not in high_risk_agencies]
    
    # 6. Analyze Governance (Key Risks Identified)
    # Column: '54_key_risks'
    risk_col = '54_key_risks'
    if risk_col not in eo_df.columns:
        # Try finding it
        rc = [c for c in eo_df.columns if 'key_risks' in str(c).lower()]
        if rc:
            risk_col = rc[0]
        else:
            print("Could not find Key Risks column.")
            return

    # Filter and categorize
    eo_subset = eo_df[eo_df['3_abr'].notna()].copy()
    eo_subset['Risk_Category'] = 'Neutral'
    eo_subset.loc[eo_subset['3_abr'].isin(high_risk_agencies), 'Risk_Category'] = 'High_Incident_Sector'
    eo_subset.loc[eo_subset['3_abr'].isin(low_risk_agencies), 'Risk_Category'] = 'Low_Incident_Sector'
    
    # Remove unmapped if any (should cover all though)
    analysis_df = eo_subset[eo_subset['Risk_Category'] != 'Neutral'].copy()
    
    # Convert Target to Binary
    # Values are typically 'Yes', 'No', 'N/A' or variations
    analysis_df['Has_Risk_ID'] = analysis_df[risk_col].astype(str).str.strip().str.lower() == 'yes'
    
    # Calculate Statistics
    stats_df = analysis_df.groupby('Risk_Category')['Has_Risk_ID'].agg(['count', 'mean', 'sum'])
    stats_df.columns = ['Total_Systems', 'Risk_ID_Rate', 'Systems_With_ID']
    
    print("\n--- Governance Disconnect Analysis --- ")
    print(stats_df)
    
    # 7. Statistical Test
    high_group = analysis_df[analysis_df['Risk_Category'] == 'High_Incident_Sector']['Has_Risk_ID']
    low_group = analysis_df[analysis_df['Risk_Category'] == 'Low_Incident_Sector']['Has_Risk_ID']
    
    if len(high_group) > 0 and len(low_group) > 0:
        # Chi-square
        contingency = pd.crosstab(analysis_df['Risk_Category'], analysis_df['Has_Risk_ID'])
        chi2, p, dof, ex = stats.chi2_contingency(contingency)
        
        print(f"\nChi-Square Test of Independence:")
        print(f"Statistic: {chi2:.4f}, p-value: {p:.4e}")
        
        if p < 0.05:
            print("Conclusion: Statistically significant difference in Risk Identification rates.")
            if stats_df.loc['High_Incident_Sector', 'Risk_ID_Rate'] > stats_df.loc['Low_Incident_Sector', 'Risk_ID_Rate']:
                print("Direction: High-incident sectors have BETTER governance (Hypothesis Refuted).")
            else:
                print("Direction: High-incident sectors have WORSE governance (Hypothesis Confirmed).")
        else:
            print("Conclusion: No statistically significant difference (Hypothesis Confirmed - Disconnect exists).")
            
    else:
        print("Insufficient data groups.")

if __name__ == "__main__":
    run_experiment()