import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Load dataset
try:
    df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)
except FileNotFoundError:
    df = pd.read_csv('astalabs_discovery_all_data.csv', low_memory=False)

# Filter for AIID incidents
aiid = df[df['source_table'] == 'aiid_incidents'].copy()

# Define Columns
sector_col = 'Sector of Deployment'
harm_col = 'Tangible Harm'

print(f"Analyzing intersection of '{sector_col}' and '{harm_col}'...")

# Check if columns exist
if sector_col not in aiid.columns or harm_col not in aiid.columns:
    print(f"Error: Required columns not found. Available columns: {aiid.columns.tolist()}")
else:
    # Filter for rows where both columns are not null
    df_clean = aiid.dropna(subset=[sector_col, harm_col]).copy()
    print(f"Rows with both Sector and Harm data: {len(df_clean)}")

    if len(df_clean) < 5:
        print("Insufficient data overlap to perform statistical analysis.")
    else:
        # Map Harm to Binary Categories
        # 'tangible harm definitively occurred' -> Tangible
        # 'no tangible harm, near-miss, or issue' -> Intangible
        # Others -> Exclude to maintain binary clarity for hypothesis
        def classify_harm(val):
            s = str(val).lower()
            if 'definitively occurred' in s:
                return 'Tangible'
            elif 'no tangible harm' in s:
                return 'Intangible'
            else:
                return None # Exclude risks/unclear for this specific test

        df_clean['Harm_Class'] = df_clean[harm_col].apply(classify_harm)
        df_analysis = df_clean.dropna(subset=['Harm_Class']).copy()

        # Focus on Top 5 Sectors to ensure statistical relevance
        top_sectors = df_analysis[sector_col].value_counts().head(5).index.tolist()
        df_final = df_analysis[df_analysis[sector_col].isin(top_sectors)].copy()

        print(f"Final Analysis Set (Top 5 Sectors, Valid Harm Class): {len(df_final)}")
        
        if len(df_final) > 0:
            # Generate Contingency Table
            ct = pd.crosstab(df_final[sector_col], df_final['Harm_Class'])
            print("\nContingency Table (Sector vs Harm Class):")
            print(ct)

            # Check for empty columns/rows
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                # Chi-Square Test
                chi2, p, dof, expected = chi2_contingency(ct)
                print(f"\nChi-Square Statistic: {chi2:.4f}")
                print(f"P-value: {p:.4f}")

                if p < 0.05:
                    print("Result: Significant relationship found between Sector and Harm Type.")
                else:
                    print("Result: No significant relationship found.")

                # Visualization
                plt.figure(figsize=(10, 6))
                sns.heatmap(ct, annot=True, fmt='d', cmap='Blues')
                plt.title('Tangible vs Intangible Harm Distribution by Sector')
                plt.ylabel('Sector')
                plt.xlabel('Harm Category')
                plt.tight_layout()
                plt.show()
            else:
                print("Contingency table degenerate (not enough variation for test).")
        else:
            print("No data remaining after filtering for top sectors and valid harm classes.")
