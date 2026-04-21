import pandas as pd
import scipy.stats as stats
import sys

def run_experiment():
    # 1. Load the dataset
    file_path = 'step2_crosswalk_matrix.csv'
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)

    # 2. Extract and Preprocess Columns
    col_audit = 'Audit Logging & Telemetry'
    col_ir = 'Incident Response & Recovery Playbooks'

    if col_audit not in df.columns or col_ir not in df.columns:
        print(f"Error: Columns '{col_audit}' or '{col_ir}' not found.")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    # Binarize: 'X' -> 1, others -> 0
    # Using fillna('') to handle NaNs safely before string manipulation
    audit_binary = df[col_audit].fillna('').astype(str).str.strip().str.upper().apply(lambda x: 1 if x == 'X' else 0)
    ir_binary = df[col_ir].fillna('').astype(str).str.strip().str.upper().apply(lambda x: 1 if x == 'X' else 0)

    # 3. Create Contingency Table
    # Format: [[Both Absent (0,0), IR Present (0,1)], [Audit Present (1,0), Both Present (1,1)]]
    # However, crosstab default is index=row_var, columns=col_var
    contingency_table = pd.crosstab(audit_binary, ir_binary)
    
    # Ensure 2x2 shape even if some combinations are missing (e.g. if no 1s exist)
    contingency_filled = contingency_table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    
    print("\nContingency Table:")
    print("Rows: Audit Logging & Telemetry (0, 1)")
    print("Cols: Incident Response & Recovery Playbooks (0, 1)")
    print(contingency_filled)

    # Extract values for clarity
    # n00 = neither
    # n01 = IR only
    # n10 = Audit only
    # n11 = Both
    n00 = contingency_filled.loc[0, 0]
    n01 = contingency_filled.loc[0, 1]
    n10 = contingency_filled.loc[1, 0]
    n11 = contingency_filled.loc[1, 1]

    print(f"\nCounts: Neither={n00}, IR_only={n01}, Audit_only={n10}, Both={n11}")

    # 4. Fisher's Exact Test
    # We use the table [[n00, n01], [n10, n11]]
    # Note: Structure affects Odds Ratio direction, but p-value remains same for independence test.
    odds_ratio, p_value = stats.fisher_exact(contingency_filled)

    print("\nFisher's Exact Test Results:")
    print(f"Odds Ratio: {odds_ratio:.4f}")
    print(f"P-value: {p_value:.4f}")

    # 5. Jaccard Index
    # J = (Intersection) / (Union) = n11 / (n10 + n01 + n11)
    union_count = n10 + n01 + n11
    if union_count == 0:
        jaccard = 0.0
    else:
        jaccard = n11 / union_count

    print(f"\nJaccard Index (Intersection over Union): {jaccard:.4f}")

    # Conclusion
    print("\nConclusion:")
    if p_value < 0.05:
        print("Reject Null: There is a statistically significant association.")
    else:
        print("Fail to Reject Null: No statistically significant association found.")

    if jaccard > 0.5:
        print("Overlap: High")
    elif jaccard > 0.2:
        print("Overlap: Moderate")
    else:
        print("Overlap: Low/Negligible")

if __name__ == "__main__":
    run_experiment()