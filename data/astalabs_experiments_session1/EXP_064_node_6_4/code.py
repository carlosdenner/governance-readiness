# [debug]
import pandas as pd

# Load datasets
df_step1 = pd.read_csv('step1_sub_competencies.csv')
df_step3 = pd.read_csv('step3_coverage_map.csv')

print("=== Step 1: Sub-Competencies ===")
print(df_step1[['id', 'bundle', 'name', 'evidence_strength']].to_string())

print("\n=== Step 3: Coverage Map ===")
print(df_step3[['sub_competency_id', 'sub_competency_name', 'bundle', 'incident_count']].to_string())

# Check Step 2 for potential bridging
df_step2_comp = pd.read_csv('step2_competency_statements.csv')
print("\n=== Step 2: Competency Statements (Head) ===")
print(df_step2_comp[['competency_id', 'bundle', 'confidence', 'applicable_controls']].head().to_string())
