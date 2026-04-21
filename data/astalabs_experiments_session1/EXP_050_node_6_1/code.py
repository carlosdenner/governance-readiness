import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

# Define file path based on the note provided
file_path = '../step2_competency_statements.csv'

# Fallback to current directory if not found (handling potential environment inconsistencies)
if not os.path.exists(file_path):
    file_path = 'step2_competency_statements.csv'

try:
    # 1. Load the dataset
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)

    # 2. Calculate word count for each 'competency_statement'
    # Using simple whitespace splitting
    df['word_count'] = df['competency_statement'].apply(lambda x: len(str(x).split()))

    # 3. Group by 'bundle'
    trust_data = df[df['bundle'] == 'Trust Readiness']['word_count']
    integration_data = df[df['bundle'] == 'Integration Readiness']['word_count']

    # 4. Descriptive Statistics
    trust_desc = trust_data.describe()
    integration_desc = integration_data.describe()

    print("\n=== Descriptive Statistics (Word Count) ===")
    print(f"Trust Readiness (n={int(trust_desc['count'])})")
    print(f"  Mean: {trust_desc['mean']:.2f}")
    print(f"  Median: {trust_desc['50%']:.2f}")
    print(f"  Std Dev: {trust_desc['std']:.2f}")
    
    print(f"\nIntegration Readiness (n={int(integration_desc['count'])})")
    print(f"  Mean: {integration_desc['mean']:.2f}")
    print(f"  Median: {integration_desc['50%']:.2f}")
    print(f"  Std Dev: {integration_desc['std']:.2f}")

    # 5. Independent Samples T-test (Welch's t-test for unequal variances/sample sizes)
    t_stat, p_val = stats.ttest_ind(trust_data, integration_data, equal_var=False)

    print("\n=== Independent Samples T-test (Welch's) ===")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val:.4f}")
    
    if p_val < 0.05:
        print("Conclusion: Reject null hypothesis. There is a significant difference in word counts.")
    else:
        print("Conclusion: Fail to reject null hypothesis. No significant difference in word counts.")

    # 6. Visualization
    plt.figure(figsize=(10, 6))
    
    # Plotting histograms with density to compare shapes despite unequal sample sizes
    plt.hist(trust_data, bins=10, alpha=0.6, label='Trust Readiness', density=True, color='blue', edgecolor='black')
    plt.hist(integration_data, bins=10, alpha=0.6, label='Integration Readiness', density=True, color='orange', edgecolor='black')
    
    # Add vertical lines for means
    plt.axvline(trust_data.mean(), color='blue', linestyle='dashed', linewidth=1, label=f'Trust Mean ({trust_data.mean():.1f})')
    plt.axvline(integration_data.mean(), color='orange', linestyle='dashed', linewidth=1, label=f'Integration Mean ({integration_data.mean():.1f})')

    plt.title('Word Count Distribution: Trust vs. Integration Competencies')
    plt.xlabel('Word Count')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
