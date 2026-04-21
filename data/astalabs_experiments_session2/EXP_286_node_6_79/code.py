import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import numpy as np
import os

def run_experiment():
    print("Starting experiment: GenAI Attack Surface Analysis (ATLAS Cases)...")

    # 1. Load the dataset
    # Trying current directory as previous attempt with ../ failed
    file_path = "astalabs_discovery_all_data.csv"
    
    if not os.path.exists(file_path):
        # Fallback to checking parent dir just in case, though previous run said it wasn't there
        if os.path.exists("../astalabs_discovery_all_data.csv"):
             file_path = "../astalabs_discovery_all_data.csv"
        else:
             print(f"Error: File not found at {file_path} or parent directory.")
             return

    try:
        # Load with low_memory=False to avoid dtype warnings
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 2. Filter for 'atlas_cases'
    atlas_df = df[df['source_table'] == 'atlas_cases'].copy()
    print(f"Loaded {len(atlas_df)} ATLAS cases.")

    if len(atlas_df) == 0:
        print("No ATLAS cases found. Exiting.")
        return

    # 3. Identify Generative AI cases
    # Keywords to identify Generative AI
    genai_keywords = ['llm', 'gpt', 'genai', 'diffusion', 'generative', 'chatbot', 'foundation model', 'hallucination', 'prompt injection', 'jailbreak', 'bard', 'bing chat', 'chatgpt']
    
    def check_genai(row):
        # Combine name and summary for keyword search
        text_content = str(row.get('name', '')) + " " + str(row.get('summary', ''))
        text_content = text_content.lower()
        return any(keyword in text_content for keyword in genai_keywords)

    atlas_df['is_genai'] = atlas_df.apply(check_genai, axis=1)
    
    # 4. Count distinct tactics
    def count_tactics(tactics_str):
        if pd.isna(tactics_str) or str(tactics_str).strip() == '':
            return 0
        # Normalize delimiters (ATLAS often uses semicolons)
        t_str = str(tactics_str).replace(';', ',')
        # Split and filter empty
        items = [t.strip() for t in t_str.split(',') if t.strip()]
        # Return unique count
        return len(set(items))

    atlas_df['tactic_count'] = atlas_df['tactics'].apply(count_tactics)

    # 5. Group data
    genai_group = atlas_df[atlas_df['is_genai']]['tactic_count']
    non_genai_group = atlas_df[~atlas_df['is_genai']]['tactic_count']

    n_genai = len(genai_group)
    n_non_genai = len(non_genai_group)

    print(f"\nGroup Sizes:\n  Generative AI Cases: {n_genai}\n  Traditional AI Cases: {n_non_genai}")

    if n_genai < 2 or n_non_genai < 2:
        print("Insufficient data in one or both groups for statistical testing.")
        return

    # 6. Statistical Analysis (Mann-Whitney U Test)
    stat, p_value = mannwhitneyu(genai_group, non_genai_group, alternative='greater') 
    # Hypothesis: GenAI > Traditional (one-sided 'greater')
    # Or two-sided to be safe, but prompt implies checking if they exhibit "significantly higher" number.
    # Let's use 'two-sided' for general difference, but interpret direction.
    
    stat_two_sided, p_value_two_sided = mannwhitneyu(genai_group, non_genai_group, alternative='two-sided')

    genai_mean = genai_group.mean()
    non_genai_mean = non_genai_group.mean()

    print(f"\nDescriptive Statistics (Tactic Counts):")
    print(f"  Generative AI: Mean={genai_mean:.2f}, Median={genai_group.median():.2f}, Std={genai_group.std():.2f}")
    print(f"  Traditional AI: Mean={non_genai_mean:.2f}, Median={non_genai_group.median():.2f}, Std={non_genai_group.std():.2f}")

    print(f"\nMann-Whitney U Test Results (Two-sided):")
    print(f"  U-statistic: {stat_two_sided}")
    print(f"  P-value: {p_value_two_sided:.4f}")
    
    alpha = 0.05
    if p_value_two_sided < alpha:
        print("  Result: Statistically Significant Difference.")
    else:
        print("  Result: No Statistically Significant Difference.")

    # 7. Visualization
    plt.figure(figsize=(10, 6))
    data_to_plot = [non_genai_group, genai_group]
    labels = [f'Traditional AI\n(n={n_non_genai})', f'Generative AI\n(n={n_genai})']
    
    plt.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))
    
    plt.title('Distribution of Attack Tactics Count: Generative vs. Traditional AI')
    plt.ylabel('Number of Distinct Tactics per Case')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Jitter plot
    for i, data in enumerate(data_to_plot):
        y = data
        x = np.random.normal(i + 1, 0.04, size=len(y))
        plt.plot(x, y, 'r.', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()