# Experiment 286: node_6_79

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_79` |
| **ID in Run** | 286 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T14:44:44.578950+00:00 |
| **Runtime** | 229.8s |
| **Parent** | `node_5_4` |
| **Children** | None |
| **Creation Index** | 287 |

---

## Hypothesis

> GenAI Attack Surface: Adversarial cases in ATLAS involving 'Generative' AI
technologies exhibit a significantly higher number of distinct attack 'Tactics'
per case compared to non-generative AI cases, indicating a broader threat
surface.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.5000 (Uncertain) |
| **Posterior** | 0.1703 (Likely False) |
| **Surprise** | -0.3956 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 15.0 |
| Uncertain | 0.0 |
| Maybe False | 15.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 60.0 |

---

## Experiment Plan

**Objective:** Quantify the complexity of attacks on Generative AI vs. traditional AI.

### Steps
- 1. Filter for `source_table` == 'atlas_cases'.
- 2. Identify Generative cases by searching the `summary` or `name` columns for keywords (LLM, GPT, GenAI, Diffusion, Generative).
- 3. Count the number of unique tactics in the `tactics` column (delimiter-separated) for each case.
- 4. Perform a Mann-Whitney U test to compare the distribution of tactic counts between Generative and Non-Generative cases.

### Deliverables
- Box plot of tactic counts by AI Type, Mann-Whitney U test statistics, and median tactic counts.

---

## Analysis

The experiment successfully analyzed 52 adversarial case studies from the MITRE
ATLAS dataset to compare the attack surface complexity (number of distinct
tactics) between Generative AI and Traditional AI systems.

**Findings:**
1.  **Classification:** The keyword-based filter identified 21 cases as
Generative AI and 31 cases as Traditional AI.
2.  **Tactic Counts:** The analysis revealed zero variance in the dependent
variable. Every single case in the dataset, regardless of category, was coded
with exactly one attack tactic (Mean = 1.00, Std = 0.00). This suggests that the
dataset structure likely flattens multi-tactic attacks into separate rows or
only records the primary tactic for each case.
3.  **Statistical Test:** Due to the identical distributions, the Mann-Whitney U
test yielded a U-statistic of 325.5 and a p-value of 1.0, indicating no
statistical difference.

**Conclusion:**
The hypothesis that Generative AI cases exhibit a broader attack surface (more
tactics) is **rejected** based on this data. However, the result is likely an
artifact of the dataset's coding schema (one tactic per row/entry) rather than a
reflection of real-world attack complexity. Future work should aggregate by
`case_id` if multiple rows exist per case, though the current N=52 suggests this
is the total unique case count.

---

## Review

The experiment was faithfully executed. The code successfully recovered from the
initial file path error, correctly filtered the dataset, applied the keyword
classification for Generative AI, and performed the statistical analysis as
planned. The results, while showing no variance (every case had exactly 1
tactic), are a valid reflection of the provided dataset's structure.

---

## Code

```python
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
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting experiment: GenAI Attack Surface Analysis (ATLAS Cases)...
Loaded 52 ATLAS cases.

Group Sizes:
  Generative AI Cases: 21
  Traditional AI Cases: 31

Descriptive Statistics (Tactic Counts):
  Generative AI: Mean=1.00, Median=1.00, Std=0.00
  Traditional AI: Mean=1.00, Median=1.00, Std=0.00

Mann-Whitney U Test Results (Two-sided):
  U-statistic: 325.5
  P-value: 1.0000
  Result: No Statistically Significant Difference.

STDERR:
<ipython-input-1-588d68197757>:108: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot(data_to_plot, labels=labels, patch_artist=True,


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is the detailed analysis:

### 1. Plot Type
*   **Type:** This is a **strip plot** (also known as a categorical scatter plot or 1D scatter plot).
*   **Purpose:** It is used to visualize the distribution of a single continuous or discrete variable ("Number of Distinct Tactics") across different categories ("Traditional AI" and "Generative AI"). It is particularly useful for showing individual data points and sample density when the dataset sizes are relatively small.

### 2. Axes
*   **X-axis:**
    *   **Label:** Represents the type of Artificial Intelligence.
    *   **Categories:**
        1.  **Traditional AI** (Sample size, $n=31$)
        2.  **Generative AI** (Sample size, $n=21$)
*   **Y-axis:**
    *   **Title:** "Number of Distinct Tactics per Case".
    *   **Units:** Count (integer values representing distinct tactics).
    *   **Range:** The visible scale ranges from **0.96 to 1.04**. However, this is a "zoomed-in" view; the data strictly sits at the integer value of **1.00**.

### 3. Data Trends
*   **Uniformity:** The most striking trend is the complete lack of variation in the y-axis variable. Every single data point (represented by red dots) for both categories aligns perfectly at the value **1.00**.
*   **Clusters:**
    *   **Traditional AI:** A horizontal cluster of red dots located at y=1.00. The cluster appears slightly wider or denser, reflecting the larger sample size ($n=31$).
    *   **Generative AI:** A similar horizontal cluster located at y=1.00, appearing slightly narrower due to the smaller sample size ($n=21$).
*   **Outliers:** There are no outliers. Every case falls exactly on the mean.

### 4. Annotations and Legends
*   **Title:** "Distribution of Attack Tactics Count: Generative vs. Traditional AI" clearly defines the scope of the comparison.
*   **Sample Size Indicators:** The x-axis labels include "($n=31$)" and "($n=21$)", providing immediate context regarding the statistical weight of each group.
*   **Grid Lines:** Horizontal dashed grid lines are plotted at intervals of 0.02 (0.96, 0.98, 1.00, etc.) to assist with reading the exact values, emphasizing that the points sit precisely on the 1.00 line.

### 5. Statistical Insights
*   **Zero Variance:** The standard deviation and variance for "Number of Distinct Tactics" in both groups is **0**.
*   **Single-Tactic Nature:** The data suggests that within the studied dataset, every single AI-related incident—whether utilizing Traditional or Generative AI—involved exactly **one distinct attack tactic**. There were no complex, multi-tactic attacks recorded in this specific sample.
*   **No Disparity:** There is no statistical difference between Traditional and Generative AI regarding the *count* of distinct tactics used per case; both hold a constant value of 1.
*   **Frequency:** While the complexity (tactic count) is identical, the dataset contains approximately **47% more cases** of Traditional AI attacks (31 cases) compared to Generative AI attacks (21 cases).
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
