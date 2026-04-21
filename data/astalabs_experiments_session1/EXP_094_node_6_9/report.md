# Experiment 94: node_6_9

| Property | Value |
|---|---|
| **Experiment ID** | `node_6_9` |
| **ID in Run** | 94 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-21T00:34:25.797435+00:00 |
| **Runtime** | 301.1s |
| **Parent** | `node_5_13` |
| **Children** | `node_7_1` |
| **Creation Index** | 95 |

---

## Hypothesis

> Incidents that involve 'Tool Use' or 'Plugin' execution techniques exhibit a
significantly higher 'Technique Count' than non-tool-use incidents, reflecting
the compounding complexity of agentic attack surfaces.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7500 (Likely True) |
| **Posterior** | 0.1921 (Likely False) |
| **Surprise** | -0.6474 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 1.0 |
| Maybe True | 29.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 90.0 |

---

## Experiment Plan

**Objective:** Test if agentic/tool-use features inherently increase attack complexity.

### Steps
- 1. Load 'step3_incident_coding.csv'.
- 2. Create a boolean column 'is_tool_use' by searching for keywords like 'Tool', 'Plugin', 'Execution', or 'Indirect Prompt Injection' in the 'techniques_used' and 'summary' columns.
- 3. Compare the 'technique_count' distribution for 'is_tool_use' = True vs False.
- 4. Perform an Independent Samples T-test or Mann-Whitney U test.
- 5. Generate a density plot (KDE) or histogram for both groups.

### Deliverables
- Density plot of Attack Complexity for Tool-Use vs Non-Tool-Use incidents and test results.

---

## Analysis

The experiment successfully tested the hypothesis that incidents involving 'Tool
Use' or 'Plugin' execution imply higher attack complexity. Using the
'step3_incident_coding.csv' dataset, incidents were classified into 'Tool-Use'
(n=18) and 'Non-Tool-Use' (n=34) groups. The analysis revealed nearly identical
mean technique counts (7.67 for Tool-Use vs. 7.47 for Non-Tool-Use). The Mann-
Whitney U test yielded a p-value of 0.4975, indicating no statistically
significant difference between the groups. Consequently, the hypothesis is
rejected. The data suggests that the complexity of an attack chain (number of
steps/techniques) is independent of whether the attacker leverages agentic tool-
use capabilities versus standard methods.

---

## Review

The experiment successfully tested the hypothesis that incidents involving 'Tool
Use' or 'Plugin' execution imply higher attack complexity. Using the
'step3_incident_coding.csv' dataset, incidents were classified into 'Tool-Use'
(n=18) and 'Non-Tool-Use' (n=34) groups. The analysis revealed nearly identical
mean technique counts (7.67 for Tool-Use vs. 7.47 for Non-Tool-Use). The Mann-
Whitney U test yielded a p-value of 0.4975, indicating no statistically
significant difference between the groups. Consequently, the hypothesis is
rejected. The data suggests that the complexity of an attack chain (number of
steps/techniques) is independent of whether the attacker leverages agentic tool-
use capabilities versus standard methods.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Load dataset
file_name = 'step3_incident_coding.csv'
# Check current then parent directory
file_path = file_name if os.path.exists(file_name) else os.path.join('..', file_name)

df = pd.read_csv(file_path)

# Define keywords for tool use / agentic features
# Expanded list based on MITRE ATLAS techniques related to execution/tools
tool_keywords = ['tool', 'plugin', 'execution', 'indirect prompt injection', 'function calling', 'api']

# Function to classify tool use
def check_tool_use(row):
    # Combine relevant text fields
    text_content = (str(row.get('techniques_used', '')).lower() + " " + 
                   str(row.get('summary', '')).lower())
    return any(keyword in text_content for keyword in tool_keywords)

# Apply classification
df['is_tool_use'] = df.apply(check_tool_use, axis=1)

# Separate groups
group_tool = df[df['is_tool_use'] == True]['technique_count'].dropna()
group_non_tool = df[df['is_tool_use'] == False]['technique_count'].dropna()

# Descriptive Statistics
print("=== Descriptive Statistics ===")
print(f"Tool-Use Incidents (n={len(group_tool)}): Mean={group_tool.mean():.2f}, Median={group_tool.median():.2f}, Std={group_tool.std():.2f}")
print(f"Non-Tool-Use Incidents (n={len(group_non_tool)}): Mean={group_non_tool.mean():.2f}, Median={group_non_tool.median():.2f}, Std={group_non_tool.std():.2f}")

# Statistical Testing
print("\n=== Statistical Analysis ===")
# Check Normality
shapiro_tool = stats.shapiro(group_tool) if len(group_tool) >= 3 else (0, 0)
shapiro_non = stats.shapiro(group_non_tool) if len(group_non_tool) >= 3 else (0, 0)

test_type = "Mann-Whitney U" if (shapiro_tool[1] < 0.05 or shapiro_non[1] < 0.05) else "T-test"
print(f"Normality Check (Shapiro-Wilk): Tool-Use p={shapiro_tool[1]:.4f}, Non-Tool p={shapiro_non[1]:.4f} -> Using {test_type}")

if test_type == "Mann-Whitney U":
    stat, p_val = stats.mannwhitneyu(group_tool, group_non_tool, alternative='two-sided')
    print(f"Mann-Whitney U Test: U={stat}, p-value={p_val:.4f}")
else:
    stat, p_val = stats.ttest_ind(group_tool, group_non_tool, equal_var=False)
    print(f"Welch's T-Test: t={stat:.4f}, p-value={p_val:.4f}")

# Visualization
plt.figure(figsize=(10, 6))
# Using histplot with kde=True is often clearer for discrete counts than pure KDE
sns.histplot(data=df, x='technique_count', hue='is_tool_use', kde=True, element="step", stat="density", common_norm=False, palette='viridis')
plt.title('Density of Attack Complexity (Technique Count)\nTool-Use vs Non-Tool-Use Incidents')
plt.xlabel('Number of Techniques Used')
plt.ylabel('Density')
plt.legend(title='Is Tool Use?', labels=['Non-Tool Use', 'Tool Use'])
plt.grid(True, alpha=0.3)
plt.show()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: === Descriptive Statistics ===
Tool-Use Incidents (n=18): Mean=7.67, Median=8.00, Std=2.79
Non-Tool-Use Incidents (n=34): Mean=7.47, Median=7.00, Std=3.05

=== Statistical Analysis ===
Normality Check (Shapiro-Wilk): Tool-Use p=0.0433, Non-Tool p=0.0558 -> Using Mann-Whitney U
Mann-Whitney U Test: U=341.5, p-value=0.4975


=== Plot Analysis (figure 1) ===
Based on the provided plot, here is the detailed analysis:

### 1. Plot Type
*   **Type:** This is a **combined Histogram and Density Plot (Kernel Density Estimate - KDE)**.
*   **Purpose:** It visualizes and compares the frequency distribution and probability density of a continuous or discrete variable (number of techniques used) across two distinct categories (Tool Use vs. Non-Tool Use). The bars represent the actual frequency counts (normalized to density), while the smooth lines estimate the underlying probability distribution.

### 2. Axes
*   **X-Axis:**
    *   **Label:** "Number of Techniques Used"
    *   **Range:** The axis spans from roughly **1 to 16**.
    *   **Meaning:** This represents the attack complexity, measured by the count of distinct techniques employed in an incident.
*   **Y-Axis:**
    *   **Label:** "Density"
    *   **Range:** The axis spans from **0.00 to 0.30**.
    *   **Meaning:** This represents the probability density function. The area under the curve (or the sum of bar areas times widths) integrates to 1. Higher values indicate a higher likelihood of that specific technique count occurring.

### 3. Data Trends
*   **Green Distribution (Non-Tool Use):**
    *   **Pattern:** This distribution is **highly concentrated (peaked)**. It shows a distinct "leptokurtic" shape, meaning it clusters tightly around the mean with smaller tails.
    *   **Peak:** The tallest bar and the peak of the curve are centered roughly between **7 and 9 techniques**. The peak density is approximately **0.30**, which is significantly higher than the other group.
    *   **Interpretation:** "Non-Tool Use" incidents tend to have a very consistent level of complexity, frequently utilizing around 8 techniques.
*   **Blue/Grey Distribution (Tool Use):**
    *   **Pattern:** This distribution is much **flatter and wider (dispersed)**. It shows a "platykurtic" shape, indicating high variance.
    *   **Peak:** The distribution does not have a single sharp peak but rather a broad plateau ranging from approximately **4 to 9 techniques**. The maximum density is much lower, peaking around **0.13 to 0.14**.
    *   **Outliers/Range:** This group covers a wider range of complexity, with visible density stretching from as low as 1 technique to as high as 16 techniques.

### 4. Annotations and Legends
*   **Title:** "Density of Attack Complexity (Technique Count) Tool-Use vs Non-Tool-Use Incidents".
*   **Legend:** Located in the top right, titled **"Is Tool Use?"**.
    *   **Non-Tool Use:** Represented by the **Light Green** filled bars and Green curve.
    *   **Tool Use:** Represented by the **Blue/Grey** filled bars and Blue curve. *(Note: The legend marker for "Tool Use" appears to display a green line in the image, likely a plotting artifact, but by process of elimination and standard color cycling, the Blue distribution corresponds to "Tool Use".)*

### 5. Statistical Insights
*   **Consistency vs. Variability:** The most significant insight is the difference in variance. Incidents classified as **"Non-Tool Use"** are highly predictable in their complexity, clustering tightly around 8 techniques. Conversely, **"Tool Use"** incidents are highly variable; they can be very simple (few techniques) or highly complex (14+ techniques).
*   **Average Complexity:** Visually, the mode (most frequent value) for "Non-Tool Use" (approx. 8) appears slightly higher than the central tendency of "Tool Use" (which centers closer to 6 or 7).
*   **Implication:** This suggests that when attackers do *not* use specific tools (perhaps relying on manual execution or "living off the land"), the attack chain requires a specific, consistent number of steps. When tools *are* used, the attack profile varies wildly—likely depending on the specific tool's capability or the automation level.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
