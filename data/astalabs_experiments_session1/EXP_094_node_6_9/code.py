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