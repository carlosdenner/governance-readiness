# Experiment 55: node_4_16

| Property | Value |
|---|---|
| **Experiment ID** | `node_4_16` |
| **ID in Run** | 55 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-20T23:17:01.694544+00:00 |
| **Runtime** | 220.6s |
| **Parent** | `node_3_18` |
| **Children** | None |
| **Creation Index** | 56 |

---

## Hypothesis

> Trust and Integration sub-competencies do not form distinct, isolated clusters
in real-world failures, but instead form mixed clusters indicating deep
interdependence.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.7500 (Likely True) |
| **Posterior** | 0.9360 (Definitely True) |
| **Surprise** | +0.2158 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
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
| Definitely True | 90.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Use unsupervised learning to analyze the co-occurrence structure of competency gaps.

### Steps
- 1. Load 'step3_incident_coding.csv' and extract 'sub_competency_ids' (semicolon-separated).
- 2. Construct a binary occurrence matrix (rows=incidents, cols=sub-competencies).
- 3. Compute a correlation or Jaccard similarity matrix between sub-competencies.
- 4. Perform Hierarchical Clustering (Linkage) and plot a Dendrogram.
- 5. Analyze if TR (Trust) and IR (Integration) codes segregate into different main branches or interleave.

### Deliverables
- Dendrogram of sub-competency clustering and a co-occurrence heatmap.

---

## Analysis

The experiment successfully tested the hypothesis that Trust and Integration
competencies are interdependent rather than isolated. The hierarchical
clustering analysis (using Jaccard distance on incident co-occurrence)
identified three primary clusters. Two of these clusters were mixed, containing
both 'Trust Readiness' (TR) and 'Integration Readiness' (IR) sub-competencies.
The largest cluster grouped IR-2 and IR-6 with TR-1, TR-2, TR-3, and TR-6, while
a smaller cluster grouped IR-1 with TR-4. Only IR-8 (Evaluation Infrastructure)
appeared as a distinct outlier. These findings support the hypothesis,
demonstrating that real-world AI incidents typically involve a tangled web of
governance (Trust) and engineering (Integration) failures, rather than isolated
breakdowns in a single domain.

---

## Review

The experiment was successfully executed and the code faithfully implemented the
planned analysis. The hierarchical clustering analysis provided clear evidence
regarding the structure of competency gaps.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import os

# Check file location
filename = 'step3_incident_coding.csv'
if not os.path.exists(filename):
    # Try one level up if not found
    filename = '../step3_incident_coding.csv'

print(f"Loading dataset from: {filename}")
df = pd.read_csv(filename)

# 2. Parse 'sub_competency_ids' to create a binary occurrence matrix
# Filter out rows with no IDs
df = df.dropna(subset=['sub_competency_ids'])

# Get all unique sub-competencies
all_ids = set()
parsed_ids = []

for val in df['sub_competency_ids']:
    # Split by semicolon, strip whitespace
    ids = [x.strip() for x in str(val).split(';') if x.strip()]
    parsed_ids.append(ids)
    all_ids.update(ids)

sorted_cols = sorted(list(all_ids))
print(f"Found {len(sorted_cols)} unique sub-competencies: {sorted_cols}")

# Create binary matrix (Rows=Incidents, Cols=Sub-competencies)
binary_matrix = pd.DataFrame(0, index=df.index, columns=sorted_cols)

for idx, ids in zip(df.index, parsed_ids):
    binary_matrix.loc[idx, ids] = 1

# 3. Compute Co-occurrence and Similarity
# Transpose so we cluster sub-competencies (columns), not incidents
X = binary_matrix.T

# Compute Jaccard Distance (1 - Jaccard Similarity)
dist_matrix = pdist(X, metric='jaccard')

# Compute Jaccard Similarity Matrix for Heatmap (1 - distance)
sim_matrix = 1 - squareform(dist_matrix)
sim_df = pd.DataFrame(sim_matrix, index=sorted_cols, columns=sorted_cols)

# 4. Hierarchical Clustering
# Using 'average' linkage
Z = linkage(dist_matrix, method='average')

# 5. Visualizations
plt.figure(figsize=(12, 12))

# Subplot 1: Heatmap of Jaccard Similarity
plt.subplot(2, 1, 1)
sns.heatmap(sim_df, annot=False, cmap='viridis', xticklabels=True, yticklabels=True)
plt.title('Jaccard Similarity Matrix of Sub-Competencies')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Subplot 2: Dendrogram
plt.subplot(2, 1, 2)
dendrogram(
    Z,
    labels=sorted_cols,
    leaf_rotation=90,
    leaf_font_size=10,
)
plt.title('Hierarchical Clustering Dendrogram (Jaccard Distance)')
plt.xlabel('Sub-Competency ID')
plt.ylabel('Distance')

plt.tight_layout()
plt.show()

# 6. Cluster Analysis
# Form clusters (e.g., k=3 to see if we get TR, IR, mixed)
k = 3
clusters = fcluster(Z, t=k, criterion='maxclust')
cluster_map = pd.DataFrame({'SubCompetency': sorted_cols, 'Cluster': clusters})

# Add bundle info
cluster_map['Bundle'] = cluster_map['SubCompetency'].apply(lambda x: 'Trust' if x.startswith('TR') else 'Integration')

print(f"\nCluster Assignments (k={k}):")
print(cluster_map.sort_values(['Cluster', 'Bundle']))

# Contingency Table
crosstab = pd.crosstab(cluster_map['Cluster'], cluster_map['Bundle'])
print("\nContingency Table of Clusters vs Bundles:")
print(crosstab)

# Check if clusters are mixed
mixed_clusters = crosstab[(crosstab['Integration'] > 0) & (crosstab['Trust'] > 0)]
if len(mixed_clusters) > 0:
    print(f"\nObservation: Found {len(mixed_clusters)} mixed clusters containing both Trust and Integration competencies.")
else:
    print("\nObservation: Clusters are perfectly separated by Bundle.")

```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from: step3_incident_coding.csv
Found 9 unique sub-competencies: ['IR-1', 'IR-2', 'IR-6', 'IR-8', 'TR-1', 'TR-2', 'TR-3', 'TR-4', 'TR-6']

Cluster Assignments (k=3):
  SubCompetency  Cluster       Bundle
0          IR-1        1  Integration
7          TR-4        1        Trust
1          IR-2        2  Integration
2          IR-6        2  Integration
4          TR-1        2        Trust
5          TR-2        2        Trust
6          TR-3        2        Trust
8          TR-6        2        Trust
3          IR-8        3  Integration

Contingency Table of Clusters vs Bundles:
Bundle   Integration  Trust
Cluster                    
1                  1      1
2                  2      4
3                  1      0

Observation: Found 2 mixed clusters containing both Trust and Integration competencies.


=== Plot Analysis (figure 1) ===
Based on the provided image, here is the detailed analysis of the two plots.

### 1. Plot Type
The image contains two related subplots illustrating data relationships:
*   **Top Plot:** A **Heatmap**. Its purpose is to visualize the Jaccard Similarity matrix, showing the pairwise similarity scores between different "Sub-Competencies."
*   **Bottom Plot:** A **Dendrogram**. Its purpose is to visualize the results of a hierarchical clustering algorithm (likely Agglomerative Clustering) applied to the data, using Jaccard Distance as the metric.

### 2. Axes

**Top Plot (Heatmap):**
*   **X-axis and Y-axis:** Both axes display categorical labels representing Sub-Competency IDs: `IR-1`, `IR-2`, `IR-6`, `IR-8`, `TR-1`, `TR-2`, `TR-3`, `TR-4`, `TR-6`.
*   **Color Scale (Z-axis equivalent):** Represented by the color bar on the right.
    *   **Range:** 0.0 to 1.0.
    *   **Meaning:** 1.0 (Yellow) represents identical sets (highest similarity), while roughly 0.0 (Dark Purple) represents no overlap (lowest similarity).

**Bottom Plot (Dendrogram):**
*   **X-axis:** Labeled "Sub-Competency ID". It lists the specific items being clustered (`IR-8`, `IR-1`, `TR-4`, `TR-1`, `TR-2`, `TR-3`, `TR-6`, `IR-2`, `IR-6`). *Note: The order here is determined by the clustering structure, not the alphanumeric order.*
*   **Y-axis:** Labeled "Distance".
    *   **Range:** 0.0 to roughly 1.0 (represented visually up to about 0.95).
    *   **Meaning:** This represents the linkage distance (dissimilarity). A lower height indicates items are more similar.

### 3. Data Trends

**Heatmap Trends:**
*   **Diagonal:** The diagonal from top-left to bottom-right is bright yellow (value 1.0), indicating that every sub-competency is perfectly similar to itself.
*   **High Similarity Areas:**
    *   **IR-2 and IR-6:** This intersection is a lighter green, indicating a high degree of similarity.
    *   **TR-1 and TR-2:** Also show a green intersection, suggesting moderate-to-high similarity.
*   **Low Similarity (Outliers):**
    *   **IR-8:** The row and column for IR-8 are extremely dark purple across the board (except for the diagonal). This indicates that IR-8 is very dissimilar to all other sub-competencies.

**Dendrogram Trends:**
*   **Closest Pairs (Lowest Branches):**
    *   **IR-2 and IR-6:** These join at the lowest point on the Y-axis (approximately 0.3 distance), confirming they are the most similar pair in the dataset.
    *   **TR-1 and TR-2:** These form the next closest pair, joining at a distance of approximately 0.4.
*   **Clustering Groups:**
    *   There is a large cluster (colored green lines) that groups together TR-1, TR-2, TR-3, TR-6, IR-2, and IR-6.
    *   There is a smaller cluster (colored orange lines) containing IR-1 and TR-4.
*   **Outliers:**
    *   **IR-8:** This item connects to the rest of the tree at the very top (distance > 0.9), visually isolating it on the far left. This confirms it is the most distinct item in the dataset.

### 4. Annotations and Legends
*   **Top Plot Title:** "Jaccard Similarity Matrix of Sub-Competencies".
*   **Bottom Plot Title:** "Hierarchical Clustering Dendrogram (Jaccard Distance)".
*   **Color Bar (Top Plot):** A vertical legend indicating the mapping of colors to Jaccard similarity values (Yellow = High, Purple = Low).
*   **Dendrogram Coloring:** The dendrogram branches are color-coded (Blue, Orange, Green) to represent the formation of primary clusters based on a threshold cut-off.

### 5. Statistical Insights
*   **Correlation between Plots:** The two plots are inverse representations of the same data relationships. The "Green" pairs in the heatmap (high similarity) correspond to the shortest vertical lines in the dendrogram (low distance).
*   **The "IR-8" Anomaly:** Statistically, **IR-8** is an outlier. It shares almost no commonality with the other competencies (Jaccard similarity near 0).
*   **Competency Groupings:** The analysis suggests distinct families of competencies:
    *   **Group A (The "Green" Cluster):** A tightly knit group involving mostly "TR" competencies (TR-1, TR-2, TR-3, TR-6) mixed with IR-2 and IR-6. This suggests that the content or skills required for `IR-2`/`IR-6` are more closely related to the `TR` series than the `IR` series.
    *   **Group B (The "Orange" Cluster):** IR-1 and TR-4 form a separate, moderately related pair.
*   **Interpretation for Use:** If these competencies represent curriculum or skill sets, a student who masters **IR-2** will likely have an easier time with **IR-6**, whereas mastering **IR-8** will provide almost no overlap/help with the other listed competencies.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
