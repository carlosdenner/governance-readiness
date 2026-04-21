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
