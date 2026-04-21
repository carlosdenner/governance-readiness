# Experiment 2: node_2_1

| Property | Value |
|---|---|
| **Experiment ID** | `node_2_1` |
| **ID in Run** | 2 |
| **Status** | FAILED |
| **Created** | 2026-02-22T00:59:38.883750+00:00 |
| **Runtime** | 739.8s |
| **Parent** | `node_1_0` |
| **Children** | `node_3_7`, `node_3_15`, `node_3_22` |
| **Creation Index** | 3 |

---

## Hypothesis

> Harm Domain Locality: The 'Physical' harm domain is almost exclusively isolated
to the 'Transportation' and 'Industrial' sectors, whereas 'Psychological' harm
is distributed broadly across 'Social Media', 'Healthcare', and 'Entertainment'.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | N/A (Unknown) |
| **Posterior** | N/A (Unknown) |
| **Surprise** | N/A |
| **Surprise Interpretation** | N/A |
| **Is Surprising?** | No |

### Prior Belief Distribution
N/A

### Posterior Belief Distribution
N/A

---

## Experiment Plan

**Objective:** Validate the hypothesis by analyzing the distribution of sectors for specific harm types ('Physical' vs. 'Psychological') using the 'Tangible Harm' column.

### Steps
- 1. Load the dataset `astalabs_discovery_all_data.csv`.
- 2. Filter the dataframe to include only rows where `source_table` is 'aiid_incidents'.
- 3. Identify the correct column for harm types. Based on previous feedback, inspect `Tangible Harm` (Column 74) instead of `Harm Domain`. Verify it contains values like 'Physical' or 'Psychological'.
- 4. Clean and explode the `Tangible Harm` and `Sector of Deployment` columns, as both may contain comma-separated lists. Remove rows with missing or empty values in these columns.
- 5. Create a cross-tabulation (contingency table) with `Tangible Harm` as the index (rows) and `Sector of Deployment` as the columns.
- 6. Calculate the Shannon Entropy of the *Sector distribution* for each Harm type (i.e., row-wise entropy of the sector probabilities given a specific harm).
- 7. Compare the entropy of 'Physical' harm (expected to be low, indicating isolation to few sectors) vs. 'Psychological' harm (expected to be high, indicating broad distribution).
- 8. Check the specific sector concentrations: For 'Physical', check the proportion of 'Transportation' and 'Industrial' (or 'Manufacturing'). For 'Psychological', check the presence in 'Social Media' (or 'Technology'), 'Healthcare', and 'Entertainment'.

### Deliverables
- 1. List of unique values found in `Tangible Harm` to confirm data validity.
- 2. Calculated Entropy scores for 'Physical' and 'Psychological' harm distributions.
- 3. A printed summary of the top 3 sectors for 'Physical' and 'Psychological' harms with their respective proportions.
- 4. A bar chart comparing the sector distribution for these two harm types.

---

## Analysis

The previous code execution encountered a `ValueError` caused by duplicate
indices in the DataFrame. This duplication happened because the `explode`
function was used to separate multiple sectors per row, but the index was not
reset afterwards. When `pd.crosstab` attempted to align the Series objects, the
duplicate indices caused a conflict. To fix this, I will ensure that
`df.reset_index(drop=True)` is called immediately after any `explode` operation.
This will guarantee a unique index for every row before the crosstabulation
step. The logic for deriving 'Physical' and 'Psychological' labels from
'Tangible Harm' and 'Special Interest Intangible Harm' columns appears sound
given the previous data exploration.

---

## Review

The previous code failed due to a `ValueError` caused by duplicate indices
generated during the `explode` operations. When `pd.crosstab` receives Series
with duplicate indices (representing the same original row split into multiple
parts), it attempts alignment which fails if the index isn't unique. To fix
this, `reset_index(drop=True)` must be called immediately after every `explode`
operation to ensure each row in the expanded DataFrame has a unique identifier.

---

## Code

```python
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import os

# 1. Load Data
filename = 'astalabs_discovery_all_data.csv'
possible_paths = [filename, f'../{filename}']
file_path = None

for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        break

if not file_path:
    file_path = filename

print(f"Loading data from: {file_path}")

try:
    df = pd.read_csv(file_path, low_memory=False)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

# 2. Filter for aiid_incidents
df_incidents = df[df['source_table'] == 'aiid_incidents'].copy()
print(f"AIID Incidents shape: {df_incidents.shape}")

# 3. Define Proxies for Physical and Psychological Harm
# Based on previous exploration, specific 'Physical'/'Psychological' labels are not in a single column.
# We hypothesize:
# - 'Physical' corresponds to 'Tangible Harm' being present.
# - 'Psychological' corresponds to 'Special Interest Intangible Harm' being 'yes'.

# Check values again to be sure
print("Tangible Harm values:", df_incidents['Tangible Harm'].dropna().unique())
if 'Special Interest Intangible Harm' in df_incidents.columns:
    print("Special Interest Intangible Harm values:", df_incidents['Special Interest Intangible Harm'].dropna().unique())

# Function to tag rows
def tag_harm_type(row):
    tags = []
    
    # Physical / Tangible
    # We consider it Physical if tangible harm definitively occurred or there was an imminent risk.
    tangible = str(row['Tangible Harm']).lower()
    if 'definitively occurred' in tangible or 'imminent risk' in tangible:
        tags.append('Physical')
        
    # Psychological / Intangible
    # We use the Special Interest Intangible Harm column
    intangible = str(row.get('Special Interest Intangible Harm', '')).lower()
    if intangible == 'yes':
        tags.append('Psychological')
    
    return tags

# Apply tagging
df_incidents['Derived_Harm_Tags'] = df_incidents.apply(tag_harm_type, axis=1)

# Explode tags so we can analyze distribution per tag
df_exploded = df_incidents.explode('Derived_Harm_Tags')
df_analysis = df_exploded.dropna(subset=['Derived_Harm_Tags'])

print(f"Rows with derived tags: {len(df_analysis)}")
print("Tag counts:\n", df_analysis['Derived_Harm_Tags'].value_counts())

if df_analysis.empty:
    print("No tags derived. Cannot proceed.")
    exit(0)

# 4. Clean Sector Column
sector_col = 'Sector of Deployment'

# Helper to explode sectors if they are lists
def clean_and_explode(dataframe, column):
    s = dataframe[column].astype(str).str.split(',')
    dataframe = dataframe.assign(**{column: s}).explode(column)
    dataframe[column] = dataframe[column].str.strip()
    return dataframe

df_analysis = clean_and_explode(df_analysis, sector_col)
df_analysis = df_analysis[~df_analysis[sector_col].isin(['', 'nan', 'NaN'])]

# 5. Analysis: Crosstab and Entropy
ct = pd.crosstab(df_analysis['Derived_Harm_Tags'], df_analysis[sector_col])
probs = ct.div(ct.sum(axis=1), axis=0)

entropy_scores = probs.apply(lambda x: scipy.stats.entropy(x, base=2), axis=1)

print("\n--- Entropy Scores (Lower = More Localized) ---")
print(entropy_scores.sort_values())

# 6. Validate Hypothesis Sectors
# Hypothesis:
# Physical -> Transportation, Industrial (Manufacturing?)
# Psychological -> Social Media, Healthcare, Entertainment

target_map = {
    'Physical': ['transportation', 'industrial', 'manufacturing'],
    'Psychological': ['social media', 'healthcare', 'entertainment']
}

print("\n--- Sector Proportions by Harm Type ---")
for harm in ['Physical', 'Psychological']:
    if harm in probs.index:
        print(f"\n{harm} Harm:")
        # Show top 5
        top = probs.loc[harm].sort_values(ascending=False).head(5)
        for s, p in top.items():
            print(f"  {s}: {p:.1%}")
            
        # Check specific hypothesis targets
        print(f"  > Hypothesis Check:")
        targets = target_map.get(harm, [])
        for t in targets:
            # Find matching sector keys
            matches = [k for k in probs.columns if t in k.lower()]
            for m in matches:
                print(f"    {m}: {probs.loc[harm, m]:.1%}")

# 7. Visualization
top_sectors_idx = ct.sum(axis=0).sort_values(ascending=False).head(10).index
plot_data = probs[top_sectors_idx]

if not plot_data.empty:
    ax = plot_data.T.plot(kind='bar', figsize=(10, 6))
    plt.title('Sector Distribution: Physical (Tangible) vs Psychological (Intangible)')
    plt.ylabel('Proportion')
    plt.xlabel('Sector')
    plt.legend(title='Derived Harm Type')
    plt.tight_layout()
    plt.show()

```

## Code Output

```
exitcode: 1 (execution failed)
Code output: Loading data from: astalabs_discovery_all_data.csv
AIID Incidents shape: (1362, 196)
Tangible Harm values: <StringArray>
[                 'no tangible harm, near-miss, or issue',
                    'tangible harm definitively occurred',
   'imminent risk of tangible harm (near miss) did occur',
 'non-imminent risk of tangible harm (an issue) occurred',
                                                'unclear']
Length: 5, dtype: str
Special Interest Intangible Harm values: <StringArray>
['yes', 'no', 'maybe']
Length: 3, dtype: str
Rows with derived tags: 208
Tag counts:
 Derived_Harm_Tags
Psychological    104
Physical         104
Name: count, dtype: int64
[31m---------------------------------------------------------------------------[39m
[31mValueError[39m                                Traceback (most recent call last)
[36mCell[39m[36m [39m[32mIn[1][39m[32m, line 92[39m
[32m     89[39m df_analysis = df_analysis[~df_analysis[sector_col].isin([[33m'[39m[33m'[39m, [33m'[39m[33mnan[39m[33m'[39m, [33m'[39m[33mNaN[39m[33m'[39m])]
[32m     91[39m [38;5;66;03m# 5. Analysis: Crosstab and Entropy[39;00m
[32m---> [39m[32m92[39m ct = [43mpd[49m[43m.[49m[43mcrosstab[49m[43m([49m[43mdf_analysis[49m[43m[[49m[33;43m'[39;49m[33;43mDerived_Harm_Tags[39;49m[33;43m'[39;49m[43m][49m[43m,[49m[43m [49m[43mdf_analysis[49m[43m[[49m[43msector_col[49m[43m][49m[43m)[49m
[32m     93[39m probs = ct.div(ct.sum(axis=[32m1[39m), axis=[32m0[39m)
[32m     95[39m entropy_scores = probs.apply([38;5;28;01mlambda[39;00m x: scipy.stats.entropy(x, base=[32m2[39m), axis=[32m1[39m)

[36mFile [39m[32m/usr/local/lib/python3.13/site-packages/pandas/core/reshape/pivot.py:1099[39m, in [36mcrosstab[39m[34m(index, columns, values, rownames, colnames, aggfunc, margins, margins_name, dropna, normalize)[39m
[32m   1093[39m [38;5;28;01mfrom[39;00m[38;5;250m [39m[34;01mpandas[39;00m[38;5;250m [39m[38;5;28;01mimport[39;00m DataFrame
[32m   1095[39m data = {
[32m   1096[39m     **[38;5;28mdict[39m([38;5;28mzip[39m(unique_rownames, index, strict=[38;5;28;01mTrue[39;00m)),
[32m   1097[39m     **[38;5;28mdict[39m([38;5;28mzip[39m(unique_colnames, columns, strict=[38;5;28;01mTrue[39;00m)),
[32m   1098[39m }
[32m-> [39m[32m1099[39m df = [43mDataFrame[49m[43m([49m[43mdata[49m[43m,[49m[43m [49m[43mindex[49m[43m=[49m[43mcommon_idx[49m[43m)[49m
[32m   1101[39m [38;5;28;01mif[39;00m values [38;5;129;01mis[39;00m [38;5;28;01mNone[39;00m:
[32m   1102[39m     df[[33m"[39m[33m__dummy__[39m[33m"[39m] = [32m0[39m

[36mFile [39m[32m/usr/local/lib/python3.13/site-packages/pandas/core/frame.py:769[39m, in [36mDataFrame.__init__[39m[34m(self, data, index, columns, dtype, copy)[39m
[32m    763[39m     mgr = [38;5;28mself[39m._init_mgr(
[32m    764[39m         data, axes={[33m"[39m[33mindex[39m[33m"[39m: index, [33m"[39m[33mcolumns[39m[33m"[39m: columns}, dtype=dtype, copy=copy
[32m    765[39m     )
[32m    767[39m [38;5;28;01melif[39;00m [38;5;28misinstance[39m(data, [38;5;28mdict[39m):
[32m    768[39m     [38;5;66;03m# GH#38939 de facto copy defaults to False only in non-dict cases[39;00m
[32m--> [39m[32m769[39m     mgr = [43mdict_to_mgr[49m[43m([49m[43mdata[49m[43m,[49m[43m [49m[43mindex[49m[43m,[49m[43m [49m[43mcolumns[49m[43m,[49m[43m [49m[43mdtype[49m[43m=[49m[43mdtype[49m[43m,[49m[43m [49m[43mcopy[49m[43m=[49m[43mcopy[49m[43m)[49m
[32m    770[39m [38;5;28;01melif[39;00m [38;5;28misinstance[39m(data, ma.MaskedArray):
[32m    771[39m     [38;5;28;01mfrom[39;00m[38;5;250m [39m[34;01mnumpy[39;00m[34;01m.[39;00m[34;01mma[39;00m[38;5;250m [39m[38;5;28;01mimport[39;00m mrecords

[36mFile [39m[32m/usr/local/lib/python3.13/site-packages/pandas/core/internals/construction.py:447[39m, in [36mdict_to_mgr[39m[34m(data, index, columns, dtype, copy)[39m
[32m    428[39m [38;5;28;01mif[39;00m copy:
[32m    429[39m     [38;5;66;03m# We only need to copy arrays that will not get consolidated, i.e.[39;00m
[32m    430[39m     [38;5;66;03m#  only EA arrays[39;00m
[32m    431[39m     arrays = [
[32m    432[39m         (
[32m    433[39m             x.copy()
[32m   (...)[39m[32m    444[39m         [38;5;28;01mfor[39;00m x [38;5;129;01min[39;00m arrays
[32m    445[39m     ]
[32m--> [39m[32m447[39m [38;5;28;01mreturn[39;00m [43marrays_to_mgr[49m[43m([49m[43marrays[49m[43m,[49m[43m [49m[43mcolumns[49m[43m,[49m[43m [49m[43mindex[49m[43m,[49m[43m [49m[43mdtype[49m[43m=[49m[43mdtype[49m[43m,[49m[43m [49m[43mconsolidate[49m[43m=[49m[43mcopy[49m[43m)[49m

[36mFile [39m[32m/usr/local/lib/python3.13/site-packages/pandas/core/internals/construction.py:117[39m, in [36marrays_to_mgr[39m[34m(arrays, columns, index, dtype, verify_integrity, consolidate)[39m
[32m    114[39m         index = ensure_index(index)
[32m    116[39m     [38;5;66;03m# don't force copy because getting jammed in an ndarray anyway[39;00m
[32m--> [39m[32m117[39m     arrays, refs = [43m_homogenize[49m[43m([49m[43marrays[49m[43m,[49m[43m [49m[43mindex[49m[43m,[49m[43m [49m[43mdtype[49m[43m)[49m
[32m    118[39m     [38;5;66;03m# _homogenize ensures[39;00m
[32m    119[39m     [38;5;66;03m#  - all(len(x) == len(index) for x in arrays)[39;00m
[32m    120[39m     [38;5;66;03m#  - all(x.ndim == 1 for x in arrays)[39;00m
[32m   (...)[39m[32m    123[39m 
[32m    124[39m [38;5;28;01melse[39;00m:
[32m    125[39m     index = ensure_index(index)

[36mFile [39m[32m/usr/local/lib/python3.13/site-packages/pandas/core/internals/construction.py:555[39m, in [36m_homogenize[39m[34m(data, index, dtype)[39m
[32m    551[39m     val = val.astype(dtype)
[32m    552[39m [38;5;28;01mif[39;00m [38;5;28misinstance[39m(val, ABCSeries) [38;5;129;01mand[39;00m val.index [38;5;129;01mis[39;00m [38;5;129;01mnot[39;00m index:
[32m    553[39m     [38;5;66;03m# Forces alignment. No need to copy data since we[39;00m
[32m    554[39m     [38;5;66;03m# are putting it into an ndarray later[39;00m
[32m--> [39m[32m555[39m     val = [43mval[49m[43m.[49m[43mreindex[49m[43m([49m[43mindex[49m[43m)[49m
[32m    556[39m refs.append(val._references)
[32m    557[39m val = val._values

[36mFile [39m[32m/usr/local/lib/python3.13/site-packages/pandas/core/series.py:5525[39m, in [36mSeries.reindex[39m[34m(self, index, axis, method, copy, level, fill_value, limit, tolerance)[39m
[32m   5299[39m [38;5;28;01mdef[39;00m[38;5;250m [39m[34mreindex[39m(  [38;5;66;03m# type: ignore[override][39;00m
[32m   5300[39m     [38;5;28mself[39m,
[32m   5301[39m     index=[38;5;28;01mNone[39;00m,
[32m   (...)[39m[32m   5309[39m     tolerance=[38;5;28;01mNone[39;00m,
[32m   5310[39m ) -> Series:
[32m   5311[39m [38;5;250m    [39m[33;03m"""[39;00m
[32m   5312[39m [33;03m    Conform Series to new index with optional filling logic.[39;00m
[32m   5313[39m 
[32m   (...)[39m[32m   5523[39m [33;03m    See the :ref:`user guide <basics.reindexing>` for more.[39;00m
[32m   5524[39m [33;03m    """[39;00m
[32m-> [39m[32m5525[39m     [38;5;28;01mreturn[39;00m [38;5;28;43msuper[39;49m[43m([49m[43m)[49m[43m.[49m[43mreindex[49m[43m([49m
[32m   5526[39m [43m        [49m[43mindex[49m[43m=[49m[43mindex[49m[43m,[49m
[32m   5527[39m [43m        [49m[43mmethod[49m[43m=[49m[43mmethod[49m[43m,[49m
[32m   5528[39m [43m        [49m[43mlevel[49m[43m=[49m[43mlevel[49m[43m,[49m
[32m   5529[39m [43m        [49m[43mfill_value[49m[43m=[49m[43mfill_value[49m[43m,[49m
[32m   5530[39m [43m        [49m[43mlimit[49m[43m=[49m[43mlimit[49m[43m,[49m
[32m   5531[39m [43m        [49m[43mtolerance[49m[43m=[49m[43mtolerance[49m[43m,[49m
[32m   5532[39m [43m        [49m[43mcopy[49m[43m=[49m[43mcopy[49m[43m,[49m
[32m   5533[39m [43m    [49m[43m)[49m

[36mFile [39m[32m/usr/local/lib/python3.13/site-packages/pandas/core/generic.py:5476[39m, in [36mNDFrame.reindex[39m[34m(self, labels, index, columns, axis, method, copy, level, fill_value, limit, tolerance)[39m
[32m   5473[39m     [38;5;28;01mreturn[39;00m [38;5;28mself[39m._reindex_multi(axes, fill_value)
[32m   5475[39m [38;5;66;03m# perform the reindex on the axes[39;00m
[32m-> [39m[32m5476[39m [38;5;28;01mreturn[39;00m [38;5;28;43mself[39;49m[43m.[49m[43m_reindex_axes[49m[43m([49m
[32m   5477[39m [43m    [49m[43maxes[49m[43m,[49m[43m [49m[43mlevel[49m[43m,[49m[43m [49m[43mlimit[49m[43m,[49m[43m [49m[43mtolerance[49m[43m,[49m[43m [49m[43mmethod[49m[43m,[49m[43m [49m[43mfill_value[49m
[32m   5478[39m [43m[49m[43m)[49m.__finalize__([38;5;28mself[39m, method=[33m"[39m[33mreindex[39m[33m"[39m)

[36mFile [39m[32m/usr/local/lib/python3.13/site-packages/pandas/core/generic.py:5498[39m, in [36mNDFrame._reindex_axes[39m[34m(self, axes, level, limit, tolerance, method, fill_value)[39m
[32m   5495[39m     [38;5;28;01mcontinue[39;00m
[32m   5497[39m ax = [38;5;28mself[39m._get_axis(a)
[32m-> [39m[32m5498[39m new_index, indexer = [43max[49m[43m.[49m[43mreindex[49m[43m([49m
[32m   5499[39m [43m    [49m[43mlabels[49m[43m,[49m[43m [49m[43mlevel[49m[43m=[49m[43mlevel[49m[43m,[49m[43m [49m[43mlimit[49m[43m=[49m[43mlimit[49m[43m,[49m[43m [49m[43mtolerance[49m[43m=[49m[43mtolerance[49m[43m,[49m[43m [49m[43mmethod[49m[43m=[49m[43mmethod[49m
[32m   5500[39m [43m[49m[43m)[49m
[32m   5502[39m axis = [38;5;28mself[39m._get_axis_number(a)
[32m   5503[39m obj = obj._reindex_with_indexers(
[32m   5504[39m     {axis: [new_index, indexer]},
[32m   5505[39m     fill_value=fill_value,
[32m   5506[39m     allow_dups=[38;5;28;01mFalse[39;00m,
[32m   5507[39m )

[36mFile [39m[32m/usr/local/lib/python3.13/site-packages/pandas/core/indexes/base.py:4253[39m, in [36mIndex.reindex[39m[34m(self, target, method, level, limit, tolerance)[39m
[32m   4250[39m     [38;5;28;01mraise[39;00m [38;5;167;01mValueError[39;00m([33m"[39m[33mcannot handle a non-unique multi-index![39m[33m"[39m)
[32m   4251[39m [38;5;28;01melif[39;00m [38;5;129;01mnot[39;00m [38;5;28mself[39m.is_unique:
[32m   4252[39m     [38;5;66;03m# GH#42568[39;00m
[32m-> [39m[32m4253[39m     [38;5;28;01mraise[39;00m [38;5;167;01mValueError[39;00m([33m"[39m[33mcannot reindex on an axis with duplicate labels[39m[33m"[39m)
[32m   4254[39m [38;5;28;01melse[39;00m:
[32m   4255[39m     indexer, _ = [38;5;28mself[39m.get_indexer_non_unique(target)

[31mValueError[39m: cannot reindex on an axis with duplicate labels

ERROR: {'type': 'ValueError', 'message': 'cannot reindex on an axis with duplicate labels', 'traceback': 'Traceback (most recent call last):\n  File "/usr/local/lib/python3.13/site-packages/IPython/core/interactiveshell.py", line 3701, in run_code\n    exec(code_obj, self.user_global_ns, self.user_ns)\n    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "<ipython-input-1-76bd3464cb5f>", line 92, in <module>\n    ct = pd.crosstab(df_analysis[\'Derived_Harm_Tags\'], df_analysis[sector_col])\n  File "/usr/local/lib/python3.13/site-packages/pandas/core/reshape/pivot.py", line 1099, in crosstab\n    df = DataFrame(data, index=common_idx)\n  File "/usr/local/lib/python3.13/site-packages/pandas/core/frame.py", line 769, in __init__\n    mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy)\n  File "/usr/local/lib/python3.13/site-packages/pandas/core/internals/construction.py", line 447, in dict_to_mgr\n    return arrays_to_mgr(arrays, columns, index, dtype=dtype, consolidate=copy)\n  File "/usr/local/lib/python3.13/site-packages/pandas/core/internals/construction.py", line 117, in arrays_to_mgr\n    arrays, refs = _homogenize(arrays, index, dtype)\n                   ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^\n  File "/usr/local/lib/python3.13/site-packages/pandas/core/internals/construction.py", line 555, in _homogenize\n    val = val.reindex(index)\n  File "/usr/local/lib/python3.13/site-packages/pandas/core/series.py", line 5525, in reindex\n    return super().reindex(\n           ~~~~~~~~~~~~~~~^\n        index=index,\n        ^^^^^^^^^^^^\n    ...<5 lines>...\n        copy=copy,\n        ^^^^^^^^^^\n    )\n    ^\n  File "/usr/local/lib/python3.13/site-packages/pandas/core/generic.py", line 5476, in reindex\n    return self._reindex_axes(\n           ~~~~~~~~~~~~~~~~~~^\n        axes, level, limit, tolerance, method, fill_value\n        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n    ).__finalize__(self, method="reindex")\n    ^\n  File "/usr/local/lib/python3.13/site-packages/pandas/core/generic.py", line 5498, in _reindex_axes\n    new_index, indexer = ax.reindex(\n                         ~~~~~~~~~~^\n        labels, level=level, limit=limit, tolerance=tolerance, method=method\n        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n    )\n    ^\n  File "/usr/local/lib/python3.13/site-packages/pandas/core/indexes/base.py", line 4253, in reindex\n    raise ValueError("cannot reindex on an axis with duplicate labels")\nValueError: cannot reindex on an axis with duplicate labels\n'}
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
