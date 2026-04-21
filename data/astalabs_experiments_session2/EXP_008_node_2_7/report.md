# Experiment 8: node_2_7

| Property | Value |
|---|---|
| **Experiment ID** | `node_2_7` |
| **ID in Run** | 8 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T01:20:31.786102+00:00 |
| **Runtime** | 230.4s |
| **Parent** | `node_1_0` |
| **Children** | `node_3_4`, `node_3_14`, `node_3_21` |
| **Creation Index** | 9 |

---

## Hypothesis

> Autonomy-Harm Escalation: In real-world incidents, higher levels of AI autonomy
are positively correlated with higher severity of harm (Harm Level), independent
of the sector of deployment.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.6210 (Maybe True) |
| **Posterior** | 0.7555 (Likely True) |
| **Surprise** | +0.1614 |
| **Surprise Interpretation** | Mild Positive (hypothesis somewhat supported) |
| **Is Surprising?** | No |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 22.0 |
| Uncertain | 1.0 |
| Maybe False | 7.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 18.0 |
| Maybe True | 42.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Investigate the relationship between system autonomy and the severity of incident outcomes.

### Steps
- 1. Load `astalabs_discovery_all_data.csv` and filter for `aiid_incidents`.
- 2. Clean `Autonomy Level` into an ordinal scale (e.g., Low, Medium, High) and `AI Harm Level` into an ordinal scale (e.g., Minor, Moderate, Severe).
- 3. Perform a Spearman rank correlation test between Autonomy and Harm Level.
- 4. Run an Ordinal Logistic Regression predicting `AI Harm Level` using `Autonomy Level`, controlling for `Sector of Deployment` (as a categorical dummy).

### Deliverables
- Correlation coefficient; Regression summary table showing the odds ratio of increased harm severity per unit increase in autonomy.

---

## Analysis

The experiment successfully cleaned and analyzed the AIID dataset, resulting in
177 incidents with valid autonomy and harm classifications. The Spearman rank
correlation indicated a statistically significant, weak positive association
between Autonomy Level and AI Harm Level (rho = 0.2331, p = 0.0018). The Ordinal
Logistic Regression further supported this, yielding a significant positive
coefficient for Autonomy Level (0.4356, p = 0.031). This converts to an Odds
Ratio of approximately 1.55, suggesting that for every one-unit increase in
system autonomy (e.g., from Low to Medium), the odds of the incident falling
into a more severe harm category increase by 55%, holding the sector constant.
While the sector control variables showed signs of instability (large standard
errors) due to data sparsity, the primary relationship between autonomy and harm
severity was statistically significant, supporting the hypothesis.

---

## Review

The experiment successfully tested the Autonomy-Harm Escalation hypothesis using
the AIID dataset. After cleaning and mapping the specific taxonomy values
('Autonomy1-3', 'none/near-miss/issue/event'), 177 incidents remained for
analysis. The results support the hypothesis:

1. **Spearman Correlation**: A statistically significant positive correlation
was found between Autonomy Level and Harm Level (rho = 0.2331, p = 0.0018).
2. **Ordinal Logistic Regression**: The model showed a significant positive
coefficient for Autonomy Level (coef = 0.4356, p = 0.031). This translates to an
Odds Ratio of approx. 1.55, indicating that a one-unit increase in autonomy is
associated with a 55% increase in the odds of higher harm severity, controlling
for sector.

Note: The sector control variables exhibited extremely large coefficients and
standard errors, indicating data sparsity or quasi-complete separation for those
specific dummy variables. However, the primary variable of interest (Autonomy)
remained statistically significant.

---

## Code

```python
import pandas as pd
import numpy as np
import sys
import subprocess
from scipy.stats import spearmanr

# Try importing statsmodels, install if missing
try:
    from statsmodels.miscmodels.ordinal_model import OrderedModel
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "statsmodels"])
    from statsmodels.miscmodels.ordinal_model import OrderedModel

def run_experiment():
    # 1. Load Data
    file_path = 'astalabs_discovery_all_data.csv'
    print(f"Loading dataset from {file_path}...")
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        # Fallback for different directory structure if needed
        df = pd.read_csv('../astalabs_discovery_all_data.csv', low_memory=False)

    # 2. Filter for AIID Incidents
    df_aiid = df[df['source_table'] == 'aiid_incidents'].copy()
    print(f"AIID Incidents subset size: {len(df_aiid)}")

    # 3. Data Cleaning and Mapping
    col_autonomy = 'Autonomy Level'
    col_harm = 'AI Harm Level'
    col_sector = 'Sector of Deployment'

    # Check unique values for debugging if needed
    print(f"Unique Autonomy Values: {df_aiid[col_autonomy].unique()}")
    print(f"Unique Harm Values: {df_aiid[col_harm].unique()}")

    # Mapping Logic
    # Autonomy: Autonomy1 (Low), Autonomy2 (Medium), Autonomy3 (High)
    def map_autonomy(val):
        val = str(val).strip()
        if val == 'Autonomy1': return 1
        if val == 'Autonomy2': return 2
        if val == 'Autonomy3': return 3
        return np.nan

    # Harm: none (0), near-miss (1), issue (2), event (3)
    def map_harm(val):
        val = str(val).strip()
        if val == 'none': return 0
        if val == 'AI tangible harm near-miss': return 1
        if val == 'AI tangible harm issue': return 2
        if val == 'AI tangible harm event': return 3
        return np.nan

    df_aiid['autonomy_ord'] = df_aiid[col_autonomy].apply(map_autonomy)
    df_aiid['harm_ord'] = df_aiid[col_harm].apply(map_harm)

    # Clean Sector
    df_aiid['sector_clean'] = df_aiid[col_sector].fillna('Unknown')

    # Drop rows with NaN in key ordinal columns
    df_clean = df_aiid.dropna(subset=['autonomy_ord', 'harm_ord']).copy()
    print(f"Data points after cleaning mappings: {len(df_clean)}")

    if len(df_clean) < 10:
        print("Insufficient data points for regression.")
        return

    # 4. Analysis
    
    # Spearman Correlation
    corr, p = spearmanr(df_clean['autonomy_ord'], df_clean['harm_ord'])
    print(f"\n--- Spearman Correlation ---")
    print(f"Correlation: {corr:.4f}, p-value: {p:.4f}")

    # Ordinal Logistic Regression
    print(f"\n--- Ordinal Logistic Regression ---")
    
    # Prepare Sector Dummies (Top 5 sectors, others grouped)
    top_sectors = df_clean['sector_clean'].value_counts().nlargest(5).index
    df_clean['sector_group'] = df_clean['sector_clean'].apply(lambda x: x if x in top_sectors else 'Other')
    
    print(f"Using sectors: {list(top_sectors)} and 'Other'")

    # Independent variables: Autonomy (treated as continuous/ordinal trend) + Sector dummies
    # We convert autonomy to numeric to see the trend effect.
    # Using drop_first=True to avoid multicollinearity
    exog = pd.get_dummies(df_clean[['autonomy_ord', 'sector_group']], columns=['sector_group'], drop_first=True, dtype=float)
    
    # Dependent variable
    endog = df_clean['harm_ord'].astype(int)

    try:
        # Fit Ordered Logit Model
        # distribution 'logit' is standard for ordered logistic regression
        model = OrderedModel(endog, exog, distr='logit')
        res = model.fit(method='bfgs', disp=False)
        
        print(res.summary())
        
        print("\n--- Odds Ratios (Effect Size) ---")
        params = res.params
        conf = res.conf_int()
        conf['OR'] = params
        conf.columns = ['2.5%', '97.5%', 'OR']
        print(np.exp(conf))

    except Exception as e:
        print(f"Regression failed: {e}")

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Loading dataset from astalabs_discovery_all_data.csv...
AIID Incidents subset size: 1362
Unique Autonomy Values: <StringArray>
['Autonomy1', 'Autonomy2', 'Autonomy3', 'unclear', nan]
Length: 5, dtype: str
Unique Harm Values: <StringArray>
[                      'none',     'AI tangible harm event',
                    'unclear', 'AI tangible harm near-miss',
                          nan,     'AI tangible harm issue']
Length: 6, dtype: str
Data points after cleaning mappings: 177

--- Spearman Correlation ---
Correlation: 0.2331, p-value: 0.0018

--- Ordinal Logistic Regression ---
Using sectors: ['information and communication', 'transportation and storage', 'Arts, entertainment and recreation, information and communication', 'human health and social work activities', 'Arts, entertainment and recreation'] and 'Other'
                             OrderedModel Results                             
==============================================================================
Dep. Variable:               harm_ord   Log-Likelihood:                -138.16
Model:                   OrderedModel   AIC:                             294.3
Method:            Maximum Likelihood   BIC:                             322.9
Date:                Sun, 22 Feb 2026                                         
Time:                        01:22:22                                         
No. Observations:                 177                                         
Df Residuals:                     168                                         
Df Model:                           6                                         
==================================================================================================================================================
                                                                                     coef    std err          z      P>|z|      [0.025      0.975]
--------------------------------------------------------------------------------------------------------------------------------------------------
autonomy_ord                                                                       0.4356      0.202      2.157      0.031       0.040       0.831
sector_group_Arts, entertainment and recreation, information and communication    12.6878    605.204      0.021      0.983   -1173.491    1198.866
sector_group_Other                                                                14.1510    605.203      0.023      0.981   -1172.026    1200.328
sector_group_human health and social work activities                              14.0108    605.204      0.023      0.982   -1172.167    1200.188
sector_group_information and communication                                        13.3304    605.203      0.022      0.982   -1172.847    1199.507
sector_group_transportation and storage                                           16.1441    605.203      0.027      0.979   -1170.033    1202.321
0/1                                                                               15.6261    605.203      0.026      0.979   -1170.551    1201.803
1/2                                                                               -1.1833      0.325     -3.646      0.000      -1.819      -0.547
2/3                                                                               -1.2245      0.345     -3.546      0.000      -1.901      -0.548
==================================================================================================================================================

--- Odds Ratios (Effect Size) ---
                                                        2.5%  ...            OR
autonomy_ord                                        1.040551  ...  1.545848e+00
sector_group_Arts, entertainment and recreation...  0.000000  ...  3.237883e+05
sector_group_Other                                  0.000000  ...  1.398650e+06
sector_group_human health and social work activ...  0.000000  ...  1.215603e+06
sector_group_information and communication          0.000000  ...  6.156076e+05
sector_group_transportation and storage             0.000000  ...  1.026390e+07
0/1                                                 0.000000  ...  6.114002e+06
1/2                                                 0.162116  ...  3.062548e-01
2/3                                                 0.149375  ...  2.938950e-01

[9 rows x 3 columns]

STDERR:
/usr/local/lib/python3.13/site-packages/pandas/core/internals/blocks.py:347: RuntimeWarning: overflow encountered in exp
  result = func(self.values, **kwargs)

```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
