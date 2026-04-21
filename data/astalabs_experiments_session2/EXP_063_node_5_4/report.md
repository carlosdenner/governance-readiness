# Experiment 63: node_5_4

| Property | Value |
|---|---|
| **Experiment ID** | `node_5_4` |
| **ID in Run** | 63 |
| **Status** | SUCCEEDED |
| **Created** | 2026-02-22T04:02:09.434617+00:00 |
| **Runtime** | 218.5s |
| **Parent** | `node_4_20` |
| **Children** | `node_6_19`, `node_6_79` |
| **Creation Index** | 64 |

---

## Hypothesis

> Generative AI Incident Surge: The proportion of reported AI incidents involving
'Generative' technologies (e.g., LLMs, Diffusion) has significantly increased in
the post-2022 period compared to the pre-2022 period.

## Belief Shift

| Metric | Value |
|---|---|
| **Prior** | 0.9839 (Definitely True) |
| **Posterior** | 0.5055 (Uncertain) |
| **Surprise** | -0.5741 |
| **Surprise Interpretation** | Strong Negative (hypothesis contradicted) |
| **Is Surprising?** | Yes |

### Prior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 30.0 |
| Maybe True | 0.0 |
| Uncertain | 0.0 |
| Maybe False | 0.0 |
| Definitely False | 0.0 |

### Posterior Belief Distribution
| Category | Count |
|---|---|
| Definitely True | 0.0 |
| Maybe True | 0.0 |
| Uncertain | 2.0 |
| Maybe False | 58.0 |
| Definitely False | 0.0 |

---

## Experiment Plan

**Objective:** Quantify the temporal shift in the AI threat landscape towards Generative AI.

### Steps
- 1. Load `aiid_incidents`.
- 2. Parse the `date` column to extract the year.
- 3. Create a period variable: 'Post-2022' (Year >= 2023) vs 'Pre-2022' (Year <= 2022).
- 4. Classify `Known AI Technology` into 'Generative' (keywords: LLM, GPT, GenAI, Diffusion, Chatbot, Generative) vs 'Other'.
- 5. Create a contingency table of Period vs. Technology Type.
- 6. Perform a Chi-square test.

### Deliverables
- Time-period comparison table, Chi-square test results, and a plot of Generative AI incident proportions over time.

---

## Analysis

The experiment successfully analyzed 240 AI incidents (filtered from 1,362 due
to missing 'Known AI Technology' labels or dates) to test the hypothesis of a
post-2022 surge in Generative AI incidents.

**Findings:**
- **Pre-2022:** 4.42% of incidents involved generative technology (10 out of
226).
- **Post-2022:** 7.14% of incidents involved generative technology (1 out of
14).

**Statistical Test:**
- The Chi-square test yielded a statistic of 0.00 and a **p-value of 1.0**,
indicating no statistically significant difference between the two periods.
- The analysis was heavily constrained by data quality: over 80% of the dataset
was excluded because the 'Known AI Technology' field was empty. The Post-2022
sample size (N=14) was insufficient to draw robust conclusions.

**Conclusion:**
The hypothesis that the proportion of Generative AI incidents has
*significantly* increased is **rejected** based on this specific dataset slice.
While the raw proportion nominally increased (from 4.4% to 7.1%), the result is
not statistically distinguishable from chance due to the sparse data labeling.

---

## Review

The experiment successfully analyzed 240 AI incidents (filtered from 1,362 due
to missing 'Known AI Technology' labels or dates) to test the hypothesis of a
post-2022 surge in Generative AI incidents.

**Findings:**
- **Pre-2022:** 4.42% of incidents involved generative technology (10 out of
226).
- **Post-2022:** 7.14% of incidents involved generative technology (1 out of
14).

**Statistical Test:**
- The Chi-square test yielded a statistic of 0.00 and a **p-value of 1.0**,
indicating no statistically significant difference between the two periods.
- The analysis was heavily constrained by data quality: over 80% of the dataset
was excluded because the 'Known AI Technology' field was empty. The Post-2022
sample size (N=14) was insufficient to draw robust conclusions.

**Conclusion:**
The hypothesis that the proportion of Generative AI incidents has
*significantly* increased is **rejected** based on this specific dataset slice.
While the raw proportion nominally increased (from 4.4% to 7.1%), the result is
not statistically distinguishable from chance due to the sparse data labeling.

---

## Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import os

def run_experiment():
    print("Starting Generative AI Incident Surge analysis...")
    
    # Load dataset
    file_path = '../astalabs_discovery_all_data.csv'
    if not os.path.exists(file_path):
        # Fallback for local testing if needed
        file_path = 'astalabs_discovery_all_data.csv'
    
    try:
        # Low memory=False to handle mixed types warning from previous steps
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Filter for AIID incidents
    aiid_df = df[df['source_table'] == 'aiid_incidents'].copy()
    print(f"AIID Incidents loaded (raw): {len(aiid_df)}")

    # Clean column names
    aiid_df.columns = [c.strip() for c in aiid_df.columns]
    
    # Check for required columns
    # Based on previous exploration, 'date' and 'Known AI Technology' should exist
    required_cols = ['date', 'Known AI Technology']
    missing_cols = [c for c in required_cols if c not in aiid_df.columns]
    if missing_cols:
        print(f"Error: Missing columns {missing_cols}. Available: {aiid_df.columns.tolist()}")
        return

    # 1. Parse Date and Extract Year
    aiid_df['date_parsed'] = pd.to_datetime(aiid_df['date'], errors='coerce')
    # Drop rows without valid dates
    aiid_df = aiid_df.dropna(subset=['date_parsed'])
    aiid_df['year'] = aiid_df['date_parsed'].dt.year
    
    # 2. Filter for Known Technology (Drop Nulls to analyze only identified tech)
    # This ensures we are comparing 'Generative' vs 'Other Known Tech'
    aiid_clean = aiid_df.dropna(subset=['Known AI Technology']).copy()
    print(f"Incidents with valid Date and Known Technology: {len(aiid_clean)}")

    # 3. Classify Technology
    # Keywords provided in prompt
    gen_keywords = ['LLM', 'GPT', 'GenAI', 'Diffusion', 'Chatbot', 'Generative']
    
    def is_generative(val):
        val_str = str(val).lower()
        return any(kw.lower() in val_str for kw in gen_keywords)

    aiid_clean['Is_Generative'] = aiid_clean['Known AI Technology'].apply(is_generative)

    # 4. Create Period Variable
    # Post-2022 (Year >= 2023) vs Pre-2022 (Year <= 2022)
    aiid_clean['Period'] = np.where(aiid_clean['year'] >= 2023, 'Post-2022', 'Pre-2022')
    
    # 5. Contingency Table
    contingency = pd.crosstab(aiid_clean['Period'], aiid_clean['Is_Generative'])
    
    # Rename columns/index for clarity
    if True in contingency.columns and False in contingency.columns:
        contingency = contingency.rename(columns={False: 'Other', True: 'Generative'})
    elif True in contingency.columns:
        contingency = contingency.rename(columns={True: 'Generative'})
        contingency['Other'] = 0
    else:
        contingency = contingency.rename(columns={False: 'Other'})
        contingency['Generative'] = 0
        
    # Ensure row order
    desired_order = ['Pre-2022', 'Post-2022']
    contingency = contingency.reindex(desired_order).fillna(0)

    print("\n--- Contingency Table (Period vs Technology) ---")
    print(contingency)

    # 6. Statistical Test (Chi-Square)
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-Square Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4e}")

    # Calculate Proportions and Odds Ratio
    try:
        pre_gen = contingency.loc['Pre-2022', 'Generative']
        pre_tot = contingency.loc['Pre-2022'].sum()
        post_gen = contingency.loc['Post-2022', 'Generative']
        post_tot = contingency.loc['Post-2022'].sum()
        
        prop_pre = pre_gen / pre_tot if pre_tot > 0 else 0
        prop_post = post_gen / post_tot if post_tot > 0 else 0
        
        print(f"\nProportion Generative (Pre-2022): {prop_pre:.2%}")
        print(f"Proportion Generative (Post-2022): {prop_post:.2%}")
        
        # Odds Ratio
        # (Gen_Post / Other_Post) / (Gen_Pre / Other_Pre)
        odds_post = post_gen / contingency.loc['Post-2022', 'Other'] if contingency.loc['Post-2022', 'Other'] > 0 else np.nan
        odds_pre = pre_gen / contingency.loc['Pre-2022', 'Other'] if contingency.loc['Pre-2022', 'Other'] > 0 else np.nan
        
        if odds_pre > 0:
            print(f"Odds Ratio: {odds_post / odds_pre:.4f}")
        else:
            print("Odds Ratio: Undefined (Zero denominator in Pre-2022)")
            
    except Exception as e:
        print(f"Error calculating stats: {e}")

    # 7. Visualization
    # Group by year to show trend
    yearly = aiid_clean.groupby('year')['Is_Generative'].agg(['sum', 'count'])
    yearly['proportion'] = yearly['sum'] / yearly['count']
    
    # Filter to relevant timeline (e.g. 2015-2024) to avoid noisy early years
    plot_data = yearly[yearly.index >= 2015]
    
    if not plot_data.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(plot_data.index, plot_data['proportion'], marker='o', linestyle='-', linewidth=2, color='darkblue')
        plt.title('Proportion of AI Incidents Involving Generative Technologies (2015-2024)')
        plt.xlabel('Year')
        plt.ylabel('Proportion (Generative / Total Known)')
        plt.axvline(x=2022.5, color='red', linestyle='--', label='End of 2022')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough data to generate plot.")

if __name__ == "__main__":
    run_experiment()
```

## Code Output

```
exitcode: 0 (execution succeeded)
Code output: Starting Generative AI Incident Surge analysis...
AIID Incidents loaded (raw): 1362
Incidents with valid Date and Known Technology: 240

--- Contingency Table (Period vs Technology) ---
Is_Generative  Other  Generative
Period                          
Pre-2022         216          10
Post-2022         13           1

Chi-Square Statistic: 0.0000
P-value: 1.0000e+00

Proportion Generative (Pre-2022): 4.42%
Proportion Generative (Post-2022): 7.14%
Odds Ratio: 1.6615


=== Plot Analysis (figure 1) ===
Based on the provided plot image, here is a detailed analysis:

### 1. Plot Type
*   **Type:** Time-series line plot with markers.
*   **Purpose:** The plot tracks the temporal evolution of a specific metric—the ratio of AI incidents involving generative technologies relative to the total number of known AI incidents—over a 10-year period.

### 2. Axes
*   **X-axis (Horizontal):**
    *   **Label:** "Year"
    *   **Range:** The axis spans from **2015 to 2024**.
    *   **Scale:** Linear, with major tick marks every two years (2016, 2018, 2020, 2022, 2024).
*   **Y-axis (Vertical):**
    *   **Label:** "Proportion (Generative / Total Known)"
    *   **Range:** Values range from **0.00 to roughly 0.16**.
    *   **Units:** The values represent a proportion (a dimensionless ratio between 0 and 1).

### 3. Data Trends
The data exhibits high volatility rather than a steady linear trend. Key observations include:
*   **2015–2016 (Zero Baseline):** The proportion starts at 0.00, indicating no recorded incidents involving generative technologies during these years.
*   **2017–2021 (Sporadic Activity):**
    *   There is a localized peak in **2017** (approx. 0.05) followed by a return to nearly zero in **2018**.
    *   Another significant rise occurs in **2019** (approx. 0.083), followed by a decline through **2020** (0.04) and **2021** (approx. 0.03).
*   **2022 (The Surge):** The most prominent feature of the plot is a dramatic spike in **2022**, reaching the global maximum of approximately **0.158**. This indicates that nearly 16% of all known AI incidents that year involved generative technologies.
*   **2023–2024 (Decline):** Following the 2022 peak, the proportion drops to roughly **0.125** in 2023. In **2024**, the value sharply returns to **0.00**.

### 4. Annotations and Legends
*   **Vertical Dashed Line:** A vertical red dashed line is positioned between 2022 and 2023 (specifically at the end of the year 2022).
*   **Legend:** A legend in the top right corner identifies the red dashed line as representing the **"End of 2022."** This annotation serves as a temporal marker, likely distinguishing a specific era or event shift in the AI landscape.
*   **Grid:** A light grey grid is overlaid on the plot to assist in estimating values.

### 5. Statistical Insights
*   **Correlation with Technological Adoption:** The sharp increase peaking in 2022 correlates strongly with the widespread public release and adoption of major generative AI models (e.g., ChatGPT, Midjourney, Stable Diffusion) which occurred in late 2022. The vertical red line ("End of 2022") effectively marks the transition into the "Generative AI era."
*   **Volatility of Incidents:** The "sawtooth" pattern between 2017 and 2021 suggests that early incidents involving generative AI were sporadic events rather than a systemic trend.
*   **The 2024 Outlier:** The drop to 0.00 in 2024 is statistically notable but requires cautious interpretation. It likely represents incomplete data collection for the current year (truncated data) rather than a genuine disappearance of generative AI incidents, given the technology's continued prevalence.
*   **Trend Significance:** While the proportion was negligible prior to 2019, the post-2021 period shows a shifting baseline where generative technologies account for a double-digit percentage (12-16%) of all AI incidents, signifying a major shift in the nature of AI risks.
==================================================
```

---

## Reproducibility Notes

- Raw JSON: `experiment.json`
- Executable code: `code.py`
- Original output: `code_output.txt`
- To reproduce figures locally, run `code.py` with the dataset at
  `data/astalabs/astalabs_discovery_all_data.csv` in the working directory.
