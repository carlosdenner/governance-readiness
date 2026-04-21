"""
Elaine Review Analyses — Feb 25, 2026
Runs four analyses requested by co-author review:

  Analysis 1 (C3): Split-TR pathway model
    Re-run Table 2 regressions with TR split into:
      TR-surface (0-2): tr_ato, tr_internal_rev
      TR-substantive (0-7): tr_impact_assess, tr_rw_testing, tr_indep_eval,
                            tr_postdeploy, tr_notice, tr_disparity, tr_appeal

  Analysis 2 (C2): TR-surface vs TR-substantive by risk tier
    Cross-tab comparing surface/substantive prevalence for rights/safety
    vs non-rights/safety use cases.

  Analysis 3 (C5): Mediation test
    Procurement -> IR (implementability) -> TR-substantive
    Does IR mediate the vendor effect on substantive safeguards?

  Analysis 4: "Opacity" granularity check (C4)
    Which specific safeguard items show vendor deficits and which don't?
    To determine if "opacity" is too broad a term.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─── Load Data ───────────────────────────────────────────────────────────────
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

df = pd.read_csv("data/processed/pathway_model_data.csv")
print(f"Loaded pathway_model_data.csv: {len(df)} rows, {len(df.columns)} cols")
print(f"Columns: {list(df.columns)}\n")

# ─── Define TR sub-indices ───────────────────────────────────────────────────
TR_SURFACE = ["tr_ato", "tr_internal_rev"]
TR_SUBSTANTIVE = ["tr_impact_assess", "tr_rw_testing", "tr_indep_eval",
                   "tr_postdeploy", "tr_notice", "tr_disparity", "tr_appeal"]

# Check columns exist
for c in TR_SURFACE + TR_SUBSTANTIVE:
    assert c in df.columns, f"Missing column: {c}"

df["TR_surface"] = df[TR_SURFACE].sum(axis=1)
df["TR_substantive"] = df[TR_SUBSTANTIVE].sum(axis=1)
df["TR_sub_x_IR"] = df["TR_substantive"] * df["IR"]
df["TR_surf_x_IR"] = df["TR_surface"] * df["IR"]

print(f"TR_surface  (0-2): mean={df['TR_surface'].mean():.3f}, median={df['TR_surface'].median():.1f}")
print(f"TR_substantive (0-7): mean={df['TR_substantive'].mean():.3f}, median={df['TR_substantive'].median():.1f}")
print(f"TR composite   (0-9): mean={df['TR'].mean():.3f}, median={df['TR'].median():.1f}")
print(f"IR             (0-7): mean={df['IR'].mean():.3f}, median={df['IR'].median():.1f}")
print()

# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 (C3): Split-TR Pathway Model
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ANALYSIS 1: SPLIT-TR PATHWAY MODEL")
print("Re-running Table 2 with TR-surface and TR-substantive as separate predictors")
print("=" * 80)

y = df["operational"]

# M1: Controls only (unchanged baseline)
X1 = sm.add_constant(df[["vendor", "mixed_dev", "rights_safety"]])
m1 = Logit(y, X1).fit(disp=0)

# M2: + Orientation (unchanged)
X2 = sm.add_constant(df[["vendor", "mixed_dev", "rights_safety", "orientation"]])
m2 = Logit(y, X2).fit(disp=0)

# M3-original: + composite TR + IR (for comparison)
X3_orig = sm.add_constant(df[["vendor", "mixed_dev", "rights_safety", "orientation", "TR", "IR"]])
m3_orig = Logit(y, X3_orig).fit(disp=0)

# M3-split: + TR_surface + TR_substantive + IR
X3_split = sm.add_constant(df[["vendor", "mixed_dev", "rights_safety", "orientation",
                                "TR_surface", "TR_substantive", "IR"]])
m3_split = Logit(y, X3_split).fit(disp=0)

# M4-original: + TR x IR interaction (for comparison)
X4_orig = sm.add_constant(df[["vendor", "mixed_dev", "rights_safety", "orientation",
                               "TR", "IR", "TR_x_IR"]])
m4_orig = Logit(y, X4_orig).fit(disp=0)

# M4-split: + TR_sub x IR interaction only
X4_split = sm.add_constant(df[["vendor", "mixed_dev", "rights_safety", "orientation",
                                "TR_surface", "TR_substantive", "IR", "TR_sub_x_IR"]])
m4_split = Logit(y, X4_split).fit(disp=0)

# M5: Full split with both interactions
X5 = sm.add_constant(df[["vendor", "mixed_dev", "rights_safety", "orientation",
                          "TR_surface", "TR_substantive", "IR",
                          "TR_surf_x_IR", "TR_sub_x_IR"]])
m5 = Logit(y, X5).fit(disp=0)

print("\n--- Original M3 (composite TR + IR) ---")
print(m3_orig.summary2().tables[1].to_string())
print(f"  Pseudo R²={m3_orig.prsquared:.4f}  AIC={m3_orig.aic:.1f}")

print("\n--- Split M3 (TR-surface + TR-substantive + IR) ---")
print(m3_split.summary2().tables[1].to_string())
print(f"  Pseudo R²={m3_split.prsquared:.4f}  AIC={m3_split.aic:.1f}")

print("\n--- Original M4 (composite TR x IR) ---")
print(m4_orig.summary2().tables[1].to_string())
print(f"  Pseudo R²={m4_orig.prsquared:.4f}  AIC={m4_orig.aic:.1f}")

print("\n--- Split M4 (TR-sub x IR) ---")
print(m4_split.summary2().tables[1].to_string())
print(f"  Pseudo R²={m4_split.prsquared:.4f}  AIC={m4_split.aic:.1f}")

print("\n--- M5 (both interactions: TR-surf x IR + TR-sub x IR) ---")
print(m5.summary2().tables[1].to_string())
print(f"  Pseudo R²={m5.prsquared:.4f}  AIC={m5.aic:.1f}")

# Odds ratios for key split models
print("\n--- ODDS RATIOS: Split M3 ---")
for var in ["TR_surface", "TR_substantive", "IR"]:
    coef = m3_split.params[var]
    ci = m3_split.conf_int().loc[var]
    p = m3_split.pvalues[var]
    print(f"  {var:20s}: OR={np.exp(coef):.3f}  95%CI [{np.exp(ci[0]):.3f}, {np.exp(ci[1]):.3f}]  p={p:.4f}")

print("\n--- ODDS RATIOS: Split M4 ---")
for var in ["TR_surface", "TR_substantive", "IR", "TR_sub_x_IR"]:
    coef = m4_split.params[var]
    ci = m4_split.conf_int().loc[var]
    p = m4_split.pvalues[var]
    print(f"  {var:20s}: OR={np.exp(coef):.3f}  95%CI [{np.exp(ci[0]):.3f}, {np.exp(ci[1]):.3f}]  p={p:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 (C2): TR-surface vs TR-substantive by Risk Tier
# ═════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("ANALYSIS 2: TR-SURFACE vs TR-SUBSTANTIVE BY RISK TIER")
print("=" * 80)

# Risk tier groups
df["risk_tier"] = np.where(df["rights_safety"] == 1, "Rights/Safety", "Non-flagged")
groups = df.groupby("risk_tier")

print("\n--- Prevalence of individual TR items by risk tier ---")
all_tr = TR_SURFACE + TR_SUBSTANTIVE
for col in all_tr:
    tier = "surface" if col in TR_SURFACE else "SUBSTANTIVE"
    rates = groups[col].mean()
    chi2, p = stats.chi2_contingency(pd.crosstab(df["risk_tier"], df[col]))[0:2]
    print(f"  [{tier:11s}] {col:22s}: Non-flagged={rates.get('Non-flagged',0)*100:5.1f}%  "
          f"Rights/Safety={rates.get('Rights/Safety',0)*100:5.1f}%  chi2={chi2:.2f} p={p:.4f}")

print("\n--- Mean sub-indices by risk tier ---")
for idx_name in ["TR_surface", "TR_substantive", "TR", "IR"]:
    means = groups[idx_name].mean()
    t_stat, t_p = stats.ttest_ind(
        df.loc[df["rights_safety"] == 1, idx_name],
        df.loc[df["rights_safety"] == 0, idx_name]
    )
    print(f"  {idx_name:20s}: Non-flagged={means.get('Non-flagged',0):.3f}  "
          f"Rights/Safety={means.get('Rights/Safety',0):.3f}  t={t_stat:.2f} p={t_p:.4f}")

# Key question: Does risk-tiering rescue substantive safeguards?
print("\n--- Governance theater prevalence by risk tier ---")
df["has_surface"] = (df["TR_surface"] > 0).astype(int)
df["has_substantive"] = (df["TR_substantive"] > 0).astype(int)
df["theater"] = ((df["has_surface"] == 1) & (df["has_substantive"] == 0)).astype(int)

theater_rates = df.groupby("risk_tier")[["has_surface", "has_substantive", "theater"]].mean()
print(theater_rates.round(3).to_string())


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 (C5): Mediation Test — Procurement → IR → TR-substantive
# ═════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("ANALYSIS 3: MEDIATION TEST — Procurement → IR → TR-substantive")
print("Baron-Kenny steps + Sobel test")
print("=" * 80)

# Filter to known dev_method (same as procurement script)
med_df = df[df["vendor"].notna() & df["mixed_dev"].notna()].copy()
# Further filter: only vendor vs in-house (drop mixed for clean comparison)
# Actually, keep mixed as control like the original procurement script
med_df = med_df.dropna(subset=["vendor", "mixed_dev", "rights_safety", "operational"])
print(f"Mediation sample: n={len(med_df)}")

# Binary outcome: has any substantive safeguard
med_df["any_substantive"] = (med_df["TR_substantive"] > 0).astype(int)
print(f"  any_substantive prevalence: {med_df['any_substantive'].mean()*100:.1f}%")

controls = ["mixed_dev", "rights_safety", "operational"]

# Step 1 (Path c): Vendor → TR-substantive (total effect)
print("\n--- Step 1 (Path c): Vendor → any_substantive ---")
Xc = sm.add_constant(med_df[["vendor"] + controls])
yc = med_df["any_substantive"]
mc = Logit(yc, Xc).fit(disp=0)
coef_c = mc.params["vendor"]
se_c = mc.bse["vendor"]
or_c = np.exp(coef_c)
p_c = mc.pvalues["vendor"]
print(f"  Vendor coef={coef_c:.4f}  OR={or_c:.3f}  p={p_c:.4f}")

# Step 2 (Path a): Vendor → IR (mediator)
print("\n--- Step 2 (Path a): Vendor → IR ---")
Xa = sm.add_constant(med_df[["vendor"] + controls])
ya = med_df["IR"]
ma = sm.OLS(ya, Xa).fit()
coef_a = ma.params["vendor"]
se_a = ma.bse["vendor"]
p_a = ma.pvalues["vendor"]
print(f"  Vendor coef={coef_a:.4f}  p={p_a:.4f}")

# Step 3 (Path c'): Vendor → TR-substantive controlling for IR
print("\n--- Step 3 (Path c'): Vendor → any_substantive | IR ---")
Xcp = sm.add_constant(med_df[["vendor", "IR"] + controls])
mcp = Logit(yc, Xcp).fit(disp=0)
coef_cp = mcp.params["vendor"]
se_cp = mcp.bse["vendor"]
or_cp = np.exp(coef_cp)
p_cp = mcp.pvalues["vendor"]
coef_b = mcp.params["IR"]
se_b = mcp.bse["IR"]
or_b = np.exp(coef_b)
p_b = mcp.pvalues["IR"]
print(f"  Vendor (c') coef={coef_cp:.4f}  OR={or_cp:.3f}  p={p_cp:.4f}")
print(f"  IR     (b)  coef={coef_b:.4f}  OR={or_b:.3f}  p={p_b:.4f}")

# Attenuation
if coef_c != 0:
    attenuation = (1 - coef_cp / coef_c) * 100
    print(f"\n  Attenuation of vendor effect: {attenuation:.1f}%")
    print(f"  (c={coef_c:.4f} → c'={coef_cp:.4f})")

# Sobel test
sobel_se = np.sqrt(coef_a**2 * se_b**2 + coef_b**2 * se_a**2)
sobel_z = (coef_a * coef_b) / sobel_se
sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))
print(f"\n  Sobel test: z={sobel_z:.3f}  p={sobel_p:.4f}")
print(f"  Indirect effect (a*b) = {coef_a * coef_b:.4f}")

# Also test: Vendor → TR-substantive count (not just binary)
print("\n--- Supplementary: Vendor → TR_substantive count (Poisson) ---")
from statsmodels.discrete.discrete_model import Poisson
Xp = sm.add_constant(med_df[["vendor", "IR"] + controls])
yp = med_df["TR_substantive"]
mp = Poisson(yp, Xp).fit(disp=0)
print(f"  Vendor coef={mp.params['vendor']:.4f}  IRR={np.exp(mp.params['vendor']):.3f}  p={mp.pvalues['vendor']:.4f}")
print(f"  IR     coef={mp.params['IR']:.4f}  IRR={np.exp(mp.params['IR']):.3f}  p={mp.pvalues['IR']:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4 (C4): Granular vendor effects per safeguard item
# ═════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("ANALYSIS 4: GRANULAR VENDOR EFFECTS PER SAFEGUARD ITEM")
print("Which items show vendor deficit (evaluability) vs. no deficit (external)?")
print("=" * 80)

# All TR + IR items
all_items = TR_SURFACE + TR_SUBSTANTIVE + [
    "ir_data_catalog", "ir_data_docs", "ir_custom_code",
    "ir_code_access", "ir_infra", "ir_timely_res", "ir_reuse"
]

# Check which exist
available_items = [c for c in all_items if c in med_df.columns]

print(f"\n{'Item':28s} {'Vendor%':>8s} {'InHouse%':>9s} {'Diff':>7s} {'OR':>7s} {'p':>8s} {'Type':>12s}")
print("-" * 90)

for item in available_items:
    # Bivariate: vendor only
    Xv = sm.add_constant(med_df[["vendor"]])
    yv = med_df[item]

    try:
        mv = Logit(yv, Xv).fit(disp=0)
        vendor_coef = mv.params["vendor"]
        vendor_or = np.exp(vendor_coef)
        vendor_p = mv.pvalues["vendor"]
    except Exception:
        vendor_or = np.nan
        vendor_p = np.nan

    rate_vendor = med_df.loc[med_df["vendor"] == 1, item].mean() * 100
    rate_inhouse = med_df.loc[med_df["vendor"] == 0, item].mean() * 100
    diff = rate_vendor - rate_inhouse

    if item in TR_SURFACE:
        item_type = "TR-surface"
    elif item in TR_SUBSTANTIVE:
        item_type = "TR-subst."
    else:
        item_type = "IR"

    sig = "***" if vendor_p < 0.001 else "**" if vendor_p < 0.01 else "*" if vendor_p < 0.05 else ""
    print(f"  {item:26s} {rate_vendor:7.1f}% {rate_inhouse:8.1f}% {diff:+6.1f}% {vendor_or:6.2f} {vendor_p:7.4f}{sig:3s} {item_type}")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("SUMMARY OF KEY FINDINGS FOR PAPER REVISION")
print("=" * 80)

print(f"""
1. SPLIT-TR REGRESSION (C3):
   - TR-surface effect on deployment: coef={m3_split.params['TR_surface']:.4f}, p={m3_split.pvalues['TR_surface']:.4f}
   - TR-substantive effect on deployment: coef={m3_split.params['TR_substantive']:.4f}, p={m3_split.pvalues['TR_substantive']:.4f}
   - IR effect: coef={m3_split.params['IR']:.4f}, p={m3_split.pvalues['IR']:.4f}
   - AIC improvement: Original M3={m3_orig.aic:.1f}, Split M3={m3_split.aic:.1f}

2. RISK-TIER ANALYSIS (C2):
   - Non-flagged theater rate: {df.loc[df['rights_safety']==0, 'theater'].mean()*100:.1f}%
   - Rights/Safety theater rate: {df.loc[df['rights_safety']==1, 'theater'].mean()*100:.1f}%
   - Surface compliance (Non-flagged): {df.loc[df['rights_safety']==0, 'has_surface'].mean()*100:.1f}%
   - Surface compliance (Rights/Safety): {df.loc[df['rights_safety']==1, 'has_surface'].mean()*100:.1f}%
   - Any substantive (Non-flagged): {df.loc[df['rights_safety']==0, 'has_substantive'].mean()*100:.1f}%
   - Any substantive (Rights/Safety): {df.loc[df['rights_safety']==1, 'has_substantive'].mean()*100:.1f}%

3. MEDIATION (C5):
   - Total effect (c): Vendor OR={or_c:.3f}, p={p_c:.4f}
   - Direct effect (c'): Vendor OR={or_cp:.3f}, p={p_cp:.4f}
   - IR mediator (b): OR={or_b:.3f}, p={p_b:.4f}
   - Attenuation: {attenuation:.1f}%
   - Sobel z={sobel_z:.3f}, p={sobel_p:.4f}

4. GRANULAR VENDOR EFFECTS (C4):
   See table above for item-level vendor effects.
""")

print("Done. Results ready for paper revision decisions.")
