"""
09_pathway_model.py
====================
Agency-level and use-case-level regression models testing the
Orientation → TR × IR → Value pathway using EO 13960 proxy variables.

Constructs
----------
Orientation (agency-level):
    - log(portfolio_size)     : log of n use cases per agency
    - topic_breadth           : n distinct topic areas per agency

Trust Readiness (TR) index (use-case level, 0-9 count):
    40_has_ato, 50_internal_review, 52_impact_assessment,
    53_real_world_testing, 55_independent_eval, 56_monitor_postdeploy,
    59_ai_notice, 62_disparity_mitigation, 65_appeal_process

Integration Readiness (IR) index (use-case level, 0-7 count):
    31_data_catalog, 34_data_docs, 37_custom_code, 38_code_access,
    43_infra_provisioned, 47_timely_resources, 49_existing_reuse

Value proxy (use-case level):
    operational  : 1 if dev_stage in {Operation and Maintenance,
                   Implementation and Assessment, In production, In mission}

Controls:
    vendor       : 1 if dev_method contains "contracting" (not "both")
    rights_safety: 1 if impact_type != "Neither"
    agency_size  : log(n use cases)  [for use-case model only]
"""

from __future__ import annotations
import pathlib, re, sys, textwrap
import numpy as np
import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent.parent
RAW_EO = BASE / "data" / "raw" / "eo13960" / "2024_consolidated_ai_inventory_raw.csv"
OUT_DIR = BASE / "data" / "processed"
FIG_DIR = BASE / "paper" / "figures"

# ── helpers ──────────────────────────────────────────────────────────────────

def _load():
    df = pd.read_csv(RAW_EO, low_memory=False)
    df.columns = [re.sub(r"\s+", "_", c.strip().lower().replace("(","").replace(")","")) for c in df.columns]
    return df

def _is_yes(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip().str.lower()
    return (~s.isin(["", "no", "n/a", "nan", "none", "na", "false"])).astype(int)

def _is_yes_strict(series: pd.Series) -> pd.Series:
    """For columns where we need 'Yes' specifically (data_catalog, infra, etc.)."""
    s = series.fillna("").astype(str).str.strip().str.lower()
    return s.str.startswith("yes").astype(int)

def _is_reuse(series: pd.Series) -> pd.Series:
    """For 49_existing_reuse: 1 if NOT 'none' and NOT blank."""
    s = series.fillna("").astype(str).str.strip().str.lower()
    return ((s != "") & (~s.str.startswith("none"))).astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
#  Build dataset
# ═══════════════════════════════════════════════════════════════════════════════

def build_dataset():
    raw = _load()
    n = len(raw)
    assert n == 1757

    # ── TR indicators (9 items) ──
    tr_cols = {
        "tr_ato":            ("40_has_ato",              _is_yes),
        "tr_internal_rev":   ("50_internal_review",      _is_yes),
        "tr_impact_assess":  ("52_impact_assessment",    _is_yes),
        "tr_rw_testing":     ("53_real_world_testing",   _is_yes),
        "tr_indep_eval":     ("55_independent_eval",     _is_yes),
        "tr_postdeploy":     ("56_monitor_postdeploy",   _is_yes),
        "tr_notice":         ("59_ai_notice",            _is_yes),
        "tr_disparity":      ("62_disparity_mitigation", _is_yes),
        "tr_appeal":         ("65_appeal_process",       _is_yes),
    }
    for new_col, (src_col, fn) in tr_cols.items():
        raw[new_col] = fn(raw[src_col])
    tr_item_cols = list(tr_cols.keys())
    raw["TR"] = raw[tr_item_cols].sum(axis=1)

    # ── IR indicators (7 items) ──
    ir_cols = {
        "ir_data_catalog":   ("31_data_catalog",       _is_yes_strict),
        "ir_data_docs":      ("34_data_docs",          _is_yes),
        "ir_custom_code":    ("37_custom_code",        _is_yes_strict),
        "ir_code_access":    ("38_code_access",        _is_yes),
        "ir_infra":          ("43_infra_provisioned",   _is_yes_strict),
        "ir_timely_res":     ("47_timely_resources",    _is_yes_strict),
        "ir_reuse":          ("49_existing_reuse",      _is_reuse),
    }
    for new_col, (src_col, fn) in ir_cols.items():
        raw[new_col] = fn(raw[src_col])
    ir_item_cols = list(ir_cols.keys())
    raw["IR"] = raw[ir_item_cols].sum(axis=1)

    # ── Outcome: operational deployment ──
    stage = raw["16_dev_stage"].fillna("").astype(str).str.strip().str.lower()
    raw["operational"] = stage.isin([
        "operation and maintenance",
        "implementation and assessment",
        "in production",
        "in mission",
    ]).astype(int)

    # ── Controls ──
    dev = raw["22_dev_method"].fillna("").astype(str).str.strip().str.lower()
    raw["vendor"] = (dev.str.contains("contracting|contractor|vendor|commercial|cots", na=False)
                     & ~dev.str.contains("in-house|both", na=False)).astype(int)
    raw["mixed_dev"] = dev.str.contains("both", na=False).astype(int)

    impact = raw["17_impact_type"].fillna("").astype(str).str.strip().str.lower()
    raw["rights_safety"] = (~impact.isin(["", "neither"])).astype(int)

    # ── Agency-level orientation proxies ──
    agency_stats = raw.groupby("3_agency").agg(
        agency_n=("2_use_case_name", "size"),
        topic_breadth=("8_topic_area", "nunique"),
    )
    agency_stats["log_agency_n"] = np.log1p(agency_stats["agency_n"])
    raw = raw.merge(agency_stats, left_on="3_agency", right_index=True, how="left")

    # Orientation composite: z-score of log_agency_n + z-score of topic_breadth
    raw["z_log_n"] = (raw["log_agency_n"] - raw["log_agency_n"].mean()) / raw["log_agency_n"].std()
    raw["z_breadth"] = (raw["topic_breadth"] - raw["topic_breadth"].mean()) / raw["topic_breadth"].std()
    raw["orientation"] = (raw["z_log_n"] + raw["z_breadth"]) / 2

    # ── Interaction ──
    raw["TR_x_IR"] = raw["TR"] * raw["IR"]

    return raw, tr_item_cols, ir_item_cols


# ═══════════════════════════════════════════════════════════════════════════════
#  Descriptive statistics
# ═══════════════════════════════════════════════════════════════════════════════

def descriptive_stats(df, tr_cols, ir_cols):
    print("=" * 72)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 72)

    desc_vars = ["TR", "IR", "orientation", "operational", "vendor",
                 "mixed_dev", "rights_safety", "agency_n", "topic_breadth"]
    print("\n  Variable-level summary:")
    for v in desc_vars:
        print(f"    {v:20s}  mean={df[v].mean():.3f}  sd={df[v].std():.3f}  "
              f"min={df[v].min():.0f}  max={df[v].max():.0f}")

    # TR/IR item prevalence
    print("\n  TR items:")
    for c in tr_cols:
        print(f"    {c:20s}  {df[c].mean()*100:5.1f}%")
    print(f"    {'TR (sum)':20s}  mean={df['TR'].mean():.2f}  sd={df['TR'].std():.2f}")

    print("\n  IR items:")
    for c in ir_cols:
        print(f"    {c:20s}  {df[c].mean()*100:5.1f}%")
    print(f"    {'IR (sum)':20s}  mean={df['IR'].mean():.2f}  sd={df['IR'].std():.2f}")

    # Correlation matrix
    corr_vars = ["operational", "orientation", "TR", "IR", "vendor", "rights_safety"]
    corr = df[corr_vars].corr()
    print("\n  Correlation matrix:")
    print("  " + " ".join(f"{v:>12s}" for v in corr_vars))
    for v in corr_vars:
        row = " ".join(f"{corr.loc[v, v2]:12.3f}" for v2 in corr_vars)
        print(f"  {v:16s} {row}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Regression models
# ═══════════════════════════════════════════════════════════════════════════════

def run_models(df):
    """Run use-case-level logistic regressions and agency-level OLS."""
    try:
        import statsmodels.api as sm
        from statsmodels.discrete.discrete_model import Logit
    except ImportError:
        print("ERROR: statsmodels not installed. Run: pip install statsmodels")
        sys.exit(1)

    results = {}

    # ── Model 1: Controls only ──
    X1 = df[["vendor", "mixed_dev", "rights_safety"]].copy()
    X1 = sm.add_constant(X1)
    y = df["operational"]
    m1 = Logit(y, X1).fit(disp=0)
    results["M1_controls"] = m1

    # ── Model 2: + Orientation ──
    X2 = df[["orientation", "vendor", "mixed_dev", "rights_safety"]].copy()
    X2 = sm.add_constant(X2)
    m2 = Logit(y, X2).fit(disp=0)
    results["M2_orientation"] = m2

    # ── Model 3: + TR + IR (main effects) ──
    X3 = df[["orientation", "TR", "IR", "vendor", "mixed_dev", "rights_safety"]].copy()
    X3 = sm.add_constant(X3)
    m3 = Logit(y, X3).fit(disp=0)
    results["M3_main"] = m3

    # ── Model 4: + TR × IR interaction ──
    X4 = df[["orientation", "TR", "IR", "TR_x_IR", "vendor", "mixed_dev", "rights_safety"]].copy()
    X4 = sm.add_constant(X4)
    m4 = Logit(y, X4).fit(disp=0)
    results["M4_interaction"] = m4

    return results


def print_model_table(results):
    """Print a regression comparison table."""
    print("\n" + "=" * 100)
    print("LOGISTIC REGRESSION RESULTS: DV = Operational Deployment (0/1)")
    print("=" * 100)

    models = list(results.keys())
    all_vars = []
    for m in models:
        for v in results[m].params.index:
            if v not in all_vars:
                all_vars.append(v)

    # Header
    header = f"{'Variable':<20s}"
    for m in models:
        header += f"  {'β':>7s}  {'SE':>6s}  {'p':>6s}"
    print(header)
    print("-" * len(header))

    for var in all_vars:
        row = f"{var:<20s}"
        for m in models:
            res = results[m]
            if var in res.params.index:
                b = res.params[var]
                se = res.bse[var]
                p = res.pvalues[var]
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "†" if p < 0.10 else ""
                row += f"  {b:7.3f}  {se:6.3f}  {p:5.3f}{stars:3s}"
            else:
                row += f"  {'':>7s}  {'':>6s}  {'':>9s}"
        print(row)

    print("-" * len(header))

    # Model fit
    row = f"{'N':<20s}"
    for m in models:
        row += f"  {results[m].nobs:7.0f}  {'':>6s}  {'':>9s}"
    print(row)

    row = f"{'Pseudo R²':<20s}"
    for m in models:
        row += f"  {results[m].prsquared:7.3f}  {'':>6s}  {'':>9s}"
    print(row)

    row = f"{'AIC':<20s}"
    for m in models:
        row += f"  {results[m].aic:7.1f}  {'':>6s}  {'':>9s}"
    print(row)

    row = f"{'Log-Likelihood':<20s}"
    for m in models:
        row += f"  {results[m].llf:7.1f}  {'':>6s}  {'':>9s}"
    print(row)

    # Odds ratios for Model 4
    print("\n── Odds Ratios (Model 4) ──")
    m4 = results["M4_interaction"]
    for var in m4.params.index:
        if var == "const":
            continue
        or_val = np.exp(m4.params[var])
        ci_lo = np.exp(m4.conf_int().loc[var, 0])
        ci_hi = np.exp(m4.conf_int().loc[var, 1])
        p = m4.pvalues[var]
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "†" if p < 0.10 else ""
        print(f"    {var:<20s}  OR={or_val:6.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]{stars}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Agency-level OLS (supplementary)
# ═══════════════════════════════════════════════════════════════════════════════

def agency_level_ols(df):
    import statsmodels.api as sm

    agg = df.groupby("3_agency").agg(
        n_cases=("operational", "size"),
        pct_operational=("operational", "mean"),
        mean_TR=("TR", "mean"),
        mean_IR=("IR", "mean"),
        orientation=("orientation", "first"),
        pct_vendor=("vendor", "mean"),
        pct_rights_safety=("rights_safety", "mean"),
        topic_breadth=("topic_breadth", "first"),
    )
    agg["mean_TR_x_IR"] = agg["mean_TR"] * agg["mean_IR"]

    print("\n" + "=" * 72)
    print(f"AGENCY-LEVEL OLS (n={len(agg)} agencies)")
    print("DV = Share of use cases in operational deployment")
    print("=" * 72)

    X = agg[["orientation", "mean_TR", "mean_IR", "mean_TR_x_IR", "pct_vendor"]].copy()
    X = sm.add_constant(X)
    y = agg["pct_operational"]

    ols = sm.OLS(y, X).fit()
    print(ols.summary())

    # Save agency dataset
    agg.to_csv(OUT_DIR / "agency_level_proxies.csv")
    print(f"\n  → saved {OUT_DIR / 'agency_level_proxies.csv'}")

    return ols


# ═══════════════════════════════════════════════════════════════════════════════
#  Marginal effects visualization
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_complementarity(df, m4):
    """Plot predicted probability of operational deployment by TR and IR levels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Create grid: TR from 0-9, IR at low/medium/high
    tr_range = np.arange(0, 10)
    ir_levels = {"Low IR (0-1)": 0.5, "Med IR (2-3)": 2.5, "High IR (4+)": 5.0}

    # Mean values for other variables
    means = {
        "orientation": df["orientation"].mean(),
        "vendor": df["vendor"].mean(),
        "mixed_dev": df["mixed_dev"].mean(),
        "rights_safety": df["rights_safety"].mean(),
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#3B82F6", "#F59E0B", "#EF4444"]

    for (label, ir_val), color in zip(ir_levels.items(), colors):
        probs = []
        for tr_val in tr_range:
            x = np.array([1, means["orientation"], tr_val, ir_val,
                          tr_val * ir_val,
                          means["vendor"], means["mixed_dev"], means["rights_safety"]])
            logit = np.dot(m4.params.values, x)
            prob = 1 / (1 + np.exp(-logit))
            probs.append(prob * 100)
        ax.plot(tr_range, probs, marker="o", color=color, linewidth=2,
                markersize=5, label=label)

    ax.set_xlabel("Trust Readiness Index (0–9 safeguards)")
    ax.set_ylabel("Predicted P(Operational Deployment), %")
    ax.set_title("TR × IR Complementarity:\nPredicted Operational Deployment by Readiness Levels")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(-0.3, 9.3)
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_complementarity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  → saved {FIG_DIR / 'fig5_complementarity.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("PATHWAY MODEL: Orientation → TR × IR → Value")
    print("=" * 72)

    df, tr_cols, ir_cols = build_dataset()
    descriptive_stats(df, tr_cols, ir_cols)

    results = run_models(df)
    print_model_table(results)

    ols = agency_level_ols(df)

    m4 = results["M4_interaction"]
    fig5_complementarity(df, m4)

    # Save use-case-level dataset
    keep_cols = (["3_agency", "2_use_case_name", "16_dev_stage", "operational",
                  "orientation", "TR", "IR", "TR_x_IR", "vendor", "mixed_dev",
                  "rights_safety", "agency_n", "topic_breadth"] +
                 tr_cols + ir_cols)
    df[keep_cols].to_csv(OUT_DIR / "pathway_model_data.csv", index=False)
    print(f"  → saved {OUT_DIR / 'pathway_model_data.csv'}")

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
