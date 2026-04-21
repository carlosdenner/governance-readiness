"""
12_procurement_confounding.py
=============================
Multivariate logistic regressions to address confounding in the
procurement-opacity claim.

For each key governance safeguard, we estimate:
    Safeguard ~ Vendor + RightsImpact + PublicFacing + Stage + AgencyFE

Then we test a mediation pathway:
    Vendor → Code Access → Appeal/IndependentEval

Outputs:
    - Console: OR + 95% CI table for each safeguard
    - data/processed/procurement_confounding_results.csv
"""

from __future__ import annotations
import pathlib, re, sys, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = pathlib.Path(__file__).resolve().parent.parent
RAW_EO = BASE / "data" / "raw" / "eo13960" / "2024_consolidated_ai_inventory_raw.csv"
OUT_DIR = BASE / "data" / "processed"

# ═══════════════════════════════════════════════════════════════════════════════
#  Data prep  (reuses logic from 09_pathway_model.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _is_yes(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip().str.lower()
    return (~s.isin(["", "no", "n/a", "nan", "none", "na", "false"])).astype(int)

def _is_yes_strict(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip().str.lower()
    return s.str.startswith("yes").astype(int)


def build_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_EO, low_memory=False)
    df.columns = [
        re.sub(r"\s+", "_", c.strip().lower().replace("(", "").replace(")", ""))
        for c in df.columns
    ]
    assert len(df) == 1757

    # --- Key safeguard outcomes ---
    df["y_code_access"]     = _is_yes(df["38_code_access"])
    df["y_impact_assess"]   = _is_yes(df["52_impact_assessment"])
    df["y_indep_eval"]      = _is_yes(df["55_independent_eval"])
    df["y_postdeploy"]      = _is_yes(df["56_monitor_postdeploy"])
    df["y_appeal"]          = _is_yes(df["65_appeal_process"])
    df["y_disparity"]       = _is_yes(df["62_disparity_mitigation"])
    df["y_rw_testing"]      = _is_yes(df["53_real_world_testing"])

    # --- Predictors ---
    # Development method: restrict to known (vendor/in-house/mixed)
    # 629 records have missing dev_method; including them in the reference group
    # creates a spurious association because they also tend to have missing safeguard data.
    dev = df["22_dev_method"].fillna("").astype(str).str.strip().str.lower()
    df["vendor"] = (
        dev.str.contains("contracting|contractor|vendor|commercial|cots", na=False)
        & ~dev.str.contains("in-house|both", na=False)
    ).astype(int)
    df["mixed_dev"] = dev.str.contains("both", na=False).astype(int)
    df["inhouse"] = dev.str.contains("in-house", na=False).astype(int) & ~df["mixed_dev"].astype(bool)
    df["dev_known"] = ((df["vendor"] == 1) | (df["mixed_dev"] == 1) | (df["inhouse"] == 1)).astype(int)

    # Filter to known dev_method only (reference = in-house)
    print(f"\n  Full dataset: {len(df)}")
    print(f"  Vendor-only: {df['vendor'].sum()}, Mixed: {df['mixed_dev'].sum()}, In-house: {df['inhouse'].sum()}, Unknown: {(df['dev_known']==0).sum()}")
    df = df[df["dev_known"] == 1].copy()
    print(f"  Analysis sample (known dev_method): {len(df)}")

    # Rights / safety impact
    impact = df["17_impact_type"].fillna("").astype(str).str.strip().str.lower()
    df["rights_impact"] = impact.str.contains("rights", na=False).astype(int)
    df["safety_impact"] = impact.str.contains("safety", na=False).astype(int)
    df["high_impact"] = ((df["rights_impact"] == 1) | (df["safety_impact"] == 1)).astype(int)

    # Public-facing
    pub = df["26_public_service"].fillna("").astype(str).str.strip()
    df["public_facing"] = (pub.str.len() > 2).astype(int)  # non-blank, non-trivial

    # Stage: operational vs. earlier
    stage = df["16_dev_stage"].fillna("").astype(str).str.strip().str.lower()
    df["operational"] = stage.isin([
        "operation and maintenance",
        "implementation and assessment",
        "in production",
        "in mission",
    ]).astype(int)

    # Agency fixed effects (top agencies + "other")
    agency_counts = df["3_agency"].value_counts()
    top_agencies = agency_counts[agency_counts >= 30].index.tolist()
    df["agency_grp"] = df["3_agency"].where(
        df["3_agency"].isin(top_agencies), other="Other"
    )
    agency_dummies = pd.get_dummies(df["agency_grp"], prefix="afe", drop_first=True).astype(int)
    df = pd.concat([df, agency_dummies], axis=1)

    return df, list(agency_dummies.columns)


# ═══════════════════════════════════════════════════════════════════════════════
#  Multivariate logistic regressions
# ═══════════════════════════════════════════════════════════════════════════════

def run_regressions(df: pd.DataFrame, afe_cols: list[str]):
    """Run logistic regression for each safeguard outcome."""

    safeguards = {
        "Code Access":          "y_code_access",
        "Impact Assessment":    "y_impact_assess",
        "Independent Eval":     "y_indep_eval",
        "Post-Deploy Monitor":  "y_postdeploy",
        "Appeal Process":       "y_appeal",
        "Disparity Mitigation": "y_disparity",
        "Real-World Testing":   "y_rw_testing",
    }

    core_predictors = ["vendor", "mixed_dev", "high_impact", "public_facing", "operational"]

    results = []

    print("=" * 90)
    print("MULTIVARIATE LOGISTIC REGRESSIONS: VENDOR EFFECT ON SAFEGUARDS")
    print("=" * 90)

    for label, ycol in safeguards.items():
        y = df[ycol]
        prev = y.mean() * 100

        # Skip if outcome too rare (< 3%) — model won't converge well
        if y.sum() < 30:
            print(f"\n{label}: skipped (n_positive={y.sum()}, <30)")
            continue

        # --- Model 1: Bivariate (vendor only) ---
        X_biv = sm.add_constant(df[["vendor"]])
        try:
            m1 = sm.Logit(y, X_biv).fit(disp=0, maxiter=100)
            or_biv = np.exp(m1.params["vendor"])
            ci_biv = np.exp(m1.conf_int().loc["vendor"])
            p_biv = m1.pvalues["vendor"]
        except Exception:
            or_biv = ci_biv = p_biv = np.nan

        # --- Model 2: + controls (no agency FE) ---
        X_ctrl = sm.add_constant(df[core_predictors])
        try:
            m2 = sm.Logit(y, X_ctrl).fit(disp=0, maxiter=100)
            or_ctrl = np.exp(m2.params["vendor"])
            ci_ctrl = np.exp(m2.conf_int().loc["vendor"])
            p_ctrl = m2.pvalues["vendor"]
        except Exception:
            or_ctrl = ci_ctrl = p_ctrl = np.nan

        # --- Model 3: + agency FE ---
        X_full = sm.add_constant(df[core_predictors + afe_cols])

        # Check for perfect separation: drop any afe columns with zero variance in y=1 or y=0
        for col in afe_cols:
            if X_full.loc[y == 1, col].nunique() < 2 and X_full.loc[y == 0, col].nunique() < 2:
                pass  # keep, both have variance
        try:
            m3 = sm.Logit(y, X_full).fit(disp=0, maxiter=200, method="bfgs")
            or_full = np.exp(m3.params["vendor"])
            ci_full = np.exp(m3.conf_int().loc["vendor"])
            p_full = m3.pvalues["vendor"]
        except Exception as e:
            print(f"  [{label}] Model 3 failed: {e}")
            or_full = ci_full = p_full = np.nan

        # Print results
        print(f"\n{'─' * 90}")
        print(f"  {label}  (prevalence: {prev:.1f}%, n_pos={y.sum()})")
        print(f"{'─' * 90}")
        print(f"  {'Model':<30s} {'OR':>8s} {'95% CI':>18s} {'p':>10s}")
        print(f"  {'─'*70}")

        if not np.isnan(or_biv) if isinstance(or_biv, float) else True:
            print(f"  {'M1: Vendor only':<30s} {or_biv:8.3f} [{ci_biv.iloc[0]:.3f}, {ci_biv.iloc[1]:.3f}]   {p_biv:10.4f}")
        if not np.isnan(or_ctrl) if isinstance(or_ctrl, float) else True:
            print(f"  {'M2: + controls':<30s} {or_ctrl:8.3f} [{ci_ctrl.iloc[0]:.3f}, {ci_ctrl.iloc[1]:.3f}]   {p_ctrl:10.4f}")
        if not (isinstance(or_full, float) and np.isnan(or_full)):
            print(f"  {'M3: + agency FE':<30s} {or_full:8.3f} [{ci_full.iloc[0]:.3f}, {ci_full.iloc[1]:.3f}]   {p_full:10.4f}")

        # Also print full M3 odds ratios for core predictors
        if not (isinstance(or_full, float) and np.isnan(or_full)):
            print(f"\n  Full M3 core predictor ORs:")
            for pred in core_predictors:
                or_p = np.exp(m3.params[pred])
                ci_p = np.exp(m3.conf_int().loc[pred])
                pv = m3.pvalues[pred]
                sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
                print(f"    {pred:<20s} OR={or_p:6.3f}  [{ci_p.iloc[0]:.3f}, {ci_p.iloc[1]:.3f}]  p={pv:.4f} {sig}")

        results.append({
            "safeguard": label,
            "prevalence_pct": round(prev, 1),
            "n_positive": int(y.sum()),
            "OR_bivariate": round(or_biv, 3) if not (isinstance(or_biv, float) and np.isnan(or_biv)) else np.nan,
            "OR_controls": round(or_ctrl, 3) if not (isinstance(or_ctrl, float) and np.isnan(or_ctrl)) else np.nan,
            "CI_controls_lo": round(ci_ctrl.iloc[0], 3) if not (isinstance(or_ctrl, float) and np.isnan(or_ctrl)) else np.nan,
            "CI_controls_hi": round(ci_ctrl.iloc[1], 3) if not (isinstance(or_ctrl, float) and np.isnan(or_ctrl)) else np.nan,
            "OR_full": round(or_full, 3) if not (isinstance(or_full, float) and np.isnan(or_full)) else np.nan,
            "CI_full_lo": round(ci_full.iloc[0], 3) if hasattr(ci_full, 'iloc') and not (isinstance(or_full, float) and np.isnan(or_full)) else np.nan,
            "CI_full_hi": round(ci_full.iloc[1], 3) if hasattr(ci_full, 'iloc') and not (isinstance(or_full, float) and np.isnan(or_full)) else np.nan,
            "p_bivariate": round(p_biv, 4) if not (isinstance(p_biv, float) and np.isnan(p_biv)) else np.nan,
            "p_controls": round(p_ctrl, 4) if not (isinstance(p_ctrl, float) and np.isnan(p_ctrl)) else np.nan,
            "p_full": round(p_full, 4) if not (isinstance(p_full, float) and np.isnan(p_full)) else np.nan,
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
#  Mediation: Vendor → Code Access → Appeal/IndepEval
# ═══════════════════════════════════════════════════════════════════════════════

def mediation_analysis(df: pd.DataFrame, afe_cols: list[str]):
    """
    Baron-Kenny-style mediation test:
    Path a: Vendor → Code Access
    Path b: Code Access → Appeal (controlling for Vendor)
    Path c: Vendor → Appeal (total)
    Path c': Vendor → Appeal (controlling for Code Access)
    """

    core = ["vendor", "mixed_dev", "high_impact", "public_facing", "operational"]

    print("\n" + "=" * 90)
    print("MEDIATION ANALYSIS: Vendor → Code Access → Appeal/IndependentEval")
    print("=" * 90)

    for outcome_label, outcome_col in [("Appeal Process", "y_appeal"),
                                        ("Independent Eval", "y_indep_eval")]:
        print(f"\n{'─' * 90}")
        print(f"  Outcome: {outcome_label}")
        print(f"{'─' * 90}")

        # Path c (total): Vendor → Outcome
        X_c = sm.add_constant(df[core + afe_cols])
        try:
            m_c = sm.Logit(df[outcome_col], X_c).fit(disp=0, maxiter=200, method="bfgs")
            or_c = np.exp(m_c.params["vendor"])
            p_c = m_c.pvalues["vendor"]
            print(f"  Path c  (total):    Vendor → {outcome_label:20s}  OR={or_c:.3f}  p={p_c:.4f}")
        except Exception as e:
            print(f"  Path c failed: {e}")
            continue

        # Path a: Vendor → Code Access (mediator)
        X_a = sm.add_constant(df[core + afe_cols])
        try:
            m_a = sm.Logit(df["y_code_access"], X_a).fit(disp=0, maxiter=200, method="bfgs")
            or_a = np.exp(m_a.params["vendor"])
            p_a = m_a.pvalues["vendor"]
            print(f"  Path a  (→mediator): Vendor → Code Access       OR={or_a:.3f}  p={p_a:.4f}")
        except Exception as e:
            print(f"  Path a failed: {e}")
            continue

        # Path c' (direct): Vendor → Outcome, controlling for Code Access
        X_cp = sm.add_constant(df[core + ["y_code_access"] + afe_cols])
        try:
            m_cp = sm.Logit(df[outcome_col], X_cp).fit(disp=0, maxiter=200, method="bfgs")
            or_cp = np.exp(m_cp.params["vendor"])
            p_cp = m_cp.pvalues["vendor"]
            or_b = np.exp(m_cp.params["y_code_access"])
            p_b = m_cp.pvalues["y_code_access"]
            print(f"  Path b  (mediator): Code Access → {outcome_label:20s}  OR={or_b:.3f}  p={p_b:.4f}")
            print(f"  Path c' (direct):   Vendor → {outcome_label:20s}  OR={or_cp:.3f}  p={p_cp:.4f}")

            # Attenuation
            log_or_c = np.log(or_c)
            log_or_cp = np.log(or_cp)
            if abs(log_or_c) > 0.001:
                attenuation = (1 - log_or_cp / log_or_c) * 100
                print(f"\n  Attenuation of vendor effect: {attenuation:.1f}%")
                if attenuation > 20:
                    print(f"  → Partial mediation: code access explains ~{attenuation:.0f}% of vendor effect on {outcome_label}")
                else:
                    print(f"  → Minimal mediation: code access explains only ~{attenuation:.0f}% of vendor effect")
            else:
                print("  → Total effect near zero; mediation not interpretable")
        except Exception as e:
            print(f"  Path c' failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Summary table for paper
# ═══════════════════════════════════════════════════════════════════════════════

def print_paper_table(results_df: pd.DataFrame):
    print("\n" + "=" * 90)
    print("TABLE FOR PAPER: Vendor effect on governance safeguards (ORs, 95% CIs)")
    print("=" * 90)
    print(f"\n  {'Safeguard':<25s} {'Prev.':>6s} {'Bivariate OR':>14s} {'Adjusted OR':>14s} {'Full OR (AFE)':>14s}")
    print(f"  {'─'*75}")
    for _, row in results_df.iterrows():
        biv = f"{row['OR_bivariate']:.2f}" if pd.notna(row['OR_bivariate']) else "—"
        ctrl = f"{row['OR_controls']:.2f}" if pd.notna(row['OR_controls']) else "—"
        full = f"{row['OR_full']:.2f}" if pd.notna(row['OR_full']) else "—"
        p_star = ""
        if pd.notna(row['p_full']):
            if row['p_full'] < 0.001: p_star = "***"
            elif row['p_full'] < 0.01: p_star = "**"
            elif row['p_full'] < 0.05: p_star = "*"
        print(f"  {row['safeguard']:<25s} {row['prevalence_pct']:5.1f}% {biv:>14s} {ctrl:>14s} {full:>13s}{p_star}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df, afe_cols = build_dataset()

    # Descriptives
    print(f"\nAnalysis dataset: n={len(df)}")
    print(f"  Vendor-only:    {df['vendor'].sum()} ({df['vendor'].mean()*100:.1f}%)")
    print(f"  Mixed dev:      {df['mixed_dev'].sum()} ({df['mixed_dev'].mean()*100:.1f}%)")
    print(f"  In-house (ref): {df['inhouse'].sum()} ({df['inhouse'].mean()*100:.1f}%)")
    print(f"  High-impact:    {df['high_impact'].sum()} ({df['high_impact'].mean()*100:.1f}%)")
    print(f"  Public-facing:  {df['public_facing'].sum()} ({df['public_facing'].mean()*100:.1f}%)")
    print(f"  Operational:    {df['operational'].sum()} ({df['operational'].mean()*100:.1f}%)")
    print(f"  Agency FE cols: {len(afe_cols)}")

    results_df = run_regressions(df, afe_cols)

    mediation_analysis(df, afe_cols)

    print_paper_table(results_df)

    # Save
    out_path = OUT_DIR / "procurement_confounding_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
