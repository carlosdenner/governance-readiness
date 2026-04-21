"""
10_ir_analysis.py
=================
Integration-Readiness deep-dive:
  1. IR distribution (histogram, quartiles) alongside TR
  2. IR–safeguard bundling: do higher-IR systems show more bundled governance?
  3. Figure 6: side-by-side TR / IR distributions
"""

from __future__ import annotations
import pathlib, re, sys
import numpy as np
import pandas as pd

BASE = pathlib.Path(__file__).resolve().parent.parent

RAW_EO = BASE / "data" / "raw" / "eo13960" / "2024_consolidated_ai_inventory_raw.csv"
OUT_DIR = BASE / "data" / "processed"
FIG_DIR = BASE / "paper" / "figures"

# ── helpers (duplicated from 09 for self-containment) ────────────────────────

def _load():
    df = pd.read_csv(RAW_EO, low_memory=False)
    df.columns = [re.sub(r"\s+", "_", c.strip().lower().replace("(","").replace(")","")) for c in df.columns]
    return df

def _is_yes(series):
    s = series.fillna("").astype(str).str.strip().str.lower()
    return (~s.isin(["", "no", "n/a", "nan", "none", "na", "false"])).astype(int)

def _is_yes_strict(series):
    s = series.fillna("").astype(str).str.strip().str.lower()
    return s.str.startswith("yes").astype(int)

def _is_reuse(series):
    s = series.fillna("").astype(str).str.strip().str.lower()
    return ((s != "") & (~s.str.startswith("none"))).astype(int)


def build():
    raw = _load()
    assert len(raw) == 1757

    # TR indicators (9)
    tr_map = {
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
    tr_cols = []
    for new_col, (src_col, fn) in tr_map.items():
        raw[new_col] = fn(raw[src_col])
        tr_cols.append(new_col)
    raw["TR"] = raw[tr_cols].sum(axis=1)

    # IR indicators (7)
    ir_map = {
        "ir_data_catalog":  ("31_data_catalog",      _is_yes_strict),
        "ir_data_docs":     ("34_data_docs",         _is_yes),
        "ir_custom_code":   ("37_custom_code",       _is_yes_strict),
        "ir_code_access":   ("38_code_access",       _is_yes),
        "ir_infra":         ("43_infra_provisioned",  _is_yes_strict),
        "ir_timely_res":    ("47_timely_resources",   _is_yes_strict),
        "ir_reuse":         ("49_existing_reuse",     _is_reuse),
    }
    ir_cols = []
    for new_col, (src_col, fn) in ir_map.items():
        raw[new_col] = fn(raw[src_col])
        ir_cols.append(new_col)
    raw["IR"] = raw[ir_cols].sum(axis=1)

    # Operational outcome
    stage = raw["16_dev_stage"].fillna("").astype(str).str.strip().str.lower()
    raw["operational"] = stage.isin([
        "operation and maintenance",
        "implementation and assessment",
        "in production",
        "in mission",
    ]).astype(int)

    # Rights/safety
    impact = raw["17_impact_type"].fillna("").astype(str).str.strip().str.lower()
    raw["rights_safety"] = (~impact.isin(["", "neither"])).astype(int)

    return raw, tr_cols, ir_cols


def main():
    df, tr_cols, ir_cols = build()
    n = len(df)

    # ================================================================
    # 1. DISTRIBUTION: TR and IR quartiles / percentiles
    # ================================================================
    print("=" * 72)
    print("1. DISTRIBUTION COMPARISON: TR vs IR")
    print("=" * 72)

    for label, col in [("TR (0-9)", "TR"), ("IR (0-7)", "IR")]:
        print(f"\n  {label}:")
        print(f"    Mean = {df[col].mean():.2f},  SD = {df[col].std():.2f}")
        print(f"    Median = {df[col].median():.1f}")
        print(f"    Q1 = {df[col].quantile(0.25):.1f},  Q3 = {df[col].quantile(0.75):.1f}")
        print(f"    Min = {df[col].min():.0f},  Max = {df[col].max():.0f}")
        print(f"    Value distribution:")
        vc = df[col].value_counts().sort_index()
        for val, cnt in vc.items():
            print(f"      {val:>3.0f}:  {cnt:>5d}  ({cnt/n*100:5.1f}%)")

    # IR item-level prevalence
    print("\n  IR item-level prevalence:")
    for c in ir_cols:
        label = c.replace("ir_", "")
        print(f"    {label:20s}  {df[c].sum():>5d}  ({df[c].mean()*100:5.1f}%)")

    # ================================================================
    # 2. IR–SAFEGUARD BUNDLING
    # ================================================================
    print("\n" + "=" * 72)
    print("2. IR–SAFEGUARD BUNDLING: Higher IR → more bundled governance?")
    print("=" * 72)

    # Deep safeguards: impact_assess, indep_eval, disparity, postdeploy, rw_testing
    deep_safeguards = ["tr_impact_assess", "tr_indep_eval", "tr_disparity",
                       "tr_postdeploy", "tr_rw_testing"]
    df["deep_TR"] = df[deep_safeguards].sum(axis=1)
    df["has_any_deep"] = (df["deep_TR"] > 0).astype(int)
    df["has_bundle"] = (df["deep_TR"] >= 2).astype(int)  # 2+ deep safeguards = bundled

    # Split IR into terciles
    ir_low = df[df["IR"] <= 2]
    ir_mid = df[(df["IR"] >= 3) & (df["IR"] <= 4)]
    ir_high = df[df["IR"] >= 5]

    print(f"\n  IR tercile n:  Low(0-2)={len(ir_low)},  Mid(3-4)={len(ir_mid)},  High(5-7)={len(ir_high)}")

    for label, subset in [("Low IR (0-2)", ir_low), ("Mid IR (3-4)", ir_mid), ("High IR (5-7)", ir_high)]:
        ns = len(subset)
        pct_any = subset["has_any_deep"].mean() * 100
        pct_bundle = subset["has_bundle"].mean() * 100
        mean_deep = subset["deep_TR"].mean()
        mean_tr = subset["TR"].mean()
        pct_oper = subset["operational"].mean() * 100
        print(f"\n  {label} (n={ns}):")
        print(f"    Mean TR          = {mean_tr:.2f}")
        print(f"    Mean deep TR     = {mean_deep:.2f}")
        print(f"    % any deep safeguard  = {pct_any:.1f}%")
        print(f"    % bundled (2+ deep)   = {pct_bundle:.1f}%")
        print(f"    % operational         = {pct_oper:.1f}%")

    # Chi-square test: IR high vs low → has_bundle
    from scipy.stats import chi2_contingency, spearmanr
    ct = pd.crosstab(
        pd.cut(df["IR"], bins=[-1, 2, 4, 7], labels=["Low", "Mid", "High"]),
        df["has_bundle"]
    )
    chi2, p_chi, dof, _ = chi2_contingency(ct)
    print(f"\n  Chi-square (IR tercile × bundled safeguards): chi2={chi2:.2f}, p={p_chi:.4f}, dof={dof}")

    # Spearman correlation IR ↔ deep_TR
    rho, p_rho = spearmanr(df["IR"], df["deep_TR"])
    print(f"  Spearman r(IR, deep_TR) = {rho:.3f}, p = {p_rho:.4f}")

    # Spearman correlation IR ↔ TR
    rho2, p_rho2 = spearmanr(df["IR"], df["TR"])
    print(f"  Spearman r(IR, TR)      = {rho2:.3f}, p = {p_rho2:.4f}")

    # ================================================================
    # 3. IR–OPERATIONAL DEPLOYMENT (supplementary)
    # ================================================================
    print("\n" + "=" * 72)
    print("3. IR → OPERATIONAL DEPLOYMENT BY TERCILE")
    print("=" * 72)
    for label, subset in [("Low IR (0-2)", ir_low), ("Mid IR (3-4)", ir_mid), ("High IR (5-7)", ir_high)]:
        print(f"  {label}: {subset['operational'].mean()*100:.1f}% operational (n={len(subset)})")

    # ================================================================
    # 4. FIGURE 6: Side-by-side TR/IR distribution
    # ================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    # TR histogram
    tr_counts = df["TR"].value_counts().sort_index()
    tr_vals = range(0, 10)
    tr_heights = [tr_counts.get(v, 0) for v in tr_vals]
    axes[0].bar(tr_vals, tr_heights, color="#3B82F6", edgecolor="white", width=0.8)
    axes[0].set_xlabel("Trust Readiness Index (0–9 safeguards)")
    axes[0].set_ylabel("Number of Use Cases")
    axes[0].set_title("Trust Readiness Distribution")
    axes[0].set_xticks(range(0, 10))
    # Add mean line
    axes[0].axvline(df["TR"].mean(), color="#EF4444", linestyle="--", linewidth=1.5,
                    label=f"Mean = {df['TR'].mean():.1f}")
    axes[0].axvline(df["TR"].median(), color="#F59E0B", linestyle=":", linewidth=1.5,
                    label=f"Median = {df['TR'].median():.0f}")
    axes[0].legend(fontsize=8)

    # IR histogram
    ir_counts = df["IR"].value_counts().sort_index()
    ir_vals = range(0, 8)
    ir_heights = [ir_counts.get(v, 0) for v in ir_vals]
    axes[1].bar(ir_vals, ir_heights, color="#10B981", edgecolor="white", width=0.8)
    axes[1].set_xlabel("Integration Readiness Index (0–7 indicators)")
    axes[1].set_title("Integration Readiness Distribution")
    axes[1].set_xticks(range(0, 8))
    axes[1].axvline(df["IR"].mean(), color="#EF4444", linestyle="--", linewidth=1.5,
                    label=f"Mean = {df['IR'].mean():.1f}")
    axes[1].axvline(df["IR"].median(), color="#F59E0B", linestyle=":", linewidth=1.5,
                    label=f"Median = {df['IR'].median():.0f}")
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Readiness Distributions: Governance Lags Architecture (EO 13960, n=1,757)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_tr_ir_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> saved {FIG_DIR / 'fig6_tr_ir_distributions.png'}")

    print("\nDONE")


if __name__ == "__main__":
    main()
