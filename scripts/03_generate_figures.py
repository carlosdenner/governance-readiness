"""
generate_figures.py
===================
Produces publication-ready figures from the analysis-ready datasets.
All figures are saved to analysis/output/figures/.

Figures generated:
  1. governance_gap_bar.png       — EO 13960 Tier-1 vs Tier-2 safeguard adoption
  2. aiid_failure_heatmap.png     — AIID technical failure × sector heatmap
  3. atlas_tactic_frequency.png   — ATLAS tactic frequency across case studies
  4. constraint_coverage.png      — McKinsey constraint coverage by data source
  5. governance_readiness_dist.png— Distribution of governance readiness scores
"""

from __future__ import annotations
import pathlib
import re
import textwrap

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

BASE = pathlib.Path(__file__).resolve().parent.parent
OUTPUT = BASE / "data" / "processed"
FIG_DIR = BASE / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
RED = "#d62728"
GREEN = "#2ca02c"
GREY = "#7f7f7f"


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 1: EO 13960 Governance Gap
# ═══════════════════════════════════════════════════════════════════════════════

def fig_governance_gap():
    print("  Fig 1: governance_gap_bar ...")
    eo = pd.read_csv(OUTPUT / "eo13960_scored.csv", low_memory=False)
    n = len(eo)

    # Re-read raw data to count per-safeguard adoption
    raw = pd.read_csv(
        BASE / "data" / "raw" / "eo13960" / "2024_consolidated_ai_inventory_raw.csv",
        low_memory=False,
    )
    raw.columns = [
        re.sub(r'\s+', '_', c.strip().lower().replace("(", "").replace(")", ""))
        for c in raw.columns
    ]

    safeguards = {
        "ATO":                  "40_has_ato",
        "Internal review":      "50_internal_review",
        "Impact assessment":    "52_impact_assessment",
        "Post-deploy monitor":  "56_monitor_postdeploy",
        "Adverse impact":       "61_adverse_impact",
        "Real-world testing":   "53_real_world_testing",
        "Autonomous impact":    "57_autonomous_impact",
        "Stakeholder consult":  "63_stakeholder_consult",
        "Opt-out mechanism":    "67_opt_out",
        "Appeal process":       "65_appeal_process",
        "AI use notice":        "59_ai_notice",
        "Independent eval":     "55_independent_eval",
        "Key risk ID":          "54_key_risks",
        "Bias mitigation":      "62_disparity_mitigation",
    }

    def pct_present(col):
        vals = raw[col].dropna()
        present = vals[~vals.astype(str).str.strip().str.lower().isin(["", "no", "n/a", "nan", "none", "na"])]
        return len(present) / n * 100

    labels = list(safeguards.keys())
    pcts = [pct_present(c) for c in safeguards.values()]
    tiers = ["Tier 1"] * 2 + ["Tier 2"] * 12
    colors = [BLUE if t == "Tier 1" else ORANGE for t in tiers]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(labels)), pcts, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("% of 1,757 use cases with safeguard present")
    ax.set_title("EO 13960 Governance Gap: Tier-1 vs. Tier-2 Safeguard Adoption\n"
                 "(McKinsey Constraints: C3 Security/Reliability, C4 Compliance)")

    # Percentage labels
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
                f"{pct:.1f}%", va="center", fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=BLUE, label="Tier 1 — Basic Controls"),
        Patch(facecolor=ORANGE, label="Tier 2 — Deep Governance"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    ax.set_xlim(0, max(pcts) * 1.15)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "governance_gap_bar.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 2: AIID Technical Failure Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def fig_aiid_heatmap():
    print("  Fig 2: aiid_failure_heatmap ...")
    df = pd.read_csv(OUTPUT / "aiid_incidents_classified.csv")
    df = df.dropna(subset=["Known AI Technical Failure", "Sector of Deployment"])

    # Explode multi-value failures
    df["failure_list"] = df["Known AI Technical Failure"].str.split(", ")
    df_exp = df.explode("failure_list")

    # Top failures and sectors
    top_fail = df_exp["failure_list"].value_counts().head(8).index.tolist()
    top_sect = df_exp["Sector of Deployment"].value_counts().head(8).index.tolist()

    df_filt = df_exp[df_exp["failure_list"].isin(top_fail) & df_exp["Sector of Deployment"].isin(top_sect)].reset_index(drop=True)
    pivot = pd.crosstab(df_filt["failure_list"], df_filt["Sector of Deployment"])
    # Reorder
    pivot = pivot.loc[[f for f in top_fail if f in pivot.index],
                      [s for s in top_sect if s in pivot.columns]]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([textwrap.fill(s, 20) for s in pivot.columns],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([textwrap.fill(f, 25) for f in pivot.index], fontsize=8)

    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if val > 0:
                color = "white" if val > pivot.values.max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8, color=color)

    ax.set_title("AIID: Technical Failure Type × Sector of Deployment\n"
                 "(Source: GMF classifications, n=" + str(len(df_filt)) + " classified links)")
    fig.colorbar(im, ax=ax, label="Incident count", shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "aiid_failure_heatmap.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 3: ATLAS Tactic Frequency
# ═══════════════════════════════════════════════════════════════════════════════

def fig_atlas_tactic():
    print("  Fig 3: atlas_tactic_frequency ...")
    import yaml

    # Load tactic names
    with open(BASE / "data" / "raw" / "atlas" / "data" / "tactics.yaml") as f:
        tactic_names = {t["id"]: t["name"] for t in yaml.safe_load(f)}

    df = pd.read_csv(OUTPUT / "atlas_cases_enriched.csv")
    # Explode tactics
    all_tactics = []
    for _, row in df.iterrows():
        if pd.notna(row["tactics"]) and row["tactics"]:
            for tid in row["tactics"].split("|"):
                all_tactics.append(tid.strip())

    tactic_counts = pd.Series(all_tactics).value_counts()
    labels = [f"{tactic_names.get(tid, tid)}" for tid in tactic_counts.index]
    constraint_tags = []
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from importlib import import_module
    mod = import_module("01_cross_taxonomy_mapping")
    ATLAS_TACTIC_TO_CONSTRAINT = mod.ATLAS_TACTIC_TO_CONSTRAINT
    for tid in tactic_counts.index:
        cids = ATLAS_TACTIC_TO_CONSTRAINT.get(tid, [])
        constraint_tags.append(", ".join(cids) if cids else "")

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(labels)), tactic_counts.values, color=BLUE,
                   edgecolor="white", height=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of case studies using this tactic")
    ax.set_title("MITRE ATLAS: Tactic Frequency Across 52 Case Studies\n"
                 "(McKinsey constraint tags shown)")

    for bar, count, ctag in zip(bars, tactic_counts.values, constraint_tags):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{count}  [{ctag}]", va="center", fontsize=8, color=GREY)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "atlas_tactic_frequency.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 4: McKinsey Constraint Coverage by Data Source
# ═══════════════════════════════════════════════════════════════════════════════

def fig_constraint_coverage():
    print("  Fig 4: constraint_coverage ...")
    mapping = pd.read_csv(OUTPUT / "cross_taxonomy_map.csv")

    constraints = ["C1", "C2", "C3", "C4", "C5", "C7", "C8"]
    sources = ["ATLAS", "AIID-GMF", "AIID-CSET", "EO13960"]
    source_labels = ["ATLAS\n(Threats)", "AIID-GMF\n(Failures)", "AIID-CSET\n(Harms)", "EO 13960\n(Practice)"]

    # Build count matrix
    matrix = np.zeros((len(constraints), len(sources)))
    for i, cid in enumerate(constraints):
        for j, src in enumerate(sources):
            matrix[i, j] = len(mapping[(mapping["target_id"] == cid) & (mapping["source_taxonomy"] == src)])

    constraint_labels = []
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from importlib import import_module
    mod = import_module("01_cross_taxonomy_mapping")
    MCKINSEY_CONSTRAINTS = mod.MCKINSEY_CONSTRAINTS
    for cid in constraints:
        info = MCKINSEY_CONSTRAINTS[cid]
        constraint_labels.append(f"{cid} ({info['pct']}%)\n{textwrap.fill(info['label'], 25)}")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(constraints))
    width = 0.2
    colors = [BLUE, ORANGE, RED, GREEN]

    for j, (src_label, color) in enumerate(zip(source_labels, colors)):
        offset = (j - 1.5) * width
        ax.bar(x + offset, matrix[:, j], width, label=src_label, color=color, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(constraint_labels, fontsize=8, ha="center")
    ax.set_ylabel("Number of mapping links")
    ax.set_title("Cross-Taxonomy Coverage: Which Data Sources Address Each McKinsey Constraint?")
    ax.legend(loc="upper right", fontsize=8)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "constraint_coverage.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 5: Governance Readiness Distribution
# ═══════════════════════════════════════════════════════════════════════════════

def fig_readiness_dist():
    print("  Fig 5: governance_readiness_dist ...")
    eo = pd.read_csv(OUTPUT / "eo13960_scored.csv", low_memory=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: histogram of total readiness %
    ax = axes[0]
    ax.hist(eo["gov_readiness_pct"], bins=20, color=BLUE, edgecolor="white", alpha=0.8)
    ax.axvline(eo["gov_readiness_pct"].mean(), color=RED, linestyle="--", linewidth=2,
               label=f"Mean = {eo['gov_readiness_pct'].mean():.1f}%")
    ax.set_xlabel("Governance Readiness Score (%)")
    ax.set_ylabel("Number of use cases")
    ax.set_title("Distribution of Governance Readiness\n(EO 13960, n=1,757)")
    ax.legend()

    # Right: Tier-1 vs Tier-2 scatter
    ax = axes[1]
    jitter1 = eo["tier1_score"] + np.random.uniform(-0.1, 0.1, len(eo))
    jitter2 = eo["tier2_score"] + np.random.uniform(-0.1, 0.1, len(eo))
    ax.scatter(jitter1, jitter2, alpha=0.15, s=10, color=BLUE)
    ax.set_xlabel(f"Tier-1 Score (Basic Controls, max={eo['tier1_max'].iloc[0]})")
    ax.set_ylabel(f"Tier-2 Score (Deep Governance, max={eo['tier2_max'].iloc[0]})")
    ax.set_title("Tier-1 vs. Tier-2 Governance Scores\n(The governance gap visualized)")

    # Quadrant labels
    ax.axhline(y=eo["tier2_max"].iloc[0] / 2, color=GREY, linestyle=":", alpha=0.5)
    ax.axvline(x=eo["tier1_max"].iloc[0] / 2, color=GREY, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "governance_readiness_dist.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("GENERATING FIGURES")
    print("=" * 72)

    fig_governance_gap()
    fig_aiid_heatmap()
    fig_atlas_tactic()
    fig_constraint_coverage()
    fig_readiness_dist()

    print("\n" + "=" * 72)
    print(f"All figures saved to {FIG_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
