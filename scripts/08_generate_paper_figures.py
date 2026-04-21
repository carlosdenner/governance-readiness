"""
08_generate_paper_figures.py
============================
Generates the four priority publication-ready figures for the AMCIS 2026 paper:

  1. fig1_governance_dropoff.png   — Surface Compliance vs. Substantive Safeguards
  2. fig2_commercial_opacity.png   — Vendor/COTS vs. In-House governance comparison
  3. fig3_sector_harm_heatmap.png  — Sector × Harm Fingerprints (AIID)
  4. fig4_threat_reality_practice.png — Three-source divergence comparison
"""

from __future__ import annotations
import pathlib
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

BASE = pathlib.Path(__file__).resolve().parent.parent
RAW_EO = BASE / "data" / "raw" / "eo13960" / "2024_consolidated_ai_inventory_raw.csv"
PROCESSED = BASE / "data" / "processed"
FIG_DIR = BASE / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette ──────────────────────────────────────────────────────────
TIER1_BLUE  = "#2563EB"
TIER2_AMBER = "#D97706"
VENDOR_RED  = "#DC2626"
INHOUSE_GRN = "#059669"
GREY        = "#6B7280"
DARK        = "#1F2937"

plt.rcParams.update({
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _load_eo_raw():
    """Load and normalise EO 13960 column names."""
    df = pd.read_csv(RAW_EO, low_memory=False)
    df.columns = [
        re.sub(r"\s+", "_", c.strip().lower().replace("(", "").replace(")", ""))
        for c in df.columns
    ]
    return df


def _is_yes(series: pd.Series) -> pd.Series:
    """Return boolean Series: True if value looks like a 'present' safeguard."""
    s = series.fillna("").astype(str).str.strip().str.lower()
    return ~s.isin(["", "no", "n/a", "nan", "none", "na", "false"])


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 1: The Governance Drop-off
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_governance_dropoff():
    print("  Fig 1: Governance Drop-off …")
    raw = _load_eo_raw()
    n = len(raw)
    assert n == 1757, f"Expected 1757 rows, got {n}"

    # Sanity-check key counts against paper_stats.csv
    ato_yes = _is_yes(raw["40_has_ato"]).sum()
    assert ato_yes == 654, f"ATO is_yes expected 654, got {ato_yes}"
    ir_yes = _is_yes(raw["50_internal_review"]).sum()
    assert ir_yes == 1067, f"Internal review expected 1067, got {ir_yes}"
    dm_yes = _is_yes(raw["62_disparity_mitigation"]).sum()
    assert dm_yes == 104, f"Disparity mitigation expected 104, got {dm_yes}"

    # Safeguards in the order from Table 2 (highest Tier 1 first)
    safeguards = [
        ("Internal review /\napproval",          "50_internal_review",      "tier1"),
        ("Authorization to\nOperate (ATO)",     "40_has_ato",              "tier1"),
        ("Post-deployment\nmonitoring",          "56_monitor_postdeploy",   "tier2"),
        ("Real-world\ntesting",                  "53_real_world_testing",   "tier2"),
        ("Appeal\nprocess",                      "65_appeal_process",       "tier2"),
        ("AI use notice\nto public",             "59_ai_notice",           "tier2"),
        ("Independent\nevaluation",              "55_independent_eval",     "tier2"),
        ("Disparity / bias\nmitigation",         "62_disparity_mitigation", "tier2"),
        ("Impact\nassessment",                   "52_impact_assessment",    "tier2"),
    ]

    labels  = [s[0] for s in safeguards]
    pcts    = [_is_yes(raw[s[1]]).sum() / n * 100 for s in safeguards]
    tiers   = [s[2] for s in safeguards]
    colors  = [TIER1_BLUE if t == "tier1" else TIER2_AMBER for t in tiers]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(labels)), pcts, color=colors, edgecolor="white", height=0.72)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% of 1,757 federal AI use cases reporting safeguard")
    ax.set_title("The Governance Drop-off:\nSurface Compliance vs. Substantive Safeguards (EO 13960)")

    # Percentage labels
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=8, fontweight="bold")

    # Annotation arrow for the drop
    drop = pcts[0] - pcts[-1]
    ax.annotate(
        f"≈ {drop:.0f} pp\ndrop-off",
        xy=(pcts[-1], len(labels) - 1),
        xytext=(pcts[0] * 0.65, len(labels) - 2.5),
        fontsize=8, color=VENDOR_RED, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=VENDOR_RED, lw=1.5),
        ha="center",
    )

    legend = [
        mpatches.Patch(facecolor=TIER1_BLUE, label="Tier 1 — Surface compliance"),
        mpatches.Patch(facecolor=TIER2_AMBER, label="Tier 2 — Substantive safeguards"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_xlim(0, max(pcts) * 1.18)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_governance_dropoff.png", bbox_inches="tight")
    plt.close(fig)
    print(f"    → saved {FIG_DIR / 'fig1_governance_dropoff.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 2: Commercial Opacity as a Governance Barrier
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_commercial_opacity():
    print("  Fig 2: Commercial Opacity …")
    raw = _load_eo_raw()

    # Classify procurement model
    dev = raw["22_dev_method"].fillna("").astype(str).str.strip().str.lower()
    raw["procurement"] = "other"
    raw.loc[dev.str.contains("contracting|contractor|vendor|commercial|cots", na=False)
            & ~dev.str.contains("in-house|both", na=False), "procurement"] = "vendor"
    raw.loc[dev.str.contains("in-house", na=False)
            & ~dev.str.contains("contracting|contractor|vendor|both", na=False), "procurement"] = "in-house"
    raw.loc[dev.str.contains("both", na=False), "procurement"] = "mixed"

    # Focus on vendor vs in-house
    subset = raw[raw["procurement"].isin(["vendor", "in-house"])].copy()
    n_vendor  = (subset["procurement"] == "vendor").sum()
    n_inhouse = (subset["procurement"] == "in-house").sum()

    # Governance indicators
    indicators = [
        ("Source code\naccess",          "38_code_access"),
        ("Data\ndocumentation",          "34_data_docs"),
        ("Impact\nassessment",           "52_impact_assessment"),
        ("Independent\nevaluation",      "55_independent_eval"),
        ("Appeal\nprocess",              "65_appeal_process"),
        ("Post-deployment\nmonitoring",  "56_monitor_postdeploy"),
        ("Disparity / bias\nmitigation", "62_disparity_mitigation"),
    ]

    labels = [i[0] for i in indicators]
    vendor_pcts = []
    inhouse_pcts = []
    for _, col in indicators:
        v = subset[subset["procurement"] == "vendor"]
        h = subset[subset["procurement"] == "in-house"]
        vendor_pcts.append(_is_yes(v[col]).sum() / n_vendor * 100)
        inhouse_pcts.append(_is_yes(h[col]).sum() / n_inhouse * 100)

    y = np.arange(len(labels))
    height = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.barh(y - height / 2, inhouse_pcts, height, color=INHOUSE_GRN,
                    edgecolor="white", label=f"In-house (n={n_inhouse})")
    bars2 = ax.barh(y + height / 2, vendor_pcts, height, color=VENDOR_RED,
                    edgecolor="white", label=f"Vendor/Contractor (n={n_vendor})")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% of use cases reporting safeguard")
    ax.set_title(
        "Commercial Opacity as a Governance Barrier:\n"
        "Vendor-Supplied vs. In-House AI Systems (EO 13960)"
    )

    # Value labels
    for bar, pct in zip(bars1, inhouse_pcts):
        if pct > 0:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", fontsize=7, color=INHOUSE_GRN,
                    fontweight="bold")
    for bar, pct in zip(bars2, vendor_pcts):
        if pct > 0:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", fontsize=7, color=VENDOR_RED,
                    fontweight="bold")

    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_xlim(0, max(max(vendor_pcts), max(inhouse_pcts)) * 1.25)

    # Experiment annotations
    ax.text(0.98, 0.02,
            "Convergent: EXP_237, 245, 247, 293 (code access)\n"
            "EXP_174, 284, 291 (documentation)\n"
            "EXP_131 (+0.402), EXP_210 (+0.204) (transparency→appeal)",
            transform=ax.transAxes, fontsize=6, ha="right", va="bottom",
            style="italic", color=GREY,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3F4F6", edgecolor=GREY, alpha=0.7))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_commercial_opacity.png", bbox_inches="tight")
    plt.close(fig)
    print(f"    → saved {FIG_DIR / 'fig2_commercial_opacity.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 3: Sector–Harm Fingerprints
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_sector_harm_heatmap():
    print("  Fig 3: Sector–Harm Fingerprints …")
    df = pd.read_csv(PROCESSED / "aiid_incidents_classified.csv", low_memory=False)

    # Simplify sector names
    sector_map = {
        "information and communication": "Information &\nCommunication",
        "transportation and storage": "Transportation",
        "wholesale and retail trade": "Retail & Trade",
        "human health and social work activities": "Healthcare",
        "law enforcement": "Law Enforcement",
        "financial and insurance activities": "Finance &\nInsurance",
        "public administration": "Public\nAdministration",
        "Education": "Education",
        "administrative and support service activities": "Admin\nServices",
        "manufacturing": "Manufacturing",
    }

    # Clean sector - take primary sector (before comma)
    df["sector_clean"] = df["Sector of Deployment"].fillna("").apply(
        lambda x: x.split(",")[0].strip() if x else ""
    )
    df = df[df["sector_clean"].isin(sector_map.keys())].copy()
    df["sector_label"] = df["sector_clean"].map(sector_map)

    # Use Known AI Technical Failure, take primary failure
    df["failure_clean"] = df["Known AI Technical Failure"].fillna("").apply(
        lambda x: x.split(",")[0].strip() if x else ""
    )
    df = df[df["failure_clean"] != ""].copy()

    # Simplify failure names
    failure_map = {
        "Generalization Failure": "Generalization\nFailure",
        "Misinformation Generation Hazard": "Misinformation\nGeneration",
        "Distributional Bias": "Distributional\nBias",
        "Context Misidentification": "Context Mis-\nidentification",
        "Lack of Transparency": "Lack of\nTransparency",
        "Unsafe Exposure or Access": "Unsafe Exposure\n/ Access",
        "Harmful Application": "Harmful\nApplication",
        "Algorithmic Bias": "Algorithmic\nBias",
    }
    df = df[df["failure_clean"].isin(failure_map.keys())].copy()
    df["failure_label"] = df["failure_clean"].map(failure_map)

    # Use tangible harm as the harm dimension
    harm_categories = {
        "tangible harm definitively occurred": "Tangible Harm\n(definitive)",
        "no tangible harm, near-miss, or issue": "No Tangible Harm\n/ Near-miss",
        "imminent risk of tangible harm (near miss) did occur": "Tangible Harm\n(near-miss)",
        "non-imminent risk of tangible harm (an issue) occurred": "Tangible Harm\n(issue)",
    }

    # Build sector × failure crosstab
    pivot = pd.crosstab(df["failure_label"], df["sector_label"])

    # Sort by total
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]
    pivot = pivot[pivot.sum(axis=0).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)

    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if val > 0:
                color = "white" if val > pivot.values.max() * 0.55 else DARK
                ax.text(j, i, str(int(val)), ha="center", va="center",
                        fontsize=8, fontweight="bold", color=color)

    ax.set_title(
        "Sector\u2013Harm Fingerprints:\n"
        f"AI Technical Failure × Sector of Deployment (AIID, n={len(df)} classified incidents)"
    )
    cbar = fig.colorbar(im, ax=ax, label="Incident count", shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=7)

    # Experiment annotations
    ax.text(0.98, -0.18,
            "Replicates: EXP_158, 170, 187, 242, 252 (sector–harm association)\n"
            "EXP_168: biometrics → civil rights (p<0.01)  |  EXP_173: χ²=12.97, p=0.0003",
            transform=ax.transAxes, fontsize=6, ha="right", va="top",
            style="italic", color=GREY)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_sector_harm_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print(f"    → saved {FIG_DIR / 'fig3_sector_harm_heatmap.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 4: Threat–Reality–Practice Divergence
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_threat_reality_practice():
    print("  Fig 4: Threat–Reality–Practice Divergence …")

    # ── ATLAS: classify case studies by target domain ──
    atlas = pd.read_csv(PROCESSED / "atlas_cases_enriched.csv", low_memory=False)

    # Manual sector classification based on ATLAS case study names/summaries
    atlas_sectors = {
        "Technology / AI Services": 18,
        "Finance / Crypto": 6,
        "Government / Defense": 8,
        "Healthcare": 3,
        "Social Media / Consumer": 7,
        "Transportation / Auto": 4,
        "Education / Research": 6,
    }

    # ── AIID: sector distribution from incidents ──
    aiid = pd.read_csv(PROCESSED / "aiid_incidents_classified.csv", low_memory=False)
    sector_clean = aiid["Sector of Deployment"].fillna("").apply(
        lambda x: x.split(",")[0].strip() if x else ""
    )
    sector_counts = sector_clean[sector_clean != ""].value_counts()

    aiid_sectors = {
        "Technology / AI Services": sector_counts.get("information and communication", 0) +
                                     sector_counts.get("administrative and support service activities", 0),
        "Finance / Crypto": sector_counts.get("financial and insurance activities", 0),
        "Government / Defense": sector_counts.get("public administration", 0) +
                                 sector_counts.get("law enforcement", 0),
        "Healthcare": sector_counts.get("human health and social work activities", 0),
        "Social Media / Consumer": sector_counts.get("Arts, entertainment and recreation", 0) +
                                    sector_counts.get("wholesale and retail trade", 0),
        "Transportation / Auto": sector_counts.get("transportation and storage", 0),
        "Education / Research": sector_counts.get("Education", 0),
    }

    # ── EO 13960: sector distribution from topic areas ──
    eo = _load_eo_raw()
    topic = eo["8_topic_area"].fillna("").astype(str).str.strip()
    topic_counts = topic.value_counts()

    eo_sectors = {
        "Technology / AI Services": topic_counts.get("Mission-Enabling", 0) +
                                     topic_counts.get("Mission-Enabling (internal agency support)", 0) +
                                     topic_counts.get("Mission-Enabling (internal agency support) ", 0),
        "Finance / Crypto": 0,  # federal - no explicit finance sector
        "Government / Defense": topic_counts.get("Law & Justice", 0) +
                                 topic_counts.get("Diplomacy & Trade", 0) +
                                 topic_counts.get("Government Services (includes Benefits and Service Delivery)", 0),
        "Healthcare": topic_counts.get("Health & Medical", 0),
        "Social Media / Consumer": topic_counts.get("Other", 0),
        "Transportation / Auto": topic_counts.get("Transportation", 0),
        "Education / Research": topic_counts.get("Education & Workforce", 0) +
                                 topic_counts.get("Science & Space", 0),
    }

    sectors = list(atlas_sectors.keys())

    # Normalise to percentages
    atlas_pcts = np.array([atlas_sectors[s] for s in sectors], dtype=float)
    atlas_pcts = atlas_pcts / atlas_pcts.sum() * 100

    aiid_pcts = np.array([aiid_sectors[s] for s in sectors], dtype=float)
    aiid_total = aiid_pcts.sum()
    aiid_pcts = aiid_pcts / (aiid_total if aiid_total > 0 else 1) * 100

    eo_pcts = np.array([eo_sectors[s] for s in sectors], dtype=float)
    eo_total = eo_pcts.sum()
    eo_pcts = eo_pcts / (eo_total if eo_total > 0 else 1) * 100

    sector_labels = [s.replace(" / ", "\n/ ") for s in sectors]

    x = np.arange(len(sectors))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars1 = ax.bar(x - width, atlas_pcts, width, color="#EF4444", edgecolor="white",
                   label="ATLAS — Threat research\n(what adversaries target)")
    bars2 = ax.bar(x, aiid_pcts, width, color="#F59E0B", edgecolor="white",
                   label="AIID — Incident reality\n(what actually fails)")
    bars3 = ax.bar(x + width, eo_pcts, width, color="#3B82F6", edgecolor="white",
                   label="EO 13960 — Governance practice\n(where AI is deployed)")

    ax.set_xticks(x)
    ax.set_xticklabels(sector_labels, fontsize=7, ha="center")
    ax.set_ylabel("% of source records in sector")
    ax.set_title(
        "Threat\u2013Reality\u2013Practice Divergence:\n"
        "Sector Distributions Across Three Data Sources"
    )
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # Mismatch annotations
    ax.annotate(
        "Threat–Reality\nMismatch\n(EXP_164)",
        xy=(0, atlas_pcts[0]), xytext=(0.8, atlas_pcts[0] + 8),
        fontsize=7, color="#EF4444", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#EF4444", lw=1.2),
        ha="center",
    )
    ax.annotate(
        "Risk–Investment\nMismatch\n(EXP_108)",
        xy=(3 + width, eo_pcts[3]), xytext=(4.2, eo_pcts[3] + 10),
        fontsize=7, color="#3B82F6", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#3B82F6", lw=1.2),
        ha="center",
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_threat_reality_practice.png", bbox_inches="tight")
    plt.close(fig)
    print(f"    → saved {FIG_DIR / 'fig4_threat_reality_practice.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("GENERATING PAPER FIGURES (Option A)")
    print("=" * 72)

    fig1_governance_dropoff()
    fig2_commercial_opacity()
    fig3_sector_harm_heatmap()
    fig4_threat_reality_practice()

    print("\n" + "=" * 72)
    print(f"All figures saved to {FIG_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
