"""
13_aiid_coverage_robustness.py
==============================
Addresses reviewer concern about small AIID heatmap subset (n=69–135).
Does three things:

1. Prints an AIID coverage cascade table for the paper.
2. Text-mines sector and harm-type from ALL 1,362 incident descriptions
   using keyword dictionaries, producing a "text-mined" sector-harm matrix.
3. Compares structured-label and text-mined associations to show consistency.

Outputs:
    - Console: coverage table + chi-square tests on both subsets
    - data/processed/aiid_coverage_robustness.csv
"""

from __future__ import annotations
import pathlib, re
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

BASE = pathlib.Path(__file__).resolve().parent.parent
AIID_CSV = BASE / "data" / "processed" / "aiid_incidents_classified.csv"
OUT_DIR = BASE / "data" / "processed"


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Coverage cascade
# ═══════════════════════════════════════════════════════════════════════════════

def coverage_cascade(df: pd.DataFrame):
    print("=" * 72)
    print("AIID COVERAGE CASCADE")
    print("=" * 72)

    total = len(df)
    has_cset = int(df["has_cset"].sum())
    has_gmf = int(df["has_gmf"].sum())
    has_either = int((df["has_cset"] | df["has_gmf"]).sum())
    has_sector = int(df["Sector of Deployment"].notna().sum())
    has_tech_failure = int(df["Known AI Technical Failure"].notna().sum())
    has_harm = int(df["Harm Domain"].notna().sum())
    has_both_sf = int((df["Sector of Deployment"].notna() & df["Known AI Technical Failure"].notna()).sum())
    has_desc = int(df["description"].notna().sum() & (df["description"].str.len() > 20).sum())

    rows = [
        ("Total AIID incidents", total, ""),
        ("With text description (>20 chars)", has_desc, f"{has_desc/total*100:.0f}%"),
        ("With CSET taxonomy", has_cset, f"{has_cset/total*100:.0f}%"),
        ("With GMF taxonomy", has_gmf, f"{has_gmf/total*100:.0f}%"),
        ("With either taxonomy", has_either, f"{has_either/total*100:.0f}%"),
        ("With sector label (CSET)", has_sector, f"{has_sector/total*100:.0f}%"),
        ("With technical failure (GMF)", has_tech_failure, f"{has_tech_failure/total*100:.0f}%"),
        ("With harm domain (CSET)", has_harm, f"{has_harm/total*100:.0f}%"),
        ("With sector + tech failure (heatmap input)", has_both_sf, f"{has_both_sf/total*100:.0f}%"),
    ]

    print(f"\n  {'Layer':<50s} {'n':>6s}  {'%':>5s}")
    print(f"  {'─'*65}")
    for label, n, pct in rows:
        print(f"  {label:<50s} {n:>6d}  {pct:>5s}")

    return {
        "total": total,
        "has_cset": has_cset,
        "has_gmf": has_gmf,
        "has_either": has_either,
        "has_sector": has_sector,
        "has_tech_failure": has_tech_failure,
        "has_harm": has_harm,
        "has_both_sf": has_both_sf,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Text-mine sector and harm from descriptions
# ═══════════════════════════════════════════════════════════════════════════════

SECTOR_KEYWORDS = {
    "Healthcare": r"\b(health|hospital|medical|patient|clinical|pharma|drug|diagnos|doctor|nurse|surgery|disease)\b",
    "Finance": r"\b(bank|financ|credit|loan|insur|trading|stock|invest|mortgage|fintech)\b",
    "Transportation": r"\b(transport|driving|autonomous.vehicle|self.driving|car|traffic|aviation|flight|drone|uber|lyft|tesla|waymo)\b",
    "Law enforcement": r"\b(police|law.enforcement|surveillance|facial.recognition|criminal|prison|parole|arrest|sentencing|recidivism)\b",
    "Education": r"\b(school|university|education|student|teacher|academic|exam|grading|classroom)\b",
    "Government": r"\b(government|federal|public.sector|municipal|military|defense|intelligence|state.agency)\b",
    "Social media": r"\b(social.media|facebook|twitter|instagram|tiktok|youtube|platform|content.moderation|misinformation)\b",
    "Retail/Commerce": r"\b(retail|commerce|e.commerce|shopping|amazon|product|recommendation|customer.service)\b",
    "Employment": r"\b(hiring|recruit|employ|resume|job.applicant|workforce|HR|human.resources)\b",
}

HARM_KEYWORDS = {
    "Physical harm": r"\b(death|killed|injury|injur|crash|accident|physical.harm|fatality|collision|bodily)\b",
    "Discrimination/bias": r"\b(bias|discriminat|racial|gender.bias|unfair|disparate|prejudice|racist|sexist|stereotyp)\b",
    "Privacy violation": r"\b(privacy|surveillance|personal.data|tracking|data.breach|biometric|facial.recognition.privacy|PII|GDPR)\b",
    "Economic harm": r"\b(financial.loss|economic.harm|fraud|scam|monetary|cost|damages|penalty|fine)\b",
    "Misinformation": r"\b(misinformation|disinformation|fake|hallucination|fabricat|false.information|deepfake|misleading)\b",
    "Civil rights": r"\b(civil.rights|civil.liberties|freedom|constitutional|due.process|equal.protection|human.rights|voting)\b",
}


def text_mine_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Assign sector and harm labels based on keyword matching in descriptions."""
    desc = df["description"].fillna("").str.lower()

    for sector, pattern in SECTOR_KEYWORDS.items():
        df[f"tm_sector_{sector}"] = desc.str.contains(pattern, regex=True, na=False).astype(int)

    for harm, pattern in HARM_KEYWORDS.items():
        df[f"tm_harm_{harm}"] = desc.str.contains(pattern, regex=True, na=False).astype(int)

    # Assign primary sector (first match, or "Other")
    sector_cols = [f"tm_sector_{s}" for s in SECTOR_KEYWORDS]
    df["tm_sector"] = "Other"
    for s in SECTOR_KEYWORDS:
        mask = df[f"tm_sector_{s}"] == 1
        df.loc[mask & (df["tm_sector"] == "Other"), "tm_sector"] = s

    # Assign primary harm (first match, or "Other")
    harm_cols = [f"tm_harm_{h}" for h in HARM_KEYWORDS]
    df["tm_harm"] = "Other"
    for h in HARM_KEYWORDS:
        mask = df[f"tm_harm_{h}"] == 1
        df.loc[mask & (df["tm_harm"] == "Other"), "tm_harm"] = h

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Robustness comparison
# ═══════════════════════════════════════════════════════════════════════════════

def structured_analysis(df: pd.DataFrame):
    """Chi-square on CSET structured labels (n≈135)."""
    print("\n" + "=" * 72)
    print("STRUCTURED LABEL ANALYSIS (CSET sector × GMF tech failure)")
    print("=" * 72)

    sub = df.dropna(subset=["Sector of Deployment", "Known AI Technical Failure"])
    print(f"  n = {len(sub)} incidents with both labels")

    # Simplify sector to primary
    sub = sub.copy()
    sub["sector_simple"] = sub["Sector of Deployment"].str.split(",").str[0].str.strip()
    top_sectors = sub["sector_simple"].value_counts().head(6).index.tolist()
    sub = sub[sub["sector_simple"].isin(top_sectors)]

    # Simplify failure to primary
    sub["fail_simple"] = sub["Known AI Technical Failure"].str.split(",").str[0].str.strip()
    top_fails = sub["fail_simple"].value_counts().head(6).index.tolist()
    sub = sub[sub["fail_simple"].isin(top_fails)]

    ct = pd.crosstab(sub["sector_simple"], sub["fail_simple"])
    print(f"\n  Cross-tab (top 6 × top 6), n={len(sub)}:")
    print(ct.to_string())

    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        chi2, p, dof, _ = chi2_contingency(ct)
        print(f"\n  Chi-square = {chi2:.2f}, df = {dof}, p = {p:.4f}")
        v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        print(f"  Cramér's V = {v:.3f}")
    return ct


def textmined_analysis(df: pd.DataFrame):
    """Chi-square on text-mined labels (n=all with sector + harm match)."""
    print("\n" + "=" * 72)
    print("TEXT-MINED ANALYSIS (keyword sector × keyword harm, full corpus)")
    print("=" * 72)

    sub = df[(df["tm_sector"] != "Other") & (df["tm_harm"] != "Other")].copy()
    print(f"  n = {len(sub)} incidents with both text-mined labels")

    ct = pd.crosstab(sub["tm_sector"], sub["tm_harm"])
    print(f"\n  Cross-tab:")
    print(ct.to_string())

    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        chi2, p, dof, _ = chi2_contingency(ct)
        print(f"\n  Chi-square = {chi2:.2f}, df = {dof}, p = {p:.4f}")
        v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        print(f"  Cramér's V = {v:.3f}")

    # Check specific fingerprints
    print("\n  Key fingerprint checks:")
    for sector, harm in [("Finance", "Economic harm"),
                          ("Healthcare", "Physical harm"),
                          ("Law enforcement", "Civil rights"),
                          ("Law enforcement", "Discrimination/bias"),
                          ("Social media", "Misinformation")]:
        if sector in ct.index and harm in ct.columns:
            val = ct.loc[sector, harm]
            row_total = ct.loc[sector].sum()
            pct = val / row_total * 100 if row_total > 0 else 0
            print(f"    {sector:20s} × {harm:25s}: {val:3d} / {row_total:3d} ({pct:.0f}%)")

    return ct


def concordance_check(df: pd.DataFrame):
    """For incidents with BOTH structured and text-mined labels, check agreement."""
    print("\n" + "=" * 72)
    print("CONCORDANCE: Structured vs. Text-Mined Labels")
    print("=" * 72)

    # Map structured sectors to simplified names
    sector_map = {
        "information and communication": "Social media",
        "transportation and storage": "Transportation",
        "human health and social work activities": "Healthcare",
        "financial and insurance activities": "Finance",
        "law enforcement": "Law enforcement",
        "Education": "Education",
        "public administration": "Government",
    }

    sub = df[df["Sector of Deployment"].notna() & (df["tm_sector"] != "Other")].copy()
    sub["struct_sector_simple"] = sub["Sector of Deployment"].str.split(",").str[0].str.strip()
    sub["struct_sector_mapped"] = sub["struct_sector_simple"].map(sector_map).fillna("Other")

    agree = (sub["struct_sector_mapped"] == sub["tm_sector"]).sum()
    total = len(sub)
    print(f"  Incidents with both labels: {total}")
    print(f"  Sector agreement: {agree}/{total} ({agree/total*100:.0f}%)" if total > 0 else "  No overlap")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df = pd.read_csv(AIID_CSV, low_memory=False)

    # 1. Coverage
    cov = coverage_cascade(df)

    # 2. Text-mine
    df = text_mine_labels(df)

    # Distribution of text-mined labels
    print("\n  Text-mined sector distribution:")
    print(df["tm_sector"].value_counts().to_string())
    print(f"\n  Text-mined harm distribution:")
    print(df["tm_harm"].value_counts().to_string())

    # 3. Analyses
    ct_struct = structured_analysis(df)
    ct_tm = textmined_analysis(df)
    concordance_check(df)

    # 4. Summary
    print("\n" + "=" * 72)
    print("SUMMARY FOR PAPER")
    print("=" * 72)
    sub_tm = df[(df["tm_sector"] != "Other") & (df["tm_harm"] != "Other")]
    sub_struct = df.dropna(subset=["Sector of Deployment", "Known AI Technical Failure"])
    print(f"  Structured subset (heatmap input): n={len(sub_struct)}")
    print(f"  Text-mined subset (robustness):    n={len(sub_tm)}")
    print(f"  Expansion factor: {len(sub_tm)/len(sub_struct):.1f}x")

    # Save
    out = OUT_DIR / "aiid_coverage_robustness.csv"
    summary = pd.DataFrame([cov])
    summary.to_csv(out, index=False)
    print(f"\n  Saved: {out}")
