"""
17_table2_split_tr.py
Generate exact coefficients, SEs, and fit stats for the revised Table 2.
M1-M3: same as original (script 09). M4: replace composite TR with TR-surface + TR-substantive.
"""
from __future__ import annotations
import pathlib, re, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
BASE = pathlib.Path(__file__).resolve().parent.parent

def _is_yes(s): return (~s.fillna("").astype(str).str.strip().str.lower().isin(["","no","n/a","nan","none","na","false"])).astype(int)
def _is_yes_strict(s): return s.fillna("").astype(str).str.strip().str.lower().str.startswith("yes").astype(int)
def _is_reuse(s):
    v = s.fillna("").astype(str).str.strip().str.lower()
    return ((v != "") & (~v.str.startswith("none"))).astype(int)

def build():
    df = pd.read_csv(BASE/"data"/"raw"/"eo13960"/"2024_consolidated_ai_inventory_raw.csv", low_memory=False)
    df.columns = [re.sub(r"\s+","_",c.strip().lower().replace("(","").replace(")","")) for c in df.columns]

    # TR items
    tr_surface = {"tr_ato":("40_has_ato",_is_yes),"tr_internal_rev":("50_internal_review",_is_yes)}
    tr_subst = {"tr_impact_assess":("52_impact_assessment",_is_yes),"tr_rw_testing":("53_real_world_testing",_is_yes),
                "tr_indep_eval":("55_independent_eval",_is_yes),"tr_postdeploy":("56_monitor_postdeploy",_is_yes),
                "tr_notice":("59_ai_notice",_is_yes),"tr_disparity":("62_disparity_mitigation",_is_yes),
                "tr_appeal":("65_appeal_process",_is_yes)}
    for n,(c,f) in {**tr_surface,**tr_subst}.items(): df[n]=f(df[c])
    df["TR"] = df[list(tr_surface)+list(tr_subst)].sum(axis=1)
    df["TR_surface"] = df[list(tr_surface)].sum(axis=1)
    df["TR_subst"] = df[list(tr_subst)].sum(axis=1)

    # IR items
    ir = {"ir_data_catalog":("31_data_catalog",_is_yes_strict),"ir_data_docs":("34_data_docs",_is_yes),
          "ir_custom_code":("37_custom_code",_is_yes_strict),"ir_code_access":("38_code_access",_is_yes),
          "ir_infra":("43_infra_provisioned",_is_yes_strict),"ir_timely_res":("47_timely_resources",_is_yes_strict),
          "ir_reuse":("49_existing_reuse",_is_reuse)}
    for n,(c,f) in ir.items(): df[n]=f(df[c])
    df["IR"] = df[list(ir)].sum(axis=1)

    # Outcome
    stage = df["16_dev_stage"].fillna("").astype(str).str.strip().str.lower()
    df["deployed"] = stage.isin(["operation and maintenance","implementation and assessment","in production","in mission"]).astype(int)

    # Dev method
    dev = df["22_dev_method"].fillna("").astype(str).str.strip().str.lower()
    df["vendor"] = (dev.str.contains("contracting|contractor|vendor|commercial|cots",na=False) & ~dev.str.contains("in-house|both",na=False)).astype(int)
    df["mixed_dev"] = dev.str.contains("both",na=False).astype(int)

    # Rights/safety (same as script 09: anything not blank or "neither")
    imp = df["17_impact_type"].fillna("").astype(str).str.strip().str.lower()
    df["rights_safety"] = (~imp.isin(["", "neither"])).astype(int)

    # Orientation (agency-level composite: z(log_n) + z(topic_breadth)) / 2 — same as script 09
    agency_stats = df.groupby("3_agency").agg(
        agency_n=("2_use_case_name", "size"),
        topic_breadth=("8_topic_area", "nunique"),
    )
    agency_stats["log_agency_n"] = np.log1p(agency_stats["agency_n"])
    df = df.merge(agency_stats, left_on="3_agency", right_index=True, how="left")
    df["z_log_n"] = (df["log_agency_n"] - df["log_agency_n"].mean()) / df["log_agency_n"].std()
    df["z_breadth"] = (df["topic_breadth"] - df["topic_breadth"].mean()) / df["topic_breadth"].std()
    df["orientation"] = (df["z_log_n"] + df["z_breadth"]) / 2

    return df

def fmt(coef, se, p):
    stars = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "+" if p<0.1 else ""
    return f"{coef:.3f}{stars} ({se:.3f})"

def run_models(df):
    y = df["deployed"]
    n = len(y)

    specs = {
        "M1": ["vendor","mixed_dev","rights_safety"],
        "M2": ["vendor","mixed_dev","rights_safety","orientation"],
        "M3": ["vendor","mixed_dev","rights_safety","orientation","TR","IR"],
        "M4": ["vendor","mixed_dev","rights_safety","orientation","TR_surface","TR_subst","IR"],
    }

    results = {}
    for name, cols in specs.items():
        X = sm.add_constant(df[cols].astype(float))
        m = sm.Logit(y, X).fit(disp=0, maxiter=200)
        results[name] = m
        print(f"\n{'='*60}")
        print(f"  {name}: {', '.join(cols)}")
        print(f"  N={n}, Pseudo R²={m.prsquared:.3f}, AIC={m.aic:.1f}")
        print(f"{'='*60}")
        for var in ["const"]+cols:
            print(f"  {var:<15s} {fmt(m.params[var], m.bse[var], m.pvalues[var])}")
        # ORs for key vars
        for var in cols:
            orv = np.exp(m.params[var])
            ci = np.exp(m.conf_int().loc[var])
            print(f"    → OR={orv:.3f} [{ci.iloc[0]:.3f}, {ci.iloc[1]:.3f}]")

    # Print markdown table
    all_vars = ["const","vendor","mixed_dev","rights_safety","orientation","TR","TR_surface","TR_subst","IR"]
    labels = {"const":"Constant","vendor":"Vendor","mixed_dev":"Mixed dev.","rights_safety":"Rights/safety",
              "orientation":"Orientation","TR":"TR (composite)","TR_surface":"TR-surface (0–2)",
              "TR_subst":"TR-substantive (0–7)","IR":"IR"}

    print("\n\n" + "="*80)
    print("MARKDOWN TABLE FOR PAPER")
    print("="*80)
    print(f"\n| Variable | M1 | M2 | M3 | M4 |")
    print(f"|:---------|:--:|:--:|:--:|:--:|")

    for var in all_vars:
        row = f"| {labels[var]} |"
        for name in ["M1","M2","M3","M4"]:
            m = results[name]
            if var in m.params.index:
                row += f" {fmt(m.params[var], m.bse[var], m.pvalues[var])} |"
            else:
                row += " |"
        print(row)

    # Fit stats
    print(f"| *N* | {n} | {n} | {n} | {n} |")
    for name in ["M1","M2","M3","M4"]:
        m = results[name]
    print(f"| Pseudo R² |", " | ".join(f"{results[n].prsquared:.3f}" for n in ["M1","M2","M3","M4"]), "|")
    print(f"| AIC |", " | ".join(f"{results[n].aic:.1f}" for n in ["M1","M2","M3","M4"]), "|")

if __name__ == "__main__":
    df = build()
    print(f"Dataset: N={len(df)}, deployed={df['deployed'].mean()*100:.1f}%")
    print(f"TR: mean={df['TR'].mean():.2f}, TR_surface: mean={df['TR_surface'].mean():.2f}, TR_subst: mean={df['TR_subst'].mean():.2f}")
    print(f"IR: mean={df['IR'].mean():.2f}")
    run_models(df)
