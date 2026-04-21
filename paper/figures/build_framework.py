import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(figsize=(17, 11))
ax.set_xlim(0, 17)
ax.set_ylim(0, 11)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Colors ─────────────────────────────────────────────────────────────────────
C_TR   = '#d8edd3'; C_TR_E  = '#3d7a52'
C_IR   = '#cfe0f5'; C_IR_E  = '#2560a0'
C_PORT = '#eaeaf5'; C_PORT_E= '#44446a'
C_BARR = '#fde8e8'; C_BARR_E= '#b33030'
C_DATA = '#f4f0e6'; C_DATA_E= '#776644'
C_OUT  = '#e4f2e4'; C_OUT_E = '#2a5a2a'
C_H    = '#b33030'
C_GR   = '#666666'

def rbox(x, y, w, h, fc, ec, lw=1.8, r=0.12):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))

def arr(x1, y1, x2, y2, color='#444444', lw=1.7):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=color, lw=lw, mutation_scale=13), zorder=5)

def txt(x, y, s, fs=9, bold=False, color='#222222', ha='center', va='center', italic=False):
    ax.text(x, y, s, fontsize=fs,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal',
            ha=ha, va=va, color=color, zorder=6)

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
# Top: Portfolio scope box
PX, PY, PW, PH = 4.5, 9.1, 6.0, 0.95
# TR box  (left)
TX, TY, TW, TH = 0.3, 5.85, 5.8, 2.85
# IR box  (right) — gap of 2.8 units between TR and IR for complementarity label
IX, IY, IW, IH = 8.9, 5.85, 5.8, 2.85
# Barrier box (centred below TR/IR)
BX, BY, BW, BH = 3.2, 4.2, 8.6, 0.95
# Outcome box: to the right of IR box, same vertical band
OX, OY, OW, OH = 15.0, 6.55, 1.75, 1.55
# Governance Theater bracket: right of barrier box
GT_X = BX + BW + 0.20
# Data sources (bottom)
DS_Y = 1.9   # centre y of ellipses

# ══════════════════════════════════════════════════════════════════════════════
# 1. TITLE
# ══════════════════════════════════════════════════════════════════════════════
txt(8.0, 10.22,
    'Figure 1. Theoretical Framework: Governance Readiness Gaps in Organizational AI Deployment',
    fs=9.5, bold=True)

# ══════════════════════════════════════════════════════════════════════════════
# 2. PORTFOLIO SCOPE BOX
# ══════════════════════════════════════════════════════════════════════════════
rbox(PX, PY, PW, PH, C_PORT, C_PORT_E, lw=2.0)
txt(PX+PW/2, PY+PH/2+0.17, 'Organizational AI Portfolio Scope', fs=11, bold=True, color='#333355')
txt(PX+PW/2, PY+PH/2-0.20, 'Agency-level proxy: z-scored log portfolio size + topic breadth',
    fs=7.5, color=C_GR)
# Orientation label (shorthand used in pathway model)
txt(PX+PW-0.25, PY+PH-0.22, '(Orientation)', fs=7.2, color='#888888', ha='right', italic=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3. ARROWS: Portfolio → TR, Portfolio → IR
# ══════════════════════════════════════════════════════════════════════════════
# Vertical stub down from centre of portfolio box
stub_y = PY
mid_x  = PX + PW/2
ax.plot([mid_x, mid_x], [stub_y, stub_y - 0.30], color='#444444', lw=1.7, zorder=4)
# Branch left to TR
arr(mid_x, stub_y-0.30, TX+TW/2, TY+TH, color='#444444')
# Branch right to IR
arr(mid_x, stub_y-0.30, IX+IW/2, IY+IH, color='#444444')

# ══════════════════════════════════════════════════════════════════════════════
# 4. TRUST READINESS BOX
# ══════════════════════════════════════════════════════════════════════════════
rbox(TX, TY, TW, TH, C_TR, C_TR_E, lw=2.0)
txt(TX+TW/2, TY+TH-0.30, 'Trust Readiness (TR)', fs=11, bold=True, color='#2d5a3d')

# TR-surface sub-box
rbox(TX+0.2, TY+1.72, TW-0.4, 0.70, '#b8dbb0', C_TR_E, lw=1.2, r=0.08)
txt(TX+0.55, TY+2.12, 'H1+', fs=8.5, bold=True, color=C_H, ha='left')
txt(TX+TW/2+0.2, TY+2.12,
    'TR-surface (0–2): Internal review  ·  Authorization to Operate (ATO)',
    fs=8.2, color='#2d5a3d')

# TR-substantive sub-box
rbox(TX+0.2, TY+0.72, TW-0.4, 0.88, '#b8dbb0', C_TR_E, lw=1.2, r=0.08)
txt(TX+0.55, TY+1.22, 'H2−', fs=8.5, bold=True, color=C_H, ha='left')
txt(TX+TW/2+0.2, TY+1.22,
    'TR-substantive (0–7): Impact assessment  ·  Independent evaluation',
    fs=7.9, color='#2d5a3d')
txt(TX+TW/2+0.1, TY+0.90,
    'Real-world testing  ·  Bias mitigation  ·  AI notice  ·  Appeal process',
    fs=7.6, color='#2d5a3d')

# TR surface→substantive continuum label
txt(TX+TW/2, TY+0.44, 'Surface compliance  ──────────→  Substantive safeguards',
    fs=7.5, color='#3d7a52', italic=True)

# ══════════════════════════════════════════════════════════════════════════════
# 5. INTEGRATION READINESS BOX
# ══════════════════════════════════════════════════════════════════════════════
rbox(IX, IY, IW, IH, C_IR, C_IR_E, lw=2.0)
txt(IX+IW/2, IY+IH-0.30, 'Integration Readiness (IR)', fs=11, bold=True, color='#1a4d7a')
txt(IX+IW-0.35, IY+IH-0.65, 'H3+', fs=8.5, bold=True, color=C_H, ha='right')

ir_rows = [
    'Data pipeline governance  ·  Documentation',
    'Source-code & model access  ·  Custom code',
    'Provisioned infrastructure  ·  Component reuse',
    'Evaluation infrastructure  ·  Timely resources',
]
for k, row in enumerate(ir_rows):
    txt(IX+IW/2, IY+IH-0.75-(k*0.43), row, fs=8.1, color='#1a4d7a')

txt(IX+IW/2, IY+0.32,
    '→ Enforcement hooks for governance implementability',
    fs=7.6, color='#2560a0', italic=True)

# ══════════════════════════════════════════════════════════════════════════════
# 6. TR × IR COMPLEMENTARITY (in the gap between boxes)
# ══════════════════════════════════════════════════════════════════════════════
gap_cx = (TX+TW + IX) / 2   # horizontal centre of gap
gap_cy = TY + TH/2           # vertical centre of TR/IR boxes

# Dashed double-headed arrow
x_left  = TX + TW + 0.15
x_right = IX - 0.15
ax.annotate('', xy=(x_right, gap_cy), xytext=(x_left, gap_cy),
            arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.6,
                            mutation_scale=12,
                            connectionstyle='arc3,rad=0'),
            zorder=5)
# Make dashed by drawing over with a dashed line
ax.plot([x_left, x_right], [gap_cy, gap_cy],
        color='#555555', lw=1.6, ls=(0, (5, 4)), zorder=6)

txt(gap_cx, gap_cy+0.38, 'TR × IR', fs=9, bold=True, color='#444444')
txt(gap_cx, gap_cy+0.04, 'Complementarity', fs=8, color='#555555', italic=True)
txt(gap_cx, gap_cy-0.28, '(conditional on', fs=7.5, color=C_GR)
txt(gap_cx, gap_cy-0.54, 'governance maturity)', fs=7.5, color=C_GR)

# ══════════════════════════════════════════════════════════════════════════════
# 7. ARROWS: TR → Barrier, IR → Barrier
# ══════════════════════════════════════════════════════════════════════════════
arr(TX+TW/2, TY, BX+BW*0.28, BY+BH, color='#555555')
arr(IX+IW/2, IY, BX+BW*0.72, BY+BH, color='#555555')

# ══════════════════════════════════════════════════════════════════════════════
# 8. EVALUABILITY CONSTRAINT / BARRIER BOX
# ══════════════════════════════════════════════════════════════════════════════
rbox(BX, BY, BW, BH, C_BARR, C_BARR_E, lw=2.0)
txt(BX+BW/2, BY+BH/2+0.20,
    'Commercial Procurement  |  Evaluability Constraint',
    fs=9.5, bold=True, color='#8b1a1a')
txt(BX+BW/2, BY+BH/2-0.18,
    'Restricts control rights  ·  Blocks model/artifact access  →  Weakens substantive safeguards',
    fs=8.0, color='#8b1a1a')
txt(BX+BW-0.22, BY+BH/2, 'H4', fs=9, bold=True, color=C_H, ha='right')

# "Governance Theater" label below barrier box (no overlap with Outcome)
txt(BX+BW/2, BY-0.32, '← Governance Theater →', fs=8.5, bold=True, color=C_BARR_E, italic=True)

# ══════════════════════════════════════════════════════════════════════════════
# 9. OPERATIONAL DEPLOYMENT BOX (right of barrier)
# ══════════════════════════════════════════════════════════════════════════════
rbox(OX, OY, OW, OH, C_OUT, C_OUT_E, lw=2.0)
txt(OX+OW/2, OY+OH/2+0.18, 'Operational', fs=9.5, bold=True, color='#2a5a2a')
txt(OX+OW/2, OY+OH/2-0.18, 'Deployment', fs=9.5, bold=True, color='#2a5a2a')

# Arrow IR → Outcome  (straight right from mid-right of IR box)
arr(IX+IW, IY+IH/2, OX, OY+OH/2, color=C_OUT_E, lw=1.8)

# ══════════════════════════════════════════════════════════════════════════════
# 10. ARROW: Barrier → Data sources
# ══════════════════════════════════════════════════════════════════════════════
arr(BX+BW/2, BY, BX+BW/2, DS_Y+0.68, color='#444444')

# ══════════════════════════════════════════════════════════════════════════════
# 11. EMPIRICAL TRIANGULATION LABEL + DATA SOURCE ELLIPSES
# ══════════════════════════════════════════════════════════════════════════════
txt(7.5, DS_Y+0.85, 'Empirical Triangulation', fs=9.5, bold=True, color='#444444')

sources = [
    (2.6,  'MITRE ATLAS\nThreats\n52 case studies'),
    (7.5,  'AI Incident Database (AIID)\n1,362 incidents'),
    (12.2, 'EO 13960 Federal\nAI Inventory\n1,757 deployments'),
]
for sx, stxt_s in sources:
    ax.add_patch(mpatches.Ellipse((sx, DS_Y), 4.0, 1.35,
                 facecolor=C_DATA, edgecolor=C_DATA_E, linewidth=1.5, zorder=3))
    txt(sx, DS_Y, stxt_s, fs=8.2, color='#554433')
    # Dashed line from barrier centre to each ellipse
    ax.plot([BX+BW/2, sx], [BY, DS_Y+0.68],
            color='#bbbbbb', lw=1.1, ls='--', zorder=2)

# ══════════════════════════════════════════════════════════════════════════════
# 12. LEGEND
# ══════════════════════════════════════════════════════════════════════════════
lg_y = 0.32
items = [
    (C_TR,   C_TR_E,  'Trust Readiness (TR)'),
    (C_IR,   C_IR_E,  'Integration Readiness (IR)'),
    (C_BARR, C_BARR_E,'Evaluability Constraint'),
    (C_OUT,  C_OUT_E, 'Operational Deployment'),
]
lx = 0.8
for fc, ec, lbl in items:
    ax.add_patch(FancyBboxPatch((lx, lg_y-0.14), 0.42, 0.30,
        boxstyle="round,pad=0,rounding_size=0.04",
        facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=5))
    txt(lx+0.60, lg_y+0.01, lbl, fs=7.8, color='#333333', ha='left')
    lx += 3.8

txt(8.0, lg_y-0.36,
    'H1+ / H2− / H3+ / H4: hypotheses with predicted directions  (+ positive,  − negative)',
    fs=7.5, color=C_H, italic=True)

plt.tight_layout(pad=0.3)
out_path = r'C:\Users\carlo\Dropbox\Projeto - Universite de Sherbrooke\AMCIS 2026\paper\figures\fig0_framework.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved: {out_path}')
