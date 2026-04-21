import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

FW, FH = 16, 11
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')

C_TR,B_TR     = '#d0ead0','#2e7d32'
C_IR,B_IR     = '#cce0f5','#1565c0'
C_PROC,B_PROC = '#f8d7d7','#b71c1c'
C_OD,B_OD     = '#fff3cd','#e65100'
C_PORT,B_PORT = '#eaeaf8','#283593'
C_SRC,B_SRC   = '#f5f0e0','#888866'

def rbox(x, y, w, h, fc, ec, lw=2.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))

def t(x, y, s, fs=9, bold=False, color='#111', ha='center', va='center'):
    ax.text(x, y, s, fontsize=fs, fontweight='bold' if bold else 'normal',
            color=color, ha=ha, va=va, zorder=5, linespacing=1.5)

def arr(x1, y1, x2, y2, color='#555', lw=1.8, dashed=False, both=False):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='<->' if both else '->',
                        color=color, lw=lw,
                        linestyle='--' if dashed else '-'), zorder=6)

def ell(cx, cy, rw, rh, fc, ec):
    ax.add_patch(mpatches.Ellipse((cx, cy), rw, rh,
        facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2))

# ── Portfolio Scope (top center) ─────────────────────────────────────────────
rbox(4.0, 9.1, 8.0, 1.3, C_PORT, B_PORT)
t(8.0, 9.82, 'Organizational AI Portfolio Scope', fs=12, bold=True, color=B_PORT)
t(8.0, 9.35, 'Agency-level proxy: z-scored log portfolio size + topic breadth',
  fs=8.5, color='#444')

# ── Trust Readiness box (left) ───────────────────────────────────────────────
TX, TY, TW, TH = 0.4, 5.4, 6.5, 3.5
rbox(TX, TY, TW, TH, C_TR, B_TR, lw=2.2)
t(TX+TW/2, TY+TH-0.32, 'Trust Readiness (TR)', fs=11, bold=True, color=B_TR)

t(TX+0.25, TY+TH-0.78, 'H1+  Surface (0-2):', fs=9.5, bold=True, color=B_TR, ha='left')
t(TX+0.25, TY+TH-1.18, 'Internal review  |  Authorization to Operate (ATO)',
  fs=8.5, ha='left', color='#333')

t(TX+0.25, TY+TH-1.65, 'H2-  Substantive (0-7):', fs=9.5, bold=True, color=B_TR, ha='left')
t(TX+0.25, TY+TH-2.05, 'Impact assessment  |  Independent evaluation',
  fs=8.5, ha='left', color='#333')
t(TX+0.25, TY+TH-2.45, 'Real-world testing  |  Bias mitigation  |  Appeal process',
  fs=8.5, ha='left', color='#333')

ax.plot([TX+0.2, TX+TW-0.2], [TY+0.78, TY+0.78],
        color=B_TR, lw=1.0, ls='--', zorder=3)
t(TX+TW/2, TY+0.43, 'Surface compliance  --------  Substantive safeguards',
  fs=8, color=B_TR)

# ── Integration Readiness box (right, ends at x=14.4 so OD fits at 14.5) ────
IX, IY, IW, IH = 8.5, 5.4, 5.7, 3.5
rbox(IX, IY, IW, IH, C_IR, B_IR, lw=2.2)
t(IX+IW/2, IY+IH-0.32, 'Integration Readiness (IR)', fs=11, bold=True, color=B_IR)
t(IX+0.25, IY+IH-0.8, 'H3+', fs=9.5, bold=True, color=B_IR, ha='left')
ir_lines = [
    'Data pipeline governance  |  Documentation',
    'Source code access  |  Custom code',
    'Provisioned infrastructure  |  Component reuse',
    'Evaluation infrastructure  |  Timely resources',
    'Enforcement hooks for governance implementability',
]
for k, line in enumerate(ir_lines):
    t(IX+0.25, IY+IH-1.2-k*0.5, line,
      fs=8.5, ha='left', color='#333', bold=(k == 4))

# ── Operational Deployment (separate box, right margin) ──────────────────────
OX, OY, OW, OH = 14.5, 6.55, 1.3, 1.5
rbox(OX, OY, OW, OH, C_OD, B_OD, lw=2.2)
t(OX+OW/2, OY+OH/2+0.18, 'Operational', fs=9, bold=True, color=B_OD)
t(OX+OW/2, OY+OH/2-0.18, 'Deployment', fs=9, bold=True, color=B_OD)

# ── TR x IR double-headed arrow (in gap between TR and IR boxes) ─────────────
arr(TX+TW+0.1, TY+TH/2, IX-0.1, IY+IH/2, color='#555', lw=2.0, both=True)
mid_x = (TX+TW + IX) / 2
t(mid_x, TY+TH/2+0.32, 'TR x IR', fs=9.5, bold=True, color='#555')
t(mid_x, TY+TH/2-0.24, '(conditional on\ngovernance maturity)', fs=8, color='#666')

# ── Commercial Procurement box (center, below TR/IR) ─────────────────────────
CPX, CPY, CPW, CPH = 2.2, 3.7, 11.5, 1.5
rbox(CPX, CPY, CPW, CPH, C_PROC, B_PROC, lw=2.2)
t(CPX+CPW/2, CPY+CPH-0.38,
  'Commercial Procurement  |  Evaluability Constraint',
  fs=11, bold=True, color=B_PROC)
t(CPX+CPW/2, CPY+CPH-0.88,
  'Restricts control rights  |  Blocks model/artifact access  |  Weakens substantive safeguards',
  fs=8.5, color='#333')
t(CPX+CPW/2, CPY+0.3, '<-- Governance Theater -->', fs=9, color=B_PROC)
t(CPX+CPW-0.4, CPY+CPH-0.38, 'H4', fs=9, bold=True, color=B_PROC, ha='right')

# ── Empirical Triangulation label ─────────────────────────────────────────────
t(FW/2, 3.25, 'Empirical Triangulation', fs=10, bold=True, color='#555')

# ── Three data source ovals ───────────────────────────────────────────────────
for cx, lb1, lb2 in [
    (3.0,  'MITRE ATLAS',           '52 case studies'),
    (8.0,  'AI Incident Database',  '1,362 incidents'),
    (13.0, 'EO 13960 Federal AI',   '1,757 deployments'),
]:
    ell(cx, 2.0, 4.2, 1.55, C_SRC, B_SRC)
    t(cx, 2.25, lb1, fs=9, bold=True, color='#333')
    t(cx, 1.75, lb2, fs=8.5, color='#555')

# ── Legend ────────────────────────────────────────────────────────────────────
lx = 0.4
for fc, ec, lab in [
    (C_TR,   B_TR,   'Trust Readiness (TR)'),
    (C_IR,   B_IR,   'Integration Readiness (IR)'),
    (C_PROC, B_PROC, 'Evaluability Constraint'),
    (C_OD,   B_OD,   'Operational Deployment'),
]:
    ax.add_patch(FancyBboxPatch((lx, 0.15), 0.5, 0.45,
        boxstyle='round,pad=0.05', facecolor=fc, edgecolor=ec,
        linewidth=1.5, zorder=3))
    t(lx+0.7, 0.38, lab, fs=8.5, ha='left', color='#333')
    lx += 3.8

# ── Directional arrows ────────────────────────────────────────────────────────
arr(5.5, 9.1,  TX+TW/2, TY+TH, color=B_PORT, lw=2.0)   # Portfolio -> TR
arr(10.5, 9.1, IX+IW/2, IY+IH, color=B_PORT, lw=2.0)   # Portfolio -> IR
arr(TX+TW/2,  TY,       CPX+CPW*0.27, CPY+CPH, color=B_TR,   lw=1.8)  # TR -> Proc
arr(IX+IW/2,  IY,       CPX+CPW*0.73, CPY+CPH, color=B_IR,   lw=1.8)  # IR -> Proc
arr(IX+IW,    IY+IH/2,  OX, OY+OH/2,           color=B_IR,   lw=2.0)  # IR -> OD
arr(CPX+CPW/2, CPY,     FW/2, 3.42,             color=B_PROC, lw=1.8)  # Proc -> Tri
for cx in [3.0, 8.0, 13.0]:
    arr(FW/2, 3.1, cx, 2.8, color='#888', lw=1.2)   # Tri -> ovals

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
OUT = r'C:\Users\carlo\Dropbox\Projeto - Universite de Sherbrooke\AMCIS 2026\paper\figures\fig0_framework.png'
plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved:', OUT)
from PIL import Image
print('Size:', Image.open(OUT).size)
