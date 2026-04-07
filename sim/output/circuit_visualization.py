#!/usr/bin/env python3
# =============================================================================
# circuit_visualization.py  —  Figure 2: L2/3 Microcircuit Organization
# Yao et al. 2022 NetPyNE replica.
#
# Pure matplotlib/numpy — no simulation, no NEURON.
# Usage:
#   python circuit_visualization.py
# Outputs:
#   output/figure2_circuit_organization.png  (300 dpi)
#   output/figure2_circuit_organization.pdf
# =============================================================================
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------------------
# Paths — import CONN_PROBS from circuit_params at runtime
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from params.circuit_params import CONN_PROBS

OUT_DIR = os.path.join(_here, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Fixed constants (from paper / circuit_params.py)
# ---------------------------------------------------------------------------
CELL_NAMES = ['HL23PYR', 'HL23SST', 'HL23PV', 'HL23VIP']
LABELS     = {'HL23PYR': 'Pyr', 'HL23SST': 'SST', 'HL23PV': 'PV',  'HL23VIP': 'VIP'}
COUNTS     = {'HL23PYR': 800,   'HL23SST': 50,    'HL23PV': 70,    'HL23VIP': 80}
COLORS     = {
    'HL23PYR': '#555555',
    'HL23SST': '#CC0000',
    'HL23PV':  '#1a7a1a',
    'HL23VIP': '#E8820C',
}
N_TOTAL = sum(COUNTS.values())      # 1000
X_EXT, Y_EXT, Z_MIN, Z_MAX = 500, 500, 250, 1200
Z_EXT = Z_MAX - Z_MIN              # 950

# Cortical layers (depth below pia, µm)
_LAYERS = [
    (0,   80,   'L1',   '#d0e8ff', 0.12),
    (80,  1200, 'L2/3', '#ffffd0', 0.10),
    (1200,1600, 'L4',   '#d0ffd0', 0.07),
    (1600,2300, 'L5',   '#ffd0d0', 0.07),
    (2300,3200, 'L6',   '#e8d0ff', 0.07),
]

# ---------------------------------------------------------------------------
# Soma positions (fixed seed for reproducibility)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
_xs = rng.uniform(0, X_EXT, N_TOTAL)
_ys = rng.uniform(0, Y_EXT, N_TOTAL)
_zs = rng.uniform(Z_MIN, Z_MAX, N_TOTAL)

# Population slices into the position arrays
_pop_sl = {}
_s = 0
for ct in CELL_NAMES:
    _pop_sl[ct] = slice(_s, _s + COUNTS[ct])
    _s += COUNTS[ct]

# ---------------------------------------------------------------------------
# Console statistics
# ---------------------------------------------------------------------------
vol_um3  = X_EXT * Y_EXT * Z_EXT
vol_mm3  = vol_um3 / 1e9
density  = N_TOTAL / vol_mm3
ei_i     = N_TOTAL - COUNTS['HL23PYR']
conn_all = [CONN_PROBS[p][q] for p in CELL_NAMES for q in CELL_NAMES]

print('=' * 58)
print('Circuit Organization — Yao et al. 2022  L2/3 Microcircuit')
print('=' * 58)
for ct in CELL_NAMES:
    print(f'  {LABELS[ct]:6s}: {COUNTS[ct]:4d} cells  '
          f'({COUNTS[ct]/N_TOTAL*100:.0f}%)')
print(f'  E/I ratio : {COUNTS["HL23PYR"]}/{ei_i}  '
      f'= {COUNTS["HL23PYR"]/N_TOTAL*100:.0f}% / '
      f'{ei_i/N_TOTAL*100:.0f}%')
print(f'  Volume    : {vol_um3:,.0f} µm³  = {vol_mm3:.5f} mm³')
print(f'  Density   : {density:,.0f} cells/mm³')
print(f'  Mean conn prob: {np.mean(conn_all)*100:.1f}%')
print(f'  Connection types: {len(conn_all)}')
print('=' * 58)

# ===========================================================================
# FIGURE  —  2 rows × 3 columns
# ===========================================================================
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    width_ratios=[2, 1, 1.2],
    hspace=0.44,
    wspace=0.38,
    left=0.06, right=0.97,
    top=0.93,  bottom=0.08,
)

# ===========================================================================
# PANEL A — 3D scatter of soma positions
# ===========================================================================
ax_a = fig.add_subplot(gs[0, 0:2], projection='3d')

# Bounding-box wireframe (12 edges of the cuboid)
_bx = [0, X_EXT]; _by = [0, Y_EXT]; _bz = [Z_MIN, Z_MAX]
_corners = [(x, y, z) for x in _bx for y in _by for z in _bz]
_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
for i, j in _edges:
    ax_a.plot([_corners[i][0], _corners[j][0]],
              [_corners[i][1], _corners[j][1]],
              [_corners[i][2], _corners[j][2]],
              color='gray', lw=0.5, ls='--', alpha=0.40, zorder=1)

# Horizontal shaded planes at L2/3 boundaries
_Xp, _Yp = np.meshgrid([0, X_EXT], [0, Y_EXT])
ax_a.plot_surface(_Xp, _Yp, np.full_like(_Xp, float(Z_MIN)),
                  color='steelblue', alpha=0.08, shade=False, zorder=1)
ax_a.plot_surface(_Xp, _Yp, np.full_like(_Xp, float(Z_MAX)),
                  color='steelblue', alpha=0.08, shade=False, zorder=1)

# Scatter — small/transparent for PYR (800 pts), larger for interneurons
_sz = {'HL23PYR': 6,  'HL23SST': 28, 'HL23PV': 28, 'HL23VIP': 28}
_al = {'HL23PYR': 0.22,'HL23SST': 0.82,'HL23PV': 0.82,'HL23VIP': 0.82}
for ct in CELL_NAMES:
    sl = _pop_sl[ct]
    ax_a.scatter(_xs[sl], _ys[sl], _zs[sl],
                 c=COLORS[ct], s=_sz[ct], alpha=_al[ct],
                 label=f'{LABELS[ct]} (n={COUNTS[ct]})',
                 depthshade=False, zorder=3)

ax_a.set_xlabel('x (µm)', fontsize=8, labelpad=3)
ax_a.set_ylabel('y (µm)', fontsize=8, labelpad=3)
ax_a.set_zlabel('Depth below pia (µm)', fontsize=8, labelpad=4)
ax_a.set_xlim(0, X_EXT); ax_a.set_ylim(0, Y_EXT)
ax_a.set_zlim(Z_MIN, Z_MAX)
ax_a.invert_zaxis()
ax_a.tick_params(labelsize=7)
ax_a.legend(fontsize=8, loc='upper right',
            bbox_to_anchor=(1.05, 1.0), framealpha=0.85)
ax_a.text2D(0.03, 0.03, '500×500×950 µm³',
            transform=ax_a.transAxes, fontsize=7, color='gray')
ax_a.set_title('A   Soma distribution in L2/3 microcircuit volume',
               fontsize=10, loc='left', pad=8)
ax_a.view_init(elev=22, azim=-55)

# ===========================================================================
# PANEL B — Population pie chart
# ===========================================================================
ax_b = fig.add_subplot(gs[0, 2])
_fracs  = [COUNTS[ct] / N_TOTAL for ct in CELL_NAMES]
_colors_b = [COLORS[ct] for ct in CELL_NAMES]
wedges, _ = ax_b.pie(
    _fracs,
    colors=_colors_b,
    explode=(0.05, 0.10, 0.10, 0.10),
    startangle=90,
    wedgeprops=dict(linewidth=0.8, edgecolor='white'),
)
for wedge, ct in zip(wedges, CELL_NAMES):
    ang = (wedge.theta1 + wedge.theta2) / 2.0
    r   = 1.30
    ax_b.text(r * np.cos(np.deg2rad(ang)),
              r * np.sin(np.deg2rad(ang)),
              f'{LABELS[ct]}\n{COUNTS[ct]} ({COUNTS[ct]/N_TOTAL*100:.0f}%)',
              ha='center', va='center',
              fontsize=8, color=COLORS[ct], fontweight='bold')
ax_b.text(0.0, -1.82, f'Total: {N_TOTAL:,} neurons',
          ha='center', va='center', fontsize=8, color='gray')
ax_b.set_title('B   Population composition', fontsize=10, loc='left', pad=8)

# ===========================================================================
# PANEL C — Depth distribution (stacked horizontal histogram)
# ===========================================================================
ax_c = fig.add_subplot(gs[1, 0])
_bins_c   = np.linspace(Z_MIN, Z_MAX, 20)   # 19 bins × 50 µm
_bcenters = (_bins_c[:-1] + _bins_c[1:]) / 2
_bw       = _bins_c[1] - _bins_c[0]

# Background layer shading (only L1 and L2/3 overlap with display range)
for z0, z1, lname, lcol, lalpha in _LAYERS:
    ylo = max(z0, 230); yhi = min(z1, 1270)
    if ylo < yhi:
        ax_c.axhspan(ylo, yhi, color=lcol, alpha=lalpha, zorder=0)

# Stacked bars
_left_c = np.zeros(len(_bcenters))
for ct in CELL_NAMES:
    h, _ = np.histogram(_zs[_pop_sl[ct]], bins=_bins_c)
    ax_c.barh(_bcenters, h, height=_bw * 0.85, left=_left_c,
              color=COLORS[ct], alpha=0.85, label=LABELS[ct], edgecolor='none')
    _left_c += h

_xmax_c = int(np.max(_left_c))
ax_c.set_xlim(0, _xmax_c * 1.20)

# L2/3 boundary dashed lines + labels
ax_c.axhline(Z_MIN,  color='steelblue', lw=1.0, ls='--', alpha=0.75)
ax_c.axhline(Z_MAX,  color='steelblue', lw=1.0, ls='--', alpha=0.75)
ax_c.text(_xmax_c * 1.18, Z_MIN + 8,  'L2/3 top',    fontsize=7,
          color='steelblue', va='bottom', ha='right')
ax_c.text(_xmax_c * 1.18, Z_MAX - 8,  'L2/3 bottom', fontsize=7,
          color='steelblue', va='top',    ha='right')

ax_c.set_xlabel('Cell count per 50 µm bin', fontsize=9)
ax_c.set_ylabel('Depth below pia (µm)', fontsize=9)
ax_c.set_ylim(230, 1270)
ax_c.invert_yaxis()
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.tick_params(labelsize=8)

# Secondary right-axis — L2/3 bracket label
ax_c2 = ax_c.twinx()
ax_c2.set_ylim(ax_c.get_ylim())
ax_c2.invert_yaxis()
ax_c2.set_yticks([(Z_MIN + Z_MAX) / 2])
ax_c2.set_yticklabels(['L2/3'], fontsize=9, color='#888800')
ax_c2.spines['top'].set_visible(False)
ax_c2.spines['left'].set_visible(False)
ax_c2.tick_params(length=0)

ax_c.legend(fontsize=8, loc='lower right', framealpha=0.85)
ax_c.set_title('C   Depth distribution', fontsize=10, loc='left', pad=8)

# ===========================================================================
# PANEL D — Connectivity matrix heatmap
# ===========================================================================
ax_d = fig.add_subplot(gs[1, 1])
n_pop   = len(CELL_NAMES)
conn_mat = np.array(
    [[CONN_PROBS[pre][post] * 100 for post in CELL_NAMES]
     for pre in CELL_NAMES]
)
im = ax_d.imshow(conn_mat, cmap='YlOrRd', aspect='auto',
                 norm=Normalize(vmin=0, vmax=60))

for i in range(n_pop):
    for j in range(n_pop):
        v = conn_mat[i, j]
        ax_d.text(j, i, f'{v:.1f}%',
                  ha='center', va='center', fontsize=8,
                  color='white' if v > 30 else 'black',
                  fontweight='bold')

_tlabels = [LABELS[ct] for ct in CELL_NAMES]
ax_d.set_xticks(range(n_pop)); ax_d.set_xticklabels(_tlabels, fontsize=9)
ax_d.set_yticks(range(n_pop)); ax_d.set_yticklabels(_tlabels, fontsize=9)
ax_d.set_xlabel('Postsynaptic', fontsize=9)
ax_d.set_ylabel('Presynaptic', fontsize=9)
cb = fig.colorbar(im, ax=ax_d, shrink=0.82, pad=0.04)
cb.set_label('Connection probability (%)', fontsize=8)
cb.ax.tick_params(labelsize=7)
ax_d.set_title('D   Connectivity matrix\n(connection probability %)',
               fontsize=9, loc='left', pad=4)

# ===========================================================================
# PANEL E — Wiring diagram schematic
# ===========================================================================
ax_e = fig.add_subplot(gs[1, 2])
ax_e.set_xlim(0, 1)
ax_e.set_ylim(0, 1)
ax_e.axis('off')
ax_e.set_title('E   Circuit connectivity schematic', fontsize=10,
               loc='left', pad=8)

# Node positions and radii
_NPOS = {
    'HL23PYR': (0.27, 0.50),
    'HL23SST': (0.73, 0.78),
    'HL23PV':  (0.73, 0.50),
    'HL23VIP': (0.73, 0.22),
}
_NRAD = {'HL23PYR': 0.09, 'HL23SST': 0.07, 'HL23PV': 0.07, 'HL23VIP': 0.07}

# Draw node circles + labels
for ct in CELL_NAMES:
    cx, cy = _NPOS[ct]
    r = _NRAD[ct]
    ax_e.add_patch(Circle((cx, cy), r,
                          facecolor=COLORS[ct], edgecolor='white',
                          linewidth=1.5, zorder=5))
    ax_e.text(cx, cy, LABELS[ct],
              ha='center', va='center',
              fontsize=9, fontweight='bold', color='white', zorder=6)
    ax_e.text(cx, cy - r - 0.036, f'n={COUNTS[ct]}',
              ha='center', va='top',
              fontsize=7, color='#666666', zorder=6)


def _bnd(ct, toward_ct):
    """Return the point on ct's circle boundary facing toward_ct."""
    cx, cy = _NPOS[ct]
    ox, oy = _NPOS[toward_ct]
    r = _NRAD[ct]
    dx, dy = ox - cx, oy - cy
    d = np.hypot(dx, dy)
    return cx + r * dx / d, cy + r * dy / d


def _arc_mid(sx, sy, ex, ey, rad):
    """Midpoint of a quadratic-Bézier arc3 curve (label placement)."""
    return (
        (sx + ex) / 2.0 - 0.5 * rad * (ey - sy),
        (sy + ey) / 2.0 + 0.5 * rad * (ex - sx),
    )


def _arrow(pre, post, exc=True, rad=0.20,
           lbl=None, lbl_col=None, lbl_dash=False):
    """Draw a directed connection arrow between two nodes."""
    px, py = _NPOS[pre]; qx, qy = _NPOS[post]
    rp = _NRAD[pre]; rq = _NRAD[post]
    dx, dy = qx - px, qy - py
    d = np.hypot(dx, dy)
    ux, uy = dx / d, dy / d
    sx, sy = px + rp * ux, py + rp * uy  # tail on pre boundary
    ex, ey = qx - rq * ux, qy - rq * uy  # head on post boundary

    col    = '#333333' if exc else COLORS[pre]
    astyle = '-|>' if exc else '-['

    ax_e.annotate('',
                  xy=(ex, ey), xytext=(sx, sy),
                  arrowprops=dict(
                      arrowstyle=astyle,
                      connectionstyle=f'arc3,rad={rad}',
                      color=col, lw=1.8, mutation_scale=10,
                  ),
                  zorder=4)

    if lbl:
        mx, my = _arc_mid(sx, sy, ex, ey, rad)
        _bbox = dict(
            facecolor='#fff8f0' if lbl_dash else 'white',
            edgecolor=(lbl_col or col) if lbl_dash else 'none',
            alpha=0.90, pad=2,
            linestyle='--' if lbl_dash else '-',
            boxstyle='round,pad=0.3',
        )
        ax_e.text(mx, my, lbl,
                  ha='center', va='center',
                  fontsize=6.5,
                  color=lbl_col or col,
                  fontweight='bold' if lbl_dash else 'normal',
                  style='normal' if lbl_dash else 'italic',
                  bbox=_bbox, zorder=7)


def _selfloop(ct):
    """Draw a small circular self-loop arc above/beside the node."""
    cx, cy = _NPOS[ct]
    r = _NRAD[ct]
    exc = (ct == 'HL23PYR')
    col = '#333333' if exc else COLORS[ct]

    # Loop circle centre — above-left for PYR, above-right for PV
    off_x = -r * 0.25 if ct == 'HL23PYR' else r * 0.35
    lx, ly = cx + off_x, cy + r + r * 0.40
    lr = r * 0.42

    # Arc from ~250° to ~-70° (nearly full circle, CCW, leaving a small gap)
    theta = np.linspace(np.deg2rad(250), np.deg2rad(290), 120)
    ax_e.plot(lx + lr * np.cos(theta),
              ly + lr * np.sin(theta),
              color=col, lw=1.8, zorder=3)

    # Arrowhead at the end of the arc
    t1, t2 = theta[-1], theta[-2]
    ax_e.annotate('',
                  xy=(lx + lr * np.cos(t1), ly + lr * np.sin(t1)),
                  xytext=(lx + lr * np.cos(t2), ly + lr * np.sin(t2)),
                  arrowprops=dict(arrowstyle='->' if exc else '-[',
                                  color=col, lw=0,
                                  mutation_scale=9),
                  zorder=4)


# ---- Draw connections (from Figure 2B of Yao et al. 2022) ----

# Excitatory  (PYR → *)
_selfloop('HL23PYR')
_arrow('HL23PYR', 'HL23SST', exc=True,  rad=-0.18)
_arrow('HL23PYR', 'HL23PV',  exc=True,  rad=0.00)
_arrow('HL23PYR', 'HL23VIP', exc=True,  rad=0.18)

# Inhibitory  (SST → *)
_arrow('HL23SST', 'HL23PYR', exc=False, rad=-0.30,
       lbl='apical', lbl_col=COLORS['HL23SST'])
_arrow('HL23SST', 'HL23PV',  exc=False, rad=0.15)
_arrow('HL23SST', 'HL23VIP', exc=False, rad=0.12)

# Inhibitory  (PV → *)
_arrow('HL23PV',  'HL23PYR', exc=False, rad=0.30,
       lbl='basal', lbl_col=COLORS['HL23PV'])
_arrow('HL23PV',  'HL23SST', exc=False, rad=-0.15)
_selfloop('HL23PV')

# Inhibitory  (VIP → *)  — VIP→SST is the key disinhibitory route
_arrow('HL23VIP', 'HL23SST', exc=False, rad=0.22,
       lbl='Disinhibitory', lbl_col=COLORS['HL23VIP'], lbl_dash=True)
_arrow('HL23VIP', 'HL23PV',  exc=False, rad=-0.12)

# Legend
ax_e.legend(
    handles=[
        mpatches.Patch(facecolor='#333333', edgecolor='none', label='Excitatory'),
        mpatches.Patch(facecolor='#888888', edgecolor='none', label='Inhibitory'),
    ],
    fontsize=7, loc='lower left',
    framealpha=0.85, bbox_to_anchor=(0.0, 0.0),
    handlelength=1.2,
)

# ===========================================================================
# Figure-level circuit parameter text box (bottom margin)
# ===========================================================================
_stats = (
    'Circuit parameters:\n'
    '  Volume: 500×500×950 µm³\n'
    '  Depth: 250–1,200 µm below pia (L2/3)\n'
    f'  N = {N_TOTAL:,} neurons\n'
    '  E/I ratio: 80% / 20%\n'
    '  dt = 0.025 ms  |  T = 34°C'
)
fig.text(0.50, 0.005, _stats,
         ha='center', va='bottom', fontsize=7, color='#444444',
         bbox=dict(facecolor='#f8f8f8', edgecolor='#cccccc',
                   alpha=0.92, pad=5, boxstyle='round'))

fig.suptitle('L2/3 Microcircuit Organization — Yao et al. 2022',
             fontsize=12, y=0.975)

# ===========================================================================
# Save
# ===========================================================================
out_png = os.path.join(OUT_DIR, 'figure2_circuit_organization.png')
out_pdf = os.path.join(OUT_DIR, 'figure2_circuit_organization.pdf')
fig.savefig(out_png, dpi=300, bbox_inches='tight')
fig.savefig(out_pdf,           bbox_inches='tight')
plt.close(fig)
print(f'\n[cv] Saved → {out_png}')
print(f'[cv] Saved → {out_pdf}')
