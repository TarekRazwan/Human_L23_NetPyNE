# h01_integration/h01_placement.py
# Loads H01 EM soma coordinates and places Yao 2022 L2/3 cells at REAL H01 positions.
#
# DESIGN DECISION 3 (2026-06-06): PRESERVE REAL H01 COORDINATES.
#   The earlier version rescaled each axis independently into a [0,500]^3 box. That was
#   WRONG for two reasons:
#     (1) It fabricated z-extent: the real H01 sample is a ~174 um THIN SLAB in z (true
#         across the entire 15,567-cell dataset, not a cropping artifact). Stretching z to
#         500 invents ~326 um of tissue that was never measured.
#     (2) DECISIVE: the distance-dependent connectivity rule (lambda values in
#         distance_rule_fit_summary_Cmin30.xlsx) was FIT on real H01 distances in micrometres.
#         To apply those lambdas correctly, inter-cell distances in the model must be in the
#         SAME real-um frame. Rescaling to [0,500]^3 puts cells in a different distance frame,
#         so the lambda would be applied to the wrong distances.
#   Therefore we now only TRANSLATE (and orient) the coordinates — a rigid transform that
#   PRESERVES ALL PAIRWISE DISTANCES exactly. No scaling. The column becomes a real-geometry
#   slab (~598 x 801 x 173 um), not a 500^3 cube. (Note for Aim 2 TMS/E-field: use these real
#   dimensions, not a 500^3 idealization.)
#
# H01 coordinate frame (columns x_new, y_new, z_new in the rotated CSV):
#   y_new = cortical depth axis; HIGHER y_new = MORE SUPERFICIAL (closer to pia).
#   x_new, z_new = lateral; z_new is the thin (~174 um) slab axis.
#   distance2Dcenter ~ 1D distance along x_new (corr 0.999 with |x_new-mean|).
#
# NetPyNE output frame:
#   x = lateral 1 (um), origin at slab min
#   y = cortical depth (um), 0 = pia (most superficial), increasing = deeper
#   z = lateral 2 (um, the thin slab axis), origin at slab min

import numpy as np
import pandas as pd

# d2d radius for the column crop (um). 300 um -> ~2516 L23 cells (E=1618, I=898),
# enough for 800 PYR + 200 interneurons with margin.
DEFAULT_D2D_RADIUS_UM = 300.0


def load_h01_l23_cells(csv_path, d2d_radius_um=DEFAULT_D2D_RADIUS_UM):
    """
    Load H01 L2/3 cells within d2d_radius_um of the column center.

    Filters to mtype in ['L23E','L23I'] and distance2Dcenter < d2d_radius_um.
    Returns a DataFrame with: id, x_new, y_new, z_new, distance2Dcenter, EI, mtype.
    """
    df = pd.read_csv(csv_path)
    df = df[df['distance2Dcenter'] < d2d_radius_um]
    df = df[df['mtype'].isin(['L23E', 'L23I'])].copy()
    df = df.reset_index(drop=True)
    return df[['id', 'x_new', 'y_new', 'z_new', 'distance2Dcenter', 'EI', 'mtype']]


def compute_netpyne_coords(h01_df):
    """
    Convert real H01 coordinates to the NetPyNE frame by TRANSLATION + Y-INVERSION ONLY.
    This is a rigid transform: all pairwise distances are preserved exactly (verified).

    Transform (computed over the loaded/cropped L23 set so the column sits at origin):
      x_np = x_new - x_new.min()                 # lateral 1, origin at 0
      z_np = z_new - z_new.min()                 # lateral 2 (thin slab), origin at 0
      y_np = y_new.max() - y_new                 # depth: pia (max y_new) -> 0, deeper -> larger
    NO scaling on any axis.

    Adds columns x_np, y_np, z_np. Returns the modified DataFrame.
    """
    df = h01_df.copy()

    # Reference extremes from the loaded set (translation/orientation anchors).
    x_min = df['x_new'].min()
    z_min = df['z_new'].min()
    y_max = df['y_new'].max()   # most superficial -> becomes depth 0 (pia)

    df['x_np'] = df['x_new'] - x_min
    df['z_np'] = df['z_new'] - z_min
    df['y_np'] = y_max - df['y_new']   # invert so depth increases away from pia

    return df


def get_all_population_positions(csv_path, globalseed, cell_counts,
                                 d2d_radius_um=DEFAULT_D2D_RADIUS_UM):
    """
    Master function. Sample real H01 soma positions for each Yao population.

    Returns a dict:
      {'HL23PYR': [[x,y,z],...],   # from L23E
       'HL23SST': [[x,y,z],...],   # from L23I
       'HL23PV':  [[x,y,z],...],   # from L23I
       'HL23VIP': [[x,y,z],...]}   # from L23I
    Coordinates are real H01 micrometres in the NetPyNE frame (translation+inversion only).

    cell_counts: e.g. {'HL23PYR':800,'HL23SST':50,'HL23PV':70,'HL23VIP':80}

    Reproducibility: a single numpy Generator seeded with globalseed drives BOTH the PYR
    draw (from the L23E pool) and a single permutation of the L23I pool, which is then sliced
    sequentially into SST | PV | VIP. The single-shuffle slice guarantees no two interneurons
    share an H01 source cell. PYR (L23E) and interneurons (L23I) draw from DISJOINT pools, so
    using the same seed for both is fine — they cannot collide.
    """
    n_pyr = cell_counts['HL23PYR']
    n_sst = cell_counts['HL23SST']
    n_pv  = cell_counts['HL23PV']
    n_vip = cell_counts['HL23VIP']
    n_inh = n_sst + n_pv + n_vip

    df = load_h01_l23_cells(csv_path, d2d_radius_um=d2d_radius_um)
    df = compute_netpyne_coords(df)

    l23e = df[df['mtype'] == 'L23E'].reset_index(drop=True)
    l23i = df[df['mtype'] == 'L23I'].reset_index(drop=True)

    if n_pyr > len(l23e):
        raise ValueError(f'Need {n_pyr} L23E (PYR) cells but only {len(l23e)} available '
                         f'after d2d<{d2d_radius_um} filter.')
    if n_inh > len(l23i):
        raise ValueError(f'Need {n_inh} L23I (interneuron) cells but only {len(l23i)} available '
                         f'after d2d<{d2d_radius_um} filter.')

    cols = ['x_np', 'y_np', 'z_np']

    # --- PYR from L23E (draw without replacement) ---
    rng_e = np.random.default_rng(globalseed)
    pyr_idx = rng_e.choice(len(l23e), size=n_pyr, replace=False)
    pyr_positions = l23e.iloc[pyr_idx][cols].values.tolist()

    # --- Interneurons from L23I: single shuffle, sequential non-overlapping slices ---
    rng_i = np.random.default_rng(globalseed)
    shuffled = rng_i.permutation(len(l23i))
    sst_positions = l23i.iloc[shuffled[0:n_sst]][cols].values.tolist()
    pv_positions  = l23i.iloc[shuffled[n_sst:n_sst + n_pv]][cols].values.tolist()
    vip_positions = l23i.iloc[shuffled[n_sst + n_pv:n_inh]][cols].values.tolist()

    # --- Report ---
    all_pos = pyr_positions + sst_positions + pv_positions + vip_positions
    ax = {i: [p[i] for p in all_pos] for i in range(3)}
    print(f'[H01] Loaded {len(df)} L23 cells (E={len(l23e)}, I={len(l23i)}) within d2d<{d2d_radius_um} um')
    print(f'[H01] HL23PYR: {n_pyr}/{len(l23e)} L23E  (seed={globalseed})')
    print(f'[H01] HL23SST: {n_sst}/{len(l23i)} L23I  (slots 0:{n_sst})')
    print(f'[H01] HL23PV:  {n_pv}/{len(l23i)} L23I  (slots {n_sst}:{n_sst+n_pv})')
    print(f'[H01] HL23VIP: {n_vip}/{len(l23i)} L23I  (slots {n_sst+n_pv}:{n_inh})')
    print(f'[H01] REAL-geometry slab (distances preserved): '
          f'x[{min(ax[0]):.1f},{max(ax[0]):.1f}] '
          f'y[{min(ax[1]):.1f},{max(ax[1]):.1f}] (depth, 0=pia) '
          f'z[{min(ax[2]):.1f},{max(ax[2]):.1f}] um')

    return {
        'HL23PYR': pyr_positions,
        'HL23SST': sst_positions,
        'HL23PV':  pv_positions,
        'HL23VIP': vip_positions,
    }


def validate_positions(positions_dict, d2d_radius_um=DEFAULT_D2D_RADIUS_UM):
    """
    Sanity-check placed positions. With translation-only there is no fixed [0,500] box to
    assert against; instead we check the geometry is physically sensible:
      - all coords non-negative (origin at slab min),
      - lateral extents <= 2*d2d_radius (x), and z within the real ~174 um slab,
      - depth (y) within the ~800 um L2/3 range.
    Prints a table; raises AssertionError on a clear violation.
    """
    print(f'\n{"Population":12s} | {"n":>5s} | {"x_min":>7s} | {"x_max":>7s} | '
          f'{"y_min":>7s} | {"y_max":>7s} | {"z_min":>7s} | {"z_max":>7s}')
    print('-' * 78)

    all_ok = True
    for pop, positions in positions_dict.items():
        xs = [p[0] for p in positions]; ys = [p[1] for p in positions]; zs = [p[2] for p in positions]
        print(f'{pop:12s} | {len(positions):>5d} | {min(xs):>7.1f} | {max(xs):>7.1f} | '
              f'{min(ys):>7.1f} | {max(ys):>7.1f} | {min(zs):>7.1f} | {max(zs):>7.1f}')
        # physical sanity (generous bounds; real slab is ~598 x 801 x 174)
        if min(xs) < -1 or min(ys) < -1 or min(zs) < -1:
            all_ok = False
        if max(xs) > 2 * d2d_radius_um + 50 or max(ys) > 1300 or max(zs) > 300:
            all_ok = False

    print()
    if not all_ok:
        raise AssertionError('H01 position validation FAILED — coords outside physically sensible '
                             'real-slab bounds. Check the translation transform.')
    print('[H01] All populations within physically sensible real-geometry bounds.')
