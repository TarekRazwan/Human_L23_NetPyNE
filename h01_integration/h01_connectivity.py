"""
h01_connectivity.py
Distance-dependent connectivity for the H01-grounded L2/3 column.

Implements DESIGN DECISIONS 1 + 2 (2026-06-06):
  D1: CLASS-LEVEL (E/I) lambda, shared across SST/PV/VIP.
      H01 does not resolve interneuron subtypes, so distance-decay is fit at the
      excitatory/inhibitory class level (within-L23 specific fits, R^2 > 0.99):
        E->E lambda=101.4   E->I lambda=69.6   I->E lambda=65.5   I->I lambda=52.8  (um)
      Cell-type-specific connection STRUCTURE (probabilities, n_contacts, weights, STP)
      stays Yao-derived (circuit_params). H01 supplies ONLY the distance-decay shape.
  D2: PRESERVE CONVERGENCE (~656 contacts/postsyn cell, the locked baseline).
      The fitted exponential is renormalized per (pre,post) pair so the EXPECTED
      connection probability averaged over real pairwise distances equals the Yao flat
      probability:  A_renorm = P_flat / mean_pairs[ exp(-d/lambda) ].
      Verified on real H01 positions: total convergence 656.5/cell vs baseline 655.7/cell.

This module produces the per-pair NetPyNE 'probability' STRING expressions
('A * exp(-dist_3D/lambda)') to drop into netParams.connParams, replacing the flat
'probability' float. dist_3D is NetPyNE's built-in pre-post soma distance (um), which is
correct ONLY because placement uses real H01 coordinates (h01_placement.py, Decision 3) —
so dist_3D is in the same real-um frame the lambdas were fit in.

NOTE: A_renorm depends on the realized positions, which depend on the seed. Two options:
  (a) recompute A_renorm per seed from that seed's positions (most exact), or
  (b) compute A_renorm once from a reference seed and reuse (simpler; convergence will vary
      by <~1% across seeds since the slab geometry is fixed).
This module supports both via compute_renorm_A(positions).
"""

import numpy as np


# Class-level distance constants (um) — within-L23 specific fits, Cmin30.
LAMBDA_CLASS = {
    ('E', 'E'): 101.4,
    ('E', 'I'): 69.6,
    ('I', 'E'): 65.5,
    ('I', 'I'): 52.8,
}

EXC_POPS = {'HL23PYR'}   # everything else is inhibitory (SST/PV/VIP)


def _ei(pop):
    return 'E' if pop in EXC_POPS else 'I'


def lambda_for(pre, post):
    """Class-level lambda (um) for a (pre,post) population pair."""
    return LAMBDA_CLASS[(_ei(pre), _ei(post))]


def compute_renorm_A(positions, conn_probs, pops=None):
    """
    Compute the renormalization amplitude A for each (pre,post) pair so that the
    distance-dependent probability A*exp(-d/lambda), averaged over all real pre->post
    pairwise distances, equals the flat Yao probability conn_probs[pre][post].

    positions: dict pop -> (N,3) array of real soma coords (um), from h01_placement.
    conn_probs: dict pre -> {post -> flat probability}.
    Returns: dict (pre,post) -> A_renorm  (only for pairs with flat prob > 0).

    Self-connections (pre==post, same cell) are excluded from the mean.
    """
    if pops is None:
        pops = list(positions.keys())
    P = {k: np.asarray(v, dtype=float) for k, v in positions.items()}
    A = {}
    for pre in pops:
        for post in pops:
            p_flat = conn_probs[pre][post]
            if p_flat <= 0:
                continue
            d = np.sqrt(((P[pre][:, None, :] - P[post][None, :, :]) ** 2).sum(-1))
            expd = np.exp(-d / lambda_for(pre, post))
            if pre == post:
                np.fill_diagonal(expd, 0.0)   # no autapse
            mean_pd = expd.mean()
            A[(pre, post)] = p_flat / mean_pd
    return A


def probability_expr(pre, post, A_renorm):
    """
    NetPyNE 'probability' string expression for a (pre,post) pair:
        'A * exp(-dist_3D/lambda)'
    A_renorm: dict from compute_renorm_A. Returns a string for netParams.connParams.
    """
    A = A_renorm[(pre, post)]
    lam = lambda_for(pre, post)
    # NetPyNE evaluates this per pre-post pair; dist_3D is the 3D soma distance (um).
    return f'{A:.6f} * exp(-dist_3D/{lam:.4f})'


def expected_convergence(positions, conn_probs, n_cont, A_renorm, pops=None):
    """
    Sanity tool: expected synaptic contacts per postsynaptic cell, per post population,
    and total per-cell — to confirm the renormalization preserves the locked convergence.
    Returns (per_post_dict, total_contacts, total_per_cell).
    """
    if pops is None:
        pops = list(positions.keys())
    P = {k: np.asarray(v, dtype=float) for k, v in positions.items()}
    conv = {p: 0.0 for p in pops}
    for pre in pops:
        for post in pops:
            if conn_probs[pre][post] <= 0:
                continue
            d = np.sqrt(((P[pre][:, None, :] - P[post][None, :, :]) ** 2).sum(-1))
            expd = np.exp(-d / lambda_for(pre, post))
            if pre == post:
                np.fill_diagonal(expd, 0.0)
            exp_conns_per_post = (A_renorm[(pre, post)] * expd).sum(0).mean()
            conv[post] += exp_conns_per_post * n_cont[pre][post]
    total = sum(conv[p] * len(P[p]) for p in pops)
    ncell = sum(len(P[p]) for p in pops)
    return conv, total, total / ncell
