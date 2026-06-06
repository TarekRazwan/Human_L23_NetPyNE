#!/usr/bin/env python3
"""
AD Cell-Loss Extraction — Driver Script
========================================

This is the thin, config-driven driver for Task B (SEA-AD cell-loss).
It creates a PipelineConfig with the confirmed AD-specific choices,
runs the pipeline, and prints a human-readable summary.

Usage:
    cd <repo_root>
    source transcriptomic/.venv/bin/activate
    python transcriptomic/scripts/run_ad_cellloss.py

All scientific choices are stated explicitly in the config below —
this is the one file to review when asking "what exactly did we compute?"

Outputs (in data/transcriptomic/):
    donor_composition.csv       — per-donor neuron counts + proportions
    cell_loss_vs_ADNC.csv       — survival curves by ADNC stage
    cell_loss_vs_Braak.csv      — survival curves by Braak stage
    cell_loss_vs_CPS.csv        — survival curves by CPS quartile
    figures/survival_vs_*.png   — corresponding plots
"""

import sys
from pathlib import Path

# Ensure the scripts directory is importable
# (allows running from repo root: python transcriptomic/scripts/run_ad_cellloss.py)
sys.path.insert(0, str(Path(__file__).parent))

from seaad_pipeline import PipelineConfig, run_pipeline


def main():
    # ---------------------------------------------------------------
    # The AD cell-loss config — every scientific choice in one place
    # ---------------------------------------------------------------
    cfg = PipelineConfig(
        # Subclass → model population mapping (confirmed Stage 2)
        # L2/3 IT = supragranular excitatory IT → HL23PYR
        # Sst + Sst Chodl → HL23SST (Chodl included: SST+, rare, same vulnerability)
        # Pvalb → HL23PV (Chandelier EXCLUDED: distinct subclass, targets AIS)
        # Vip → HL23VIP
        cell_type_mapping={
            'L2/3 IT':   'PYR',
            'Sst':       'SST',
            'Sst Chodl': 'SST',
            'Pvalb':     'PV',
            'Vip':       'VIP',
        },

        # Run all three severity axes:
        #   ADNC (discrete, 4 bins) — the robust primary
        #   Braak (discrete, 6 stages) — secondary
        #   CPS (continuous → quartiles) — the upgrade for f(CPS)
        severity_axes=['ADNC', 'Braak', 'CPS'],

        # Denominator: fraction of ALL neurons (not just the 4 mapped pops,
        # and not all nuclei including glia — avoids confounding by
        # reactive gliosis in AD)
        denominator='neurons',

        # Normalization anchors:
        #   ADNC "Not AD" (9 donors) — the cleanest healthy baseline
        #   Braak II (4 donors) — Braak 0 has only 2, too noisy for a stable mean
        #   CPS 'lowest' → lowest quartile of CPS scores
        normalization_anchor={
            'ADNC':  'Not AD',
            'Braak': 'Braak II',
            'CPS':   'lowest',
        },

        # File paths (relative to repo root)
        output_dir=Path('data/transcriptomic'),
        cache_dir=Path('transcriptomic/cache/sea-ad'),
    )

    # ---------------------------------------------------------------
    # Run the pipeline
    # ---------------------------------------------------------------
    results = run_pipeline(cfg)

    # ---------------------------------------------------------------
    # Print the terminal survival values for quick reference
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Terminal survival fractions (most-severe bin)")
    print("=" * 60)
    for axis, sanity in results['sanity_checks'].items():
        print(f"\n  {axis}:")
        for pop, val in sorted(sanity['terminal_survivals'].items()):
            # Floor at 1.0 for spared populations (closure artifact)
            floored = min(val, 1.0)
            note = " (floored from {:.3f} — closure artifact)".format(val) if val > 1.0 else ""
            print(f"    {pop}: {floored:.3f}{note}")

    # ---------------------------------------------------------------
    # Summary of scCODA cross-check
    # ---------------------------------------------------------------
    if len(results['sccoda_comparison']) > 0:
        print("\n" + "=" * 60)
        print("scCODA cross-check (CPS axis, compositional model)")
        print("=" * 60)
        print("\n  Raw proportions say → scCODA says:")
        for _, row in results['sccoda_comparison'].iterrows():
            pop = row['population']
            # Get our raw-proportion direction
            if pop in results['sanity_checks'].get('CPS', {}).get('terminal_survivals', {}):
                raw_surv = results['sanity_checks']['CPS']['terminal_survivals'][pop]
                if raw_surv < 0.95:
                    raw_dir = f"loss ({raw_surv:.3f})"
                elif raw_surv > 1.05:
                    raw_dir = f"apparent gain ({raw_surv:.3f}, closure artifact)"
                else:
                    raw_dir = f"stable ({raw_surv:.3f})"
            else:
                raw_dir = "n/a"

            # Median effect from scCODA (compositional model)
            med_eff = row.get('median_effect', row.get('mean_log2fc', float('nan')))
            print(
                f"    {pop:4s}: raw = {raw_dir:40s} | "
                f"scCODA = {row['direction_summary']} "
                f"(median effect: {med_eff:+.3f})"
            )

    return results


if __name__ == '__main__':
    main()
