"""
SEA-AD Compositional Analysis Pipeline
=======================================

Core module for extracting cell-type proportions and survival curves from
the Seattle Alzheimer's Disease Brain Cell Atlas (Gabitto et al., 2024).

This module is designed as a reusable, config-driven pipeline:
  - Any cell-type mapping can be passed in (not just the AD four-population case).
  - Any severity axis available in the donor metadata can be selected.
  - Denominator and normalization anchor are explicit parameters.

The pipeline operates on METADATA ONLY — it never touches the 36 GB expression
matrix.  All data comes from:
  1. The scCODA neuronal supertype abundances h5ad (~0.1 MB) from the SEA-AD
     Supplementary Information on AWS S3.
  2. Supplementary Table 1 (donor demographics + neuropathology, ~46 KB).
  3. A library→donor mapping extracted from the first 10 MB of the per-nucleus
     metadata CSV (S3 range request).

Terminology used throughout:
  - "supertype": finest-grained cell-type label in SEA-AD (109 neuronal types)
  - "subclass":  intermediate grouping (18 neuronal subclasses, e.g. "L2/3 IT")
  - "population": model-level grouping (e.g. PYR, SST, PV, VIP)
  - "library":   one 10x Chromium run (the unit in the scCODA file, 170 total)
  - "donor":     one brain (69 with snRNA-seq data, each with 1–5 libraries)

Data source: SEA-AD MTG snRNA-seq, release 2024-02-13
S3 bucket: sea-ad-single-cell-profiling (AWS Open Data, no auth required)
"""

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anndata
import boto3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    All tunable parameters for one run of the compositional pipeline.

    This is the single place where scientific choices are stated explicitly.
    The driver script (run_ad_cellloss.py) creates one of these with the
    AD-specific defaults; a future cohort would create a different one.

    Attributes
    ----------
    cell_type_mapping : dict
        Maps SEA-AD subclass names → model population labels.
        Only subclasses listed here become tracked populations; all others
        still count toward the denominator but are not reported individually.
    severity_axes : list of str
        Which donor-level severity columns to analyze.  Each must be either
        a column in the scCODA obs (e.g. 'Continuous_Pseudo-progression_Score')
        or a column in Supplementary Table 1 (e.g. 'Braak').
    denominator : str
        What to divide by when computing proportions:
          'neurons'      — total neuronal nuclei per donor (default)
          'all_nuclei'   — would require the non-neuronal abundances too
          'within_class' — fraction within GABAergic or Glutamatergic class
    normalization_anchor : dict
        Per-axis: which severity bin = 1.0 (baseline).
        Example: {'ADNC': 'Not AD', 'Braak': 'Braak II', 'CPS': 'lowest'}
        For continuous axes like CPS, 'lowest' means the lowest-CPS donor bin.
    braak_anchor_note : str
        Why we don't default to Braak 0 (only 2 donors — too noisy).
    output_dir : Path
        Where to write CSVs and figures.
    cache_dir : Path
        Where to cache downloaded S3 files.
    s3_bucket : str
        S3 bucket name (allows pointing at a different release or cohort).
    s3_region : str
        AWS region for the S3 bucket.
    abundances_key : str
        S3 key for the scCODA neuronal abundances h5ad.
    table1_key : str
        S3 key for Supplementary Table 1 (donor metadata).
    metadata_csv_key : str
        S3 key for the per-nucleus metadata CSV (only first 10 MB read).
    sccoda_results_key : str
        S3 key for the scCODA neuronal results CSV (for cross-check).
    """
    cell_type_mapping: dict = field(default_factory=lambda: {
        'L2/3 IT':   'PYR',
        'Sst':       'SST',
        'Sst Chodl': 'SST',   # included: SST-expressing, rare (~0.1%), same vulnerability
        'Pvalb':     'PV',
        'Vip':       'VIP',
        # Chandelier excluded: Pvalb-expressing but distinct subclass, targets AIS not soma
        # All other subclasses (L4/5/6 IT, Lamp5, Sncg, Pax6, etc.) → denominator only
    })
    severity_axes: list = field(default_factory=lambda: [
        'ADNC', 'Braak', 'CPS',
    ])
    denominator: str = 'neurons'
    normalization_anchor: dict = field(default_factory=lambda: {
        'ADNC':  'Not AD',
        # Braak 0 has only 2 donors — too few for a stable baseline.
        # Braak II (4 donors) is the lowest stage with reasonable n.
        'Braak': 'Braak II',
        'CPS':   'lowest',
    })
    braak_anchor_note: str = (
        'Braak 0 has only 2 donors in this cohort, making it an unstable '
        'anchor.  Braak II (n=4) is the lowest stage with enough donors for '
        'a meaningful mean.  This is a pragmatic choice, not a biological claim '
        'that Braak II is "healthy".'
    )
    output_dir: Path = field(default_factory=lambda: Path('data/transcriptomic'))
    cache_dir: Path = field(default_factory=lambda: Path('transcriptomic/cache/sea-ad'))
    s3_bucket: str = 'sea-ad-single-cell-profiling'
    s3_region: str = 'us-west-2'
    abundances_key: str = (
        'MTG/RNAseq/Supplementary Information/scCODA Results/MTG_RNAseq/'
        'objects/Neuronal: Glutamatergic Neuronal: GABAergic_Supertype_abundances.h5ad'
    )
    table1_key: str = (
        'MTG/RNAseq/Supplementary Information/Supplementary Table 1.xlsx'
    )
    metadata_csv_key: str = (
        'MTG/RNAseq/Supplementary Information/'
        'SEAAD_MTG_RNAseq_all-nuclei_metadata.2024-02-13.csv'
    )
    sccoda_results_key: str = (
        'MTG/RNAseq/Supplementary Information/scCODA Results/MTG_RNAseq/'
        'Continuous_Pseudo-progression_Score/'
        'Neuronal: Glutamatergic Neuronal: GABAergic_Supertype_results.csv'
    )


# ---------------------------------------------------------------------------
# 1. Data access — download small files from S3, build the donor table
# ---------------------------------------------------------------------------

def _get_s3_client(region: str) -> boto3.client:
    """Create an anonymous S3 client (SEA-AD data is public, no auth needed)."""
    return boto3.client(
        's3',
        config=BotoConfig(signature_version=UNSIGNED),
        region_name=region,
    )


def _download_if_missing(s3, bucket: str, key: str, local_path: Path) -> Path:
    """
    Download a file from S3 only if it isn't already cached locally.

    This avoids re-downloading on every run.  Delete the cached file
    to force a fresh download.

    Parameters
    ----------
    s3 : boto3 S3 client
    bucket : S3 bucket name
    key : S3 object key
    local_path : where to save locally

    Returns
    -------
    Path to the local file.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading {key.split('/')[-1]} ...")
        s3.download_file(bucket, str(key), str(local_path))
        print(f"  → saved to {local_path} ({local_path.stat().st_size / 1e3:.0f} KB)")
    else:
        print(f"  Cached: {local_path.name}")
    return local_path


def load_abundances(cfg: PipelineConfig) -> anndata.AnnData:
    """
    Load the scCODA neuronal supertype abundances from S3.

    This is a tiny file (~0.1 MB): 170 libraries × 109 neuronal supertypes,
    with per-library donor-level covariates (CPS, ADNC codes, etc.) in .obs.
    The .X matrix contains raw cell COUNTS (not proportions).

    Returns
    -------
    AnnData with .X = counts, .obs = library-level metadata, .var = supertypes.
    """
    s3 = _get_s3_client(cfg.s3_region)
    local = _download_if_missing(
        s3, cfg.s3_bucket, cfg.abundances_key,
        cfg.cache_dir / 'neuronal_supertype_abundances.h5ad',
    )
    return anndata.read_h5ad(local)


def load_donor_metadata(cfg: PipelineConfig) -> pd.DataFrame:
    """
    Load Supplementary Table 1: donor demographics + neuropathology.

    This is the authoritative source for text-valued ADNC, Braak, Thal,
    CERAD, cognitive status, and other donor-level clinical variables.

    Returns
    -------
    DataFrame indexed by Donor ID, with 70+ columns.
    """
    s3 = _get_s3_client(cfg.s3_region)
    local = _download_if_missing(
        s3, cfg.s3_bucket, cfg.table1_key,
        cfg.cache_dir / 'Supplementary_Table_1.xlsx',
    )
    df = pd.read_excel(local)
    df = df.set_index('Donor ID')
    return df


def build_library_to_donor_map(cfg: PipelineConfig) -> dict:
    """
    Extract the library_id → donor_id mapping from the per-nucleus metadata.

    We only need the first ~10 MB of the 2.1 GB CSV — enough to see thousands
    of cells and extract every unique library→donor pair.  This uses an S3
    range request to avoid downloading the full file.

    The per-nucleus CSV has barcodes like:
        TTCATGTCAATGTTGC-L8TX_210429_01_D03-1142430416
    where the middle segment (L8TX_...) is the library ID, and column 3
    is the Donor ID (e.g. H21.33.003).

    Returns
    -------
    dict mapping library_id (str) → donor_id (str).
    """
    cache_path = cfg.cache_dir / 'library_to_donor.csv'

    if cache_path.exists():
        print("  Cached: library_to_donor.csv")
        cached = pd.read_csv(cache_path)
        return dict(zip(cached['library_id'], cached['donor_id']))

    print("  Extracting library→donor mapping via S3 range request (first 10 MB)...")
    s3 = _get_s3_client(cfg.s3_region)
    resp = s3.get_object(
        Bucket=cfg.s3_bucket,
        Key=cfg.metadata_csv_key,
        Range='bytes=0-10485760',  # 10 MB — covers thousands of cells
    )
    text = resp['Body'].read().decode('utf-8')

    lib_to_donor = {}
    for line in text.split('\n')[1:]:  # skip header
        if not line.strip():
            continue
        fields = line.split(',')
        if len(fields) < 4:
            continue
        # sample_id format: BARCODE-LIBRARY_ID-NUMBER
        parts = fields[1].split('-')
        if len(parts) >= 2:
            lib_to_donor[parts[1]] = fields[3]

    # Cache for future runs
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        list(lib_to_donor.items()), columns=['library_id', 'donor_id']
    ).to_csv(cache_path, index=False)
    print(f"  → {len(lib_to_donor)} libraries mapped to "
          f"{len(set(lib_to_donor.values()))} donors")

    return lib_to_donor


# ---------------------------------------------------------------------------
# 2. Aggregation — libraries → donors, supertypes → populations
# ---------------------------------------------------------------------------

def aggregate_to_donors(
    adata: anndata.AnnData,
    lib_to_donor: dict,
    donor_meta: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """
    Aggregate library-level supertype counts to donor-level population counts.

    This function does three things:
      1. Sums each donor's libraries into a single count vector
         (critical: avoids pseudoreplication from treating libraries as
         independent observations).
      2. Collapses 109 supertypes → subclasses → model populations using
         the configured cell_type_mapping.
      3. Joins donor-level severity variables (ADNC text, Braak, CPS, etc.).

    Parameters
    ----------
    adata : AnnData from load_abundances() — 170 libraries × 109 supertypes
    lib_to_donor : dict from build_library_to_donor_map()
    donor_meta : DataFrame from load_donor_metadata() — indexed by Donor ID
    cfg : PipelineConfig with the cell_type_mapping

    Returns
    -------
    DataFrame with one row per donor, columns:
      - One column per mapped population (PYR, SST, PV, VIP) = raw count
      - 'other_neurons' = neurons not in any mapped population
      - 'total_neurons' = sum of all neuronal counts
      - 'n_libraries' = how many 10x libraries this donor contributed
      - Severity columns: 'ADNC', 'Braak', 'CPS', etc.
    """
    # --- Step 1: Map supertypes → subclasses ---
    # Supertypes are named like "L2/3 IT_1", "Sst_10", "Sst Chodl_2".
    # The subclass is everything before the last "_N" suffix.
    supertype_to_subclass = {}
    for st in adata.var.index:
        # rsplit on '_' gets: ("L2/3 IT", "1") or ("Sst Chodl", "2")
        parts = st.rsplit('_', 1)
        supertype_to_subclass[st] = parts[0]

    # --- Step 2: Map subclasses → populations (from config) ---
    # Invert: for each supertype, what population (if any) does it belong to?
    supertype_to_pop = {}
    for st, subclass in supertype_to_subclass.items():
        supertype_to_pop[st] = cfg.cell_type_mapping.get(subclass, None)

    # Get the set of populations we're tracking
    populations = sorted(set(cfg.cell_type_mapping.values()))

    # --- Step 3: Add donor_id to each library ---
    obs = adata.obs.copy()
    obs['donor_id'] = obs.index.map(lib_to_donor)
    unmapped = obs['donor_id'].isna().sum()
    if unmapped > 0:
        warnings.warn(f"{unmapped} libraries could not be mapped to a donor")
    obs = obs.dropna(subset=['donor_id'])

    # --- Step 4: Sum libraries → donors ---
    # Convert AnnData .X to a DataFrame of counts per supertype
    counts_df = pd.DataFrame(
        adata.X,
        index=adata.obs.index,
        columns=adata.var.index,
    )
    counts_df['donor_id'] = obs['donor_id']

    # Sum all libraries belonging to the same donor
    donor_counts = counts_df.groupby('donor_id').sum()

    # --- Step 5: Collapse supertypes → populations ---
    result = pd.DataFrame(index=donor_counts.index)
    for pop in populations:
        # Which supertypes belong to this population?
        cols = [st for st, p in supertype_to_pop.items() if p == pop]
        result[pop] = donor_counts[cols].sum(axis=1)

    # "other_neurons": all supertypes NOT mapped to any population
    unmapped_cols = [st for st, p in supertype_to_pop.items() if p is None]
    result['other_neurons'] = donor_counts[unmapped_cols].sum(axis=1)

    # Total neurons = mapped + unmapped (the denominator for proportions)
    result['total_neurons'] = donor_counts.sum(axis=1)

    # How many libraries per donor (useful for diagnostics)
    result['n_libraries'] = counts_df.groupby('donor_id').size()

    # --- Step 6: Join severity variables from donor metadata ---
    # ADNC (text label from Table 1)
    if 'Overall AD neuropathological Change' in donor_meta.columns:
        result['ADNC'] = result.index.map(
            donor_meta['Overall AD neuropathological Change']
        )

    # Braak (from Table 1)
    if 'Braak' in donor_meta.columns:
        result['Braak'] = result.index.map(donor_meta['Braak'])

    # Thal (from Table 1)
    if 'Thal' in donor_meta.columns:
        result['Thal'] = result.index.map(donor_meta['Thal'])

    # CERAD (from Table 1)
    if 'CERAD score' in donor_meta.columns:
        result['CERAD'] = result.index.map(donor_meta['CERAD score'])

    # CPS (from scCODA obs — it's donor-level, take the first library's value)
    cps_per_lib = obs.set_index('donor_id')['Continuous_Pseudo-progression_Score']
    # Verify CPS is truly donor-level (same for all libraries of a donor)
    cps_check = cps_per_lib.groupby(level=0).nunique()
    if (cps_check > 1).any():
        warnings.warn("CPS varies across libraries for some donors!")
    result['CPS'] = cps_per_lib.groupby(level=0).first()

    # Cognitive status
    if 'Cognitive Status' in donor_meta.columns:
        result['Cognitive_Status'] = result.index.map(
            donor_meta['Cognitive Status']
        )

    # APOE4 carrier status (for Direction 4 stratification)
    if 'APOE Genotype' in donor_meta.columns:
        # Carrier = any genotype containing '4' (e.g. e3/e4, e4/e4)
        apoe_geno = donor_meta['APOE Genotype']
        is_carrier = apoe_geno.str.contains('4', na=False)
        result['APOE4_carrier'] = result.index.map(is_carrier)

    result.index.name = 'donor_id'
    return result


# ---------------------------------------------------------------------------
# 3. Analysis — proportions, survival curves, sanity checks
# ---------------------------------------------------------------------------

def compute_proportions(
    donor_df: pd.DataFrame,
    populations: list,
    denominator: str = 'neurons',
) -> pd.DataFrame:
    """
    Compute per-donor cell-type proportions.

    For each donor, proportion = (count of population X) / (denominator).

    Parameters
    ----------
    donor_df : DataFrame from aggregate_to_donors()
    populations : list of population names (e.g. ['PYR', 'SST', 'PV', 'VIP'])
    denominator : 'neurons' (default) — divide by total neuronal nuclei.
                  Other options reserved for future use.

    Returns
    -------
    DataFrame with columns like 'PYR_prop', 'SST_prop', etc.,
    plus all original columns from donor_df.
    """
    df = donor_df.copy()

    if denominator == 'neurons':
        denom = df['total_neurons']
    else:
        raise ValueError(f"Unsupported denominator: {denominator}")

    for pop in populations:
        # Proportion = count / denominator
        df[f'{pop}_prop'] = df[pop] / denom

    return df


def compute_survival_curves(
    donor_df: pd.DataFrame,
    populations: list,
    severity_col: str,
    anchor_value: str,
    ordered_bins: Optional[list] = None,
) -> pd.DataFrame:
    """
    Compute survival fractions for each population across severity bins.

    "Survival fraction" = mean proportion in this bin / mean proportion in
    the anchor (healthiest) bin.  A value of 1.0 means no change from
    baseline; 0.5 means the population halved.

    IMPORTANT — compositional closure caveat:
    Because proportions sum to 1, when one population drops (e.g. SST),
    others MUST rise in relative terms even if their absolute count is stable.
    This means spared populations (PYR, VIP) may show survival > 1.0.
    That's a mathematical artifact, not biological growth.  The output
    includes a 'closure_artifact' flag for bins where survival > 1.0.

    Parameters
    ----------
    donor_df : DataFrame with proportion columns (from compute_proportions())
    populations : list of population names
    severity_col : column name to bin by (e.g. 'ADNC', 'Braak', 'CPS')
    anchor_value : which bin = baseline 1.0 (e.g. 'Not AD').
                   For continuous axes, pass 'lowest' to use the lowest-value bin.
    ordered_bins : optional explicit ordering of severity bins for the output.
                   If None, bins are sorted alphabetically/numerically.

    Returns
    -------
    DataFrame with columns:
      population, severity_bin, n_donors, mean_proportion, sem,
      survival_fraction, closure_artifact
    """
    prop_cols = {pop: f'{pop}_prop' for pop in populations}

    # For continuous axes like CPS, bin into quartiles
    if severity_col == 'CPS':
        df = donor_df.copy()
        # Create CPS quartile bins
        df['_severity_bin'] = pd.qcut(
            df['CPS'], q=4,
            labels=['CPS Q1 (lowest)', 'CPS Q2', 'CPS Q3', 'CPS Q4 (highest)'],
        )
        if anchor_value == 'lowest':
            anchor_value = 'CPS Q1 (lowest)'
        if ordered_bins is None:
            ordered_bins = ['CPS Q1 (lowest)', 'CPS Q2', 'CPS Q3', 'CPS Q4 (highest)']
    else:
        df = donor_df.copy()
        df['_severity_bin'] = df[severity_col]

    # Drop rows with missing severity values
    df = df.dropna(subset=['_severity_bin'])

    # Group by severity bin, compute mean ± SEM for each population
    rows = []
    for bin_val, grp in df.groupby('_severity_bin', observed=True):
        for pop in populations:
            prop_values = grp[prop_cols[pop]]
            rows.append({
                'population': pop,
                'severity_bin': bin_val,
                'n_donors': len(grp),
                'mean_proportion': prop_values.mean(),
                'sem': prop_values.sem(),
            })

    result = pd.DataFrame(rows)

    # Compute survival fraction: normalize to anchor bin
    for pop in populations:
        mask = (result['population'] == pop)
        anchor_mask = mask & (result['severity_bin'] == anchor_value)
        anchor_mean = result.loc[anchor_mask, 'mean_proportion'].values

        if len(anchor_mean) == 0:
            warnings.warn(
                f"Anchor '{anchor_value}' not found for {severity_col}. "
                f"Available bins: {result.loc[mask, 'severity_bin'].unique()}"
            )
            result.loc[mask, 'survival_fraction'] = np.nan
        else:
            result.loc[mask, 'survival_fraction'] = (
                result.loc[mask, 'mean_proportion'].values / anchor_mean[0]
            )

    # Flag compositional closure artifact: survival > 1.0
    # When lost populations (SST/PV) shrink, spared populations (PYR/VIP)
    # can appear to GROW in relative composition.  This is mathematical,
    # not biological.  Flag it so downstream consumers know.
    result['closure_artifact'] = result['survival_fraction'] > 1.0

    # Sort output
    if ordered_bins is not None:
        bin_order = {b: i for i, b in enumerate(ordered_bins)}
        result['_sort'] = result['severity_bin'].map(bin_order)
        result = result.sort_values(['population', '_sort']).drop(columns='_sort')
    else:
        result = result.sort_values(['population', 'severity_bin'])

    result = result.reset_index(drop=True)
    return result


def run_sanity_check(survival_df: pd.DataFrame, severity_col: str) -> dict:
    """
    Check whether the survival curves match known AD cell-loss biology.

    The expected pattern (from Gabitto et al. 2024 and replicated in
    Barker et al. 2024 across 1,373 donors):
      - SST: drops EARLY and substantially (the most vulnerable)
      - PV:  drops LATE (less vulnerable than SST, more than PYR)
      - PYR: relatively SPARED until late stages
      - VIP: relatively SPARED throughout

    This function checks the curves at the most-severe bin and reports
    pass/fail.  A failure is NOT necessarily a bug — it could be a real
    finding — but it should trigger manual review before shipping.

    Parameters
    ----------
    survival_df : DataFrame from compute_survival_curves()
    severity_col : which axis this is (for reporting)

    Returns
    -------
    dict with keys: 'passed', 'details' (str), 'terminal_survivals' (dict)
    """
    # Get the most-severe bin (last bin) survival for each population
    pops = survival_df['population'].unique()
    terminal = {}
    for pop in pops:
        pop_data = survival_df[survival_df['population'] == pop]
        # Last row is most-severe bin
        terminal[pop] = pop_data.iloc[-1]['survival_fraction']

    details = []
    passed = True

    # Check 1: SST should drop the most (lowest terminal survival)
    if 'SST' in terminal and 'PV' in terminal:
        if terminal['SST'] > terminal['PV']:
            details.append(
                f"WARNING: SST ({terminal['SST']:.3f}) > PV ({terminal['PV']:.3f}) "
                f"at highest severity — expected SST to drop MORE than PV"
            )
            passed = False
        else:
            details.append(
                f"OK: SST ({terminal['SST']:.3f}) ≤ PV ({terminal['PV']:.3f}) "
                f"— SST drops more, as expected"
            )

    # Check 2: VIP should be relatively spared (survival close to 1.0)
    if 'VIP' in terminal:
        if terminal['VIP'] < 0.85:
            details.append(
                f"WARNING: VIP ({terminal['VIP']:.3f}) drops below 0.85 "
                f"— expected VIP to be relatively spared"
            )
            passed = False
        else:
            details.append(
                f"OK: VIP ({terminal['VIP']:.3f}) ≥ 0.85 — relatively spared"
            )

    # Check 3: PYR should be more spared than SST
    if 'PYR' in terminal and 'SST' in terminal:
        if terminal['PYR'] < terminal['SST']:
            details.append(
                f"WARNING: PYR ({terminal['PYR']:.3f}) < SST ({terminal['SST']:.3f}) "
                f"— expected PYR to be MORE spared than SST"
            )
            passed = False
        else:
            details.append(
                f"OK: PYR ({terminal['PYR']:.3f}) ≥ SST ({terminal['SST']:.3f}) "
                f"— PYR more spared"
            )

    return {
        'passed': passed,
        'severity_axis': severity_col,
        'details': '\n'.join(details),
        'terminal_survivals': terminal,
    }


# ---------------------------------------------------------------------------
# 4. scCODA cross-check
# ---------------------------------------------------------------------------

def load_sccoda_results(cfg: PipelineConfig) -> pd.DataFrame:
    """
    Load the pre-computed scCODA compositional analysis results.

    scCODA (Büttner et al. 2021) models cell-type composition using a
    Bayesian framework that properly handles compositional closure — unlike
    raw proportions, it accounts for the fact that cell-type fractions must
    sum to 1.  The SEA-AD team ran scCODA against CPS and ADNC.

    The results CSV contains, for each supertype:
      - 'Final Parameter': log-fold-change effect of the covariate
      - 'Inclusion probability': probability that this effect is non-zero
      - 'Expected Sample': expected cell count under the model
      - 'log2-fold change': effect size in log2 scale

    Returns
    -------
    DataFrame with scCODA results per supertype.
    """
    s3 = _get_s3_client(cfg.s3_region)
    local = _download_if_missing(
        s3, cfg.s3_bucket, cfg.sccoda_results_key,
        cfg.cache_dir / 'sccoda_neuronal_cps_results.csv',
    )
    return pd.read_csv(local)


def cross_check_sccoda(
    sccoda_df: pd.DataFrame,
    cell_type_mapping: dict,
) -> pd.DataFrame:
    """
    Summarize scCODA credible-change results at the population level.

    scCODA is a Bayesian compositional model (Büttner et al. 2021) that
    properly accounts for compositional closure.  The SEA-AD results CSV
    is a 109×109 matrix: for each (reference_cell_type, target_cell_type)
    pair and each covariate, it reports a "Final Parameter" (effect size)
    and an "Inclusion probability" (credibility of the effect).

    Because results depend on the reference cell type, we take the
    MEDIAN effect across all 109 references for each target supertype.
    This is robust to the reference choice.  We then aggregate supertypes
    → populations using the cell_type_mapping.

    A supertype is considered "credibly changing" if its median inclusion
    probability > 0.5 across references.

    Parameters
    ----------
    sccoda_df : DataFrame from load_sccoda_results()
    cell_type_mapping : the subclass→population mapping dict

    Returns
    -------
    DataFrame with columns: population, n_supertypes, n_credible_changes,
    median_effect, direction_summary
    """
    # --- Filter to CPS covariate only ---
    cps_mask = sccoda_df['Covariate'] == 'Continuous_Pseudo-progression_Score'
    cps_df = sccoda_df[cps_mask].copy()

    if len(cps_df) == 0:
        warnings.warn("No CPS rows found in scCODA results")
        return pd.DataFrame()

    # --- For each target cell type, compute MEDIAN across all references ---
    # This makes the result robust to reference-cell-type choice.
    per_target = cps_df.groupby('Cell Type').agg(
        median_param=('Final Parameter', 'median'),
        median_incl=('Inclusion probability', 'median'),
        mean_param=('Final Parameter', 'mean'),
    ).reset_index()

    # --- Map supertypes → populations ---
    def supertype_to_pop(st_name):
        parts = st_name.rsplit('_', 1)
        subclass = parts[0] if len(parts) == 2 else st_name
        return cell_type_mapping.get(subclass)

    per_target['population'] = per_target['Cell Type'].apply(supertype_to_pop)

    # Keep only supertypes that map to a model population
    mapped = per_target.dropna(subset=['population'])

    # --- Aggregate to population level ---
    rows = []
    for pop in sorted(mapped['population'].unique()):
        pop_data = mapped[mapped['population'] == pop]
        n_st = len(pop_data)
        # A supertype "credibly changes" if median inclusion prob > 0.5
        n_credible = (pop_data['median_incl'] > 0.5).sum()
        median_eff = pop_data['median_param'].median()

        # Direction: negative = loss with increasing CPS (worsening AD)
        if n_credible == 0:
            direction = 'no credible change (compositionally stable)'
        elif median_eff < -0.1:
            direction = 'LOSS (decreasing with CPS)'
        elif median_eff > 0.1:
            direction = 'GAIN (increasing with CPS)'
        else:
            direction = 'weak / mixed change'

        rows.append({
            'population': pop,
            'n_supertypes': n_st,
            'n_credible_changes': int(n_credible),
            'median_effect': median_eff,
            'direction_summary': direction,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------------

def plot_survival_curves(
    survival_df: pd.DataFrame,
    severity_col: str,
    title_suffix: str = '',
    output_path: Optional[Path] = None,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    Plot survival fraction vs severity for each population.

    Includes a dashed line at survival=1.0 (baseline) and annotates
    the compositional-closure caveat for any population that rises above 1.0.

    Parameters
    ----------
    survival_df : DataFrame from compute_survival_curves()
    severity_col : label for x-axis (e.g. 'ADNC', 'Braak')
    title_suffix : appended to plot title
    output_path : if provided, saves figure to this path
    figsize : figure dimensions

    Returns
    -------
    matplotlib Figure object.
    """
    # Color scheme: match the biological story
    # SST=red (most vulnerable), PV=orange, PYR=blue (spared), VIP=green (spared)
    pop_colors = {
        'SST': '#d62728',
        'PV':  '#ff7f0e',
        'PYR': '#1f77b4',
        'VIP': '#2ca02c',
    }
    pop_markers = {'SST': 'o', 'PV': 's', 'PYR': '^', 'VIP': 'D'}

    fig, ax = plt.subplots(figsize=figsize)

    populations = survival_df['population'].unique()
    bins = survival_df['severity_bin'].unique()

    for pop in populations:
        pop_data = survival_df[survival_df['population'] == pop]
        x = range(len(pop_data))
        y = pop_data['survival_fraction'].values
        err = pop_data['sem'].values / pop_data.iloc[0]['mean_proportion']  # normalized SEM

        color = pop_colors.get(pop, '#333333')
        marker = pop_markers.get(pop, 'o')

        ax.errorbar(
            x, y, yerr=err,
            label=pop, color=color, marker=marker,
            linewidth=2, markersize=8, capsize=4,
        )

    # Baseline reference
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Compositional closure caveat annotation
    has_closure = survival_df['closure_artifact'].any()
    if has_closure:
        ax.text(
            0.98, 0.02,
            'Note: values > 1.0 reflect compositional\n'
            'closure (relative, not absolute increase)',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=7, fontstyle='italic', color='gray',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      edgecolor='gray', alpha=0.8),
        )

    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins, rotation=30, ha='right')
    ax.set_xlabel(severity_col)
    ax.set_ylabel('Survival fraction (normalized to baseline)')
    ax.set_title(f'Cell-type survival vs {severity_col}{title_suffix}')
    ax.legend(loc='best')
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# 5b. Direction 1 — f(CPS) curve-shape fitting
# ---------------------------------------------------------------------------

def _sigmoid(x, L, k, x0, b):
    """Logistic/sigmoid: L / (1 + exp(-k*(x - x0))) + b."""
    return L / (1.0 + np.exp(-k * (x - x0))) + b


def _hinge(x, x0, slope_before, slope_after, y0):
    """
    Piecewise-linear hinge: flat/shallow before x0, steeper decline after.

    This models a "threshold" onset — the population is stable until CPS
    reaches x0, then starts declining.  Many biological loss curves have
    this shape (compensated homeostasis until a tipping point).
    """
    return np.where(
        x < x0,
        y0 + slope_before * (x - x0),
        y0 + slope_after * (x - x0),
    )


def fit_cps_curves(
    donor_df: pd.DataFrame,
    populations: list,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Fit candidate functional forms to each population's survival-vs-CPS curve.

    For each population, we fit four models to the per-donor proportion data
    (NOT the binned means — we use all 69 individual donor values for the fit):
      1. Linear:    survival = a * CPS + b
      2. Sigmoid:   survival = L / (1 + exp(-k*(CPS - x0))) + b
      3. Exponential: survival = a * exp(b * CPS) + c
      4. Hinge:     piecewise-linear with breakpoint at x0

    We compare models using AIC (Akaike Information Criterion).  Lower AIC
    is better; AIC penalizes extra parameters, so a sigmoid only "wins" if
    it fits meaningfully better than a line.

    The "onset CPS" is defined as:
      - For sigmoid: the inflection point x0 (where loss accelerates)
      - For hinge: the breakpoint x0
      - For linear: no onset (loss is gradual from the start)
      - For exponential: the CPS where survival drops to 0.95 (5% loss)

    Parameters
    ----------
    donor_df : DataFrame with CPS and proportion columns (from compute_proportions)
    populations : list of population names (e.g. ['PYR', 'SST', 'PV', 'VIP'])
    output_dir : if provided, saves the figure and CSV there

    Returns
    -------
    DataFrame with one row per population, columns:
      population, best_model, aic_linear, aic_sigmoid, aic_exponential,
      aic_hinge, onset_cps, parameters (as string), r_squared
    """
    from scipy.optimize import curve_fit
    from scipy.stats import linregress

    df = donor_df.dropna(subset=['CPS']).copy()
    cps = df['CPS'].values
    # Normalize CPS to [0,1] range for fitting stability
    cps_min, cps_max = cps.min(), cps.max()

    results = []

    for pop in populations:
        prop_col = f'{pop}_prop'
        if prop_col not in df.columns:
            continue

        y = df[prop_col].values
        # Normalize y to survival fraction (relative to low-CPS mean)
        # Use donors in the lowest CPS quartile as the baseline
        q25 = np.percentile(cps, 25)
        baseline = y[cps <= q25].mean()
        if baseline == 0:
            continue
        y_surv = y / baseline

        n = len(y_surv)
        fits = {}

        # --- Model 1: Linear ---
        slope, intercept, r, p, se = linregress(cps, y_surv)
        y_pred_lin = slope * cps + intercept
        rss_lin = np.sum((y_surv - y_pred_lin) ** 2)
        k_lin = 2  # parameters: slope, intercept
        # AIC = n * ln(RSS/n) + 2k  (for least-squares models)
        aic_lin = n * np.log(rss_lin / n) + 2 * k_lin
        fits['linear'] = {
            'aic': aic_lin,
            'params': {'slope': slope, 'intercept': intercept},
            'r_squared': r ** 2,
            'y_pred': y_pred_lin,
            # Onset: CPS where survival crosses 0.95 (5% loss threshold)
            'onset_cps': (0.95 - intercept) / slope if slope != 0 else np.nan,
        }

        # --- Model 2: Sigmoid ---
        try:
            # Initial guesses: curve drops from ~1.0 to ~0.7 across CPS range
            p0 = [-0.3, 10, 0.5, 1.0]  # L, k, x0, b
            bounds = ([-1.0, 0.1, 0.0, 0.0], [0.0, 50, 1.0, 2.0])
            popt, pcov = curve_fit(_sigmoid, cps, y_surv, p0=p0, bounds=bounds,
                                   maxfev=5000)
            y_pred_sig = _sigmoid(cps, *popt)
            rss_sig = np.sum((y_surv - y_pred_sig) ** 2)
            k_sig = 4
            aic_sig = n * np.log(rss_sig / n) + 2 * k_sig
            ss_tot = np.sum((y_surv - y_surv.mean()) ** 2)
            r2_sig = 1 - rss_sig / ss_tot if ss_tot > 0 else 0

            fits['sigmoid'] = {
                'aic': aic_sig,
                'params': {'L': popt[0], 'k': popt[1], 'x0': popt[2], 'b': popt[3]},
                'r_squared': r2_sig,
                'y_pred': y_pred_sig,
                # Onset = inflection point x0 (where loss accelerates)
                'onset_cps': popt[2],
            }
        except (RuntimeError, ValueError):
            fits['sigmoid'] = {'aic': np.inf, 'params': {}, 'r_squared': 0,
                               'y_pred': np.full_like(cps, np.nan), 'onset_cps': np.nan}

        # --- Model 3: Exponential decay ---
        try:
            def exp_decay(x, a, b, c):
                return a * np.exp(b * x) + c
            p0_exp = [1.0, -1.0, 0.0]
            bounds_exp = ([0.0, -10, -1.0], [3.0, 0, 2.0])
            popt_e, _ = curve_fit(exp_decay, cps, y_surv, p0=p0_exp,
                                   bounds=bounds_exp, maxfev=5000)
            y_pred_exp = exp_decay(cps, *popt_e)
            rss_exp = np.sum((y_surv - y_pred_exp) ** 2)
            k_exp = 3
            aic_exp = n * np.log(rss_exp / n) + 2 * k_exp
            ss_tot = np.sum((y_surv - y_surv.mean()) ** 2)
            r2_exp = 1 - rss_exp / ss_tot if ss_tot > 0 else 0

            # Onset: CPS where predicted survival = 0.95
            try:
                onset_e = np.log((0.95 - popt_e[2]) / popt_e[0]) / popt_e[1]
                if not (0 <= onset_e <= 1):
                    onset_e = np.nan
            except (ValueError, ZeroDivisionError):
                onset_e = np.nan

            fits['exponential'] = {
                'aic': aic_exp,
                'params': {'a': popt_e[0], 'b': popt_e[1], 'c': popt_e[2]},
                'r_squared': r2_exp,
                'y_pred': y_pred_exp,
                'onset_cps': onset_e,
            }
        except (RuntimeError, ValueError):
            fits['exponential'] = {'aic': np.inf, 'params': {}, 'r_squared': 0,
                                    'y_pred': np.full_like(cps, np.nan), 'onset_cps': np.nan}

        # --- Model 4: Hinge (piecewise linear) ---
        try:
            p0_h = [0.5, 0.0, -0.5, 1.0]  # x0, slope_before, slope_after, y0
            bounds_h = ([0.1, -2, -5, 0], [0.9, 2, 0, 2])
            popt_h, _ = curve_fit(_hinge, cps, y_surv, p0=p0_h,
                                   bounds=bounds_h, maxfev=5000)
            y_pred_h = _hinge(cps, *popt_h)
            rss_h = np.sum((y_surv - y_pred_h) ** 2)
            k_h = 4
            aic_h = n * np.log(rss_h / n) + 2 * k_h
            ss_tot = np.sum((y_surv - y_surv.mean()) ** 2)
            r2_h = 1 - rss_h / ss_tot if ss_tot > 0 else 0

            fits['hinge'] = {
                'aic': aic_h,
                'params': {'x0': popt_h[0], 'slope_before': popt_h[1],
                           'slope_after': popt_h[2], 'y0': popt_h[3]},
                'r_squared': r2_h,
                'y_pred': y_pred_h,
                # Onset = breakpoint x0
                'onset_cps': popt_h[0],
            }
        except (RuntimeError, ValueError):
            fits['hinge'] = {'aic': np.inf, 'params': {}, 'r_squared': 0,
                              'y_pred': np.full_like(cps, np.nan), 'onset_cps': np.nan}

        # --- Pick best model by AIC ---
        best_name = min(fits, key=lambda m: fits[m]['aic'])
        best = fits[best_name]

        results.append({
            'population': pop,
            'best_model': best_name,
            'onset_cps': best['onset_cps'],
            'r_squared': best['r_squared'],
            'aic_linear': fits['linear']['aic'],
            'aic_sigmoid': fits.get('sigmoid', {}).get('aic', np.inf),
            'aic_exponential': fits.get('exponential', {}).get('aic', np.inf),
            'aic_hinge': fits.get('hinge', {}).get('aic', np.inf),
            'parameters': str(best['params']),
            '_fits': fits,  # keep for plotting
            '_cps': cps,
            '_y_surv': y_surv,
        })

    result_df = pd.DataFrame(results)

    # --- Plot: data + fitted curves ---
    if output_dir and len(result_df) > 0:
        _plot_cps_fits(result_df, output_dir)

    # Drop internal columns before returning
    return result_df.drop(columns=['_fits', '_cps', '_y_surv'], errors='ignore')


def _plot_cps_fits(result_df: pd.DataFrame, output_dir: Path):
    """
    Plot per-donor survival data with best-fit curves overlaid.

    One subplot per population, showing raw data points (jittered) and
    the best-fit curve with its functional form labeled.
    """
    pop_colors = {'SST': '#d62728', 'PV': '#ff7f0e', 'PYR': '#1f77b4', 'VIP': '#2ca02c'}
    n_pops = len(result_df)
    fig, axes = plt.subplots(1, n_pops, figsize=(4 * n_pops, 4), sharey=True)
    if n_pops == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, result_df.iterrows()):
        pop = row['population']
        cps = row['_cps']
        y_surv = row['_y_surv']
        fits = row['_fits']
        best = row['best_model']
        color = pop_colors.get(pop, '#333333')

        # Scatter: raw per-donor data
        ax.scatter(cps, y_surv, alpha=0.4, s=20, color=color, zorder=2)

        # Smooth x for fitted curve
        x_fit = np.linspace(cps.min(), cps.max(), 200)

        # Plot best-fit curve
        best_fit = fits[best]
        if best == 'linear':
            p = best_fit['params']
            y_fit = p['slope'] * x_fit + p['intercept']
        elif best == 'sigmoid':
            p = best_fit['params']
            y_fit = _sigmoid(x_fit, p['L'], p['k'], p['x0'], p['b'])
        elif best == 'exponential':
            p = best_fit['params']
            y_fit = p['a'] * np.exp(p['b'] * x_fit) + p['c']
        elif best == 'hinge':
            p = best_fit['params']
            y_fit = _hinge(x_fit, p['x0'], p['slope_before'], p['slope_after'], p['y0'])
        else:
            y_fit = np.full_like(x_fit, np.nan)

        ax.plot(x_fit, y_fit, color=color, linewidth=2.5, zorder=3)

        # Mark onset CPS with a vertical line
        onset = row['onset_cps']
        if not np.isnan(onset) and cps.min() <= onset <= cps.max():
            ax.axvline(onset, color=color, linestyle=':', linewidth=1.5, alpha=0.6)
            ax.text(onset, ax.get_ylim()[0] + 0.02, f'onset\n{onset:.2f}',
                    ha='center', va='bottom', fontsize=7, color=color)

        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_title(f"{pop}: {best}\n(R²={row['r_squared']:.3f})", fontsize=10)
        ax.set_xlabel('CPS')
        if ax == axes[0]:
            ax.set_ylabel('Survival fraction')

    fig.suptitle('f(CPS) curve fits — per-population', fontsize=12, y=1.02)
    fig.tight_layout()

    out_path = Path(output_dir) / 'figures' / 'cps_curve_fits.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Figure saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5c. Direction 2 — Supertype-resolved vulnerability
# ---------------------------------------------------------------------------

def compute_supertype_survival(
    adata: anndata.AnnData,
    lib_to_donor: dict,
    donor_meta: pd.DataFrame,
    subclass_prefix: str,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """
    Compute per-supertype survival vs CPS within a single subclass.

    Instead of collapsing all Sst_* into "SST", this function keeps each
    supertype separate and computes its survival curve against CPS.  This
    reveals whether loss is uniform across subtypes or concentrated in a
    few molecularly distinct populations.

    Parameters
    ----------
    adata : AnnData from load_abundances() — 170 libraries × 109 supertypes
    lib_to_donor : dict from build_library_to_donor_map()
    donor_meta : DataFrame from load_donor_metadata()
    subclass_prefix : e.g. 'Sst' or 'Pvalb' — matches supertype names like Sst_1
    cfg : PipelineConfig

    Returns
    -------
    DataFrame with columns:
      supertype, cps_quartile, n_donors, mean_proportion, sem,
      survival_fraction, total_nuclei, nuclei_per_donor_median
    """
    # Find all supertypes matching this subclass
    # Handle "Sst Chodl" separately from "Sst" — include Chodl if prefix is "Sst"
    matching_sts = []
    for st in adata.var.index:
        parts = st.rsplit('_', 1)
        subclass = parts[0] if len(parts) == 2 else st
        if subclass == subclass_prefix:
            matching_sts.append(st)
        # Also include "Sst Chodl" supertypes when analyzing SST
        elif subclass_prefix == 'Sst' and subclass == 'Sst Chodl':
            matching_sts.append(st)

    if not matching_sts:
        return pd.DataFrame()

    # Build per-donor counts for each supertype
    counts_df = pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var.index)
    obs = adata.obs.copy()
    obs['donor_id'] = obs.index.map(lib_to_donor)
    counts_df['donor_id'] = obs['donor_id']
    counts_df = counts_df.dropna(subset=['donor_id'])

    # Sum libraries → donors
    donor_counts = counts_df.groupby('donor_id').sum()

    # Total neurons per donor (for proportion denominator)
    total_neurons = donor_counts.sum(axis=1)

    # Get CPS per donor
    cps_map = obs.groupby('donor_id')['Continuous_Pseudo-progression_Score'].first()
    donor_counts['CPS'] = cps_map
    donor_counts['total_neurons'] = total_neurons
    donor_counts = donor_counts.dropna(subset=['CPS'])

    # CPS quartile bins
    donor_counts['cps_quartile'] = pd.qcut(
        donor_counts['CPS'], q=4,
        labels=['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)'],
    )
    q1_label = 'Q1 (lowest)'

    rows = []
    for st in matching_sts:
        # Per-donor proportion of this supertype
        donor_counts[f'{st}_prop'] = donor_counts[st] / donor_counts['total_neurons']

        total_nuclei = donor_counts[st].sum()
        median_per_donor = donor_counts[st].median()

        for q, grp in donor_counts.groupby('cps_quartile', observed=True):
            prop_vals = grp[f'{st}_prop']
            baseline_props = donor_counts.loc[
                donor_counts['cps_quartile'] == q1_label, f'{st}_prop'
            ]
            baseline_mean = baseline_props.mean()

            rows.append({
                'supertype': st,
                'cps_quartile': q,
                'n_donors': len(grp),
                'mean_proportion': prop_vals.mean(),
                'sem': prop_vals.sem(),
                'survival_fraction': prop_vals.mean() / baseline_mean if baseline_mean > 0 else np.nan,
                'total_nuclei': int(total_nuclei),
                'nuclei_per_donor_median': median_per_donor,
            })

    result = pd.DataFrame(rows)
    # Flag rare supertypes (< 100 total nuclei) — too noisy for reliable curves
    result['low_count_warning'] = result['total_nuclei'] < 100
    return result


def plot_supertype_vulnerability(
    st_df: pd.DataFrame,
    subclass_name: str,
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot per-supertype survival curves within a subclass.

    Supertypes are colored by their terminal (Q4) survival: red = most lost,
    blue = most spared.  Low-count supertypes are shown with dashed lines.
    """
    supertypes = st_df['supertype'].unique()
    n_st = len(supertypes)

    # Get terminal survival for color mapping
    terminal = {}
    for st in supertypes:
        st_data = st_df[st_df['supertype'] == st]
        terminal[st] = st_data.iloc[-1]['survival_fraction']

    # Sort supertypes by terminal survival (most lost first)
    sorted_sts = sorted(terminal, key=terminal.get)

    # Color map: red (most lost) → blue (most spared)
    cmap = plt.cm.RdYlBu
    colors = {st: cmap(i / max(n_st - 1, 1)) for i, st in enumerate(sorted_sts)}

    fig, ax = plt.subplots(figsize=(10, 6))

    for st in sorted_sts:
        st_data = st_df[st_df['supertype'] == st].sort_values('cps_quartile')
        x = range(len(st_data))
        y = st_data['survival_fraction'].values
        is_rare = st_data['low_count_warning'].iloc[0]
        total = st_data['total_nuclei'].iloc[0]

        linestyle = '--' if is_rare else '-'
        linewidth = 1.0 if is_rare else 1.8
        label = f"{st} ({total:,} nuclei)" + (" ⚠" if is_rare else "")

        ax.plot(x, y, color=colors[st], linewidth=linewidth,
                linestyle=linestyle, label=label, marker='o', markersize=4)

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)'])
    ax.set_xlabel('CPS quartile')
    ax.set_ylabel('Survival fraction')
    ax.set_title(f'{subclass_name} supertype vulnerability vs CPS\n'
                 f'(dashed = <100 nuclei, treat with caution)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    if output_dir:
        out_path = Path(output_dir) / 'figures' / f'supertype_vulnerability_{subclass_name}.png'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"  Figure saved: {out_path}")

    return fig


# ---------------------------------------------------------------------------
# 5d. Direction 4 — APOE stratification (exploratory)
# ---------------------------------------------------------------------------

def compute_apoe_stratified_survival(
    donor_df: pd.DataFrame,
    populations: list,
) -> pd.DataFrame:
    """
    Compute cell-loss survival curves stratified by APOE4 carrier status.

    EXPLORATORY / likely UNDERPOWERED.  With 69 donors split by genotype
    × CPS quartile, many cells will have n < 5.  This function reports
    n explicitly so the reader can judge whether any pattern is real.

    The APOE4 status comes from Supplementary Table 1 ('APOE Genotype').
    Carriers = any genotype containing '4' (e.g. e3/e4, e4/e4).

    Parameters
    ----------
    donor_df : DataFrame from compute_proportions() with 'ADNC' and CPS columns
    populations : list of population names

    Returns
    -------
    DataFrame with survival fractions per population × APOE4 status × CPS quartile
    """
    df = donor_df.copy()

    # Get APOE4 carrier status from the scCODA obs (already joined)
    # The scCODA obs has 'APOE4_Status' but it's not in donor_df —
    # we need to detect from the APOE Genotype column if available,
    # or fall back to a direct column
    if 'APOE4_carrier' not in df.columns:
        # Try to derive from APOE Genotype in donor_meta
        # For now, we'll use a simple heuristic from the data
        # The pipeline already has CPS; we need to add APOE4 status
        warnings.warn("APOE4 carrier status not found in donor_df — "
                       "add it during aggregation")
        return pd.DataFrame()

    # CPS quartiles
    df['cps_quartile'] = pd.qcut(
        df['CPS'], q=4,
        labels=['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)'],
    )

    rows = []
    for apoe_status in ['non-carrier', 'carrier']:
        sub = df[df['APOE4_carrier'] == (apoe_status == 'carrier')]
        if len(sub) == 0:
            continue

        # Baseline = Q1 for this genotype group
        q1 = sub[sub['cps_quartile'] == 'Q1 (lowest)']

        for q, grp in sub.groupby('cps_quartile', observed=True):
            for pop in populations:
                prop_col = f'{pop}_prop'
                baseline_mean = q1[prop_col].mean() if len(q1) > 0 else np.nan

                rows.append({
                    'population': pop,
                    'APOE4_status': apoe_status,
                    'cps_quartile': q,
                    'n_donors': len(grp),
                    'mean_proportion': grp[prop_col].mean(),
                    'sem': grp[prop_col].sem(),
                    'survival_fraction': (grp[prop_col].mean() / baseline_mean
                                          if baseline_mean > 0 else np.nan),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Output — CSVs and summary
# ---------------------------------------------------------------------------

def save_survival_csv(
    survival_df: pd.DataFrame,
    severity_col: str,
    output_dir: Path,
) -> Path:
    """
    Save survival-curve data to a CSV file.

    File naming convention: cell_loss_vs_{AXIS}.csv
    Columns: population, severity_bin, n_donors, mean_proportion, sem,
             survival_fraction, closure_artifact

    Returns the path to the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f'cell_loss_vs_{severity_col}.csv'
    survival_df.to_csv(path, index=False, float_format='%.6f')
    print(f"  CSV saved: {path}")
    return path


def save_donor_composition(donor_df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Save the per-donor composition table (the tidy intermediate).

    This is the "raw" data from which survival curves are derived — useful
    for the notebook walkthrough and for any custom re-analysis.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'donor_composition.csv'
    donor_df.to_csv(path, float_format='%.6f')
    print(f"  CSV saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 7. Top-level runner — orchestrates the full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg: PipelineConfig) -> dict:
    """
    Execute the full compositional analysis pipeline.

    Steps:
      1. Download data (scCODA abundances, donor metadata, library mapping)
      2. Aggregate libraries → donors, supertypes → populations
      3. Compute proportions and survival curves for each severity axis
      4. Run the scCODA cross-check
      5. Run sanity checks
      6. Save CSVs and figures

    Parameters
    ----------
    cfg : PipelineConfig with all parameters for this run.

    Returns
    -------
    dict with keys:
      'donor_df': per-donor composition DataFrame
      'survival_curves': dict of axis → survival DataFrame
      'sanity_checks': dict of axis → sanity-check result
      'sccoda_comparison': DataFrame comparing raw vs scCODA
    """
    populations = sorted(set(cfg.cell_type_mapping.values()))
    print(f"\n{'='*60}")
    print(f"SEA-AD Compositional Analysis Pipeline")
    print(f"{'='*60}")
    print(f"Populations: {populations}")
    print(f"Severity axes: {cfg.severity_axes}")
    print(f"Denominator: {cfg.denominator}")
    print(f"Output: {cfg.output_dir}")
    print()

    # --- 1. Load data ---
    print("Step 1: Loading data from S3...")
    adata = load_abundances(cfg)
    donor_meta = load_donor_metadata(cfg)
    lib_to_donor = build_library_to_donor_map(cfg)
    print(f"  Loaded: {adata.shape[0]} libraries × {adata.shape[1]} supertypes")
    print(f"  Donor metadata: {len(donor_meta)} donors")
    print(f"  Library→donor map: {len(lib_to_donor)} entries")
    print()

    # --- 2. Aggregate ---
    print("Step 2: Aggregating to donor level...")
    donor_df = aggregate_to_donors(adata, lib_to_donor, donor_meta, cfg)
    donor_df = compute_proportions(donor_df, populations, cfg.denominator)
    print(f"  Result: {len(donor_df)} donors")
    print(f"  Populations: {', '.join(populations)}")
    for pop in populations:
        total = donor_df[pop].sum()
        print(f"    {pop}: {total:,.0f} nuclei total")
    print(f"    Total neurons: {donor_df['total_neurons'].sum():,.0f}")
    save_donor_composition(donor_df, cfg.output_dir)
    print()

    # --- 3. Survival curves ---
    print("Step 3: Computing survival curves...")
    survival_curves = {}
    sanity_checks = {}

    # Set up ordered bins for each axis
    axis_bins = {
        'ADNC': ['Not AD', 'Low', 'Intermediate', 'High'],
        'Braak': ['Braak 0', 'Braak II', 'Braak III', 'Braak IV', 'Braak V', 'Braak VI'],
        'Thal': ['Thal 0', 'Thal 1', 'Thal 2', 'Thal 3', 'Thal 4', 'Thal 5'],
        'CERAD': ['Absent', 'Sparse', 'Moderate', 'Frequent'],
    }

    fig_dir = cfg.output_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    for axis in cfg.severity_axes:
        anchor = cfg.normalization_anchor.get(axis, 'lowest')
        bins = axis_bins.get(axis)

        print(f"\n  --- {axis} (anchor = '{anchor}') ---")

        surv = compute_survival_curves(
            donor_df, populations,
            severity_col=axis,
            anchor_value=anchor,
            ordered_bins=bins,
        )
        survival_curves[axis] = surv
        save_survival_csv(surv, axis, cfg.output_dir)

        # Sanity check
        sanity = run_sanity_check(surv, axis)
        sanity_checks[axis] = sanity
        status = "PASSED" if sanity['passed'] else "FAILED"
        print(f"  Sanity check: {status}")
        print(f"    {sanity['details']}")

        # Plot
        plot_survival_curves(
            surv, axis,
            output_path=fig_dir / f'survival_vs_{axis}.png',
        )

    print()

    # --- 4. scCODA cross-check ---
    print("Step 4: scCODA cross-check (CPS axis)...")
    try:
        sccoda_df = load_sccoda_results(cfg)
        sccoda_comparison = cross_check_sccoda(sccoda_df, cfg.cell_type_mapping)
        if len(sccoda_comparison) > 0:
            print("\n  scCODA population-level summary (vs CPS):")
            for _, row in sccoda_comparison.iterrows():
                print(
                    f"    {row['population']:4s}: "
                    f"median effect = {row['median_effect']:+.3f}, "
                    f"{row['n_credible_changes']}/{row['n_supertypes']} "
                    f"supertypes credible → {row['direction_summary']}"
                )
        else:
            print("  WARNING: Could not parse scCODA results")
            sccoda_comparison = pd.DataFrame()
    except Exception as e:
        print(f"  WARNING: scCODA cross-check failed: {e}")
        sccoda_comparison = pd.DataFrame()

    print()

    # --- 5. Summary ---
    print(f"{'='*60}")
    print("Pipeline complete.")
    print(f"{'='*60}")
    all_passed = all(s['passed'] for s in sanity_checks.values())
    if all_passed:
        print("All sanity checks PASSED.")
    else:
        failed = [k for k, v in sanity_checks.items() if not v['passed']]
        print(f"SANITY CHECK FAILURES on: {failed}")
        print("Review the curves before using these results.")
    print()

    return {
        'donor_df': donor_df,
        'survival_curves': survival_curves,
        'sanity_checks': sanity_checks,
        'sccoda_comparison': sccoda_comparison,
    }
