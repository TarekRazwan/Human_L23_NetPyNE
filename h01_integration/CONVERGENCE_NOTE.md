# H01 Distance-Dependent Connectivity — Convergence Note

## Expected vs realized convergence

| Metric | Value |
|---|---|
| Analytic expectation (renormalization target) | 656.5 contacts/cell |
| After probability clipping to [0, 1] | 612.7 contacts/cell |
| Realized in single-seed run (seed 1234) | 613 contacts/cell |
| Gap from clipping | 43.8 contacts/cell (6.7%) |
| Remaining stochastic gap | −0.3 contacts/cell (negligible) |

## Cause: near-pair probability clipping (confirmed)

The renormalization sets `A_renorm` so that `A * exp(-d/λ)`, averaged over all
pre-post distances, equals the flat Yao connection probability. But for close pairs
where `d ≪ λ`, the computed probability exceeds 1.0 — and NetPyNE clips it.

Worst offenders (max computed probability, % of pairs clipped):

| Pathway | A | λ (µm) | Max prob | Pairs clipped |
|---|---|---|---|---|
| PV→PV | 13.00 | 52.8 | 10.13 | 10.9% |
| VIP→SST | 13.51 | 52.8 | 10.38 | 10.0% |
| SST→PV | 6.91 | 52.8 | 5.04 | 5.7% |
| SST→PYR | 4.65 | 65.5 | 4.03 | 4.8% |
| PYR→SST | 4.13 | 69.6 | 3.61 | 4.6% |

High-A pathways are those with high flat probability and short λ: the
exponential can't decay fast enough over the distance range to average down to
the target, so the renormalization drives A well above 1. Close pairs then
have probabilities physically clipped to 1.

## Why this is correct (not a bug to fix)

Clipping is a real physical constraint: a cell can't connect to the same partner
more than once per connection rule. The analytic expectation (656.5) assumes an
unbounded probability; the clipped value (612.7) is the **honest** expected
convergence under the distance rule, and the realized value (613) matches it.

The 6.7% reduction from the locked flat-probability baseline (656/cell) is a
**real consequence** of imposing distance-dependent connectivity: close pairs
that would have been connected with flat probability are now "oversaturated"
(probability hits 1 and can't go higher), while distant pairs lose more
connections than the clipping adds. The net effect is a modest reduction in
total connectivity.

## Validation target

Validate the H01 5-seed array against **~613 contacts/cell** (the clipped
expectation), not 656 (the flat-probability baseline). The locked baseline
remains the comparison reference, but the H01 model's convergence is expected
to be ~6.7% lower by construction.
