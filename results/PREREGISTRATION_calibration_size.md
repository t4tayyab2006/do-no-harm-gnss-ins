# Pre-registration: calibration-size analysis (secondary, pre-registered)

Written **2026-09-16T08:41:10+02:00**, before `code/calibration_size.py` was run for the first time.

## Motivation

The pre-registered confirmatory test (`PREREGISTRATION_confirmatory.md`,
`results/confirmatory.json`) **rejected H2**: with 300 calibration trajectories the
certified predictive switch beat the classical filter in 75.5% of splits, not the
required 95%. A post hoc inspection attributed this to the certificate failing to
certify any threshold beyond the classical fallback in 24.5% of splits. That
explanation is a hypothesis, not a result. This analysis tests it.

## Data

No new filter runs and no retraining. The frozen per-trajectory RMSE matrix from the
confirmatory run is re-used exactly as produced (`results/confirmatory_rows.json`,
600 fresh trajectories, seeds 5000-5599, fingerprint below). Only the split sizes
change.

## Design (fixed before running)

* Calibration sizes m in {150, 300, 450}.
* Test size fixed at 150 trajectories for every m, drawn disjoint from the
  calibration set, so that test-set noise is identical across m and the only thing
  that varies is how much data the certificate gets.
* 200 random splits per m, from one permutation stream (`numpy` PCG64, seed 1);
  the same 200 permutations are used for every m.
* Unchanged from the confirmatory test: threshold grid `PS_GRID`, two-level budget
  (alpha = 20% at eps = 5%, alpha = 2% at eps = 50%), delta = 10% split by
  Bonferroni, fixed-sequence Learn-then-Test, both learned filters (learned-R and
  the KalmanNet-style filter).

## Hypotheses and pass criteria

* **H4 (primary; KalmanNet-style filter, certified predictive switch).** More
  calibration data raises the share of splits in which the certified system beats the
  classical filter: the share is non-decreasing across m = 150, 300, 450 and is
  strictly larger at m = 450 than at m = 150. **Pass** if both hold.
* **H5 (validity retained).** At every m and for both learned filters, the two-level
  budget is exceeded in at most delta = 10% of splits. **Pass** if this holds
  everywhere.

H4 is a statement about the certificate's sample efficiency. It does **not** rescue
H2: H2 was tested at m = 300 and failed, and that failure stands as reported.

Results will be reported whatever the outcome, including if H4 fails, and the
outcome will be labelled a secondary pre-registered analysis, never as a
re-test of H2.

## Fingerprints (SHA-256, at the time of writing)

| File | SHA-256 |
|---|---|
| `code/calibration_size.py` | `fc5739eed76d8b0e30f7c2a70fcede21850110382136695bf9bc083a8ff31dde` |
| `code/certify.py` | `e56246a7f61d7a05d9adf4feed041adcb74e98ec57687c6acbc87be9ecbdfd52` |
| `code/benchmark_pred.py` | `6ca0995e6b6930079d7b76ec7950bd6ab8e0497b1a5037996987e5d492df074a` |
| `code/confirmatory.py` | `41032727bc6db53550ec52546f4cead37301650cacfec5a244aa249af9756c36` |
| `results/confirmatory_rows.json` | `173e2b9c7e8fcfd79f7005146df3addd60abde3d10c62748690b2316d42f9d6a` |
