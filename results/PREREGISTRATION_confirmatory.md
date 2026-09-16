# Pre-registration: confirmatory test of the certified predictive switch

Written 2026-09-16T00:16:28+02:00, BEFORE `python confirmatory.py run` was executed. No result of this test existed when this file was written.

## Why
The certified predictive switch was introduced after the benchmark results were known (manuscript Section 3.8), so its
benchmark numbers are exploratory. This test re-evaluates it, unchanged, on data never used before.

## Design (fixed)
- Fresh pool: 600 simulated trajectories, seeds 5000-5599 (disjoint from every earlier seed set), receiver indicator d' = 3.
- Frozen components, no retraining or retuning: learned-R detector + classical fallback (d' = 3), KalmanNet-style filter
  (knet_sim.npz), tuned process noise, threshold grids (pred_switch: -inf,-8,-4,-2,-1,-0.5,0,0.5,1,2,4,8,inf; metre gate: 0..24 m, inf).
- Certificate: two-level, (alpha 0.20, eps 0.05) and (alpha 0.02, eps 0.50), delta = 0.10, fixed-sequence Learn-then-Test.
- 200 random splits, 300 calibration / 300 test, RNG seed 0.
- Comparators reported alongside: ungated learned filter, IMM heuristic (pred_switch at lambda = 0, uncalibrated),
  certified metre gate.

## Hypotheses and pass criteria (fixed)
- **H1 (validity), each learned filter:** the share of splits in which the certified predictive switch exceeds its
  two-level budget on the test half (share >5% worse > 0.20, or share >50% worse > 0.02) is <= 0.10.
  Note: this observable includes test-set sampling noise, so it is a strict check of the nominal level.
- **H2 (gain), KalmanNet-style filter:** the certified predictive switch's mean RMSE ratio to the classical fallback is < 1
  on average over splits AND in at least 95% of the splits.
- **H3 (vs pre-specified gate), KalmanNet-style filter:** its mean ratio is lower than the certified metre gate's.
- All numbers are reported whatever the outcome; a FAIL is reported as a FAIL in the manuscript.

## SHA-256 of code and frozen models at registration

| file | sha256 |
|---|---|
| `confirmatory.py` | `41032727bc6db53550ec52546f4cead37301650cacfec5a244aa249af9756c36` |
| `gated.py` | `4ee810ab9ba7f1d50cb72fdb7f7afd138cf2771ef46a23723942716ce7b7bc7e` |
| `certify.py` | `e56246a7f61d7a05d9adf4feed041adcb74e98ec57687c6acbc87be9ecbdfd52` |
| `knet.py` | `af67aa7b7636745c415ae079589541a7e50b5fc2d13b89f8c5c1cbe01fc5d6ef` |
| `benchmark.py` | `81507ca7e85474d689d1cec9d62ca87d5eb11640930ed5628512ee1eae9af68b` |
| `benchmark_pred.py` | `6ca0995e6b6930079d7b76ec7950bd6ab8e0497b1a5037996987e5d492df074a` |
| `ankf.py` | `ae05c9af10e367352d4f7a0131ad92aeebd6ea5a01df7e2aa3c06ffad2fa0c63` |
| `simulate.py` | `fc43ff64b380c018e6489710270ee4e2903d6b229264c1f57be44545fad64d0f` |
| `../results/detector_k3.0.pt` | `a1f3f6defc8cb2bcea7764fcc560d43e8ef7da9c48a45dd40e39332d302048ae` |
| `../results/knet_sim.npz` | `a0e5bdf96a13b7e422c5b423359fe079596a1c3349ce9f755e0bf296aa260af0` |
| `../results/quality_k3.0.json` | `aabecc4bdd7862bf26b44ab763b55bfae69de71cf14d4506420352360df68cf7` |
| `../results/bench_sim_setup.json` | `a677e7f8c21a767e0d559d9941af8805c6523e0d64f170b756a5710308ba291f` |
