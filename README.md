# Do No Harm: certified deployment of learned adaptive filters for GNSS/INS

Code and results for the manuscript

> M. Tayyab, *Do No Harm: Distribution-Free Certification of Learned Adaptive Filters for
> GNSS/INS Integration via Learned-versus-Classical Solution Separation*, submitted to
> **IEEE Sensors Journal**, 2026 — under review.

**Archived release:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22861254.svg)](https://doi.org/10.5281/zenodo.22861254)

**Author:** Muhammad Tayyab — Department of Industrial and Information Engineering and
Economics (DIIIE), University of L'Aquila, Italy — muhammad.tayyab@graduate.univaq.it

## What this is

Learned components that adapt the GNSS measurement-noise covariance are often better than a
tuned classical filter *on average*, yet much worse on individual trajectories. This
repository implements a model-agnostic safety wrapper for such components:

1. the learned filter and a trusted classical adaptive filter run in lockstep;
2. at each GNSS epoch a ground-truth-free statistic compares their solutions (solution
   separation in metres, or a predictive switching statistic);
3. above a threshold the system outputs the classical solution and optionally resets the
   learned filter;
4. the threshold is calibrated by **Learn-then-Test**, which guarantees — finite-sample and
   distribution-free — that with probability at least 1 − δ the deployed system is more than
   ε worse than the classical filter on at most a fraction α of trajectories. A two-level
   variant also bounds the severity of harm.

## Main results

| Setting | Ungated worst case | Certified worst case |
|---|---|---|
| Simulation, learned detector (98% accuracy) | 10.50× | **1.69×** (4.9% better on average) |
| Real IMU, trained in Nagoya, certified in Tokyo | 1.86× | **1.38×** (full mean gain kept) |

The guarantee held in every tested setting. It was confirmed in a pre-registered test on
600 fresh trajectories (`results/PREREGISTRATION_confirmatory.md`) and in a pre-registered
calibration-size analysis (`results/PREREGISTRATION_calibration_size.md`); both files carry
SHA-256 fingerprints of the code they govern.

## Layout

```
code/
  simulate.py            trajectories, MEMS IMU and GNSS with degradations, quality indicator
  ekf.py                 classical loosely coupled EKF
  gated.py               lockstep filters, separation / predictive gates, reset   <- the method
  certify.py             Learn-then-Test certification (single and two-level)
  ankf.py                learned degradation detector, classical adaptive rules
  knet.py                KalmanNet-style learned filter (re-implementation)
  realdata.py            PPC-Dataset loader (real IMU, real trajectories)
  experiment_*.py        simulation and real-data experiments
  benchmark*.py          classical baselines, learned filters, safety mechanisms, runtime
  confirmatory.py        pre-registered confirmatory test
  calibration_size.py    pre-registered calibration-size analysis
  make_*figure*.py       figures (figstyle.py holds the shared style)
  smoke_test.py          automated checks of invariants, statistics and reported numbers
results/                 result files, trained models, figures, logs, pre-registrations
```

## Requirements

Python 3.9 with the versions used for the paper:

```bash
pip install -r requirements.txt
```

## Reproducing

All commands run from `code/`. A 12-core laptop needs about 2–3 hours for everything.

```bash
python smoke_test.py                  # fast check that the checkout is consistent

python train_eval.py
python dev_gate.py
python experiment_quality.py 0.0      # and 1.0, 2.0, 3.0
python experiment_passthrough.py
python ablation.py
python ablation_learned.py 3.0
python experiment_real.py             # needs the PPC-Dataset, see below
python benchmark.py sim-setup
python benchmark.py sim-pool 0 2
python benchmark.py sim-pool 1 2
python benchmark.py real
python benchmark.py runtime
python benchmark.py analyze
python benchmark_pred.py dev-sim
python benchmark_pred.py pool-sim 0 2
python benchmark_pred.py pool-sim 1 2
python benchmark_pred.py real
python benchmark_pred.py analyze
python confirmatory.py run
python confirmatory.py analyze
python calibration_size.py
python make_figures.py
python make_real_figures.py
python make_method_figure.py
python make_graphical_abstract.py
```

## Real data

The real-data experiments use the **PPC-Dataset** (T. Suzuki, MIT licence), which is not
redistributed here. Download it from <https://github.com/taroz/PPC-Dataset> and place it at
`data/PPC-Dataset/` with the folders `nagoya/` and `tokyo/`. Without it, the smoke test skips
the real-data loader check and everything else runs from the saved results.

## Citation

If you use this code, please cite the manuscript (see `CITATION.cff`). The citation will be
updated with the journal reference once the article is published.

## Licence

MIT — see `LICENSE`.
