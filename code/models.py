"""
models.py
---------
Two learned models compared against the classical EKF baseline:

1. LSTMOnly      - pure end-to-end deep-learning baseline. No physics model:
                   consumes raw (accel, gyro, gps_x, gps_y, gps_flag) at every
                   timestep and regresses absolute (x, y) position directly.
                   Represents the "deep learning only" family in the
                   literature (e.g. IONet-style approaches).

2. ANKFCorrector - the proposed hybrid: a lightweight GRU that sits on top of
                   the classical EKF and learns to predict the EKF's *residual
                   error* from causal features (time-since-GPS-fix, local IMU
                   noise level, EKF covariance trace, last GPS innovation).
                   final_estimate = ekf_estimate + learned_correction.
                   This keeps the interpretable, data-efficient physics model
                   as the backbone (per KalmanNet / AI-IMU / deep-Kalman-filter
                   design philosophy from the reviewed papers) and only asks
                   the network to learn what the fixed-covariance EKF gets
                   wrong -- primarily during GPS outages / multipath.
"""
import torch
import torch.nn as nn


class LSTMOnly(nn.Module):
    def __init__(self, in_dim=5, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out)


class ANKFCorrector(nn.Module):
    def __init__(self, in_dim=6, hidden=32):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.Tanh(), nn.Linear(16, 2))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out)
