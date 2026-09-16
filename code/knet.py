"""
knet.py
-------
KalmanNet-style learned filter for the benchmark: a re-implementation following
Revach et al. (IEEE TSP 2022, ref [3]), NOT the authors' code. The state model is
the same bicycle-type kinematics used by every other filter here; KalmanNet
replaces the Kalman gain by a recurrent network.

Architecture ("Architecture 1" of [3]): at every available GNSS fix the four
KalmanNet features are formed,
    F1 = z_t - z_prev                      observation difference         (2)
    F2 = z_t - H x_prior_t                 innovation                     (2)
    F3 = x_post_prev - x_post_prevprev     forward evolution difference   (6)
    F4 = x_post_prev - x_prior_prev        forward update difference      (6)
scaled, passed through FC -> GRU -> FC, and mapped to a 6x2 gain K; the update is
x_post = x_prior + K (z_t - H x_prior). Missing fixes skip the update and freeze the
GRU. Training: end-to-end back-propagation through the filter on position MSE.
One implementation choice beyond [3], to make training reliable with little data:
the gain head's bias is initialised at the median gain of the classical EKF, and its
output is scaled per state row by that EKF's typical gain magnitude.

KalmanNet provides no error covariance, so the certified gate (metre-scale
separation) needs none; the Gaussian solution-separation benchmark uses 2 P_C.
"""
import numpy as np
import torch
import torch.nn as nn

from simulate import DT

FSCALE = np.array([10, 10, 5, 5, 10, 10, 0.1, 1.0, 0.05, 0.005, 5, 5, 0.05, 0.5, 0.02, 0.002], float)
HIDDEN = 64


def _wrap_t(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def _wrap_np(a):
    return np.arctan2(np.sin(a), np.cos(a))


class KNetGain(nn.Module):
    def __init__(self, kscale, kbias, hidden=HIDDEN):
        super().__init__()
        self.fc_in = nn.Sequential(nn.Linear(16, 64), nn.ReLU())
        self.gru = nn.GRUCell(64, hidden)
        self.fc_mid = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU())
        self.fc_out = nn.Linear(64, 12)
        nn.init.normal_(self.fc_out.weight, std=1e-3)
        with torch.no_grad():
            self.fc_out.bias.copy_(torch.tensor((kbias / kscale[:, None]).reshape(-1), dtype=torch.float32))
        self.register_buffer("kscale", torch.tensor(np.repeat(kscale[:, None], 2, axis=1), dtype=torch.float32))
        self.register_buffer("fscale", torch.tensor(FSCALE, dtype=torch.float32))

    def forward(self, feat, h):
        f = torch.clamp(feat / self.fscale, -10.0, 10.0)
        h = self.gru(self.fc_in(f), h)
        K = self.fc_out(self.fc_mid(h)).view(-1, 6, 2) * self.kscale
        return K, h


def rollout(model, batch, return_track=False):
    """Differentiable batched KalmanNet filter.
    batch: dict of tensors  acc (B,n), gyr (B,n), x0 (B,6), fix (B,n) bool,
    z (B,n,2), gt (B,n,2). Returns position MSE (and optionally tracks)."""
    acc, gyr, fix, z, gt = batch["acc"], batch["gyr"], batch["fix"], batch["z"], batch["gt"]
    B, n = acc.shape
    x = batch["x0"].clone()
    h = torch.zeros(B, model.gru.hidden_size, dtype=x.dtype)
    z_prev = x[:, :2].clone(); post_prev = x.clone(); post_pp = x.clone(); prior_prev = x.clone()
    loss, cnt = 0.0, 0
    track = [] if return_track else None
    for k in range(n):
        th = _wrap_t(x[:, 2] + (gyr[:, k] - x[:, 5]) * DT)
        v = x[:, 3] + (acc[:, k] - x[:, 4]) * DT
        x = torch.stack([x[:, 0] + v * torch.cos(th) * DT, x[:, 1] + v * torch.sin(th) * DT,
                         th, v, x[:, 4], x[:, 5]], dim=1)
        m = fix[:, k]
        if bool(m.any()):
            zk = z[:, k]
            d3 = post_prev - post_pp; d3 = torch.cat([d3[:, :2], _wrap_t(d3[:, 2:3]), d3[:, 3:]], 1)
            d4 = post_prev - prior_prev; d4 = torch.cat([d4[:, :2], _wrap_t(d4[:, 2:3]), d4[:, 3:]], 1)
            feat = torch.cat([zk - z_prev, zk - x[:, :2], d3, d4], dim=1)
            K, h_new = model(feat, h)
            mf = m.to(x.dtype)[:, None]
            x_post = x + (K @ (zk - x[:, :2]).unsqueeze(-1)).squeeze(-1)
            x_post = torch.cat([x_post[:, :2], _wrap_t(x_post[:, 2:3]), x_post[:, 3:]], 1)
            prior = x
            x = mf * x_post + (1 - mf) * x
            h = mf * h_new + (1 - mf) * h
            post_pp = mf * post_prev + (1 - mf) * post_pp
            post_prev = mf * x + (1 - mf) * post_prev
            prior_prev = mf * prior + (1 - mf) * prior_prev
            z_prev = mf * zk + (1 - mf) * z_prev
        if k % 10 == 0:
            loss = loss + ((x[:, :2] - gt[:, k]) ** 2).sum(1).mean(); cnt += 1
        if return_track:
            track.append(x[:, :2].detach())
    loss = loss / cnt
    return (loss, torch.stack(track, 1)) if return_track else loss


def make_batch(pairs, dtype=torch.float32):
    """(ref, sens) pairs of equal length -> batch dict of tensors."""
    n = len(pairs[0][0]["t"])
    B = len(pairs)
    fix = np.zeros((B, n), bool); z = np.zeros((B, n, 2))
    for b, (ref, s) in enumerate(pairs):
        for j, idx in enumerate(s["gps_idx"]):
            if s["gps_available"][j]:
                fix[b, idx] = True; z[b, idx] = (s["gps_x"][j], s["gps_y"][j])
    x0 = np.stack([s["x0"] if "x0" in s else np.zeros(6) for _, s in pairs])
    T = lambda a: torch.tensor(np.asarray(a), dtype=dtype)
    return {"acc": T([s["accel_meas"] for _, s in pairs]), "gyr": T([s["gyro_meas"] for _, s in pairs]),
            "x0": T(x0), "fix": torch.tensor(fix), "z": T(z),
            "gt": T(np.stack([np.stack([r["x"], r["y"]], 1) for r, _ in pairs]))}


def gain_stats(pairs, q):
    """Median signed gain and typical magnitude per state row of the classical
    fixed-R EKF on training data (used to scale and initialise the gain head)."""
    from gated import StepEKF, H
    Ks = []
    for ref, s in pairs:
        f = StepEKF(q["q_accel"], q["q_gyro"], q_bias=q.get("q_bias", 1e-5), x0=s.get("x0"), P0=s.get("P0"))
        gps = {idx: j for j, idx in enumerate(s["gps_idx"])}
        for k in range(len(ref["t"])):
            f.predict(s["accel_meas"][k], s["gyro_meas"][k])
            if k in gps and s["gps_available"][gps[k]]:
                S = H @ f.P @ H.T + np.eye(2) * 2.5 ** 2
                Ks.append(f.P @ H.T @ np.linalg.inv(S))
                f.update(np.array([s["gps_x"][gps[k]], s["gps_y"][gps[k]]]))
    Ks = np.array(Ks)
    kbias = np.median(Ks, axis=0)
    kscale = np.maximum(np.median(np.abs(Ks), axis=(0, 2)) * 2.0, 1e-6)
    return kscale, kbias


def train_knet(train_pairs, val_pairs, q, epochs=60, lr=1e-3, patience=10, seed=0, log=print, ckpt=None):
    torch.manual_seed(seed)
    kscale, kbias = gain_stats(train_pairs, q)
    model = KNetGain(kscale, kbias)
    tr, va = make_batch(train_pairs), make_batch(val_pairs)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, stale = float("inf"), None, 0
    with torch.no_grad():
        v0 = rollout(model, va).item()
    log(f"  [KNet] init val MSE {v0:.3f} m^2 (gain head at median classical gain)")
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = rollout(model, tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            v = rollout(model, va).item()
        if v < best - 1e-6:
            best, stale = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
            if ckpt:  # survive interruption: always keep the best model on disk
                np.savez(ckpt, **{k: v.detach().cpu().numpy().astype(np.float64) for k, v in best_state.items()})
        else:
            stale += 1
        if ep % 5 == 0 or stale == 0:
            log(f"  [KNet] epoch {ep+1:3d} train {loss.item():.3f}  val {v:.3f}  best {best:.3f}")
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model


def export_params(model):
    return {k: v.detach().cpu().numpy().astype(np.float64) for k, v in model.state_dict().items()}


class StepKNet:
    """NumPy inference version of the trained KalmanNet filter, with the StepEKF
    interface (predict / update / reset_to, x, P=None) so it can sit inside the gate."""

    def __init__(self, params, x0=None):
        self.p = params
        self.x = np.zeros(6) if x0 is None else np.array(x0, float)
        self.P = None
        self.h = np.zeros(params["gru.weight_hh"].shape[1])
        self.z_prev = self.x[:2].copy(); self.post_prev = self.x.copy()
        self.post_pp = self.x.copy(); self.prior_prev = self.x.copy()

    def predict(self, accel, gyro):
        x = self.x
        th = _wrap_np(x[2] + (gyro - x[5]) * DT)
        v = x[3] + (accel - x[4]) * DT
        self.x = np.array([x[0] + v * np.cos(th) * DT, x[1] + v * np.sin(th) * DT, th, v, x[4], x[5]])

    def _gain(self, feat):
        p = self.p
        f = np.clip(feat / p["fscale"], -10.0, 10.0)
        a = np.maximum(p["fc_in.0.weight"] @ f + p["fc_in.0.bias"], 0.0)
        gi = p["gru.weight_ih"] @ a + p["gru.bias_ih"]
        gh = p["gru.weight_hh"] @ self.h + p["gru.bias_hh"]
        H_ = self.h.shape[0]
        r = 1 / (1 + np.exp(-(gi[:H_] + gh[:H_])))
        zg = 1 / (1 + np.exp(-(gi[H_:2 * H_] + gh[H_:2 * H_])))
        nn_ = np.tanh(gi[2 * H_:] + r * gh[2 * H_:])
        self.h = (1 - zg) * nn_ + zg * self.h
        m = np.maximum(p["fc_mid.0.weight"] @ self.h + p["fc_mid.0.bias"], 0.0)
        return (p["fc_out.weight"] @ m + p["fc_out.bias"]).reshape(6, 2) * p["kscale"]

    def update(self, z, quality=None, true_std=None):
        def wr(d):
            d = d.copy(); d[2] = _wrap_np(d[2]); return d
        feat = np.concatenate([z - self.z_prev, z - self.x[:2],
                               wr(self.post_prev - self.post_pp), wr(self.post_prev - self.prior_prev)])
        K = self._gain(feat)
        prior = self.x.copy()
        post = prior + K @ (z - prior[:2])
        post[2] = _wrap_np(post[2])
        self.x = post
        self.post_pp, self.post_prev, self.prior_prev, self.z_prev = self.post_prev, post.copy(), prior, z.copy()

    def reset_to(self, other):
        self.x = other.x.copy()
        self.post_prev = self.x.copy(); self.post_pp = self.x.copy(); self.prior_prev = self.x.copy()
