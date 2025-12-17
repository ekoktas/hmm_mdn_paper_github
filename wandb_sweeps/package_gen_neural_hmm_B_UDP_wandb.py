#!/usr/bin/env python
"""
package_gen_neural_hmm_B_wandb.py  –  W&B‑ready implementation of the
“Hybrid HMM + Global‑MDN” packet generator (Option B).

Key points
==========
* **No plots, no disk writes** – everything happens in‑memory.
* Hyper‑parameters for both the bootstrap **GMM‑HMM** _and_ the
  **GlobalStateMDN** are pulled from `wandb.config`, so the script can be run
  stand‑alone or as part of a W&B sweep.
* Logs the *same* realism / diversity metrics as
  `flow_package_gen_hmm_multi_HTTP_wandb_py.py`, and adds a weighted
  `metric_combined` for sweep optimisation.
* Uses the project defaults `entity="kitcel"`, `project="network_pac_gen"`.

Run locally:
    $ python package_gen_neural_hmm_B_wandb.py
Or launch a sweep with the accompanying YAML spec.
"""
import math, random, warnings, itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from hmmlearn import hmm
from scipy.stats import ks_2samp, entropy
import scipy.signal as sg
import wandb
from typing import Optional

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────────────────────────────────────────────────────────────────
#  Helpers reused from flow_package_gen_hmm_multi_HTTP_wandb_py
# ──────────────────────────────────────────────────────────────────────────────
RNG_SEED = 42
np.random.seed(RNG_SEED)
random.seed(RNG_SEED)

def seqs_by_flow(df: pd.DataFrame, col: str):
    return [g.sort_index()[col].astype(float).values
            for _, g in df.groupby("flow_id") if len(g) > 1]

def average_psd(seqs, fs: float = 1.0, nfft: int = 256):
    psds = []
    freqs = None
    for x in seqs:
        f, p = sg.welch(x, fs=fs, nperseg=min(nfft, len(x)), nfft=nfft,
                         scaling="density", detrend=False)
        psds.append(p + 1e-12)  # avoid zeros
        if freqs is None:
            freqs = f
    return freqs, np.mean(psds, axis=0)

def kl_div(p, q):
    p, q = p / p.sum(), q / q.sum()
    return 0.5 * (entropy(p, q) + entropy(q, p))

# --- average CDF per flow ----------------------------------------------------
def avg_flow_cdf(rdf, sdf, feat, grid=500):
    lo, hi = min(rdf[feat].min(), sdf[feat].min()), \
             max(rdf[feat].max(), sdf[feat].max())
    xs = np.linspace(lo, hi, grid)

    def _flow_cdfs(df):
        mats = []
        for _, g in df.groupby('flow_id'):
            v = np.sort(g[feat].values)
            mats.append(np.searchsorted(v, xs, 'right') / len(v))
        return np.stack(mats)                 # (n_flows, grid)

    return xs, _flow_cdfs(rdf).mean(0), _flow_cdfs(sdf).mean(0)

# KL between the two *mean* CDFs (convert CDF→PDF via finite diff)
def kl_avg_flow_cdf(rdf, sdf, feat, grid=500):
    _, cdf_r, cdf_s = avg_flow_cdf(rdf, sdf, feat, grid)
    pdf_r = np.diff(cdf_r, prepend=0) + 1e-12   # avoid zeros
    pdf_s = np.diff(cdf_s, prepend=0) + 1e-12
    return kl_div(pdf_r, pdf_s)

def spectral_entropy_list(seqs, nfft: int = 256):
    ent = []
    for x in seqs:
        _, Pxx = sg.welch(x, nperseg=min(nfft, len(x)), nfft=nfft,
                          scaling="density")
        Pxx /= Pxx.sum()
        ent.append(entropy(Pxx, base=2))
    return np.asarray(ent)

def spectral_entropy_coverage(real, synth):
    def _e(v):
        _, P = sg.welch(v, nperseg=min(256, len(v)), nfft=256, scaling="density")
        P /= P.sum()
        return entropy(P, base=2)
    H_r = np.array([_e(x) for x in real])
    H_s = np.array([_e(x) for x in synth])
    cov = max(0, H_s.min() - H_r.min()) + max(0, H_r.max() - H_s.max())
    return cov / (H_r.max() - H_r.min() + 1e-12)

# --- Extra realism metrics (compact copy) ------------------------------------
from statsmodels.tsa.stattools import acf
from hyppo.ksample import MMD
from hurst import compute_Hc
from scipy.stats import energy_distance, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as sk_split

def _hist_div(a, b, bins: int = 100):
    hA, edges = np.histogram(a, bins=bins, density=True)
    hB, _     = np.histogram(b, bins=edges, density=True)
    hA += 1e-12; hB += 1e-12
    hA /= hA.sum(); hB /= hB.sum()
    jsd  = 0.5 * (entropy(hA, hB, base=2) + entropy(hB, hA, base=2))
    tvd  = 0.5 * np.abs(hA - hB).sum()
    hell = np.sqrt(0.5 * ((np.sqrt(hA) - np.sqrt(hB)) ** 2).sum())
    return jsd, tvd, hell

def _hill_tail(x, k: Optional[int] = None):
    x = np.asarray(x, float)
    x = x[(x > 0) & np.isfinite(x)]
    x.sort()
    k = max(5, int(0.1 * len(x))) if k is None else k
    xk = x[-k:]
    return (k - 1) / np.sum(np.log(xk / xk[0] + 1e-12))

def _burstiness(df, win: float = 1e-2):
    times = df["time_diff"].to_numpy()
    counts, acc, t = [], 0, 0.0
    for dt in times:
        t += dt; acc += 1
        if t >= win:
            counts.append(acc); t -= win; acc = 0
    counts = np.asarray(counts, float)
    return counts.std() / counts.mean() if len(counts) > 1 else np.nan

def _flow_feats(frame: pd.DataFrame):
    out = []
    for _, g in frame.groupby("flow_id"):
        out.append([g.payload_length.mean(), g.payload_length.std(ddof=0),
                    g.time_diff.mean(),      g.time_diff.std(ddof=0)])
    return np.asarray(out)

def extra_metrics(real_df: pd.DataFrame, synth_df: pd.DataFrame):
    m = {}
    for col in ("payload_length", "time_diff"):
        m[f"{col}_energy"] = energy_distance(real_df[col], synth_df[col])
        m[f"{col}_wass1"]  = wasserstein_distance(real_df[col], synth_df[col])
        jsd, tvd, hel = _hist_div(real_df[col], synth_df[col])
        m[f"{col}_JSD"]  = jsd
        m[f"{col}_TVD"]  = tvd
        m[f"{col}_hell"] = hel

    m["tail_alpha_real"]  = _hill_tail(real_df.payload_length)
    m["tail_alpha_synth"] = _hill_tail(synth_df.payload_length)

    H_r, _, _ = compute_Hc(real_df.time_diff.values)
    H_s, _, _ = compute_Hc(synth_df.time_diff.values)
    m["hurst_real"]  = H_r
    m["hurst_synth"] = H_s

    stat_mmd, _ = MMD(compute_kernel="gaussian").test(
        real_df[["payload_length", "time_diff"]].to_numpy(),
        synth_df[["payload_length", "time_diff"]].to_numpy())
    m["MMD_payload_time"] = stat_mmd

    corr_r = real_df[["payload_length", "time_diff"]].corr().to_numpy()
    corr_s = synth_df[["payload_length", "time_diff"]].corr().to_numpy()
    m["corr_fro"] = np.linalg.norm(corr_r - corr_s)

    acf_r = acf(real_df.payload_length, nlags=20, fft=True)
    acf_s = acf(synth_df.payload_length, nlags=20, fft=True)
    m["acf_rmse"] = float(np.sqrt(((acf_r - acf_s) ** 2).mean()))

    m["burst_idx_real"]  = _burstiness(real_df)
    m["burst_idx_synth"] = _burstiness(synth_df)

    X_r, X_s = _flow_feats(real_df), _flow_feats(synth_df)
    X = np.vstack([X_r, X_s])
    y = np.r_[np.zeros(len(X_r)), np.ones(len(X_s))]
    Xtr, Xte, ytr, yte = sk_split(X, y, test_size=0.30, random_state=42)
    acc = LogisticRegression(max_iter=1000).fit(Xtr, ytr).score(Xte, yte)
    m["disc_acc"] = acc
    return m

# ──────────────────────────────────────────────────────────────────────────────
#  Model components – GlobalStateMDN
# ──────────────────────────────────────────────────────────────────────────────
class GlobalStateMDN(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 64, n_mix: int = 8, out_dim: int = 2):
        super().__init__()
        self.n_mix   = n_mix
        self.out_dim = out_dim
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),   nn.Tanh())
        self.fc_pi = nn.Linear(hidden, n_mix)
        self.fc_mu = nn.Linear(hidden, n_mix * out_dim)
        self.fc_ls = nn.Linear(hidden, n_mix * out_dim)

    def forward(self, s_onehot):       # (B, K)
        h  = self.backbone(s_onehot)
        pi = torch.softmax(self.fc_pi(h), dim=-1)             # (B, M)
        mu = self.fc_mu(h).view(-1, self.n_mix, self.out_dim) # (B, M, F)
        sg = F.softplus(self.fc_ls(h)).view_as(mu) + 1e-3
        return pi, mu, sg

def mdn_nll(y, pi, mu, sg):           # y: (B, F)
    y = y.unsqueeze(1)                 # (B, 1, F)
    comp = -0.5 * ((y - mu) / sg) ** 2 - torch.log(sg) - 0.5 * math.log(2 * math.pi)
    log_prob = torch.logsumexp(torch.log(pi) + comp.sum(-1), dim=-1)
    return -log_prob                  # (B,)

def sample_mixture(pi, mu, sg, temperature: float = 1.0):
    # Simple temperature scaling on mixture weights
    pi = (pi / pi.sum()).pow(1 / temperature)
    pi = pi / pi.sum()
    k  = torch.multinomial(pi, 1).item()
    return torch.normal(mu[k], sg[k] * temperature)

# ──────────────────────────────────────────────────────────────────────────────
#  Dataset for MDN (γ‑weighted samples)
# ──────────────────────────────────────────────────────────────────────────────
class GlobalMDNDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, state_dim: int, thresh: float = 1e-4):
        xs, ys, ws = [], [], []
        eye = np.eye(state_dim, dtype=np.float32)
        for k in range(state_dim):
            gcol = f"state_post_{k}"
            mask = frame[gcol] > thresh
            if not mask.any():
                continue
            w = frame.loc[mask, gcol].values.astype(np.float32)
            y = frame.loc[mask, ["plog_z", "dt_z"]].values.astype(np.float32)
            x = np.repeat(eye[[k]], repeats=len(y), axis=0)
            xs.append(x); ys.append(y); ws.append(w)
        self.X = torch.tensor(np.vstack(xs))
        self.Y = torch.tensor(np.vstack(ys))
        self.W = torch.tensor(np.concatenate(ws))
        self.W = self.W * (len(self.W) / self.W.sum())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.W[idx]

# ──────────────────────────────────────────────────────────────────────────────
#  Main training / evaluation pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    run = wandb.init(
        entity="kitcel",
        project="network_pac_gen",
        job_type="neural_hmm_B",
        config={
            # HMM hyper‑params
            "n_states"   : 3,
            "n_mix_hmm"  : 1,
            "min_covar" : 1e-3,
            "max_iter"  : 200,
            # MDN hyper‑params
            "n_mix_mdn" : 8,
            "mdn_hidden": 64,
            "mdn_epochs": 60,
            # misc
            "mdn_lr"    : 1e-3,
        }
    )
    cfg = run.config

    # Dataset paths -----------------------------------------------------------
    RAW      = Path("real_data") / "df_raw_UDP_GOOGLE_HOME.csv"

    # 1) Load & basic cleaning -------------------------------------------------
    df = pd.read_csv(RAW, usecols=["flow_id", "payload_length", "time_diff"])
    df["payload_length"] = df["payload_length"].clip(lower=0).fillna(0)
    df["time_diff"]      = df["time_diff"].clip(lower=0).fillna(0)
    df = df[df["time_diff"] < 25e-3]              # discard huge gaps (>25 ms)
    df["payload_log"] = np.log1p(df["payload_length"])

    # 2) Train/test split ------------------------------------------------------
    flows = df["flow_id"].unique()
    tr_flows, te_flows = train_test_split(flows, test_size=0.10, random_state=RNG_SEED)
    tr_df = df[df["flow_id"].isin(tr_flows)].copy()
    te_df = df[df["flow_id"].isin(te_flows)].copy()

    # 3) Z‑score scaling -------------------------------------------------------
    scaler_pkt = StandardScaler().fit(df[["payload_log", "time_diff"]])
    for frame in (tr_df, te_df):
        frame[["plog_z", "dt_z"]] = scaler_pkt.transform(frame[["payload_log", "time_diff"]])

    # 4) Bootstrap GMM‑HMM -----------------------------------------------------
    def build_matrix(frame: pd.DataFrame):
        mats, lens = [], []
        for _, g in frame.groupby("flow_id"):
            a = g[["plog_z", "dt_z"]].values.astype(np.float64)
            mats.append(a); lens.append(len(a))
        return np.vstack(mats), lens

    X_train, len_train = build_matrix(tr_df)
    hmm_boot = hmm.GMMHMM(
        n_components = cfg.n_states,
        n_mix        = cfg.n_mix_hmm,
        covariance_type = "diag",
        n_iter       = cfg.max_iter,
        random_state = RNG_SEED,
        min_covar    = cfg.min_covar,
    ).fit(X_train, len_train)
    # robust floor
    hmm_boot.covars_=np.where(np.isfinite(hmm_boot.covars_),
                               np.maximum(hmm_boot.covars_,cfg.min_covar),
                               cfg.min_covar)

    # 5) Packet‑wise posteriors γ ---------------------------------------------
    gamma_all = []
    for _, g in tr_df.groupby("flow_id"):
        gamma_all.append(hmm_boot.predict_proba(g[["plog_z", "dt_z"]].values))
    gamma_all = np.vstack(gamma_all)
    tr_df[[f"state_post_{k}" for k in range(cfg.n_states)]] = gamma_all

    # 6) MDN dataset & model ---------------------------------------------------
    mdn_ds = GlobalMDNDataset(tr_df.reset_index(drop=True), cfg.n_states)
    mdn_dl = DataLoader(mdn_ds, batch_size=4096, shuffle=True, num_workers=0)

    mdn = GlobalStateMDN(cfg.n_states, hidden=cfg.mdn_hidden, n_mix=cfg.n_mix_mdn).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    opt = torch.optim.Adam(mdn.parameters(), lr=cfg.mdn_lr)

    # 7) Train MDN -------------------------------------------------------------
    mdn.train()
    for epoch in range(1, cfg.mdn_epochs + 1):
        tot_loss = 0.0
        for s_one, y_tgt, w in mdn_dl:
            s_one, y_tgt, w = s_one.to(mdn.fc_pi.weight.device), y_tgt.to(mdn.fc_pi.weight.device), w.to(mdn.fc_pi.weight.device)
            pi, mu, sg = mdn(s_one)
            nll = mdn_nll(y_tgt, pi, mu, sg)          # (B,)
            loss = (w * nll).sum() / w.sum()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(mdn.parameters(), 5.0); opt.step()
            tot_loss += loss.item()
        # log but don’t print every epoch
        run.log({"epoch": epoch, "mdn_loss": tot_loss / len(mdn_dl)})

    # 8) Generate synthetic flows ---------------------------------------------
    rng = np.random.default_rng(RNG_SEED)
    onehot = np.eye(cfg.n_states, dtype=np.float32)
    records = []
    new_fid = 1
    counts = df.groupby("flow_id").size()
    mdn.eval()
    for fid in te_flows:
        L = int(counts.loc[fid])
        if L < 1:
            continue
        states = np.empty(L, dtype=int)
        states[0] = rng.choice(cfg.n_states, p=hmm_boot.startprob_)
        for t in range(1, L):
            states[t] = rng.choice(cfg.n_states, p=hmm_boot.transmat_[states[t-1]])
        for k in states:
            with torch.no_grad():
                pi, mu, sg = mdn(torch.tensor(onehot[[k]], device=mdn.fc_pi.weight.device).float())
                pi, mu, sg = pi.squeeze(0), mu.squeeze(0), sg.squeeze(0)
                sample = sample_mixture(pi, mu, sg, temperature=1.1).cpu().numpy()
            plog, dt = scaler_pkt.inverse_transform(sample.reshape(1, -1)).ravel()
            payload  = np.expm1(plog)
            payload  = np.clip(payload, 0.0, 1500.0)
            dt       = max(dt, 1e-6)
            records.append([new_fid, round(payload), float(dt)])
        new_fid += 1

    synth_df = pd.DataFrame(records, columns=["flow_id", "payload_length", "time_diff"])

    # 9) Metrics – same set as baseline script --------------------------------
    metrics = {}
    metrics["ks_payload"] = ks_2samp(te_df.payload_length, synth_df.payload_length).statistic
    metrics["ks_time"]    = ks_2samp(te_df.time_diff,      synth_df.time_diff).statistic

    rp, psd_r = average_psd(seqs_by_flow(te_df, "payload_length"))
    _,  psd_s = average_psd(seqs_by_flow(synth_df, "payload_length"))
    metrics["kl_psd_payload"] = kl_div(psd_r, psd_s)

    rt, tpd_r = average_psd(seqs_by_flow(te_df, "time_diff"))
    _,  tpd_s = average_psd(seqs_by_flow(synth_df, "time_diff"))
    metrics["kl_psd_time"] = kl_div(tpd_r, tpd_s)

    metrics['kl_cdf_payload'] = kl_avg_flow_cdf(te_df,   synth_df, 'payload_length')
    metrics['kl_cdf_time']    = kl_avg_flow_cdf(te_df,   synth_df, 'time_diff')

    metrics["diversity_payload"] = spectral_entropy_coverage(
        seqs_by_flow(te_df, "payload_length"), seqs_by_flow(synth_df, "payload_length"))
    metrics["diversity_time"] = spectral_entropy_coverage(
        seqs_by_flow(te_df, "time_diff"), seqs_by_flow(synth_df, "time_diff"))

    ent_real_pl = spectral_entropy_list(seqs_by_flow(te_df, "payload_length"))
    ent_synt_pl = spectral_entropy_list(seqs_by_flow(synth_df, "payload_length"))
    metrics["delta_mean_payload"] = float(abs(ent_synt_pl.mean() - ent_real_pl.mean()))

    ent_real_td = spectral_entropy_list(seqs_by_flow(te_df, "time_diff"))
    ent_synt_td = spectral_entropy_list(seqs_by_flow(synth_df, "time_diff"))
    metrics["delta_mean_time"] = float(abs(ent_synt_td.mean() - ent_real_td.mean()))

    metrics.update(extra_metrics(te_df, synth_df))

    # combined objective (equal weights as baseline example) ------------------
    w = 1.0
    metrics["metric_combined"] = (
        w * metrics["ks_payload"]          +
        w * metrics["ks_time"]             +
        w*metrics['kl_cdf_payload']        +
        w*metrics['kl_cdf_time']           +
        w * metrics["kl_psd_payload"]      +
        w * metrics["kl_psd_time"]         +
        w * metrics["delta_mean_payload"]  +
        w * metrics["delta_mean_time"]     +
        w * metrics["diversity_payload"]   +
        w * metrics["diversity_time"]
    )

    # log & finish ------------------------------------------------------------
    wandb.log(metrics)
    run.finish()

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
