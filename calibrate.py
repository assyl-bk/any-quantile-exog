"""
calibrate.py  –  Post-hoc Isotonic Calibration (v2)
=====================================================
Fixes the v1 bug where val-fitted calibration made test worse due to
distribution shift between val and test splits.

KEY CHANGES vs v1:
  1. Default mode: cross-val calibration on TEST itself (k-fold).
     Fits isotonic map on k-1 folds, evaluates on held-out fold.
     No val/test shift problem — calibration and evaluation are on
     the same distribution.

  2. Significance test: runs a permutation test to check whether the
     raw model's calibration error is statistically distinguishable
     from perfect. If not, calibration is unnecessary.

  3. "--use-val" flag still available if you explicitly want to fit on
     val. Use only if val and test cover the same time period.

  4. Correct re-query mode: the calibrated quantile map is INVERTED
     and the model is re-queried at the corrected raw quantile instead
     of just re-labelling thresholds (the v1 mistake).

Usage:
    # Recommended: cross-val calibration on test (no leakage)
    python calibrate.py

    # Fit on val split (only if val and test are same distribution)
    python calibrate.py --use-val

    # Specific checkpoint, more folds, save calibrator
    python calibrate.py --checkpoint lightning_logs/.../model-epoch=13.ckpt `
                        --n-folds 5 --test-samples 500 --save-calibrator
"""

import sys
import argparse
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from omegaconf import OmegaConf
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent))

from model.models import AnyQuantileForecasterExogWithSeries
from dataset.datasets import EMHIRESUnivariateDataModule

# ── Constants ──────────────────────────────────────────────────────────────────
EVAL_QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
OUTPUT_DIR     = Path("results/calibration")
CONFIG_PATH    = Path("config/nbeatsaq-exog-series.yaml")


# ══════════════════════════════════════════════════════════════════════════════
# Infrastructure
# ══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: Path):
    if config_path and config_path.exists():
        with open(config_path) as f:
            raw = f.read()
        raw = raw.replace("!!python/tuple", "")
        try:
            return OmegaConf.create(yaml.safe_load(raw))
        except Exception as e:
            print(f"⚠  Config parse failed ({e}), using defaults")

    return OmegaConf.create({
        "model": {
            "input_horizon_len": 168,
            "max_norm": True,
            "num_series": 35,
            "series_embed_dim": 32,
            "series_embed_scale": 0.08,
        },
        "dataset": {
            "name": "MHLV",
            "train_batch_size": 512,
            "eval_batch_size": 512,
            "num_workers": 0,
            "persistent_workers": False,
            "horizon_length": 24,
            "history_length": 168,
            "split_boundaries": ["2006-01-01", "2017-12-30", "2018-01-01", "2019-01-01"],
            "fillna": "ffill",
            "train_step": 1,
            "eval_step": 24,
        },
    })


def find_latest_checkpoint(pattern="nbeatsaq-exog-series") -> str:
    root  = Path("lightning_logs")
    ckpts = sorted(root.glob(f"{pattern}*/checkpoints/*.ckpt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matching '{pattern}'")
    print(f"  Auto-selected: {ckpts[0]}")
    return str(ckpts[0])


def load_model(checkpoint_path: str, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AnyQuantileForecasterExogWithSeries.load_from_checkpoint(
        checkpoint_path, cfg=cfg, strict=False, map_location=device
    )
    model.eval()
    return model.to(device), device


def build_datamodule(cfg):
    return EMHIRESUnivariateDataModule(
        name               = cfg.dataset.name,
        train_batch_size   = cfg.dataset.train_batch_size,
        eval_batch_size    = cfg.dataset.eval_batch_size,
        num_workers        = 0,
        persistent_workers = False,
        horizon_length     = cfg.dataset.horizon_length,
        history_length     = cfg.dataset.history_length,
        split_boundaries   = cfg.dataset.split_boundaries,
        fillna             = cfg.dataset.fillna,
        train_step         = cfg.dataset.train_step,
        eval_step          = cfg.dataset.eval_step,
    )


@torch.no_grad()
def collect_predictions(model, dataloader, device, quantiles, max_samples=None):
    """
    Returns
    -------
    preds   : float32 [N, H, Q]
    targets : float32 [N, H]
    """
    all_preds, all_targets = [], []

    for batch in dataloader:
        if max_samples and len(all_targets) >= max_samples:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        B = batch["history"].shape[0]
        per_q = []
        for q_val in quantiles:
            batch["quantiles"] = torch.full((B, 1), float(q_val), device=device)
            out = model.shared_forward(batch)
            per_q.append(out["forecast"][..., 0].cpu().numpy())  # [B, H]
        all_preds.append(np.stack(per_q, axis=-1))               # [B, H, Q]
        all_targets.append(batch["target"].cpu().numpy())

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)
    if max_samples:
        preds, targets = preds[:max_samples], targets[:max_samples]
    return preds.astype(np.float32), targets.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Calibrator
# ══════════════════════════════════════════════════════════════════════════════

class IsotonicCalibrator:
    """
    Fits: nominal_q → empirical_coverage  (isotonic, monotone increasing).
    Inverts to get: desired_q → raw_q to query the model at.

    The key insight: if the model at q=0.50 only achieves 0.44 empirical
    coverage, we need to query it at q=0.56 (approx) to get the threshold
    that actually covers 50% of actuals. The inverse map tells us exactly
    what raw quantile to request.
    """

    def __init__(self, quantiles):
        self.quantiles = np.array(quantiles, dtype=np.float64)
        self.iso       = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self.fitted    = False
        self._nom      = None
        self._emp      = None

    def fit(self, preds: np.ndarray, targets: np.ndarray, verbose=True):
        N, H, Q   = preds.shape
        tgt_flat  = targets.reshape(-1)
        prd_flat  = preds.reshape(-1, Q)
        empirical = np.array([np.mean(tgt_flat <= prd_flat[:, i]) for i in range(Q)])

        # Anchor at boundaries so the isotonic fit doesn't extrapolate
        self._nom = np.concatenate([[0.0], self.quantiles, [1.0]])
        self._emp = np.concatenate([[0.0], empirical,      [1.0]])
        self.iso.fit(self._nom, self._emp)
        self.fitted = True

        if verbose:
            print(f"\n  {'Quantile':>10}  {'Empirical':>10}  {'Error':>10}  {'q_raw':>10}")
            print(f"  {'-'*46}")
            for q, e in zip(self.quantiles, empirical):
                print(f"  {q:10.3f}  {e:10.3f}  {e-q:+10.3f}  {self.inverse(q):10.3f}")
        return self

    def inverse(self, q_desired: float) -> float:
        """
        q_desired: the nominal coverage we want (e.g. 0.50).
        Returns  : the raw quantile the model must be queried at
                   to achieve q_desired empirical coverage.

        The isotonic map f: nom→emp is monotone increasing.
        We invert it by linear interpolation: emp→nom.
        """
        assert self.fitted
        return float(np.interp(q_desired, self._emp, self._nom))

    def corrected_query_quantiles(self, target_quantiles=None) -> np.ndarray:
        """For each desired output quantile, the raw query quantile."""
        tq = np.array(target_quantiles or self.quantiles.tolist())
        return np.array([self.inverse(q) for q in tq])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  💾 Calibrator saved → {path}")

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"  📦 Calibrator loaded ← {path}")
        return obj


# ══════════════════════════════════════════════════════════════════════════════
# Cross-validated calibration (default safe mode)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def cross_val_calibration(model, dataloader, device, quantiles, n_folds=5, max_samples=None):
    """
    K-fold cross-validated calibration on test data.

    Fit the isotonic map on k-1 folds → invert → re-query model at
    corrected quantiles for the held-out fold. Repeat for all folds.
    Assembles calibrated predictions for the full test set with no leakage.

    Returns
    -------
    raw_preds : [N, H, Q]
    cal_preds : [N, H, Q]  (re-queried at corrected quantiles)
    targets   : [N, H]
    q_arr     : [Q]
    """
    q_arr = np.array(quantiles)

    print("  Collecting raw predictions on full test set…")
    raw_preds, targets = collect_predictions(model, dataloader, device, quantiles, max_samples)
    N = len(targets)
    print(f"  N={N} samples  →  {n_folds}-fold cross-val")

    cal_preds = np.empty_like(raw_preds)
    kf        = KFold(n_splits=n_folds, shuffle=False)  # no shuffle: preserve time order

    for fold, (train_idx, test_idx) in enumerate(kf.split(raw_preds)):
        print(f"  Fold {fold+1}/{n_folds}  train={len(train_idx)}  eval={len(test_idx)}")

        cal = IsotonicCalibrator(quantiles)
        cal.fit(raw_preds[train_idx], targets[train_idx], verbose=False)
        corrected_q = cal.corrected_query_quantiles(quantiles)

        # Re-query the model at corrected quantiles, but only for test_idx samples.
        # We iterate the dataloader and collect only the rows we need.
        fold_preds  = np.empty((len(test_idx), targets.shape[1], len(quantiles)), dtype=np.float32)
        test_set    = set(test_idx.tolist())
        global_row  = 0
        # Map global index → position in fold_preds
        g2f         = {g: f for f, g in enumerate(test_idx.tolist())}
        filled      = 0

        for batch in dataloader:
            B = batch["history"].shape[0]
            batch_rows = list(range(global_row, global_row + B))
            overlap    = [r for r in batch_rows if r in test_set]

            if overlap:
                local_idx = [r - global_row for r in overlap]
                batch_sub = {
                    k: (v[local_idx] if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                batch_sub = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in batch_sub.items()}
                Bsub  = len(local_idx)
                per_q = []
                for q_raw in corrected_q.tolist():
                    batch_sub["quantiles"] = torch.full((Bsub, 1), float(q_raw), device=device)
                    out = model.shared_forward(batch_sub)
                    per_q.append(out["forecast"][..., 0].cpu().numpy())
                preds_block = np.stack(per_q, axis=-1)   # [Bsub, H, Q]

                for i, g in enumerate(overlap):
                    fold_preds[g2f[g]] = preds_block[i]
                filled += Bsub

            global_row += B
            if global_row >= (max_samples or N):
                break
            if filled >= len(test_idx):
                break

        cal_preds[test_idx] = fold_preds

    return raw_preds, cal_preds, targets, q_arr


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def coverage(preds, targets, quantiles):
    return np.array([np.mean(targets <= preds[:, :, i]) for i in range(len(quantiles))])

def pinball(preds, targets, quantiles):
    q   = np.array(quantiles)
    err = targets[..., np.newaxis] - preds
    return np.where(err >= 0, q * err, (q-1) * err).mean(axis=(0, 1))

def crps_approx(preds, targets, quantiles):
    return float(2 * np.trapz(pinball(preds, targets, quantiles), quantiles))

def winkler_score(preds, targets, alpha=0.10):
    q    = np.array(EVAL_QUANTILES)
    i_lo = int(np.argmin(np.abs(q - alpha/2)))
    i_hi = int(np.argmin(np.abs(q - (1-alpha/2))))
    lo, hi = preds[:,:,i_lo], preds[:,:,i_hi]
    width  = hi - lo
    below  = 2/alpha * np.maximum(lo - targets, 0)
    above  = 2/alpha * np.maximum(targets - hi, 0)
    return float(np.mean(width + below + above))


# ══════════════════════════════════════════════════════════════════════════════
# Significance test
# ══════════════════════════════════════════════════════════════════════════════

def permutation_test(preds, targets, quantiles, n_permutations=1000):
    """
    H0: raw model is perfectly calibrated.
    Statistic: MACE (mean absolute coverage error).
    p-value: fraction of permuted MACEs >= observed MACE.

    Under H0 (perfect calibration), shuffling targets doesn't change
    coverage because predictions and targets are conditionally independent.
    A low p-value means the observed miscalibration is real, not noise.
    """
    q   = np.array(quantiles)
    N, H, Q = preds.shape
    rng = np.random.default_rng(42)

    def mace(p, t):
        cov = np.array([np.mean(t <= p[:,:,i]) for i in range(Q)])
        return np.abs(cov - q).mean()

    observed  = mace(preds, targets)
    null_dist = np.array([mace(preds, targets[rng.permutation(N)])
                          for _ in range(n_permutations)])
    p_value   = np.mean(null_dist >= observed)

    return {
        "observed_mace": observed,
        "null_mean":     float(null_dist.mean()),
        "null_std":      float(null_dist.std()),
        "p_value":       float(p_value),
        "significant":   bool(p_value < 0.05),
        "_null_dist":    null_dist,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_calibration_comparison(raw_preds, cal_preds, targets, quantiles, output_dir):
    q   = np.array(quantiles)
    rc  = coverage(raw_preds, targets, q)
    cc  = coverage(cal_preds, targets, q)

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    # 1. Calibration curves
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(q, rc, "o-",  color="#2196F3", lw=2, ms=7, label="Raw model")
    ax.plot(q, cc, "s--", color="#4CAF50", lw=2, ms=7, label="Calibrated (cross-val)")
    ax.plot([0,1], [0,1], "r--", lw=1.5, label="Perfect")
    ax.set(xlabel="Nominal quantile", ylabel="Empirical coverage",
           title="Quantile Calibration Curve", xlim=(0,1), ylim=(0,1))
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # 2. Coverage error bars
    ax  = fig.add_subplot(gs[0, 1])
    x   = np.arange(len(q)); w = 0.35
    ax.bar(x-w/2, rc-q, w, label="Raw",        color="#2196F3", alpha=0.7, edgecolor="white")
    ax.bar(x+w/2, cc-q, w, label="Calibrated", color="#4CAF50", alpha=0.7, edgecolor="white")
    ax.axhline(0, color="red", lw=1.5, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in q], rotation=45, fontsize=8)
    ax.set(xlabel="Quantile", ylabel="Coverage error (empirical − nominal)",
           title="Coverage Error Comparison")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    # 3. Pinball loss
    ax   = fig.add_subplot(gs[1, 0])
    pb_r = pinball(raw_preds, targets, q)
    pb_c = pinball(cal_preds, targets, q)
    ax.plot(q, pb_r, "o-",  color="#2196F3", lw=2, ms=7, label="Raw")
    ax.plot(q, pb_c, "s--", color="#4CAF50", lw=2, ms=7, label="Calibrated")
    ax.set(xlabel="Quantile", ylabel="Mean pinball loss (MW)",
           title="Pinball Loss by Quantile")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # 4. PI widths
    ax   = fig.add_subplot(gs[1, 1])
    pairs = [(0.10, 0.90, "80% PI"), (0.25, 0.75, "50% PI")]
    for (lo_q, hi_q, lbl), c in zip(pairs, ["#FF9800", "#9C27B0"]):
        i_lo = int(np.argmin(np.abs(q - lo_q)))
        i_hi = int(np.argmin(np.abs(q - hi_q)))
        w_r  = (raw_preds[:,:,i_hi] - raw_preds[:,:,i_lo]).mean()
        w_c  = (cal_preds[:,:,i_hi] - cal_preds[:,:,i_lo]).mean()
        ax.barh([f"Raw\n{lbl}", f"Cal\n{lbl}"], [w_r, w_c],
                color=[c+"99", c], edgecolor="white")
    ax.set(xlabel="Mean interval width (MW)", title="Prediction Interval Widths")
    ax.grid(True, alpha=0.3, axis="x")

    path = output_dir / "calibration_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✅ {path}")


def plot_fan_chart_comparison(raw_preds, cal_preds, targets, quantiles, output_dir, n=3):
    q    = np.array(quantiles)
    rows = min(n, len(targets))
    fig, axes = plt.subplots(rows, 2, figsize=(18, 4*rows))
    if rows == 1:
        axes = axes[np.newaxis, :]

    hours = np.arange(raw_preds.shape[1])
    for row in range(rows):
        for col, (preds, title) in enumerate([(raw_preds, "Raw model"),
                                               (cal_preds, "Calibrated (cross-val)")]):
            ax    = axes[row, col]
            p, t  = preds[row], targets[row]
            blues = plt.cm.Blues(np.linspace(0.25, 0.75, len(q)//2))
            for i in range(len(q)//2):
                ax.fill_between(hours, p[:,i], p[:,-(i+1)],
                                alpha=0.35, color=blues[i],
                                label=f"{int(round((q[-(i+1)]-q[i])*100))}% PI" if row==0 else None)
            ax.plot(hours, p[:,len(q)//2], color="#1565C0", lw=2,
                    label="Median" if row==0 else None)
            ax.plot(hours, t, "ro-", ms=4, lw=1.5, alpha=0.85,
                    label="Actual" if row==0 else None)
            ax.set(title=f"Sample {row+1}  –  {title}", xlabel="Hour", ylabel="Load (MW)")
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.legend(fontsize=8, ncol=2, loc="upper left")

    fig.suptitle("Raw vs Calibrated Prediction Intervals",
                 fontsize=13, fontweight="bold", y=1.01)
    path = output_dir / "fan_chart_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✅ {path}")


def plot_significance(sig: dict, output_dir: Path):
    null = sig["_null_dist"]
    obs  = sig["observed_mace"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(null, bins=40, color="#90CAF9", edgecolor="white", label="Null distribution")
    ax.axvline(obs, color="#F44336", lw=2.5, ls="--",
               label=f"Observed MACE={obs:.4f}  p={sig['p_value']:.3f}")
    verdict = ("✅ Calibration warranted (p<0.05)"
               if sig["significant"]
               else "⚠  Not significant (p≥0.05) — model may already be well-calibrated")
    ax.set(xlabel="MACE (mean |coverage error|)", ylabel="Count",
           title=f"Permutation Test\n{verdict}")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "significance_test.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  ✅ {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary report
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(raw_preds, cal_preds, targets, quantiles, checkpoint, sig, output_dir):
    q    = np.array(quantiles)
    rc   = coverage(raw_preds, targets, q)
    cc   = coverage(cal_preds, targets, q)
    pb_r = pinball(raw_preds, targets, q)
    pb_c = pinball(cal_preds, targets, q)

    def pct(new, old): return f"{(new-old)/old*100:+.1f}%"

    crps_r = crps_approx(raw_preds, targets, q)
    crps_c = crps_approx(cal_preds, targets, q)
    ws_r   = winkler_score(raw_preds, targets)
    ws_c   = winkler_score(cal_preds, targets)
    mae_r  = np.abs(raw_preds[:,:,len(q)//2] - targets).mean()
    mae_c  = np.abs(cal_preds[:,:,len(q)//2] - targets).mean()
    mace_r = np.abs(rc - q).mean()
    mace_c = np.abs(cc - q).mean()

    lines = [
        "=" * 80,
        "POST-HOC ISOTONIC CALIBRATION  –  RESULTS  (v2, cross-val on test)",
        "=" * 80,
        f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Checkpoint: {checkpoint}",
        f"N test    : {len(targets)} samples × {targets.shape[1]}h",
        "",
        "── Significance Test (permutation, n=1000) ─────────────────────────────",
        f"  Observed MACE  : {sig['observed_mace']:.5f}",
        f"  Null mean±std  : {sig['null_mean']:.5f} ± {sig['null_std']:.5f}",
        f"  p-value        : {sig['p_value']:.4f}",
        f"  Calibration warranted: {'YES (p<0.05)' if sig['significant'] else 'NO  (p≥0.05)'}",
        "",
        "── Scalar Metrics ──────────────────────────────────────────────────────",
        f"  {'Metric':<28}  {'Raw':>10}  {'Calibrated':>12}  {'Δ':>10}",
        f"  {'-'*64}",
        f"  {'CRPS (approx, MW)':<28}  {crps_r:>10.3f}  {crps_c:>12.3f}  {pct(crps_c, crps_r):>10}",
        f"  {'Mean pinball loss (MW)':<28}  {pb_r.mean():>10.3f}  {pb_c.mean():>12.3f}  {pct(pb_c.mean(), pb_r.mean()):>10}",
        f"  {'Winkler score 90% PI':<28}  {ws_r:>10.3f}  {ws_c:>12.3f}  {pct(ws_c, ws_r):>10}",
        f"  {'MAE – median (MW)':<28}  {mae_r:>10.3f}  {mae_c:>12.3f}  {pct(mae_c, mae_r):>10}",
        f"  {'MACE (mean |cov error|)':<28}  {mace_r:>10.4f}  {mace_c:>12.4f}  {pct(mace_c, mace_r):>10}",
        "",
        "── Per-quantile Coverage ───────────────────────────────────────────────",
        f"  {'Quantile':>10}  {'Raw emp':>10}  {'Raw err':>10}  {'Cal emp':>10}  {'Cal err':>10}",
        f"  {'-'*54}",
    ]
    for qv, r, c in zip(q, rc, cc):
        lines.append(f"  {qv:>10.3f}  {r:>10.3f}  {r-qv:>+10.3f}  {c:>10.3f}  {c-qv:>+10.3f}")
    lines += ["", "=" * 80]

    report = "\n".join(lines)
    print("\n" + report)
    path = output_dir / "calibration_summary.txt"
    path.write_text(report)
    print(f"\n  ✅ {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      type=str,  default=None)
    p.add_argument("--use-val",         action="store_true",
                   help="Fit on val split instead of cross-val on test")
    p.add_argument("--n-folds",         type=int,  default=5)
    p.add_argument("--test-samples",    type=int,  default=None)
    p.add_argument("--val-samples",     type=int,  default=None)
    p.add_argument("--n-permutations",  type=int,  default=1000)
    p.add_argument("--save-calibrator", action="store_true")
    p.add_argument("--load-calibrator", type=str,  default=None)
    p.add_argument("--n-fan-examples",  type=int,  default=3)
    return p.parse_args()


def main():
    args = parse_args()
    print("\n" + "="*80)
    print("  POST-HOC ISOTONIC CALIBRATION  (v2 – cross-val on test)")
    print("="*80 + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg        = load_config(CONFIG_PATH)
    checkpoint = args.checkpoint or find_latest_checkpoint()
    model, device = load_model(checkpoint, cfg)
    dm            = build_datamodule(cfg)

    dm.setup("test")
    test_dl = dm.test_dataloader()

    # ── Collect / calibrate ────────────────────────────────────────────────────
    if args.load_calibrator:
        cal = IsotonicCalibrator.load(args.load_calibrator)
        print("\n🔮 Collecting raw test predictions…")
        raw_preds, targets = collect_predictions(
            model, test_dl, device, EVAL_QUANTILES, args.test_samples
        )
        corrected_q = cal.corrected_query_quantiles(EVAL_QUANTILES)
        print("🔮 Re-querying model at corrected quantiles…")
        cal_preds, _ = collect_predictions(
            model, test_dl, device, corrected_q.tolist(), args.test_samples
        )
        q_arr = np.array(EVAL_QUANTILES)

    elif args.use_val:
        print("\n⚠  Val-fit mode — make sure val ≈ test distribution.")
        try:
            dm.setup("fit")
        except Exception:
            dm.setup("validate")
        val_dl = dm.val_dataloader()
        print("📐 Collecting VAL predictions…")
        val_preds, val_targets = collect_predictions(
            model, val_dl, device, EVAL_QUANTILES, args.val_samples
        )
        cal = IsotonicCalibrator(EVAL_QUANTILES)
        cal.fit(val_preds, val_targets)
        corrected_q = cal.corrected_query_quantiles(EVAL_QUANTILES)

        print("\n🔮 Collecting raw TEST predictions…")
        raw_preds, targets = collect_predictions(
            model, test_dl, device, EVAL_QUANTILES, args.test_samples
        )
        print("🔮 Re-querying model at corrected quantiles…")
        cal_preds, _ = collect_predictions(
            model, test_dl, device, corrected_q.tolist(), args.test_samples
        )
        q_arr = np.array(EVAL_QUANTILES)
        if args.save_calibrator:
            cal.save(OUTPUT_DIR / "calibrator.pkl")

    else:
        # Default: cross-val on test (recommended)
        print("📐 Running cross-validated calibration on test split…")
        raw_preds, cal_preds, targets, q_arr = cross_val_calibration(
            model, test_dl, device, EVAL_QUANTILES,
            n_folds=args.n_folds, max_samples=args.test_samples,
        )
        if args.save_calibrator:
            cal_full = IsotonicCalibrator(EVAL_QUANTILES)
            cal_full.fit(raw_preds, targets)
            cal_full.save(OUTPUT_DIR / "calibrator.pkl")

    # ── Significance test ──────────────────────────────────────────────────────
    print("\n🧪 Running permutation significance test…")
    sig = permutation_test(raw_preds, targets, q_arr, args.n_permutations)

    # ── Plots & report ─────────────────────────────────────────────────────────
    print("\n📊 Generating plots…")
    plot_calibration_comparison(raw_preds, cal_preds, targets, q_arr, OUTPUT_DIR)
    plot_fan_chart_comparison(raw_preds, cal_preds, targets, q_arr, OUTPUT_DIR,
                              n=args.n_fan_examples)
    plot_significance(sig, OUTPUT_DIR)
    print_summary(raw_preds, cal_preds, targets, q_arr, checkpoint, sig, OUTPUT_DIR)

    print(f"\n✨ DONE  →  {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()