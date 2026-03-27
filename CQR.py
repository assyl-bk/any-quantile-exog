"""
Conformalized Quantile Regression (CQR) — V2 Sequential Curriculum
====================================================================
Post-hoc calibration for guaranteed coverage without retraining.
Romano, Patterson, and Candès — NeurIPS 2019.

Algorithm:
  1. Run the model on the calibration set (first half of 2018)
  2. For each quantile level q, compute the signed residuals:
       r_i = y_i - q_hat_q(x_i)
  3. Take the q*(1+1/n)-th empirical quantile of residuals → offset_q
  4. At test time: shift each quantile prediction by its offset
     → guaranteed coverage at each nominal level

"""

import sys
import pickle
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from model.models import AQNBEATSPlusPlus
from utils.model_factory import instantiate

# ── Config ────────────────────────────────────────────────────────────────────
EVAL_QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
OUTPUT_DIR     = "results/CQR"

MODELS = {
    
    "stage2_v2": {
        "label":      "V2 Stage 2 v2 — Exog+Adaptive",
        "checkpoint": "lightning_logs/nbeatsaq-stage2-seed0/checkpoints/model-epoch=4.ckpt",
        "config":     "config/stage2_v2.yaml",
        "color":      "#1976D2",
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_cfg(config_path: Path) -> OmegaConf:
    if config_path.exists():
        with open(config_path) as f:
            raw = f.read().replace("!!python/tuple", "")
        cfg = OmegaConf.create(yaml.safe_load(raw))
        print(f"  ✅ Config: {config_path}")
        return cfg
    raise FileNotFoundError(f"Config not found: {config_path}")


def load_model(ckpt_path: Path, cfg: OmegaConf, device: torch.device):
    model = AQNBEATSPlusPlus.load_from_checkpoint(
        str(ckpt_path), cfg=cfg, strict=False, map_location=device
    )
    model.eval().to(device)
    print(f"  ✅ Model loaded on {device}")
    return model


def get_loader(cfg: OmegaConf, stage: str):
    """
    stage: 'cal' or 'test'

    The original split boundaries give only 2 days for val (2017-12-30 to 2018-01-01)
    which is far too small for conformal calibration.

    Fix: split the test year (2018) in half:
      - cal  = 2018-01-01 to 2018-07-01  (first 6 months, ~6,300 samples)
      - test = 2018-07-01 to 2019-01-01  (second 6 months, ~6,300 samples)
    This gives a proper calibration set while keeping a clean held-out test set.
    """
    from omegaconf import OmegaConf

    cal_boundaries  = ["2006-01-01", "2017-12-30", "2018-01-01", "2018-07-01"]
    test_boundaries = ["2006-01-01", "2018-06-30", "2018-07-01", "2019-01-01"]

    boundaries = cal_boundaries if stage == "cal" else test_boundaries

    # Build a modified cfg with new split boundaries
    cfg_mod = OmegaConf.to_container(cfg, resolve=True)
    cfg_mod["dataset"]["split_boundaries"] = boundaries
    cfg_mod = OmegaConf.create(cfg_mod)

    dm = instantiate(cfg_mod.dataset)
    dm.setup(stage="test")
    loader = dm.test_dataloader()
    n = len(dm.test_dataset)
    print(f"  ✅ {stage} set ({boundaries[2]} : {n:,} samples")
    return loader


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_all(model, loader, device, quantiles=EVAL_QUANTILES):
    """
    Returns:
        preds   : np.ndarray [N, H, Q]
        targets : np.ndarray [N, H]
    """
    q_tensor = torch.tensor(quantiles, dtype=torch.float32, device=device)
    all_preds, all_targets = [], []

    for batch in tqdm(loader, desc="  inference", leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        B = batch["history"].shape[0]
        per_q = []
        for q_val in q_tensor:
            batch["quantiles"] = q_val.view(1, 1).expand(B, 1)
            pred = model(batch).squeeze(-1)   # [B, H]
            per_q.append(pred)
        all_preds.append(torch.stack(per_q, dim=-1).cpu())   # [B, H, Q]
        all_targets.append(batch["target"].cpu())

    preds   = torch.cat(all_preds,   dim=0).numpy()   # [N, H, Q]
    targets = torch.cat(all_targets, dim=0).numpy()   # [N, H]
    print(f"  ✅ Predicted: {preds.shape[0]:,} samples")
    return preds, targets


# ── Conformal Calibration ─────────────────────────────────────────────────────

class ConformalCalibrator:
    """
    Split Conformal Prediction calibrator.

    Stores per-quantile correction offsets fitted on the calibration
    (validation) set. At test time, call .correct(preds) to get
    coverage-guaranteed prediction intervals.
    """

    def __init__(self, quantiles=EVAL_QUANTILES, alpha=0.05):
        self.quantiles   = quantiles
        self.alpha       = alpha
        self.q_idx_lo    = None
        self.q_idx_hi    = None
        self.q_per_level = None   # dict: per-quantile CQR offsets
        self.n_cal       = None

    def fit(self, preds_cal: np.ndarray, targets_cal: np.ndarray):
        """
        Fit per-quantile CQR offsets on calibration predictions.

        preds_cal  : [N, H, Q]
        targets_cal: [N, H]
        """
        N, H, Q = preds_cal.shape
        self.n_cal = N
        valid      = np.isfinite(targets_cal)

        # Store lo/hi indices for interval width tracking
        q_arr          = np.array(self.quantiles)
        self.q_idx_lo  = int(np.argmin(np.abs(q_arr - self.alpha / 2)))
        self.q_idx_hi  = int(np.argmin(np.abs(q_arr - (1 - self.alpha / 2))))

        print(f"\n  Fitting CQR on {N:,} calibration samples...")
        print(f"  {'Quantile':>10}  {'Offset (MW)':>12}  {'Pre-cal cov':>14}  {'Post-cal cov':>14}")
        print(f"  {'─'*56}")

        self.q_per_level = {}
        for i, q in enumerate(self.quantiles):
            pred_q      = preds_cal[:, :, i]
            resid       = targets_cal - pred_q          # signed residual [N, H]
            resid_valid = resid[valid]
            cal_level   = min(q * (1 + 1 / N), 1.0)
            offset      = float(np.quantile(resid_valid, cal_level))
            self.q_per_level[q] = offset

            pre_cov  = float(np.mean(targets_cal[valid] <= pred_q[valid]))
            post_cov = float(np.mean(targets_cal[valid] <= (pred_q + offset)[valid]))
            print(f"  {q:>10.3f}  {offset:>+12.3f}  {pre_cov:>14.4f}  {post_cov:>14.4f}")

        print(f"\n  ✅ CQR calibrator fitted on {N:,} samples")

    def correct_cqr(self, preds: np.ndarray) -> np.ndarray:
        """
        Apply per-quantile (CQR) correction.
        Each quantile level gets its own offset.
        Returns corrected preds [N, H, Q] — sorted.
        """
        corrected = preds.copy()
        for i, q in enumerate(self.quantiles):
            corrected[:, :, i] += self.q_per_level[q]
        # Sort to maintain monotonicity after independent shifts
        corrected = np.sort(corrected, axis=-1)
        return corrected

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"   Calibrator saved: {path}")

    @classmethod
    def load(cls, path: str) -> "ConformalCalibrator":
        with open(path, "rb") as f:
            cal = pickle.load(f)
        print(f"   Calibrator loaded: {path}")
        return cal


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_coverage_and_crps(preds, targets, quantiles=EVAL_QUANTILES):
    q_arr      = np.array(quantiles)
    valid      = np.isfinite(targets)

    # CRPS
    err    = targets[..., None] - preds
    pb     = np.where(err >= 0, q_arr * err, (q_arr - 1) * err)
    crps   = float(np.nanmean(np.where(valid[..., None], pb, np.nan)) * 2)

    # Per-quantile coverage
    coverage = {}
    for i, q in enumerate(quantiles):
        v_tgt  = targets[valid]
        v_pred = preds[:, :, i][valid]
        coverage[q] = float(np.mean(v_tgt <= v_pred))

    mace = float(np.mean([abs(coverage[q] - q) for q in quantiles]))
    return {"crps": crps, "coverage": coverage, "mace": mace}


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_cqr_results(m_raw: dict, m_cqr: dict, save_dir: str, model_label: str):
    q_arr = np.array(EVAL_QUANTILES)
    x     = np.arange(len(EVAL_QUANTILES))

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 2, hspace=0.5, wspace=0.35)

    # ── Calibration curve ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect', zorder=0)
    ax.plot(q_arr, [m_raw["coverage"][q] for q in EVAL_QUANTILES],
            'o-', lw=2, markersize=7, color='#e74c3c', label='Raw')
    ax.plot(q_arr, [m_cqr["coverage"][q] for q in EVAL_QUANTILES],
            '^-', lw=2, markersize=7, color='#2ecc71', label='CQR')
    ax.set_xlabel("Nominal quantile", fontsize=11)
    ax.set_ylabel("Empirical coverage", fontsize=11)
    ax.set_title("Calibration Curve — Raw vs CQR", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # ── Coverage error bars ──────────────────────────────────────────────────
    ax  = fig.add_subplot(gs[0, 1])
    w   = 0.35
    raw_errs = [m_raw["coverage"][q] - q for q in EVAL_QUANTILES]
    cqr_errs = [m_cqr["coverage"][q] - q for q in EVAL_QUANTILES]
    ax.bar(x - w/2, raw_errs, width=w * 0.9,
           color='#e74c3c', alpha=0.8, edgecolor='black', lw=0.4, label='Raw')
    ax.bar(x + w/2, cqr_errs, width=w * 0.9,
           color='#2ecc71', alpha=0.8, edgecolor='black', lw=0.4, label='CQR')
    ax.axhline(0, color='black', lw=1.5, ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{q:.2f}' for q in EVAL_QUANTILES], rotation=45, fontsize=9)
    ax.set_xlabel("Quantile", fontsize=11)
    ax.set_ylabel("Coverage error (empirical − nominal)", fontsize=11)
    ax.set_title("Coverage Error — Raw vs CQR\n(green=over-cover, red=under-cover)",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Empirical vs nominal bar chart ───────────────────────────────────────
    ax = fig.add_subplot(gs[1, :])
    ax.bar(x - w/2, [m_raw["coverage"][q] for q in EVAL_QUANTILES],
           width=w * 0.9, color='#e74c3c', alpha=0.8,
           edgecolor='black', lw=0.4, label='Raw')
    ax.bar(x + w/2, [m_cqr["coverage"][q] for q in EVAL_QUANTILES],
           width=w * 0.9, color='#2ecc71', alpha=0.8,
           edgecolor='black', lw=0.4, label='CQR')
    ax.step(np.arange(-0.5, len(EVAL_QUANTILES)),
            np.append(q_arr, q_arr[-1]),
            where='post', color='black', lw=2, ls='--', label='Nominal (ideal)')
    ax.axhline(0.95, color='purple', lw=1, ls=':', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Q{q:.2f}' for q in EVAL_QUANTILES], rotation=45, fontsize=9)
    ax.set_ylabel("Empirical coverage", fontsize=11)
    ax.set_title("Empirical vs Nominal Coverage — Full Evaluation Set",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    # ── Summary table ────────────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.axis('off')
    col_labels = ["Method", "CRPS ↓", "Coverage-0.05 ↑", "Coverage-0.50",
                  "Coverage-0.95 ↑", "MACE ↓", "Interval width change"]
    rows_data = [
        ["Raw",
         f"{m_raw['crps']:.3f}",
         f"{m_raw['coverage'].get(0.05, float('nan')):.4f}",
         f"{m_raw['coverage'].get(0.50, float('nan')):.4f}",
         f"{m_raw['coverage'].get(0.95, float('nan')):.4f}",
         f"{m_raw['mace']:.4f}",
         "—"],
        ["CQR",
         f"{m_cqr['crps']:.3f}",
         f"{m_cqr['coverage'].get(0.05, float('nan')):.4f}",
         f"{m_cqr['coverage'].get(0.50, float('nan')):.4f}",
         f"{m_cqr['coverage'].get(0.95, float('nan')):.4f}",
         f"{m_cqr['mace']:.4f}",
         f"+{m_cqr.get('mean_width_change', 0):.2f} MW"],
        ["AQ-NBEATS (paper)", "211.22", "—", "—", "—", "—", "—"],
    ]

    tbl = ax_tbl.table(cellText=rows_data, colLabels=col_labels,
                       cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.6)
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#2E75B6")
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')
    for j in range(len(col_labels)):   # highlight CQR row
        tbl[(2, j)].set_facecolor("#E2EFDA")
    for j in range(len(col_labels)):   # paper baseline
        tbl[(3, j)].set_facecolor("#FFF2CC")

    ax_tbl.set_title("CQR Calibration — Metrics Summary", fontsize=12,
                     fontweight='bold', pad=12)

    plt.suptitle(f"Conformalized Quantile Regression — {model_label}\n"
                 "Red = Raw model   Green = CQR (per-quantile calibrated)",
                 fontsize=13, fontweight='bold', y=1.01)

    out = f"{save_dir}/cqr_results.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  ✅ Saved: {out}")
    plt.close()


def save_cqr_report(m_raw: dict, m_cqr: dict, calibrator: ConformalCalibrator,
                    save_dir: str, model_label: str):
    lines = [
        "=" * 80,
        f"CQR CALIBRATION REPORT — {model_label}",
        "=" * 80,
        f"\nCalibration set : 2018-01-01 → 2018-07-01  ({calibrator.n_cal:,} samples)",
        f"Evaluation set  : 2018-07-01 → 2019-01-01",
        f"\nPer-quantile CQR offsets:",
        f"  {'Quantile':>10}  {'Offset (MW)':>14}",
        f"  {'─'*28}",
    ]
    for q, offset in calibrator.q_per_level.items():
        lines.append(f"  {q:>10.3f}  {offset:>+14.4f}")

    for label, m in [("RAW", m_raw), ("CQR", m_cqr)]:
        lines += [
            f"\n{'─'*80}", f"  {label}", f"{'─'*80}",
            f"  CRPS : {m['crps']:.4f} MW",
            f"  MACE : {m['mace']:.6f}",
            f"\n  {'Quantile':>10}  {'Nominal':>10}  {'Empirical':>12}  {'Error':>10}",
            f"  {'─'*48}",
        ]
        for q in EVAL_QUANTILES:
            emp = m["coverage"][q]
            lines.append(f"  {q:>10.3f}  {q:>10.3f}  {emp:>12.4f}  {emp-q:>+10.4f}")

    lines += [
        "\n" + "=" * 80,
        "  PAPER BASELINES",
        "=" * 80,
        "  AQ-NBEATS  CRPS=211.22  MAPE=2.47%",
        "  AQ-ESRNN   CRPS=195.94  MAPE=2.32%",
        "=" * 80,
    ]

    out = f"{save_dir}/cqr_report.txt"
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✅ Report: {out}")
    print("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), default="stage2")
    parser.add_argument("--mode",  choices=["fit", "eval", "both"],  default="both")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Miscoverage level (default 0.05 → 95% coverage)")
    args = parser.parse_args()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info       = MODELS[args.model]
    cal_path   = f"{OUTPUT_DIR}/calibrator_{args.model}.pkl"

    print(f"\n{'='*70}")
    print(f"  Conformal Prediction — {info['label']}")
    print(f"  alpha = {args.alpha}  (target coverage = {1-args.alpha:.0%})")
    print(f"{'='*70}\n")

    cfg   = load_cfg(Path(info["config"]))
    model = load_model(Path(info["checkpoint"]), cfg, device)

    # ── FIT ──────────────────────────────────────────────────────────────────
    if args.mode in ("fit", "both"):
        print("\n📐 STEP 1: Fitting calibrator on first half of test year (2018-01 to 2018-07)...")
        cal_loader          = get_loader(cfg, "cal")
        preds_cal, tgts_cal = predict_all(model, cal_loader, device)

        calibrator = ConformalCalibrator(quantiles=EVAL_QUANTILES, alpha=args.alpha)
        calibrator.fit(preds_cal, tgts_cal)
        calibrator.save(cal_path)

    # ── EVAL ─────────────────────────────────────────────────────────────────
    if args.mode in ("eval", "both"):
        print("\n📊 STEP 2: Evaluating on second half of test year (2018-07 to 2019-01)...")
        calibrator = ConformalCalibrator.load(cal_path)

        test_loader           = get_loader(cfg, "test")
        preds_test, tgts_test = predict_all(model, test_loader, device)

        # Raw (uncorrected)
        m_raw = compute_coverage_and_crps(preds_test, tgts_test)

        # CQR correction
        preds_cqr = calibrator.correct_cqr(preds_test)
        m_cqr     = compute_coverage_and_crps(preds_cqr, tgts_test)

        # Interval width change (Q0.05 → Q0.95)
        lo, hi = calibrator.q_idx_lo, calibrator.q_idx_hi
        raw_width = np.nanmean(preds_test[:, :, hi] - preds_test[:, :, lo])
        cqr_width = np.nanmean(preds_cqr[:, :, hi]  - preds_cqr[:, :, lo])
        m_cqr["mean_width_change"] = cqr_width - raw_width

        print(f"\n  Coverage-0.95 :  Raw={m_raw['coverage'][0.95]:.4f}  →  CQR={m_cqr['coverage'][0.95]:.4f}  (target {1-args.alpha:.2f})")
        print(f"  CRPS          :  Raw={m_raw['crps']:.4f}  →  CQR={m_cqr['crps']:.4f}")
        print(f"  MACE          :  Raw={m_raw['mace']:.4f}  →  CQR={m_cqr['mace']:.4f}")

        plot_cqr_results(m_raw, m_cqr, OUTPUT_DIR, info["label"])
        save_cqr_report(m_raw, m_cqr, calibrator, OUTPUT_DIR, info["label"])

    print(f"\n✨ Done. Outputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()