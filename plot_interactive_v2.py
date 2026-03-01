"""
Full Evaluation Script — V2 Sequential Curriculum
Evaluates on ALL test samples to get proper calibration curves.

Usage:
    python full_eval_v2.py --model stage2   # Stage 2-v2 (best, default)
    python full_eval_v2.py --model stage3   # Stage 3-v2
    python full_eval_v2.py --model both     # Compare both side by side
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
import yaml
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from model.models import AnyQuantileForecasterExogSeriesAdaptive
from utils.model_factory import instantiate   # same pattern as run_stage3_v2.py

# ── Constants ─────────────────────────────────────────────────────────────────
EVAL_QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
OUTPUT_DIR     = "results/full_eval_v2"

MODELS = {
    "stage2": {
        "label":      "V2 Stage 2 — Exog+Adaptive (BEST)",
        "checkpoint": "lightning_logs/nbeatsaq-v2-stage2-seed0/checkpoints/model-epoch=07.ckpt",
        "config":     "config/stage2_v2.yaml",
        "color":      "#2196F3",
        "crps_full":  172.21,
        "cov_full":   0.9070,
    },
    "stage3": {
        "label":      "V2 Stage 3 — Exog+Adaptive+Series",
        "checkpoint": "lightning_logs/nbeatsaq-v2-stage3-seed0/checkpoints/model-epoch=03.ckpt",
        "config":     "config/stage3_v2.yaml",
        "color":      "#FF9800",
        "crps_full":  172.75,
        "cov_full":   0.8954,
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_cfg(config_path: Path) -> OmegaConf:
    if config_path.exists():
        with open(config_path) as f:
            raw = f.read().replace("!!python/tuple", "")
        try:
            cfg = OmegaConf.create(yaml.safe_load(raw))
            print(f"  ✅ Config loaded: {config_path}")
            return cfg
        except Exception as e:
            print(f"  ⚠️  YAML parse failed ({e}), using fallback")
    else:
        print(f"  ⚠️  Config not found: {config_path}, using fallback")

    # Fallback mirrors stage2_v2.yaml exactly
    return OmegaConf.create({
        "model": {
            "_target_": "model.AnyQuantileForecasterExogSeriesAdaptive",
            "input_horizon_len": 168, "max_norm": True,
            "num_series": 35, "series_embed_dim": 32,
            "series_embed_scale": 0.0,
            "q_sampling": "adaptive", "q_distribution": "beta", "q_parameter": 0.3,
            "adaptive_sampling": {
                "num_adaptive_quantiles": 4, "num_bins": 30,
                "momentum": 0.99, "temperature": 1.2, "min_prob": 0.002,
            },
        },
        "dataset": {
            "_target_": "dataset.EMHIRESUnivariateDataModule",
            "name": "MHLV",
            "train_batch_size": 512, "eval_batch_size": 512,
            "num_workers": 4, "persistent_workers": True,
            "horizon_length": 24, "history_length": 168,
            "split_boundaries": ["2006-01-01", "2017-12-30", "2018-01-01", "2019-01-01"],
            "fillna": "ffill", "train_step": 1, "eval_step": 24,
            "exog_features": ["temperature", "humidity", "pressure", "wind_speed"],
            "calendar_features": True,
        },
    })


def get_dataloader(cfg):
    """Use instantiate() — avoids the unexpected keyword argument error."""
    dm = instantiate(cfg.dataset)
    dm.setup("test")
    print(f"  ✅ Test set: {len(dm.test_dataset):,} samples")
    return dm.test_dataloader()


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_full_eval(model_key: str, device: torch.device):
    info      = MODELS[model_key]
    ckpt_path = Path(info["checkpoint"])
    cfg_path  = Path(info["config"])

    print(f"\n{'='*70}")
    print(f"  Evaluating : {info['label']}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"{'='*70}")

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run from your project root directory."
        )

    cfg   = load_cfg(cfg_path)
    model = AnyQuantileForecasterExogSeriesAdaptive.load_from_checkpoint(
        str(ckpt_path), cfg=cfg, strict=False, map_location=device
    )
    model.eval().to(device)
    print(f"  ✅ Model loaded on {device}")

    loader   = get_dataloader(cfg)
    q_tensor = torch.tensor(EVAL_QUANTILES, dtype=torch.float32, device=device)

    all_preds, all_targets, all_series = [], [], []

    print("  🔮 Inference on all batches...")
    for batch in tqdm(loader, desc="  batches"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        B = batch["history"].shape[0]

        per_q = []
        for q_val in q_tensor:
            batch["quantiles"] = q_val.view(1, 1).expand(B, 1)
            pred = model(batch)
            per_q.append(pred.squeeze(-1))          # [B, H]

        all_preds.append(torch.stack(per_q, dim=-1).cpu())   # [B, H, Q]
        all_targets.append(batch["target"].cpu())
        if "series_id" in batch:
            sid = batch["series_id"]
            all_series.append(
                sid.cpu().numpy() if isinstance(sid, torch.Tensor) else np.array(sid)
            )

    preds_all   = torch.cat(all_preds,   dim=0).numpy()   # [N, H, Q]
    targets_all = torch.cat(all_targets, dim=0).numpy()   # [N, H]
    series_all  = np.concatenate(all_series) if all_series else None

    N, H, Q = preds_all.shape
    print(f"  ✅ Done: {N:,} samples × {H}h × {Q} quantiles")
    return preds_all, targets_all, series_all


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(preds, targets, quantiles=EVAL_QUANTILES):
    q_arr      = np.array(quantiles)
    valid_mask = np.isfinite(targets)

    # CRPS = 2 * mean pinball loss
    err       = targets[..., None] - preds
    pb        = np.where(err >= 0, q_arr * err, (q_arr - 1) * err)
    pb_masked = np.where(valid_mask[..., None], pb, np.nan)
    crps      = float(np.nanmean(pb_masked) * 2)

    # Per-quantile coverage
    coverage = {}
    for i, q in enumerate(quantiles):
        valid_tgt  = targets[valid_mask]
        valid_pred = preds[:, :, i][valid_mask]
        coverage[q] = float(np.mean(valid_tgt <= valid_pred))

    # Point metrics at median
    mid_idx    = quantiles.index(0.50) if 0.50 in quantiles else len(quantiles) // 2
    valid_flat = valid_mask.ravel()
    pm_flat    = preds[:, :, mid_idx].ravel()[valid_flat]
    tg_flat    = targets.ravel()[valid_flat]
    mae        = float(np.mean(np.abs(pm_flat - tg_flat)))
    mape       = float(np.mean(np.abs((pm_flat - tg_flat) / (np.abs(tg_flat) + 1e-8)))) * 100
    rmse       = float(np.sqrt(np.mean((pm_flat - tg_flat) ** 2)))
    mace       = float(np.mean([abs(coverage[q] - q) for q in quantiles]))

    return {"crps": crps, "mae": mae, "rmse": rmse, "mape": mape,
            "mace": mace, "coverage": coverage, "N": int(preds.shape[0])}


def compute_ncrps_per_country(preds, targets, series_ids, quantiles=EVAL_QUANTILES):
    q_arr = np.array(quantiles)
    out   = {}
    for sid in np.unique(series_ids):
        mask  = series_ids == sid
        tgt   = targets[mask]
        pred  = preds[mask]
        y_bar = np.nanmean(tgt)
        if y_bar == 0:
            continue
        err = tgt[..., None] - pred
        pb  = np.where(err >= 0, q_arr * err, (q_arr - 1) * err)
        out[int(sid)] = float(np.nanmean(pb) / y_bar * 100)
    return out


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_calibration_full(results: dict, save_dir: str):
    n_models  = len(results)
    fig       = plt.figure(figsize=(18, 14))
    gs        = gridspec.GridSpec(3, 2, hspace=0.5, wspace=0.35)
    q_nominal = np.array(EVAL_QUANTILES)
    x         = np.arange(len(EVAL_QUANTILES))
    w         = 0.35 if n_models == 2 else 0.55
    offsets   = [-w / 2, w / 2] if n_models == 2 else [0]

    # Calibration curve
    ax_cal = fig.add_subplot(gs[0, 0])
    ax_cal.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect', zorder=0)
    for key, (_, _, m, info) in results.items():
        cov_vals = [m["coverage"][q] for q in EVAL_QUANTILES]
        ax_cal.plot(q_nominal, cov_vals, 'o-', lw=2, markersize=7,
                    color=info["color"], label=info["label"].split("—")[0].strip())
    ax_cal.set_xlabel("Nominal quantile", fontsize=11)
    ax_cal.set_ylabel("Empirical coverage", fontsize=11)
    ax_cal.set_title("Quantile Calibration Curve\n(full test set)", fontsize=12, fontweight='bold')
    ax_cal.legend(fontsize=9)
    ax_cal.grid(True, alpha=0.3)
    ax_cal.set_xlim(0, 1); ax_cal.set_ylim(0, 1)

    # Coverage error bars
    ax_err = fig.add_subplot(gs[0, 1])
    for (key, (_, _, m, info)), offset in zip(results.items(), offsets):
        errs   = [m["coverage"][q] - q for q in EVAL_QUANTILES]
        colors = ['#2ecc71' if e >= 0 else '#e74c3c' for e in errs]
        ax_err.bar(x + offset, errs, width=w * 0.9, color=colors,
                   alpha=0.8, edgecolor='black', lw=0.5,
                   label=info["label"].split("—")[0].strip())
    ax_err.axhline(0, color='black', lw=1.5, ls='--')
    ax_err.set_xticks(x)
    ax_err.set_xticklabels([f'{q:.2f}' for q in EVAL_QUANTILES], rotation=45, fontsize=9)
    ax_err.set_xlabel("Quantile", fontsize=11)
    ax_err.set_ylabel("Coverage error", fontsize=11)
    ax_err.set_title("Coverage Error per Quantile\n(green = over-cover, red = under-cover)",
                     fontsize=12, fontweight='bold')
    if n_models == 2:
        ax_err.legend(fontsize=9)
    ax_err.grid(True, alpha=0.3, axis='y')

    # Empirical vs nominal bar chart
    ax_cov = fig.add_subplot(gs[1, :])
    for (key, (_, _, m, info)), offset in zip(results.items(), offsets):
        cov_vals = [m["coverage"][q] for q in EVAL_QUANTILES]
        ax_cov.bar(x + offset, cov_vals, width=w * 0.9,
                   color=info["color"], alpha=0.75, edgecolor='black', lw=0.5,
                   label=info["label"].split("—")[0].strip())
    ax_cov.step(np.arange(-0.5, len(EVAL_QUANTILES)),
                np.append(q_nominal, q_nominal[-1]),
                where='post', color='black', lw=2, ls='--', label='Nominal (ideal)')
    ax_cov.set_xticks(x)
    ax_cov.set_xticklabels([f'Q{q:.2f}' for q in EVAL_QUANTILES], rotation=45, fontsize=9)
    ax_cov.set_ylabel("Empirical coverage", fontsize=11)
    ax_cov.set_title("Empirical vs Nominal Coverage — Full Test Set",
                     fontsize=12, fontweight='bold')
    ax_cov.legend(fontsize=9)
    ax_cov.grid(True, alpha=0.3, axis='y')
    ax_cov.set_ylim(0, 1.05)

    # Metrics summary table
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.axis('off')
    col_labels = ["Model", "N samples", "CRPS ↓", "MAE ↓", "MAPE % ↓",
                  "RMSE ↓", "Coverage-0.95 ↑", "MACE ↓"]
    rows_data = []
    for key, (_, _, m, info) in results.items():
        rows_data.append([
            info["label"].split("—")[0].strip(),
            f"{m['N']:,}",
            f"{m['crps']:.3f}",
            f"{m['mae']:.2f}",
            f"{m['mape']:.2f}",
            f"{m['rmse']:.2f}",
            f"{m['coverage'].get(0.95, float('nan')):.4f}",
            f"{m['mace']:.4f}",
        ])
    rows_data.append(["AQ-NBEATS (paper)", "—", "211.22", "—", "2.47", "—", "—", "—"])
    rows_data.append(["AQ-ESRNN  (paper)", "—", "195.94", "—", "2.32", "—", "—", "—"])

    tbl = ax_tbl.table(cellText=rows_data, colLabels=col_labels,
                       cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#2E75B6")
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')
    for j in range(len(col_labels)):
        tbl[(1, j)].set_facecolor("#E2EFDA")
    for r in [len(rows_data) - 1, len(rows_data)]:
        for j in range(len(col_labels)):
            tbl[(r, j)].set_facecolor("#FFF2CC")

    ax_tbl.set_title("Metrics Summary", fontsize=12, fontweight='bold', pad=12)
    plt.suptitle("V2 Sequential Curriculum — Full Test Evaluation\n"
                 "Exog → Exog+Adaptive → Exog+Adaptive+Series",
                 fontsize=14, fontweight='bold', y=1.01)

    out = f"{save_dir}/full_calibration_report.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  ✅ Saved: {out}")
    plt.close()


def plot_per_country_ncrps(results: dict, save_dir: str):
    fig, axes = plt.subplots(len(results), 1,
                             figsize=(18, 5 * len(results)), squeeze=False)
    for ax, (key, (_, _, m, info)) in zip(axes[:, 0], results.items()):
        if "ncrps_per_country" not in m:
            ax.text(0.5, 0.5, "No series_id in batch",
                    ha='center', va='center', transform=ax.transAxes)
            continue
        countries = sorted(m["ncrps_per_country"].keys())
        vals      = [m["ncrps_per_country"][c] for c in countries]
        x         = np.arange(len(countries))
        ax.bar(x, vals, color=info["color"], alpha=0.8, edgecolor='black', lw=0.5)
        ax.axhline(1.84, color='red',    lw=1.5, ls='--', label='AQ-NBEATS (1.84)')
        ax.axhline(1.72, color='orange', lw=1.5, ls='--', label='AQ-ESRNN (1.72)')
        ax.axhline(np.mean(vals), color='navy', lw=2,
                   label=f'Your mean ({np.mean(vals):.3f})')
        ax.set_xticks(x)
        ax.set_xticklabels(countries, rotation=45, fontsize=8)
        ax.set_xlabel("Series ID (country)", fontsize=11)
        ax.set_ylabel("N-CRPS", fontsize=11)
        ax.set_title(f"Per-Country N-CRPS — {info['label']}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    out = f"{save_dir}/ncrps_per_country.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {out}")
    plt.close()


def save_text_report(results: dict, save_dir: str):
    lines = ["=" * 80,
             "V2 SEQUENTIAL CURRICULUM — FULL TEST EVALUATION REPORT",
             "=" * 80]
    for key, (_, _, m, info) in results.items():
        lines += [
            f"\n{'─'*80}", f"  {info['label']}", f"{'─'*80}",
            f"  N samples : {m['N']:,}",
            f"  CRPS      : {m['crps']:.4f} MW  (trainer reported: {info['crps_full']})",
            f"  MAE       : {m['mae']:.4f} MW",
            f"  RMSE      : {m['rmse']:.4f} MW",
            f"  MAPE      : {m['mape']:.4f} %",
            f"  MACE      : {m['mace']:.6f}",
            f"\n  Per-Quantile Coverage:",
            f"  {'Quantile':>10}  {'Nominal':>10}  {'Empirical':>12}  {'Error':>10}",
            f"  {'─'*48}",
        ]
        for q in EVAL_QUANTILES:
            emp = m["coverage"][q]
            lines.append(f"  {q:>10.3f}  {q:>10.3f}  {emp:>12.4f}  {emp-q:>+10.4f}")
        if "ncrps_per_country" in m:
            vals = list(m["ncrps_per_country"].values())
            lines += [f"\n  N-CRPS mean         : {np.mean(vals):.4f}",
                      f"  Paper AQ-NBEATS     : 1.84",
                      f"  Paper AQ-ESRNN      : 1.72"]
    lines += ["\n" + "=" * 80, "  PAPER BASELINES", "=" * 80,
              "  AQ-NBEATS  CRPS=211.22  N-CRPS=1.84  MAPE=2.47%",
              "  AQ-ESRNN   CRPS=195.94  N-CRPS=1.72  MAPE=2.32%",
              "=" * 80]
    out = f"{save_dir}/full_eval_report.txt"
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  ✅ Report saved: {out}")
    print("\n".join(lines))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["stage2", "stage3", "both"],
                        default="both", help="Which model(s) to evaluate")
    args   = parser.parse_args()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    keys = ["stage2", "stage3"] if args.model == "both" else [args.model]

    results = {}
    for key in keys:
        preds, targets, series = run_full_eval(key, device)
        m = compute_metrics(preds, targets)
        if series is not None:
            m["ncrps_per_country"] = compute_ncrps_per_country(preds, targets, series)
        results[key] = (preds, targets, m, MODELS[key])

    print(f"\n\n{'='*70}\n  📊 Generating plots\n{'='*70}")
    plot_calibration_full(results, OUTPUT_DIR)
    plot_per_country_ncrps(results, OUTPUT_DIR)
    save_text_report(results, OUTPUT_DIR)
    print(f"\n✨ All outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()