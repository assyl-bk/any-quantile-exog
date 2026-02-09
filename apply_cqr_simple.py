"""
Simple CQR Application Script

Applies CQR to predictions saved from a model run.
This version doesn't require loading the model itself.

Usage:
    1. First, save predictions: python save_predictions.py
    2. Then apply CQR: python apply_cqr_simple.py
"""

import torch
import numpy as np
import pickle
from utils.cqr import ConformizedQuantileRegression


def compute_crps(preds: np.ndarray, targets: np.ndarray, quantiles: np.ndarray) -> float:
    """Compute CRPS using trapezoidal integration."""
    
    N, H, Q = preds.shape
    
    crps_sum = 0.0
    for i in range(Q - 1):
        tau_i = quantiles[i]
        tau_ip1 = quantiles[i + 1]
        
        q_i = preds[:, :, i]
        q_ip1 = preds[:, :, i + 1]
        
        width = q_ip1 - q_i
        
        ind_i = (targets > q_i).astype(float)
        ind_ip1 = (targets > q_ip1).astype(float)
        
        crps_sum += np.sum(width * ((tau_i + tau_ip1) / 2 - (ind_i + ind_ip1) / 2) ** 2)
    
    crps = crps_sum / (N * H)
    
    return crps


def evaluate_predictions(preds: np.ndarray, targets: np.ndarray, quantiles: np.ndarray, label: str = ""):
    """Evaluate predictions with metrics."""
    
    print(f"\n{label}")
    print("=" * 60)
    
    # CRPS
    crps = compute_crps(preds, targets, quantiles)
    print(f"CRPS: {crps:.2f}")
    
    # Coverage
    q_lo_idx = np.argmin(np.abs(quantiles - 0.025))
    q_hi_idx = np.argmin(np.abs(quantiles - 0.975))
    
    q_lo = preds[:, :, q_lo_idx]
    q_hi = preds[:, :, q_hi_idx]
    
    coverage_95 = np.mean((targets >= q_lo) & (targets <= q_hi))
    print(f"Coverage (95%): {coverage_95:.3f}")
    
    # Interval width
    width_95 = np.mean(q_hi - q_lo)
    print(f"Interval Width: {width_95:.2f}")
    
    # Per-quantile metrics
    print(f"\nPer-Quantile Coverage:")
    for i, tau in enumerate(quantiles):
        empirical_coverage = np.mean(targets <= preds[:, :, i])
        print(f"  τ={tau:.3f}: coverage={empirical_coverage:.3f} (target={tau:.3f})")
    
    return {
        'crps': crps,
        'coverage_95': coverage_95,
        'width_95': width_95
    }


def main():
    print("=" * 80)
    print("ConformQualified Quantile Regression - Simple Application")
    print("=" * 80)
    
    # Load predictions (you need to generate these first)
    # For now, let's use the multihead model outputs
    
    # Try to load from saved predictions file
    try:
        with open('predictions_multihead.pkl', 'rb') as f:
            data = pickle.load(f)
            cal_preds = data['cal_preds']
            cal_targets = data['cal_targets']
            test_preds = data['test_preds']
            test_targets = data['test_targets']
            quantiles = data['quantiles']
        print("[OK] Loaded predictions from file")
    except FileNotFoundError:
        print("ERROR: predictions_multihead.pkl not found")
        print("\nPlease run the model first to generate predictions, or use:")
        print("  python save_model_predictions.py")
        return
    
    print(f"\nCalibration set: {cal_preds.shape}")
    print(f"Test set: {test_preds.shape}")
    print(f"Quantiles: {quantiles}")
    
    # Evaluate BEFORE CQR
    metrics_before = evaluate_predictions(test_preds, test_targets, quantiles, "BEFORE CQR")
    
    # Calibrate CQR
    print("\n" + "=" * 80)
    print("Calibrating CQR")
    print("=" * 80)
    
    alpha = 0.05  # 95% coverage
    cqr = ConformizedQuantileRegression(alpha=alpha)
    
    adjustments = cqr.calibrate_all_quantiles(cal_targets, cal_preds, quantiles)
    print(f"Quantile adjustments: {adjustments}")
    
    info = cqr.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("[OK] CQR calibrated")
    
    # Apply CQR to test predictions
    print("\n" + "=" * 80)
    print("Applying CQR to Test Predictions")
    print("=" * 80)
    
    test_preds_cqr = cqr.apply_adjustments(test_preds)
    
    print("[OK] CQR applied")
    
    # Evaluate AFTER CQR
    metrics_after = evaluate_predictions(test_preds_cqr, test_targets, quantiles, "AFTER CQR")
    
    # Compare
    print("\n" + "=" * 80)
    print("CQR IMPACT SUMMARY")
    print("=" * 80)
    
    crps_change = metrics_after['crps'] - metrics_before['crps']
    crps_pct = crps_change / metrics_before['crps'] * 100
    
    coverage_change = metrics_after['coverage_95'] - metrics_before['coverage_95']
    
    width_change = metrics_after['width_95'] - metrics_before['width_95']
    width_pct = width_change / metrics_before['width_95'] * 100
    
    print(f"\nCRPS:              {metrics_before['crps']:.2f} → {metrics_after['crps']:.2f} ({crps_pct:+.1f}%)")
    print(f"Coverage (95%):    {metrics_before['coverage_95']:.3f} → {metrics_after['coverage_95']:.3f} ({coverage_change:+.3f})")
    print(f"Interval Width:    {metrics_before['width_95']:.2f} → {metrics_after['width_95']:.2f} ({width_pct:+.1f}%)")
    
    print("\n" + "=" * 80)
    
    # Check if target achieved
    if metrics_after['crps'] < 211:
        print(f"[SUCCESS!] CRPS = {metrics_after['crps']:.2f} < 211 (baseline)")
    else:
        print(f"[WARNING] CRPS = {metrics_after['crps']:.2f} >= 211 (baseline)")
    
    target_coverage = 1 - alpha
    if abs(metrics_after['coverage_95'] - target_coverage) < 0.02:
        print(f"[OK] Coverage = {metrics_after['coverage_95']:.3f} ~= {target_coverage:.3f} (target)")
    else:
        print(f"[WARNING] Coverage = {metrics_after['coverage_95']:.3f} vs {target_coverage:.3f} (target)")
    
    print("=" * 80)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
