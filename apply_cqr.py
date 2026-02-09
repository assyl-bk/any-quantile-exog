"""
Apply CQR Post-Processing to Trained Model

This script applies Conformalized Quantile Regression to any trained model
without retraining. Use this to get guaranteed coverage on existing checkpoints.

Usage:
    python apply_cqr.py --checkpoint <path> --config <config_file> --alpha 0.05
    
Example:
    python apply_cqr.py --checkpoint lightning_logs/version_42/checkpoints/best.ckpt --config config/nbeatsaq-multihead.yaml --alpha 0.05
"""

import torch
import numpy as np
import argparse
from pathlib import Path
import sys
import yaml

from utils.cqr import ConformizedQuantileRegression


def load_model_and_config(checkpoint_path: str, config_path: str):
    """Load trained model and configuration."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert to object with dot notation
    class DotDict(dict):
        def __getattr__(self, key):
            val = self[key]
            if isinstance(val, dict):
                return DotDict(val)
            return val
        def __setattr__(self, key, val):
            self[key] = val
    
    config = DotDict(config)
    
    # Load checkpoint directly
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Import model class dynamically
    model_target = config.model._target_
    model_module = '.'.join(model_target.split('.')[:-1])
    model_class = model_target.split('.')[-1]
    
    # Import the module
    import importlib
    module = importlib.import_module(model_module)
    ModelClass = getattr(module, model_class)
    
    # Load from checkpoint
    model = ModelClass.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    return model, config


def get_dataloader(config, split='val'):
    """Create dataloader from config."""
    
    # Import dataset class
    dataset_module = config.dataset._target_.split('.')[0]
    dataset_class = config.dataset._target_.split('.')[-1]
    
    exec(f"from {dataset_module} import {dataset_class}")
    DatasetClass = eval(dataset_class)
    
    # Create dataset
    dataset_args = {k: v for k, v in config.dataset.items() if k != '_target_'}
    dataset = DatasetClass(**dataset_args)
    
    # Setup
    dataset.setup('fit' if split in ['train', 'val'] else 'test')
    
    # Get dataloader
    if split == 'train':
        return dataset.train_dataloader()
    elif split == 'val':
        return dataset.val_dataloader()
    else:
        return dataset.test_dataloader()


def collect_predictions(model, dataloader, device='cuda'):
    """Collect predictions and targets from dataloader."""
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['history'].to(device)
            y = batch['target']
            
            # Get quantiles
            if 'quantiles' in batch:
                quantiles = batch['quantiles'].to(device)
            elif hasattr(model, 'quantile_levels'):
                quantiles = model.quantile_levels.unsqueeze(0).expand(x.size(0), -1).to(device)
            else:
                # Assume 7 quantiles
                quantiles = torch.tensor([0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975], device=device)
                quantiles = quantiles.unsqueeze(0).expand(x.size(0), -1)
            
            # Forward pass
            preds = model(x, quantiles)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    return preds, targets


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
        
        delta_tau = tau_ip1 - tau_i
        
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
    print(f"\nPer-Quantile Metrics:")
    for i, tau in enumerate(quantiles):
        errors = targets - preds[:, :, i]
        pinball = np.where(errors >= 0, tau * errors, (tau - 1) * errors)
        mean_pinball = np.mean(pinball)
        
        empirical_coverage = np.mean(targets <= preds[:, :, i])
        
        print(f"  τ={tau:.3f}: pinball={mean_pinball:.4f}, coverage={empirical_coverage:.3f}")
    
    return {
        'crps': crps,
        'coverage_95': coverage_95,
        'width_95': width_95
    }


def main():
    parser = argparse.ArgumentParser(description='Apply CQR to trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--alpha', type=float, default=0.05, help='Miscoverage level (default: 0.05 for 95% coverage)')
    parser.add_argument('--method', type=str, default='all_quantiles', choices=['all_quantiles', 'interval'],
                        help='CQR calibration method')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--no-cqr', action='store_true', help='Evaluate without CQR (baseline)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Conformalized Quantile Regression Post-Processing")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Alpha: {args.alpha} (Target coverage: {1 - args.alpha:.1%})")
    print(f"Method: {args.method}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Load model and config
    print("\nLoading model...")
    model, config = load_model_and_config(args.checkpoint, args.config)
    print("✓ Model loaded")
    
    # Get quantile levels
    if hasattr(model, 'quantile_levels'):
        quantiles = model.quantile_levels.cpu().numpy()
    else:
        quantiles = np.array([0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975])
    
    print(f"Quantile levels: {quantiles}")
    
    # Load calibration data (validation set)
    print("\nLoading calibration data (validation set)...")
    cal_loader = get_dataloader(config, 'val')
    print(f"✓ Calibration data loaded: {len(cal_loader)} batches")
    
    # Load test data
    print("\nLoading test data...")
    test_loader = get_dataloader(config, 'test')
    print(f"✓ Test data loaded: {len(test_loader)} batches")
    
    # Collect calibration predictions
    print("\n" + "=" * 80)
    print("Collecting Calibration Predictions")
    print("=" * 80)
    cal_preds, cal_targets = collect_predictions(model, cal_loader, args.device)
    print(f"Shape: {cal_preds.shape}")
    
    # Collect test predictions
    print("\n" + "=" * 80)
    print("Collecting Test Predictions")
    print("=" * 80)
    test_preds, test_targets = collect_predictions(model, test_loader, args.device)
    print(f"Shape: {test_preds.shape}")
    
    # Evaluate BEFORE CQR
    metrics_before = evaluate_predictions(test_preds, test_targets, quantiles, "BEFORE CQR")
    
    if not args.no_cqr:
        # Calibrate CQR
        print("\n" + "=" * 80)
        print("Calibrating CQR")
        print("=" * 80)
        
        cqr = ConformizedQuantileRegression(alpha=args.alpha)
        
        if args.method == 'all_quantiles':
            adjustments = cqr.calibrate_all_quantiles(cal_targets, cal_preds, quantiles)
            print(f"Quantile adjustments: {adjustments}")
        elif args.method == 'interval':
            adjustment = cqr.calibrate_interval(cal_targets, cal_preds[:, :, 0], cal_preds[:, :, -1])
            print(f"Interval adjustment: {adjustment:.4f}")
        
        info = cqr.get_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("✓ CQR calibrated")
        
        # Apply CQR to test predictions
        print("\n" + "=" * 80)
        print("Applying CQR to Test Predictions")
        print("=" * 80)
        
        if args.method == 'all_quantiles':
            test_preds_cqr = cqr.apply_adjustments(test_preds)
        elif args.method == 'interval':
            q_lo, q_hi = cqr.predict_interval(test_preds[:, :, 0], test_preds[:, :, -1])
            test_preds_cqr = test_preds.copy()
            test_preds_cqr[:, :, 0] = q_lo
            test_preds_cqr[:, :, -1] = q_hi
        
        print("✓ CQR applied")
        
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
            print(f"🎯 SUCCESS! CRPS = {metrics_after['crps']:.2f} < 211 (baseline)")
        else:
            print(f"⚠️  CRPS = {metrics_after['crps']:.2f} ≥ 211 (baseline)")
        
        target_coverage = 1 - args.alpha
        if abs(metrics_after['coverage_95'] - target_coverage) < 0.02:
            print(f"✓ Coverage = {metrics_after['coverage_95']:.3f} ≈ {target_coverage:.3f} (target)")
        else:
            print(f"⚠️  Coverage = {metrics_after['coverage_95']:.3f} vs {target_coverage:.3f} (target)")
        
        print("=" * 80)
        
    else:
        print("\n(CQR not applied - baseline evaluation only)")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
