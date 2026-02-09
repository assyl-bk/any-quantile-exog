"""
CQR Lightning Module

Wraps any quantile forecasting model with Conformalized Quantile Regression.
Provides guaranteed coverage through post-hoc calibration.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from typing import Dict, Any, Optional
import sys
sys.path.append('..')

from utils.cqr import ConformizedQuantileRegression
from metrics.metrics import quantile_loss, coverage, interval_width


class CQRLightningModule(pl.LightningModule):
    """
    Lightning module that wraps any model with CQR post-processing.
    
    Two-stage process:
    1. Train the base model normally (or load pretrained)
    2. Calibrate CQR on validation set
    3. Apply CQR adjustments during testing
    
    Args:
        base_model: The underlying quantile forecasting model
        quantile_levels: List of quantile levels
        alpha: Miscoverage level (0.1 = 90% coverage)
        calibration_method: 'all_quantiles' or 'interval'
        apply_cqr: Whether to apply CQR (disable for baseline comparison)
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        quantile_levels: list,
        alpha: float = 0.05,  # 95% coverage
        calibration_method: str = 'all_quantiles',
        apply_cqr: bool = True,
        loss_fn: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__()
        
        self.base_model = base_model
        self.quantile_levels = torch.tensor(quantile_levels, dtype=torch.float32)
        self.alpha = alpha
        self.calibration_method = calibration_method
        self.apply_cqr = apply_cqr
        self.loss_fn = loss_fn
        
        # CQR object (calibrated after first validation epoch)
        self.cqr = ConformizedQuantileRegression(alpha=alpha)
        self.cqr_calibrated = False
        
        # Store calibration data during validation
        self.cal_predictions = []
        self.cal_targets = []
        
        # Store test predictions
        self.test_predictions = []
        self.test_targets = []
        
        self.save_hyperparameters(ignore=['base_model', 'loss_fn'])
    
    def forward(self, x, quantiles=None):
        """Forward pass through base model."""
        if quantiles is None:
            quantiles = self.quantile_levels.to(x.device)
        return self.base_model(x, quantiles)
    
    def training_step(self, batch, batch_idx):
        """Training step - train base model normally."""
        x = batch['history']
        y = batch['target']
        
        # Random quantiles for training
        if 'quantiles' in batch:
            quantiles = batch['quantiles']
        else:
            B = x.size(0)
            quantiles = torch.rand(B, len(self.quantile_levels), device=x.device)
            quantiles = torch.sort(quantiles, dim=1)[0]
        
        # Forward pass
        preds = self(x, quantiles)
        
        # Compute loss
        if self.loss_fn is not None:
            loss = self.loss_fn(preds, y, quantiles)
        else:
            loss = quantile_loss(preds, y, quantiles)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step - collect predictions for CQR calibration."""
        x = batch['history']
        y = batch['target']
        
        # Use fixed quantile levels
        quantiles = self.quantile_levels.unsqueeze(0).expand(x.size(0), -1).to(x.device)
        
        # Forward pass
        preds = self(x, quantiles)
        
        # Compute loss
        if self.loss_fn is not None:
            loss = self.loss_fn(preds, y, quantiles)
        else:
            loss = quantile_loss(preds, y, quantiles)
        
        self.log('val_loss', loss, prog_bar=True)
        
        # Store for CQR calibration
        self.cal_predictions.append(preds.detach().cpu().numpy())
        self.cal_targets.append(y.detach().cpu().numpy())
        
        return loss
    
    def on_validation_epoch_end(self):
        """Calibrate CQR at end of validation epoch."""
        
        if not self.apply_cqr or self.cqr_calibrated:
            # Clear buffers
            self.cal_predictions = []
            self.cal_targets = []
            return
        
        if len(self.cal_predictions) == 0:
            return
        
        # Concatenate all calibration data
        cal_preds = np.concatenate(self.cal_predictions, axis=0)  # [N, H, Q]
        cal_targets = np.concatenate(self.cal_targets, axis=0)  # [N, H]
        
        print(f"\n{'='*60}")
        print(f"Calibrating CQR with {len(cal_targets)} samples...")
        print(f"Method: {self.calibration_method}, alpha={self.alpha}")
        print(f"Target coverage: {1 - self.alpha:.1%}")
        
        # Calibrate based on method
        if self.calibration_method == 'interval':
            # Use first and last quantile
            q_lo = cal_preds[:, :, 0]
            q_hi = cal_preds[:, :, -1]
            adjustment = self.cqr.calibrate_interval(cal_targets, q_lo, q_hi)
            print(f"Interval adjustment: {adjustment:.4f}")
            
        elif self.calibration_method == 'all_quantiles':
            quantiles_np = self.quantile_levels.cpu().numpy()
            adjustments = self.cqr.calibrate_all_quantiles(
                cal_targets, cal_preds, quantiles_np
            )
            print(f"Quantile adjustments: {adjustments}")
        
        # Print calibration info
        info = self.cqr.get_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        self.cqr_calibrated = True
        print(f"✓ CQR calibration complete!")
        print(f"{'='*60}\n")
        
        # Clear buffers
        self.cal_predictions = []
        self.cal_targets = []
    
    def test_step(self, batch, batch_idx):
        """Test step - apply CQR if calibrated."""
        x = batch['history']
        y = batch['target']
        
        # Use fixed quantile levels
        quantiles = self.quantile_levels.unsqueeze(0).expand(x.size(0), -1).to(x.device)
        
        # Forward pass
        preds = self(x, quantiles)
        
        # Store for final evaluation
        self.test_predictions.append(preds.detach().cpu().numpy())
        self.test_targets.append(y.detach().cpu().numpy())
        
        return {'predictions': preds, 'targets': y}
    
    def on_test_epoch_end(self):
        """Apply CQR and compute final metrics."""
        
        if len(self.test_predictions) == 0:
            return
        
        # Concatenate all test data
        test_preds = np.concatenate(self.test_predictions, axis=0)  # [N, H, Q]
        test_targets = np.concatenate(self.test_targets, axis=0)  # [N, H]
        
        print(f"\n{'='*60}")
        print(f"Test Evaluation")
        print(f"{'='*60}")
        
        # Evaluate BEFORE CQR
        metrics_before = self._compute_metrics(test_preds, test_targets, "Before CQR")
        
        # Apply CQR if calibrated
        if self.apply_cqr and self.cqr_calibrated:
            print(f"\nApplying CQR adjustments...")
            
            if self.calibration_method == 'all_quantiles':
                test_preds_cqr = self.cqr.apply_adjustments(test_preds)
            elif self.calibration_method == 'interval':
                q_lo = test_preds[:, :, 0]
                q_hi = test_preds[:, :, -1]
                cal_lo, cal_hi = self.cqr.predict_interval(q_lo, q_hi)
                test_preds_cqr = test_preds.copy()
                test_preds_cqr[:, :, 0] = cal_lo
                test_preds_cqr[:, :, -1] = cal_hi
            
            # Evaluate AFTER CQR
            metrics_after = self._compute_metrics(test_preds_cqr, test_targets, "After CQR")
            
            # Log comparison
            print(f"\n{'='*60}")
            print(f"CQR Impact Summary")
            print(f"{'='*60}")
            
            crps_improvement = (metrics_before['crps'] - metrics_after['crps']) / metrics_before['crps'] * 100
            coverage_improvement = metrics_after['coverage_95'] - metrics_before['coverage_95']
            width_reduction = (metrics_before['interval_width'] - metrics_after['interval_width']) / metrics_before['interval_width'] * 100
            
            print(f"CRPS:     {metrics_before['crps']:.2f} → {metrics_after['crps']:.2f} ({crps_improvement:+.1f}%)")
            print(f"Coverage: {metrics_before['coverage_95']:.3f} → {metrics_after['coverage_95']:.3f} ({coverage_improvement:+.3f})")
            print(f"Width:    {metrics_before['interval_width']:.2f} → {metrics_after['interval_width']:.2f} ({width_reduction:+.1f}%)")
            
            if metrics_after['crps'] < 211:
                print(f"\n🎯 SUCCESS! CRPS = {metrics_after['crps']:.2f} < 211 (baseline)")
            
        else:
            print("\nCQR not applied (baseline evaluation)")
        
        print(f"{'='*60}\n")
        
        # Clear buffers
        self.test_predictions = []
        self.test_targets = []
    
    def _compute_metrics(self, preds: np.ndarray, targets: np.ndarray, label: str) -> Dict[str, float]:
        """Compute and log metrics."""
        
        # CRPS (using all quantiles)
        quantiles_np = self.quantile_levels.cpu().numpy()
        crps = self._compute_crps(preds, targets, quantiles_np)
        
        # Coverage (95% interval = [0.025, 0.975])
        q_lo_idx = np.argmin(np.abs(quantiles_np - 0.025))
        q_hi_idx = np.argmin(np.abs(quantiles_np - 0.975))
        
        q_lo = preds[:, :, q_lo_idx]
        q_hi = preds[:, :, q_hi_idx]
        
        cov_95 = np.mean((targets >= q_lo) & (targets <= q_hi))
        
        # Interval width
        width_95 = np.mean(q_hi - q_lo)
        
        # Quantile losses
        pinball_losses = []
        for i, tau in enumerate(quantiles_np):
            errors = targets - preds[:, :, i]
            pinball = np.where(errors >= 0, tau * errors, (tau - 1) * errors)
            pinball_losses.append(np.mean(pinball))
        mean_pinball = np.mean(pinball_losses)
        
        metrics = {
            'crps': crps,
            'coverage_95': cov_95,
            'interval_width': width_95,
            'mean_pinball': mean_pinball
        }
        
        # Print
        print(f"\n{label}:")
        print(f"  CRPS:              {crps:.2f}")
        print(f"  Coverage (95%):    {cov_95:.3f}")
        print(f"  Interval Width:    {width_95:.2f}")
        print(f"  Mean Pinball Loss: {mean_pinball:.4f}")
        
        # Log to tensorboard
        for key, value in metrics.items():
            self.log(f'test_{label.lower().replace(" ", "_")}_{key}', value)
        
        return metrics
    
    def _compute_crps(self, preds: np.ndarray, targets: np.ndarray, quantiles: np.ndarray) -> float:
        """Compute CRPS using trapezoidal integration."""
        
        # CRPS = integral of (F(y) - 1(y >= target))^2 dy
        # Approximated using quantile predictions
        
        N, H, Q = preds.shape
        
        crps_sum = 0.0
        for i in range(Q - 1):
            tau_i = quantiles[i]
            tau_ip1 = quantiles[i + 1]
            
            q_i = preds[:, :, i]
            q_ip1 = preds[:, :, i + 1]
            
            # Trapezoidal rule
            # Contribution from interval [tau_i, tau_ip1]
            
            # For y in [q_i, q_ip1], approximate integral
            width = q_ip1 - q_i
            
            # Indicator: is target above this quantile?
            ind_i = (targets > q_i).astype(float)
            ind_ip1 = (targets > q_ip1).astype(float)
            
            # Expected value of (F - I)^2 over this interval
            delta_tau = tau_ip1 - tau_i
            
            # Simplified: width * squared error
            crps_sum += np.sum(width * ((tau_i + tau_ip1) / 2 - (ind_i + ind_ip1) / 2) ** 2)
        
        crps = crps_sum / (N * H)
        
        return crps
    
    def configure_optimizers(self):
        """Configure optimizers - delegate to base model if it has the method."""
        if hasattr(self.base_model, 'configure_optimizers'):
            return self.base_model.configure_optimizers()
        
        # Default: Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
