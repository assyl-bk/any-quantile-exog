"""
Conformalized Quantile Regression (CQR)

Based on Romano, Patterson, and Candès (NeurIPS 2019).
Post-processing for distribution-free coverage guarantees.

Evidence: 31% shorter intervals than split conformal
"""

import numpy as np
from typing import Tuple


class ConformizedQuantileRegression:
    """
    Conformalized Quantile Regression (CQR)
    Post-processing for distribution-free coverage guarantees.
    
    Apply AFTER training - no retraining needed.
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Miscoverage level (0.1 = 90% coverage, 0.05 = 95% coverage)
        """
        self.alpha = alpha
        self.q_adjustment = None
        self.adjustments = None
    
    def calibrate(
        self,
        y_true: np.ndarray,
        q_lo: np.ndarray,
        q_hi: np.ndarray,
    ) -> float:
        """
        Calibrate using held-out calibration set.
        
        Args:
            y_true: [N] true values
            q_lo: [N] lower quantile predictions (α/2)
            q_hi: [N] upper quantile predictions (1-α/2)
            
        Returns:
            adjustment: the conformity score quantile
        """
        # Conformity scores (Eq. 12)
        scores = np.maximum(q_lo - y_true, y_true - q_hi)
        
        # Compute quantile (Eq. 13)
        n = len(scores)
        q_level = np.ceil((1 - self.alpha) * (n + 1)) / n
        q_level = min(q_level, 1.0)
        
        self.q_adjustment = np.quantile(scores, q_level)
        return self.q_adjustment
    
    def predict(
        self,
        q_lo: np.ndarray,
        q_hi: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply conformal adjustment to get calibrated intervals.
        
        Args:
            q_lo, q_hi: Quantile predictions from trained model
            
        Returns:
            calibrated_lo, calibrated_hi: Intervals with guaranteed coverage
        """
        if self.q_adjustment is None:
            raise ValueError("Must call calibrate() first")
        
        # Apply adjustment (Eq. 14)
        calibrated_lo = q_lo - self.q_adjustment
        calibrated_hi = q_hi + self.q_adjustment
        
        return calibrated_lo, calibrated_hi
    
    def calibrate_all_quantiles(
        self,
        y_cal: np.ndarray,
        q_preds_cal: np.ndarray,
        quantile_levels: np.ndarray,
    ) -> np.ndarray:
        """
        Calibrate all quantile levels, not just intervals.
        
        Args:
            y_cal: [N, H] calibration targets
            q_preds_cal: [N, H, Q] calibration predictions
            quantile_levels: [Q] quantile levels
            
        Returns:
            adjustments: [Q] adjustment for each quantile
        """
        N, H, Q = q_preds_cal.shape
        adjustments = np.zeros(Q)
        
        for q_idx, tau in enumerate(quantile_levels):
            # Residuals for this quantile
            residuals = y_cal - q_preds_cal[:, :, q_idx]  # [N, H]
            residuals_flat = residuals.flatten()
            
            # For coverage at level tau
            n = len(residuals_flat)
            q_level = np.ceil(tau * (n + 1)) / n
            q_level = min(q_level, 1.0)
            
            adjustments[q_idx] = np.quantile(residuals_flat, q_level)
        
        self.adjustments = adjustments
        return adjustments
    
    def apply_adjustments(
        self,
        q_preds: np.ndarray,
    ) -> np.ndarray:
        """Apply per-quantile adjustments."""
        if self.adjustments is None:
            raise ValueError("Must call calibrate_all_quantiles() first")
        
        # Add adjustments
        calibrated = q_preds + self.adjustments.reshape(1, 1, -1)
        
        # Enforce monotonicity
        calibrated = np.sort(calibrated, axis=-1)
        
        return calibrated
