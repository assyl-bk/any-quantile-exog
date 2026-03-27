"""Conformal Quantile Regression calibrator module."""
import pickle
import numpy as np


class ConformalCalibrator:
    """
    Split Conformal Prediction calibrator.

    Stores per-quantile correction offsets fitted on the calibration
    (validation) set. At test time, call .correct(preds) to get
    coverage-guaranteed prediction intervals.
    """

    def __init__(self, quantiles=None, alpha=0.05):
        self.quantiles = quantiles or [0.05, 0.25, 0.5, 0.75, 0.9]
        self.alpha = alpha
        self.q_idx_lo = None
        self.q_idx_hi = None
        self.q_per_level = None  # dict: per-quantile CQR offsets
        self.n_cal = None

    def fit(self, preds_cal: np.ndarray, targets_cal: np.ndarray):
        """
        Fit per-quantile CQR offsets on calibration predictions.

        preds_cal  : [N, H, Q]
        targets_cal: [N, H]
        """
        N, H, Q = preds_cal.shape
        self.n_cal = N
        valid = np.isfinite(targets_cal)

        # Store lo/hi indices for interval width tracking
        q_arr = np.array(self.quantiles)
        self.q_idx_lo = int(np.argmin(np.abs(q_arr - self.alpha / 2)))
        self.q_idx_hi = int(np.argmin(np.abs(q_arr - (1 - self.alpha / 2))))

        print(f"\n  Fitting CQR on {N:,} calibration samples...")
        print(
            f"  {'Quantile':>10}  {'Offset (MW)':>12}  {'Pre-cal cov':>14}  {'Post-cal cov':>14}"
        )
        print(f"  {'─'*56}")

        self.q_per_level = {}
        for i, q in enumerate(self.quantiles):
            pred_q = preds_cal[:, :, i]
            resid = targets_cal - pred_q  # signed residual [N, H]
            resid_valid = resid[valid]
            cal_level = min(q * (1 + 1 / N), 1.0)
            offset = float(np.quantile(resid_valid, cal_level))
            self.q_per_level[q] = offset

            pre_cov = float(np.mean(targets_cal[valid] <= pred_q[valid]))
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
