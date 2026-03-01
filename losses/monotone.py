import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotonicityLoss(nn.Module):
    """
    Penalize quantile crossing during training.

    When predicting multiple quantiles simultaneously, higher quantile levels
    should predict higher values. This loss penalizes violations of this constraint.
    """

    def __init__(self, margin: float = 0.0, reduction: str = "mean"):
        """
        Args:
            margin: Minimum gap between adjacent quantile predictions
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, quantiles: torch.Tensor):
        """
        Args:
            predictions: [B, H, Q] - predicted values for Q quantiles
            quantiles: [B, 1, Q] or [B, Q] - quantile levels

        Returns:
            Scalar loss penalizing monotonicity violations
        """

        # Ensure quantiles are 3D: [B, 1, Q]
        if quantiles.dim() == 2:
            quantiles = quantiles.unsqueeze(1)

        # Handle invalid values in predictions
        predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Sort quantiles and get sorting indices
        q_sorted, sort_idx = quantiles.sort(dim=-1)

        # Apply same sorting to predictions
        sort_idx_expanded = sort_idx.expand_as(predictions)
        pred_sorted = predictions.gather(-1, sort_idx_expanded)

        # Compute differences between adjacent quantile predictions
        # For proper ordering: pred[q_i+1] >= pred[q_i] + margin
        diffs = pred_sorted[..., 1:] - pred_sorted[..., :-1]  # [B, H, Q-1]

        # Penalize negative differences (quantile crossings)
        # Clamp to prevent numerical instability
        violations = F.relu(-diffs + self.margin)
        violations = torch.clamp(violations, max=1e6)  # Prevent explosion
        
        # Apply reduction
        if self.reduction == "mean":
            result = violations.mean()
        elif self.reduction == "sum":
            result = violations.sum()
        else:
            result = violations
        
        # Final check for NaN/Inf
        result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result


from .pinball import PinballLoss


class MQNLossWithMonotonicity(nn.Module):
    """Combined MQN Loss + Monotonicity penalty for non-crossing quantiles."""
    
    def __init__(self, quantiles, monotone_weight: float = 0.1, monotone_margin: float = 0.01):
        super().__init__()
        self.mqn_loss = nn.ModuleDict()
        for q in quantiles:
            self.mqn_loss[str(q)] = PinballLoss(quantile=q, reduction="none")
        self.monotonicity_loss = MonotonicityLoss(margin=monotone_margin, reduction="mean")
        self.monotone_weight = monotone_weight
        
    def forward(self, predictions, targets, quantiles):
        """
        Args:
            predictions: [B, H, Q]
            targets: [B, H]
            quantiles: [B, Q]
        """
        # Primary loss: MQN (pinball)
        mqn = 0.0
        Q = predictions.shape[-1]
        for i, q in enumerate(quantiles[0].tolist()):
            q_str = str(round(q, 4))
            if q_str in self.mqn_loss:
                mqn += self.mqn_loss[q_str](predictions[..., i], targets).mean()
        mqn = mqn / Q
        
        # Secondary loss: Monotonicity penalty
        mono = self.monotonicity_loss(predictions, quantiles)
        
        # Combined loss
        total = mqn + self.monotone_weight * mono
        return total
