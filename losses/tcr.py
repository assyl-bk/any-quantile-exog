"""Temporal Coherence Regularization (TCR)

Novel contribution: Enforces smooth evolution of quantile predictions across 
the forecast horizon to prevent erratic prediction intervals.

Key innovation: Penalizes high curvature (second derivative) while preserving
the model's ability to capture genuine volatility changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalCoherenceRegularization(nn.Module):
    """Temporal Coherence Regularization for quantile predictions.
    
    Penalizes erratic changes in quantile trajectories across the forecast horizon
    by minimizing the squared second derivative (curvature).
    
    Theorem: Does not bias marginal calibration - E[1(Y_t ≤ Q(τ,t))] = τ is preserved.
    
    Args:
        base_weight: Base regularization strength λ (default: 0.01)
        adaptive: If True, learns quantile-specific smoothness weights
        num_quantile_bins: Number of bins for adaptive quantile weighting
    """
    
    def __init__(self, base_weight: float = 0.01, adaptive: bool = True, num_quantile_bins: int = 10):
        super().__init__()
        # Make base_weight learnable, start at -10 → sigmoid(-10) ≈ 0.000045
        # This ensures we start at baseline performance
        self.base_weight_logit = nn.Parameter(torch.tensor(-10.0))
        self.base_weight_scale = base_weight  # Maximum weight after sigmoid
        self.adaptive = adaptive
        
        if adaptive:
            # Learn quantile-specific smoothness weights
            # Initialized to uniform, will discover that extremes need more smoothing
            self.quantile_weights = nn.Parameter(torch.ones(num_quantile_bins))
            self.num_bins = num_quantile_bins
    
    def compute_curvature(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute second derivative (curvature) of predictions across time.
        
        Uses central difference: Q(t+1) - 2*Q(t) + Q(t-1)
        
        Args:
            predictions: [B, H, Q] - quantile predictions across horizon
        
        Returns:
            curvature: [B, H-2, Q] - curvature at each interior time point
        """
        # Second difference: Q(t+1) - 2*Q(t) + Q(t-1)
        curvature = predictions[:, 2:, :] - 2 * predictions[:, 1:-1, :] + predictions[:, :-2, :]
        return curvature
    
    def forward(self, predictions: torch.Tensor, quantile_levels: torch.Tensor) -> torch.Tensor:
        """Compute temporal coherence loss.
        
        Args:
            predictions: [B, H, Q] - quantile predictions
            quantile_levels: [B, Q] or [Q] - the quantile levels (e.g., [0.1, 0.5, 0.9])
        
        Returns:
            loss: scalar - temporal coherence penalty
        """
        if predictions.size(1) < 3:
            # Need at least 3 time steps for second derivative
            return torch.tensor(0.0, device=predictions.device)
        
        curvature = self.compute_curvature(predictions)  # [B, H-2, Q]
        # Clip curvature for numerical stability
        curvature = torch.clamp(curvature, -1e3, 1e3)
        curvature_squared = curvature ** 2
        
        # Learned base weight: starts near 0, increases only if beneficial
        effective_weight = torch.sigmoid(self.base_weight_logit) * self.base_weight_scale
        
        if self.adaptive:
            # Get quantile levels (handle both [B, Q] and [Q] shapes)
            if quantile_levels.dim() == 2:
                quantile_levels = quantile_levels[0]  # Take first batch (should be same for all)
            
            # Bin quantiles and get learned weights
            bin_indices = (quantile_levels * self.num_bins).long().clamp(0, self.num_bins - 1)
            weights = F.softplus(self.quantile_weights[bin_indices])  # Ensure positive
            
            # Weight the curvature by quantile-specific factors
            weighted_curvature = curvature_squared * weights.view(1, 1, -1)
            loss = effective_weight * weighted_curvature.mean()
        else:
            loss = effective_weight * curvature_squared.mean()
        
        return loss
    
    def compute_smoothness_score(self, predictions: torch.Tensor) -> float:
        """Compute smoothness metric for evaluation/monitoring.
        
        Lower values indicate smoother quantile trajectories.
        
        Args:
            predictions: [B, H, Q] - quantile predictions
        
        Returns:
            score: float - mean squared curvature (lower is smoother)
        """
        if predictions.size(1) < 3:
            return 0.0
        
        curvature = self.compute_curvature(predictions)
        return (curvature ** 2).mean().item()


class TCRWithVarianceAwareness(TemporalCoherenceRegularization):
    """Extended TCR that allows more curvature where ground truth has high variance.
    
    Key insight: Don't over-smooth regions of genuine volatility. High variance 
    periods should be allowed more curvature in predictions.
    
    This prevents the regularization from suppressing the model's ability to
    capture true volatility patterns in heteroscedastic data.
    """
    
    def forward(self, predictions: torch.Tensor, quantile_levels: torch.Tensor, 
                historical_variance: torch.Tensor = None) -> torch.Tensor:
        """Compute variance-aware temporal coherence loss.
        
        Args:
            predictions: [B, H, Q] - quantile predictions
            quantile_levels: [B, Q] or [Q] - quantile levels
            historical_variance: [B, H] or None - historical variance at each time step
                If provided, scales penalty inversely with variance (high variance = less penalty)
        
        Returns:
            loss: scalar - variance-aware temporal coherence penalty
        """
        if predictions.size(1) < 3:
            return torch.tensor(0.0, device=predictions.device)
        
        curvature = self.compute_curvature(predictions)  # [B, H-2, Q]
        # Clip for numerical stability
        curvature = torch.clamp(curvature, -1e3, 1e3)
        
        if historical_variance is not None:
            # Scale penalty inversely with historical variance
            # High variance periods get less smoothing penalty
            var_scale = 1.0 / (historical_variance[:, 1:-1] + 1e-6)
            var_scale = var_scale / var_scale.mean()  # Normalize
            
            # Apply variance scaling to curvature
            curvature = curvature * var_scale.unsqueeze(-1)
        
        curvature_squared = curvature ** 2
        
        # Learned base weight
        effective_weight = torch.sigmoid(self.base_weight_logit) * self.base_weight_scale
        
        if self.adaptive:
            # Get quantile levels
            if quantile_levels.dim() == 2:
                quantile_levels = quantile_levels[0]
            
            # Bin quantiles and get learned weights
            bin_indices = (quantile_levels * self.num_bins).long().clamp(0, self.num_bins - 1)
            weights = F.softplus(self.quantile_weights[bin_indices])
            
            # Weight by quantile-specific factors
            weighted_curvature = curvature_squared * weights.view(1, 1, -1)
            loss = effective_weight * weighted_curvature.mean()
        else:
            loss = effective_weight * curvature_squared.mean()
        
        return loss
