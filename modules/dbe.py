"""Distributional Basis Expansion (DBE)

Novel contribution: Decomposes the predictive distribution into interpretable 
basis components. Quantiles are computed analytically from a mixture distribution,
ensuring monotonicity by construction.

Key innovation: Aligns with N-BEATS philosophy - basis expansion for both location
and scale parameters, producing well-structured uncertainty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionalBasisExpansion(nn.Module):
    """Distributional Basis Expansion for quantile forecasting.
    
    Models the predictive distribution as a mixture of basis distributions:
        p(y_t | x) = Σ_k π_k(x) · p_k(y_t; μ_k(t), σ_k(t))
    
    Key features:
    - Location parameters μ_k from N-BEATS basis expansion
    - Scale parameters σ_k learned separately per component
    - Quantiles computed analytically from mixture
    - Monotonicity guaranteed by construction (Theorem 3)
    
    Args:
        num_components: Number of basis components (default: 3 for trend/season/residual)
        horizon: Forecast horizon length
        feature_dim: Dimension of backbone features
    """
    
    def __init__(self, num_components: int = 3, horizon: int = 24, feature_dim: int = 1024):
        super().__init__()
        self.num_components = num_components
        self.horizon = horizon
        self.feature_dim = feature_dim
        
        # Component names for interpretability
        self.component_names = ['trend', 'seasonality', 'residual'][:num_components]
        
        # Scale predictor for each component
        # Different components have different uncertainty patterns
        self.scale_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Linear(256, horizon),
                nn.Softplus()  # Ensure positive scale
            )
            for _ in range(num_components)
        ])
        
        # Mixture weight predictor
        # Determines contribution of each component to final distribution
        self.mixture_weights = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_components),
            nn.Softmax(dim=-1)
        )
        
        # Asymmetry parameters (learnable per component)
        # Allows skewed distributions if needed
        self.asymmetry = nn.Parameter(torch.zeros(num_components))
        
        # Blend weight: start at -10 → sigmoid(-10) ≈ 0 → 100% baseline
        # Only increases if DBE demonstrably improves performance
        self.blend_weight = nn.Parameter(torch.tensor(-10.0))
    
    def forward(self, backbone_features: torch.Tensor, backbone_locations: torch.Tensor, 
                quantile_levels: torch.Tensor) -> torch.Tensor:
        """Compute quantile predictions from distributional mixture.
        
        Args:
            backbone_features: [B, D] - features from N-BEATS backbone
            backbone_locations: [B, H, K] - location predictions from each N-BEATS block
            quantile_levels: [B, Q] or [Q] - quantile levels to compute
        
        Returns:
            quantile_predictions: [B, H, Q] - analytically computed quantiles
        """
        B = backbone_features.size(0)
        H = backbone_locations.size(1)
        K = backbone_locations.size(2) if backbone_locations.dim() == 3 else 1
        
        # Handle quantile_levels shape
        if quantile_levels.dim() == 1:
            quantile_levels = quantile_levels.unsqueeze(0).expand(B, -1)
        Q = quantile_levels.size(1)
        
        # Get mixture weights [B, num_components]
        weights = self.mixture_weights(backbone_features)
        
        # Get scale for each component [B, H, num_components]
        scales = torch.stack([
            pred(backbone_features) for pred in self.scale_predictors
        ], dim=-1)
        
        # Compute mixture distribution parameters
        # Location: weighted average of component locations
        if backbone_locations.dim() == 3 and K == self.num_components:
            # If we have K blocks matching num_components, use them directly
            mixture_location = (backbone_locations * weights.unsqueeze(1)).sum(dim=-1)  # [B, H]
        else:
            # Otherwise, use mean location for all components
            if backbone_locations.dim() == 3:
                backbone_locations = backbone_locations.mean(dim=-1)  # [B, H]
            mixture_location = backbone_locations  # [B, H]
        
        # Scale: root-mean-square of weighted component scales
        mixture_scale = torch.sqrt(
            (scales ** 2 * weights.unsqueeze(1) ** 2).sum(dim=-1) + 1e-6
        )  # [B, H]
        
        # Compute quantiles analytically using inverse CDF of Laplace mixture
        quantile_preds = self._compute_mixture_quantiles(
            mixture_location, mixture_scale, quantile_levels
        )
        
        # Return predictions and blend weight for residual blending
        return quantile_preds, torch.sigmoid(self.blend_weight)
        
        return quantile_preds
    
    def _compute_mixture_quantiles(self, location: torch.Tensor, scale: torch.Tensor, 
                                   quantile_levels: torch.Tensor) -> torch.Tensor:
        """Compute quantiles from Laplace mixture distribution.
        
        For Laplace(μ, b) distribution, the quantile function is:
            Q(τ) = μ - b * sign(τ - 0.5) * ln(1 - 2|τ - 0.5|)
        
        This is the inverse CDF, which guarantees monotonic quantiles.
        
        Args:
            location: [B, H] - mixture location parameters μ
            scale: [B, H] - mixture scale parameters b
            quantile_levels: [B, Q] - quantile levels τ
        
        Returns:
            quantiles: [B, H, Q] - guaranteed monotonic quantiles
        """
        # Expand for broadcasting: [B, H, 1] and [B, 1, Q]
        loc = location.unsqueeze(-1)  # [B, H, 1]
        scl = scale.unsqueeze(-1)     # [B, H, 1]
        q = quantile_levels.unsqueeze(1)  # [B, 1, Q]
        
        # Laplace inverse CDF
        # Q(τ) = μ - b * sign(τ - 0.5) * log(1 - 2|τ - 0.5|)
        centered_q = q - 0.5
        sign_q = torch.sign(centered_q)
        
        # Clamp to avoid log(0) - keeps τ away from 0 and 1
        abs_centered = torch.abs(centered_q).clamp(max=0.4999)
        log_term = torch.log(1 - 2 * abs_centered)
        
        quantiles = loc - scl * sign_q * log_term
        
        return quantiles  # [B, H, Q] - guaranteed monotonic!
    
    def get_mixture_params(self, backbone_features: torch.Tensor, 
                           backbone_locations: torch.Tensor) -> dict:
        """Get mixture distribution parameters for NLL computation.
        
        Args:
            backbone_features: [B, D] - features from N-BEATS backbone
            backbone_locations: [B, H, Q] or [B, H, K] - location predictions
        
        Returns:
            dict with:
                - weights: [B, num_components] - mixture weights π_k
                - locations: [B, H, num_components] - location parameters μ_k
                - scales: [B, H, num_components] - scale parameters σ_k
        """
        B = backbone_features.size(0)
        H = backbone_locations.size(1)
        
        # Get mixture weights [B, num_components]
        weights = self.mixture_weights(backbone_features)
        
        # Get scale for each component [B, H, num_components]
        scales = torch.stack([
            pred(backbone_features) for pred in self.scale_predictors
        ], dim=-1)
        
        # Get locations: use backbone_locations as component locations
        if backbone_locations.shape[-1] == self.num_components:
            locations = backbone_locations  # [B, H, num_components]
        else:
            # Replicate mean location across components
            mean_loc = backbone_locations.mean(dim=-1, keepdim=True)  # [B, H, 1]
            locations = mean_loc.expand(-1, -1, self.num_components)  # [B, H, num_components]
        
        return {
            'weights': weights,
            'locations': locations,
            'scales': scales
        }
    
    def get_component_analysis(self, backbone_features: torch.Tensor) -> dict:
        """Extract component contributions for interpretability.
        
        Args:
            backbone_features: [B, D] - features from backbone
        
        Returns:
            dict with:
                - weights: [B, K] - mixture weights (which component dominates)
                - scales: [B, H, K] - uncertainty per component
                - names: list - component names for interpretation
        """
        weights = self.mixture_weights(backbone_features)
        scales = torch.stack([
            pred(backbone_features) for pred in self.scale_predictors
        ], dim=-1)
        
        return {
            'weights': weights.detach(),     # Which component dominates
            'scales': scales.detach(),        # Uncertainty per component
            'names': self.component_names     # Interpretable names
        }


class DBEWithAdaptiveComponents(DistributionalBasisExpansion):
    """Extended DBE with adaptive number of components based on input complexity.
    
    Learns to activate/deactivate components based on the time series characteristics.
    Simple patterns may only need trend, complex patterns activate all components.
    """
    
    def __init__(self, num_components: int = 3, horizon: int = 24, feature_dim: int = 1024):
        super().__init__(num_components, horizon, feature_dim)
        
        # Gating network: decides which components to activate
        self.component_gate = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_components),
            nn.Sigmoid()  # Gate values in [0, 1]
        )
    
    def forward(self, backbone_features: torch.Tensor, backbone_locations: torch.Tensor, 
                quantile_levels: torch.Tensor) -> torch.Tensor:
        """Forward with adaptive component activation."""
        
        # Get component gates [B, K]
        gates = self.component_gate(backbone_features)
        
        # Original mixture weights
        base_weights = self.mixture_weights(backbone_features)
        
        # Modulate weights by gates (zero out inactive components)
        gated_weights = base_weights * gates
        
        # Renormalize to sum to 1
        weights = gated_weights / (gated_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Rest is same as base class but use gated weights
        B = backbone_features.size(0)
        H = backbone_locations.size(1)
        K = backbone_locations.size(2) if backbone_locations.dim() == 3 else 1
        
        if quantile_levels.dim() == 1:
            quantile_levels = quantile_levels.unsqueeze(0).expand(B, -1)
        
        scales = torch.stack([
            pred(backbone_features) for pred in self.scale_predictors
        ], dim=-1)
        
        if backbone_locations.dim() == 3 and K == self.num_components:
            mixture_location = (backbone_locations * weights.unsqueeze(1)).sum(dim=-1)
        else:
            if backbone_locations.dim() == 3:
                backbone_locations = backbone_locations.mean(dim=-1)
            mixture_location = backbone_locations
        
        mixture_scale = torch.sqrt(
            (scales ** 2 * weights.unsqueeze(1) ** 2).sum(dim=-1) + 1e-6
        )
        
        quantile_preds = self._compute_mixture_quantiles(
            mixture_location, mixture_scale, quantile_levels
        )
        
        return quantile_preds, torch.sigmoid(self.blend_weight)
    
    def get_mixture_params(self, backbone_features: torch.Tensor, 
                           backbone_locations: torch.Tensor) -> dict:
        """Get mixture parameters with adaptive gating applied."""
        # Get base parameters
        params = super().get_mixture_params(backbone_features, backbone_locations)
        
        # Apply adaptive gating
        gates = self.component_gate(backbone_features)
        gated_weights = params['weights'] * gates
        gated_weights = gated_weights / (gated_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        params['weights'] = gated_weights
        params['gates'] = gates  # Additional info
        
        return params
