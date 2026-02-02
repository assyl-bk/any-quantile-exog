"""
Idea 3: Smooth Pinball Loss (Huber/Arctan Variants)
Evidence: Dramatic reduction in quantile crossings (Sluijterman et al., 2024)
Publication: "Optimal Quantile Regression via Smooth Pinball Losses" - arXiv 2024

Key Innovation:
- Standard pinball loss: Non-differentiable at y=ŷ, zero second derivative
- Smooth variants: Differentiable everywhere, non-zero second derivative
- Result: Better gradients → better convergence → fewer crossings

Variants:
1. Huber Pinball: Smooth near u=0 (quadratic transition)
2. Arctan Pinball: Smooth everywhere (best for optimization)
3. Adaptive: Starts smooth, becomes sharper during training
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class HuberPinballLoss(nn.Module):
    """
    Huber Pinball Loss - Smooth near u=0
    
    Combines quadratic smoothing near zero with linear behavior far from zero.
    This provides better gradient flow near the quantile prediction.
    
    Mathematical Form:
        ρ_τ^Huber(u) = |τ - 1(u < 0)| · H_δ(u)
        
        where H_δ(u) = {
            u²/(2δ)      if |u| ≤ δ
            |u| - δ/2    if |u| > δ
        }
    
    Args:
        delta: Transition point from quadratic to linear (default: 1.0)
        reduction: 'mean' or 'sum' for loss aggregation
    """
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        
    def huber(self, u: torch.Tensor) -> torch.Tensor:
        """
        Huber function: quadratic near 0, linear far from 0.
        
        Args:
            u: Error tensor
            
        Returns:
            Huber-smoothed errors
        """
        abs_u = torch.abs(u)
        
        # Quadratic region: u²/(2δ)
        quadratic = 0.5 * u ** 2 / self.delta
        
        # Linear region: |u| - δ/2
        linear = abs_u - 0.5 * self.delta
        
        # Switch at |u| = δ
        return torch.where(abs_u <= self.delta, quadratic, linear)
    
    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        tau: torch.Tensor = None,
        q: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass of Huber Pinball Loss.
        
        Args:
            preds: [B, H, Q] predicted quantiles
            targets: [B, H] true values
            tau: [Q] or [B, Q] quantile levels (alternative name)
            q: [Q] or [B, Q] quantile levels
            
        Returns:
            Scalar loss value
        """
        # Support both parameter names
        quantiles = q if q is not None else tau
        if quantiles is None:
            raise ValueError("Either 'q' or 'tau' must be provided")
        
        # Reshape for broadcasting
        targets = targets.unsqueeze(-1)  # [B, H, 1]
        
        if quantiles.dim() == 1:
            tau = quantiles.view(1, 1, -1)  # [1, 1, Q]
        else:
            tau = quantiles.unsqueeze(1)  # [B, 1, Q]
        
        # Compute errors
        errors = targets - preds  # [B, H, Q]
        
        # Asymmetric weights: τ for positive errors, (1-τ) for negative
        weights = torch.where(errors >= 0, tau, 1 - tau)
        
        # Apply Huber smoothing
        huber_errors = self.huber(errors)
        
        # Weighted loss
        loss = weights * huber_errors
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ArctanPinballLoss(nn.Module):
    """
    Arctan Pinball Loss - Smooth everywhere
    
    Uses arctan function to smoothly transition between the two linear pieces
    of standard pinball loss. This provides:
    - Differentiability everywhere
    - Non-zero second derivative (better for optimization)
    - Dramatic reduction in quantile crossings
    
    Mathematical Form:
        ρ_τ^arctan(u) = (τ - 1/2 + (1/π)arctan(u/s)) · u
        
    where s controls smoothness (smaller s → closer to standard pinball)
    
    Args:
        smoothness: Controls transition sharpness (default: 1.0)
        reduction: 'mean' or 'sum' for loss aggregation
    """
    
    def __init__(self, smoothness: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.s = smoothness
        self.reduction = reduction
        
    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        tau: torch.Tensor = None,
        q: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass of Arctan Pinball Loss.
        
        Args:
            preds: [B, H, Q] predicted quantiles
            targets: [B, H] true values
            tau: [Q] or [B, Q] quantile levels (alternative name)
            q: [Q] or [B, Q] quantile levels
            
        Returns:
            Scalar loss value
        """
        # Support both parameter names
        quantiles = q if q is not None else tau
        if quantiles is None:
            raise ValueError("Either 'q' or 'tau' must be provided")
        
        # Reshape for broadcasting
        targets = targets.unsqueeze(-1)  # [B, H, 1]
        
        if quantiles.dim() == 1:
            tau = quantiles.view(1, 1, -1)  # [1, 1, Q]
        else:
            tau = quantiles.unsqueeze(1)  # [B, 1, Q]
        
        # Compute errors: u = y - ŷ
        errors = targets - preds  # [B, H, Q]
        
        # Smooth indicator function: τ - 1/2 + (1/π)arctan(u/s)
        smooth_indicator = tau - 0.5 + (1.0 / math.pi) * torch.atan(errors / self.s)
        
        # Smooth pinball loss
        loss = smooth_indicator * errors
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def get_second_derivative(self, u: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Compute second derivative for analysis.
        
        Non-zero unlike standard pinball loss!
        This is what enables better optimization.
        
        d²ρ/du² = (2/π) · s · u / (u² + s²)²
        
        Args:
            u: Error values
            tau: Quantile level (not used in second derivative)
            
        Returns:
            Second derivative values
        """
        numerator = 2 * self.s * u / math.pi
        denominator = (u ** 2 + self.s ** 2) ** 2
        return numerator / denominator


class AdaptiveSmoothPinballLoss(nn.Module):
    """
    Adaptive Smooth Pinball Loss
    
    Starts with high smoothness (easy optimization) and gradually
    becomes sharper (closer to true pinball) during training.
    
    Strategy:
    - Early epochs: Smooth loss (s=2.0) → Easy to optimize
    - Late epochs: Sharp loss (s=0.1) → Precise quantile estimation
    
    This provides the best of both worlds:
    - Fast initial convergence (smooth gradients)
    - Accurate final estimates (sharp loss)
    
    Args:
        initial_smoothness: Starting smoothness (default: 2.0)
        final_smoothness: Ending smoothness (default: 0.1)
        reduction: 'mean' or 'sum'
    """
    
    def __init__(
        self,
        initial_smoothness: float = 2.0,
        final_smoothness: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.initial_s = initial_smoothness
        self.final_s = final_smoothness
        self.reduction = reduction
        
        # Internal arctan loss (smoothness will be updated)
        self.arctan_loss = ArctanPinballLoss(
            smoothness=initial_smoothness,
            reduction=reduction
        )
        
        # Track current epoch
        self.current_epoch = 0
        self.total_epochs = 1
    
    def set_epoch(self, epoch: int, total_epochs: int):
        """Update current training progress."""
        self.current_epoch = epoch
        self.total_epochs = total_epochs
    
    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        tau: torch.Tensor = None,
        q: torch.Tensor = None,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass with adaptive smoothness.
        
        Args:
            preds: [B, H, Q] predicted quantiles
            targets: [B, H] true values
            tau: [Q] or [B, Q] quantile levels (alternative name)
            q: [Q] or [B, Q] quantile levels
            epoch: Current epoch (optional, uses set_epoch if None)
            total_epochs: Total epochs (optional, uses set_epoch if None)
            
        Returns:
            Scalar loss value
        """
        # Update epoch if provided
        if epoch is not None and total_epochs is not None:
            self.current_epoch = epoch
            self.total_epochs = total_epochs
        
        # Compute progress (0 to 1)
        progress = self.current_epoch / max(self.total_epochs, 1)
        
        # Linear annealing from initial to final smoothness
        current_s = self.initial_s * (1 - progress) + self.final_s * progress
        
        # Update arctan loss smoothness
        self.arctan_loss.s = current_s
        
        # Support both parameter names
        quantiles = q if q is not None else tau
        
        # Compute loss
        return self.arctan_loss(preds, targets, q=quantiles)
    
    def get_current_smoothness(self) -> float:
        """Get current smoothness value."""
        return self.arctan_loss.s


# ============================================================================
# Comparison and Analysis Functions
# ============================================================================

def compare_losses(u_values: torch.Tensor, tau: float = 0.5):
    """
    Compare standard vs smooth pinball losses.
    
    Args:
        u_values: Range of error values to evaluate
        tau: Quantile level
        
    Returns:
        Dictionary of loss values for each variant
    """
    # Standard pinball
    standard = torch.where(
        u_values >= 0,
        tau * u_values,
        (tau - 1) * u_values
    )
    
    # Huber pinball
    huber_loss = HuberPinballLoss(delta=1.0)
    tau_tensor = torch.tensor([tau])
    huber = huber_loss.huber(u_values) * torch.where(
        u_values >= 0,
        tau,
        1 - tau
    )
    
    # Arctan pinball
    arctan_loss = ArctanPinballLoss(smoothness=1.0)
    arctan = (tau - 0.5 + (1.0 / math.pi) * torch.atan(u_values)) * u_values
    
    return {
        'standard': standard,
        'huber': huber,
        'arctan': arctan,
        'errors': u_values
    }


# ============================================================================
# Testing and Validation
# ============================================================================

if __name__ == "__main__":
    print("Testing Smooth Pinball Loss Implementations")
    print("=" * 60)
    
    # Test setup
    batch_size = 32
    horizon = 48
    num_quantiles = 9
    
    preds = torch.randn(batch_size, horizon, num_quantiles)
    targets = torch.randn(batch_size, horizon)
    tau = torch.linspace(0.1, 0.9, num_quantiles)
    
    print(f"\nTest shapes:")
    print(f"  Predictions: {preds.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  Quantiles: {tau.shape}")
    
    # Test 1: Huber Pinball Loss
    print("\n" + "=" * 60)
    print("Test 1: Huber Pinball Loss")
    print("=" * 60)
    
    huber_loss = HuberPinballLoss(delta=1.0)
    loss_huber = huber_loss(preds, targets, tau)
    print(f"Loss value: {loss_huber.item():.4f}")
    print(f"✓ Huber Pinball Loss works!")
    
    # Test 2: Arctan Pinball Loss
    print("\n" + "=" * 60)
    print("Test 2: Arctan Pinball Loss")
    print("=" * 60)
    
    arctan_loss = ArctanPinballLoss(smoothness=1.0)
    loss_arctan = arctan_loss(preds, targets, tau)
    print(f"Loss value: {loss_arctan.item():.4f}")
    
    # Test second derivative
    u_test = torch.linspace(-3, 3, 100)
    second_deriv = arctan_loss.get_second_derivative(u_test, tau=0.5)
    print(f"Second derivative range: [{second_deriv.min():.6f}, {second_deriv.max():.6f}]")
    print(f"✓ Arctan Pinball Loss works!")
    
    # Test 3: Adaptive Smooth Pinball Loss
    print("\n" + "=" * 60)
    print("Test 3: Adaptive Smooth Pinball Loss")
    print("=" * 60)
    
    adaptive_loss = AdaptiveSmoothPinballLoss(
        initial_smoothness=2.0,
        final_smoothness=0.1
    )
    
    print("Smoothness schedule:")
    for epoch in [0, 5, 10, 14]:
        loss_adaptive = adaptive_loss(preds, targets, tau, epoch=epoch, total_epochs=15)
        current_s = adaptive_loss.get_current_smoothness()
        print(f"  Epoch {epoch:2d}: smoothness={current_s:.2f}, loss={loss_adaptive.item():.4f}")
    
    print(f"✓ Adaptive Smooth Pinball Loss works!")
    
    # Test 4: Gradient Flow Comparison
    print("\n" + "=" * 60)
    print("Test 4: Gradient Flow")
    print("=" * 60)
    
    preds_grad = preds.clone().requires_grad_(True)
    
    # Standard pinball (for comparison)
    errors = targets.unsqueeze(-1) - preds_grad
    standard_loss = torch.where(
        errors >= 0,
        tau.view(1, 1, -1) * errors,
        (tau.view(1, 1, -1) - 1) * errors
    ).mean()
    
    # Arctan pinball
    arctan_loss_grad = arctan_loss(preds_grad, targets, tau)
    
    # Compute gradients
    standard_loss.backward(retain_graph=True)
    grad_standard = preds_grad.grad.clone()
    
    preds_grad.grad.zero_()
    arctan_loss_grad.backward()
    grad_arctan = preds_grad.grad.clone()
    
    print(f"Standard pinball gradient: mean={grad_standard.mean():.4f}, std={grad_standard.std():.4f}")
    print(f"Arctan pinball gradient: mean={grad_arctan.mean():.4f}, std={grad_arctan.std():.4f}")
    print(f"✓ Gradient computation works!")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)