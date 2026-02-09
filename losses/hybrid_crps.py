"""
Hybrid CRPS-Pinball Loss

Combines MQNLoss (which works - achieves CRPS=211) with CRPS sharpness penalty.

Key insight: MQNLoss already optimizes quantiles properly. We just add a 
sharpness penalty from CRPS to tighten intervals when appropriate.

Loss = α * MQNLoss + β * SharpnessPenalty

where SharpnessPenalty = E[|Q(τ_i) - Q(τ_j)|] penalizes excessive spread
"""

import torch
import torch.nn as nn
from typing import Optional


class HybridCRPSPinballLoss(nn.Module):
    """
    Hybrid Loss: MQNLoss + CRPS Sharpness Penalty
    
    MQNLoss ensures proper quantile calibration (coverage ≈ 0.95)
    CRPS penalty encourages tighter intervals when appropriate
    
    Loss = α * MQNLoss + β * SharpnessPenalty
    
    Args:
        alpha: Weight for MQNLoss (default: 1.0)
        beta: Weight for sharpness penalty (default: 0.1)
        reduction: 'mean' or 'sum'
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    
    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        q: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            preds: [B, H, Q] predicted quantiles
            targets: [B, H] true values
            q: [B, Q] or [B, 1, Q] quantile levels
            
        Returns:
            Hybrid loss value
        """
        # Reshape for broadcasting
        if targets.dim() != preds.dim():
            targets = targets.unsqueeze(-1)  # [B, H, 1]
        
        if q.dim() == 3 and q.shape[1] == 1:
            tau = q  # [B, 1, Q]
        elif q.dim() == 2:
            tau = q.unsqueeze(1)  # [B, 1, Q]
        else:
            tau = q
        
        # 1. MQNLoss (normalized pinball) - ensures proper quantile calibration
        denominator = targets.clone()
        denominator[denominator.abs() < 1] = 1
        
        pinball = torch.where(
            targets >= preds,
            tau * (targets - preds) / denominator,
            (1 - tau) * (preds - targets) / denominator
        )
        mqn_loss = pinball.mean()
        
        # 2. Sharpness penalty - encourages tighter intervals
        # E[|Q(τ_i) - Q(τ_j)|] - penalizes excessive spread
        q_i = preds.unsqueeze(-1)  # [B, H, Q, 1]
        q_j = preds.unsqueeze(-2)  # [B, H, 1, Q]
        pairwise_dist = torch.abs(q_i - q_j)  # [B, H, Q, Q]
        sharpness_penalty = pairwise_dist.mean()
        
        # Combined loss
        total_loss = self.alpha * mqn_loss + self.beta * sharpness_penalty
        
        return total_loss


class AdaptiveHybridCRPSLoss(nn.Module):
    """
    Adaptive Hybrid Loss with Annealing
    
    Starts with pure MQNLoss (β=0) for stable quantile learning,
    gradually increases sharpness penalty (β increases) to tighten intervals.
    
    Early epochs: Focus on calibration
    Late epochs: Add sharpness constraint
    
    Args:
        alpha: Weight for MQNLoss (default: 1.0)
        beta_initial: Starting sharpness weight (default: 0.0)
        beta_final: Final sharpness weight (default: 0.2)
        reduction: 'mean' or 'sum'
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta_initial: float = 0.0,
        beta_final: float = 0.2,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.beta_initial = beta_initial
        self.beta_final = beta_final
        self.reduction = reduction
        
        # Track epoch
        self.current_epoch = 0
        self.total_epochs = 1
        self.current_beta = beta_initial
    
    def set_epoch(self, epoch: int, total_epochs: int):
        """Update training progress for beta annealing."""
        self.current_epoch = epoch
        self.total_epochs = total_epochs
        
        # Linear annealing of beta
        progress = epoch / max(total_epochs, 1)
        self.current_beta = self.beta_initial + (self.beta_final - self.beta_initial) * progress
    
    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        q: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            preds: [B, H, Q] predicted quantiles
            targets: [B, H] true values
            q: [B, Q] or [B, 1, Q] quantile levels
            
        Returns:
            Adaptive hybrid loss value
        """
        # Reshape for broadcasting
        if targets.dim() != preds.dim():
            targets = targets.unsqueeze(-1)  # [B, H, 1]
        
        if q.dim() == 3 and q.shape[1] == 1:
            tau = q  # [B, 1, Q]
        elif q.dim() == 2:
            tau = q.unsqueeze(1)  # [B, 1, Q]
        else:
            tau = q
        
        # 1. MQNLoss (normalized pinball)
        denominator = targets.clone()
        denominator[denominator.abs() < 1] = 1
        
        pinball = torch.where(
            targets >= preds,
            tau * (targets - preds) / denominator,
            (1 - tau) * (preds - targets) / denominator
        )
        mqn_loss = pinball.mean()
        
        # 2. Sharpness penalty (weighted by current_beta)
        q_i = preds.unsqueeze(-1)  # [B, H, Q, 1]
        q_j = preds.unsqueeze(-2)  # [B, H, 1, Q]
        pairwise_dist = torch.abs(q_i - q_j)  # [B, H, Q, Q]
        sharpness_penalty = pairwise_dist.mean()
        
        # Combined loss with adaptive beta
        total_loss = self.alpha * mqn_loss + self.current_beta * sharpness_penalty
        
        return total_loss
    
    def get_current_beta(self) -> float:
        """Get current beta value."""
        return self.current_beta


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing Hybrid CRPS-Pinball Loss")
    print("=" * 60)
    
    # Test setup
    batch_size = 32
    horizon = 48
    num_quantiles = 9
    
    preds = torch.randn(batch_size, horizon, num_quantiles).sort(dim=-1)[0]
    targets = torch.randn(batch_size, horizon)
    tau = torch.linspace(0.1, 0.9, num_quantiles)
    q = tau.unsqueeze(0).expand(batch_size, -1)  # [B, Q]
    
    # Test 1: Hybrid Loss
    print("\nTest 1: Hybrid CRPS-Pinball Loss")
    print("-" * 60)
    
    hybrid_loss = HybridCRPSPinballLoss(alpha=1.0, beta=0.1)
    loss_hybrid = hybrid_loss(preds, targets, q)
    print(f"Loss value: {loss_hybrid.item():.4f}")
    print(f"✓ Hybrid loss works!")
    
    # Test 2: Adaptive Loss
    print("\nTest 2: Adaptive Hybrid Loss")
    print("-" * 60)
    
    adaptive_loss = AdaptiveHybridCRPSLoss(
        alpha=1.0,
        beta_initial=0.0,
        beta_final=0.2
    )
    
    print("Beta annealing schedule:")
    for epoch in [0, 5, 10, 14]:
        adaptive_loss.set_epoch(epoch, 15)
        loss_val = adaptive_loss(preds, targets, q)
        beta = adaptive_loss.get_current_beta()
        print(f"  Epoch {epoch:2d}: β={beta:.3f}, loss={loss_val.item():.4f}")
    
    print(f"✓ Adaptive loss works!")
    
    # Test 3: Gradients
    print("\nTest 3: Gradient Flow")
    print("-" * 60)
    
    preds_grad = preds.clone().requires_grad_(True)
    loss_grad = hybrid_loss(preds_grad, targets, q)
    loss_grad.backward()
    
    print(f"Gradient: mean={preds_grad.grad.mean():.4f}, std={preds_grad.grad.std():.4f}")
    print(f"✓ Gradients work!")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
