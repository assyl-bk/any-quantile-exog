"""
Multi-Head Quantile Network (MQ-RNN Style)

Research Foundation:
Wen et al. (Amazon, NeurIPS Workshop 2017) - "A Multi-Horizon Quantile Recurrent Forecaster"
- Won GEFCom2014 Electricity Price: QL = 2.63 (beat official winner's 2.72)
- Production deployment at Amazon for large-scale demand forecasting

Key Innovation:
Shared backbone with quantile-specific output heads. Each quantile learns
completely separate transformations, allowing extreme quantiles to specialize
in different patterns (e.g., lower quantiles learn base patterns, upper quantiles
learn peak patterns).

Architecture:
    h = Backbone(x)                    # Shared encoder
    Q̂(τk) = MLPk(h)                    # Separate head per quantile
"""

import torch
import torch.nn as nn
from typing import List, Optional


class MultiHeadQuantileNBEATS(nn.Module):
    """
    Multi-Head Quantile Network with N-BEATS Backbone
    
    Shared N-BEATS backbone with separate MLP heads for each quantile level.
    This allows different quantiles to learn specialized patterns while sharing
    common temporal features.
    
    Args:
        backbone: Base N-BEATS architecture (without quantile conditioning)
        backbone_output_dim: Output dimension from backbone
        horizon: Forecast horizon length
        quantile_levels: List of quantile levels to predict
        head_hidden_dims: Hidden dimensions for quantile heads (default: [256, 128])
        dropout: Dropout rate in heads (default: 0.1)
        use_shared_first_layer: Share first layer across heads for efficiency
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        backbone_output_dim: int,
        horizon: int,
        quantile_levels: List[float] = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975],
        head_hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1,
        use_shared_first_layer: bool = False,
    ):
        super().__init__()
        
        self.backbone = backbone
        self.backbone_output_dim = backbone_output_dim
        self.horizon = horizon
        self.quantile_levels = sorted(quantile_levels)  # Ensure ascending order
        self.num_quantiles = len(quantile_levels)
        self.use_shared_first_layer = use_shared_first_layer
        
        if use_shared_first_layer:
            # Shared first layer for computational efficiency
            self.shared_layer = nn.Sequential(
                nn.Linear(backbone_output_dim, head_hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            
            # Quantile-specific output layers
            self.quantile_outputs = nn.ModuleDict({
                f'q_{int(q*1000):04d}': self._create_output_head(
                    head_hidden_dims[0], 
                    head_hidden_dims[1:], 
                    horizon,
                    dropout
                )
                for q in quantile_levels
            })
        else:
            # Completely separate heads for maximum flexibility
            self.quantile_heads = nn.ModuleDict({
                f'q_{int(q*1000):04d}': self._create_full_head(
                    backbone_output_dim,
                    head_hidden_dims,
                    horizon,
                    dropout
                )
                for q in quantile_levels
            })
    
    def _create_full_head(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float
    ) -> nn.Module:
        """Create a complete MLP head for a quantile."""
        layers = []
        
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        
        # Final output layer
        layers.append(nn.Linear(dims[-1], output_dim))
        
        return nn.Sequential(*layers)
    
    def _create_output_head(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float
    ) -> nn.Module:
        """Create output layers (after shared first layer)."""
        if not hidden_dims:
            return nn.Linear(input_dim, output_dim)
        
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        
        layers.append(nn.Linear(dims[-1], output_dim))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, q: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through multi-head network.
        
        Args:
            x: [B, T] input time series
            q: [B, Q] or [B, 1, Q] quantile levels
               - If None or matches self.quantile_levels: use fixed heads directly
               - If different: interpolate between fixed heads
            
        Returns:
            quantiles: [B, H, Q] predicted quantiles
        """
        B = x.shape[0]
        
        # Shared backbone
        features = self.backbone(x)  # [B, D] or [B, H, 1] - need to handle both
        
        # If backbone outputs [B, H, 1], take the mean or last
        if features.dim() == 3:
            features = features.mean(dim=1)  # [B, D]
        
        # Generate predictions for all fixed quantile levels
        if self.use_shared_first_layer:
            shared_features = self.shared_layer(features)  # [B, hidden_dim]
            fixed_outputs = []
            for q_level in self.quantile_levels:
                key = f'q_{int(q_level*1000):04d}'
                out = self.quantile_outputs[key](shared_features)  # [B, H]
                fixed_outputs.append(out)
        else:
            fixed_outputs = []
            for q_level in self.quantile_levels:
                key = f'q_{int(q_level*1000):04d}'
                out = self.quantile_heads[key](features)  # [B, H]
                fixed_outputs.append(out)
        
        # Stack fixed predictions: [B, H, num_fixed_quantiles]
        fixed_quantiles = torch.stack(fixed_outputs, dim=-1)
        
        # Post-hoc monotonicity enforcement
        fixed_quantiles, _ = torch.sort(fixed_quantiles, dim=-1)
        
        # If no q provided or q matches fixed levels, return directly
        if q is None:
            return fixed_quantiles
        
        # Extract requested quantile levels
        if q.dim() == 3 and q.shape[1] == 1:
            q_levels = q[:, 0, :]  # [B, Q]
        elif q.dim() == 2:
            q_levels = q  # [B, Q]
        else:
            # Assume q.dim() == 3 with shape [B, H, Q] - take first horizon
            q_levels = q[:, 0, :]  # [B, Q]
        
        # Check if q matches our fixed levels (use first batch)
        requested_q =  q_levels[0].cpu().numpy() if q_levels.shape[0] > 0 else []
        if len(requested_q) == len(self.quantile_levels):
            import numpy as np
            if np.allclose(requested_q, self.quantile_levels, atol=1e-6):
                return fixed_quantiles
        
        # Interpolate to requested quantiles
        # This handles training with random quantiles
        interpolated = self._interpolate_quantiles(fixed_quantiles, q_levels)
        
        return interpolated
    
    def _interpolate_quantiles(
        self, 
        fixed_quantiles: torch.Tensor, 
        target_q_levels: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate between fixed quantile predictions to arbitrary levels.
        
        Args:
            fixed_quantiles: [B, H, Q_fixed] predictions at fixed levels
            target_q_levels: [B, Q_target] requested quantile levels
            
        Returns:
            [B, H, Q_target] interpolated predictions
        """
        B, H, Q_fixed = fixed_quantiles.shape
        Q_target = target_q_levels.shape[1]
        
        device = fixed_quantiles.device
        fixed_levels = torch.tensor(
            self.quantile_levels, 
            device=device, 
            dtype=fixed_quantiles.dtype
        )  # [Q_fixed]
        
        # For each target quantile, find surrounding fixed quantiles
        interpolated = []
        
        for b in range(B):
            batch_preds = []
            for q_idx in range(Q_target):
                target_q = target_q_levels[b, q_idx]
                
                # Find surrounding fixed quantiles
                if target_q <= fixed_levels[0]:
                    # Extrapolate below minimum
                    pred = fixed_quantiles[b, :, 0]
                elif target_q >= fixed_levels[-1]:
                    # Extrapolate above maximum
                    pred = fixed_quantiles[b, :, -1]
                else:
                    # Interpolate
                    # Find the two fixed quantiles surrounding target_q
                    idx_high = (fixed_levels >= target_q).nonzero()[0, 0]
                    idx_low = idx_high - 1
                    
                    q_low = fixed_levels[idx_low]
                    q_high = fixed_levels[idx_high]
                    
                    # Linear interpolation weight
                    weight = (target_q - q_low) / (q_high - q_low + 1e-8)
                    
                    # Interpolate predictions
                    pred = (1 - weight) * fixed_quantiles[b, :, idx_low] + \
                           weight * fixed_quantiles[b, :, idx_high]
                
                batch_preds.append(pred)
            
            # Stack predictions for this batch: [H, Q_target]
            batch_preds = torch.stack(batch_preds, dim=-1)
            interpolated.append(batch_preds)
        
        # Stack all batches: [B, H, Q_target]
        result = torch.stack(interpolated, dim=0)
        
        # Ensure monotonicity
        result, _ = torch.sort(result, dim=-1)
        
        return result
    
    def get_quantile_head(self, quantile_level: float) -> nn.Module:
        """Get specific head for analysis/fine-tuning."""
        key = f'q_{int(quantile_level*1000):04d}'
        if self.use_shared_first_layer:
            return self.quantile_outputs[key]
        return self.quantile_heads[key]


class MultiHeadNBEATSWrapper(nn.Module):
    """
    Wrapper to integrate Multi-Head architecture with existing N-BEATS modules.
    
    Takes any NBEATS variant (standard, FiLM, etc.) and wraps it with
    multiple quantile-specific heads.
    
    Args:
        nbeats_config: Configuration dict for base N-BEATS
        quantile_levels: List of quantile levels
        head_hidden_dims: Hidden dimensions for heads
        dropout: Dropout in heads
        use_shared_first_layer: Share first layer
    """
    
    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        layer_width: int,
        size_in: int,
        size_out: int,
        quantile_levels: List[float] = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975],
        head_hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1,
        use_shared_first_layer: bool = False,
        share: bool = False,
    ):
        super().__init__()
        
        from modules.nbeats import NBEATS, NbeatsBlock
        
        # Create standard NBEATS backbone (no quantile conditioning)
        self.backbone = NBEATS(
            num_blocks=num_blocks,
            num_layers=num_layers,
            layer_width=layer_width,
            share=share,
            size_in=size_in,
            size_out=size_out,
            block_class=NbeatsBlock
        )
        
        self.size_out = size_out
        self.quantile_levels = sorted(quantile_levels)
        
        # Multi-head output
        self.multi_head = MultiHeadQuantileNBEATS(
            backbone=self.backbone,
            backbone_output_dim=size_out,  # NBEATS outputs [B, H, 1]
            horizon=size_out,
            quantile_levels=quantile_levels,
            head_hidden_dims=head_hidden_dims,
            dropout=dropout,
            use_shared_first_layer=use_shared_first_layer,
        )
    
    def forward(self, x: torch.Tensor, q: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T] input history
            q: [B, Q] quantile levels (optional, ignored)
            
        Returns:
            [B, H, Q] predicted quantiles
        """
        return self.multi_head(x, q)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing Multi-Head Quantile Network")
    print("=" * 60)
    
    # Test parameters
    batch_size = 4
    history_len = 168
    horizon = 48
    hidden_dim = 512
    quantiles = [0.1, 0.5, 0.9]
    
    # Mock backbone
    class MockBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(history_len, hidden_dim)
        
        def forward(self, x):
            return self.fc(x)
    
    backbone = MockBackbone()
    
    # Test 1: Basic multi-head
    print("\nTest 1: Basic Multi-Head Network")
    print("-" * 60)
    
    model = MultiHeadQuantileNBEATS(
        backbone=backbone,
        backbone_output_dim=hidden_dim,
        horizon=horizon,
        quantile_levels=quantiles,
        head_hidden_dims=[256, 128],
        use_shared_first_layer=False,
    )
    
    x = torch.randn(batch_size, history_len)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, {horizon}, {len(quantiles)})")
    assert output.shape == (batch_size, horizon, len(quantiles))
    print("✓ Basic multi-head works!")
    
    # Test 2: Shared first layer
    print("\nTest 2: Shared First Layer")
    print("-" * 60)
    
    model_shared = MultiHeadQuantileNBEATS(
        backbone=backbone,
        backbone_output_dim=hidden_dim,
        horizon=horizon,
        quantile_levels=quantiles,
        head_hidden_dims=[256, 128],
        use_shared_first_layer=True,
    )
    
    output_shared = model_shared(x)
    print(f"Output shape: {output_shared.shape}")
    assert output_shared.shape == (batch_size, horizon, len(quantiles))
    print("✓ Shared layer variant works!")
    
    # Test 3: Monotonicity
    print("\nTest 3: Monotonicity Check")
    print("-" * 60)
    
    # Check if quantiles are ordered
    diffs = output[:, :, 1:] - output[:, :, :-1]
    is_monotonic = (diffs >= 0).all()
    print(f"All quantiles monotonic: {is_monotonic}")
    assert is_monotonic, "Quantiles not monotonically increasing!"
    print("✓ Monotonicity enforced!")
    
    # Test 4: Gradients
    print("\nTest 4: Gradient Flow")
    print("-" * 60)
    
    x_grad = x.clone().requires_grad_(True)
    output_grad = model(x_grad)
    loss = output_grad.mean()
    loss.backward()
    
    print(f"Input gradient: mean={x_grad.grad.mean():.4f}, std={x_grad.grad.std():.4f}")
    print("✓ Gradients flow correctly!")
    
    # Test 5: Parameter count
    print("\nTest 5: Parameter Count Comparison")
    print("-" * 60)
    
    params_separate = sum(p.numel() for p in model.parameters())
    params_shared = sum(p.numel() for p in model_shared.parameters())
    
    print(f"Separate heads: {params_separate:,} parameters")
    print(f"Shared first layer: {params_shared:,} parameters")
    print(f"Reduction: {(1 - params_shared/params_separate)*100:.1f}%")
    print("✓ Shared layer reduces parameters!")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
