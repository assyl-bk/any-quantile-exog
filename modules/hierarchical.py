import torch
import torch.nn as nn


class HierarchicalQuantilePredictor(nn.Module):
    """
    FIXED Two-stage hierarchical quantile prediction:

    Key fixes:
    1. Use backbone median/IQR as anchors (stable initialization)
    2. Predict RESIDUALS not absolute values
    3. Constrain scale to be close to backbone scale
    4. Use tanh for bounded offsets
    
    Stage 1: Predict location (median) and scale (IQR or std)
    Stage 2: Predict quantile offsets conditioned on location/scale

    This ensures extreme quantiles are coherent with central tendency.
    """

    def __init__(self, backbone, size_in, size_out, layer_width):
        super().__init__()

        self.backbone = backbone
        self.size_out = size_out

        # FIX: Smaller heads with dropout for stability
        # Stage 1: Predict median OFFSET (residual from backbone median)
        self.location_head = nn.Sequential(
            nn.Linear(layer_width, layer_width // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(layer_width // 4, size_out)
        )

        # Stage 1: Predict scale MULTIPLIER (relative to backbone IQR)
        # FIXED: Wider range for better coverage
        self.scale_head = nn.Sequential(
            nn.Linear(layer_width, layer_width // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(layer_width // 4, size_out),
            nn.Sigmoid()  # Output in [0, 1], scale will be 0.8 + 2.2*output (range 0.8-3x)
        )

        # Stage 2: Quantile offset predictor with MONOTONICITY GUARANTEE
        # +1 corresponds to the quantile level q
        # CRITICAL FIX: Use Softplus to ensure positive increments
        self.offset_net = nn.Sequential(
            nn.Linear(layer_width + 1, layer_width // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(layer_width // 2, size_out),
            nn.Softplus()  # POSITIVE outputs for monotonicity
        )

    def forward(self, x, q):
        """
        Args:
            x: Tensor of shape [B, T] — input history (normalized)
            q: Tensor of shape [B, Q] — quantile levels

        Returns:
            Tensor of shape [B, H, Q] — quantile predictions
        """
        
        # FIX: Get backbone anchors for stability
        q_median = torch.ones(x.shape[0], 1).to(x) * 0.5
        q_25 = torch.ones(x.shape[0], 1).to(x) * 0.25
        q_75 = torch.ones(x.shape[0], 1).to(x) * 0.75
        
        backbone_median = self.backbone(x, q_median).squeeze(-1)  # [B, H]
        backbone_q25 = self.backbone(x, q_25).squeeze(-1)
        backbone_q75 = self.backbone(x, q_75).squeeze(-1)
        backbone_iqr = torch.abs(backbone_q75 - backbone_q25).clamp(min=1e-4)  # [B, H]

        # Backbone feature extraction
        features = self.backbone.encode(x)  # [B, W]

        # Stage 1: Location and scale (as RESIDUALS)
        median_offset = self.location_head(features)  # [B, H]
        median = backbone_median + 0.05 * median_offset  # Very small residual (was 0.1)
        
        scale_multiplier = self.scale_head(features)  # [B, H] in [0, 1]
        # FIXED: Wider scale range for better coverage (0.8x to 3x IQR)
        scale = backbone_iqr * (0.8 + 2.2 * scale_multiplier)  # Range [0.8*IQR, 3*IQR]
        scale = torch.clamp(scale, min=1e-4)  # Prevent collapse

        # Stage 2: Quantile-specific offsets
        Q = q.shape[-1]

        features_expanded = features.unsqueeze(1).expand(-1, Q, -1)  # [B, Q, W]
        q_expanded = q.unsqueeze(-1)                                 # [B, Q, 1]

        offset_input = torch.cat([features_expanded, q_expanded], dim=-1)
        raw_offsets = self.offset_net(offset_input)  # [B, Q, H] - POSITIVE values
        raw_offsets = raw_offsets.transpose(1, 2)    # [B, H, Q]

        # MONOTONICITY GUARANTEE: Convert to cumulative offsets
        # For sorted quantiles q1 < q2 < ... < qn, ensure pred1 < pred2 < ... < predn
        # Method: cumsum of positive increments
        cumulative_offsets = torch.cumsum(raw_offsets, dim=-1)  # [B, H, Q]
        
        # Center around median by subtracting median offset
        median_idx = Q // 2
        median_offset = cumulative_offsets[..., median_idx:median_idx+1]  # [B, H, 1]
        centered_offsets = cumulative_offsets - median_offset  # [B, H, Q]
        
        # Scale offsets for better tail coverage
        # Amplify based on distance from median quantile
        q_dist_from_median = torch.abs(q - 0.5).unsqueeze(1)  # [B, 1, Q]
        scale_factor = 1.0 + 2.0 * q_dist_from_median  # Range [1.0, 2.0]
        
        predictions = (
            median.unsqueeze(-1)
            + scale.unsqueeze(-1) * centered_offsets * scale_factor
        )

        return predictions
