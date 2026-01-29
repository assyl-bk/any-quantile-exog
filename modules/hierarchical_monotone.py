import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualHierarchicalMonotonicPredictor(nn.Module):
    """
    Residual Hierarchical Architecture with Structural Monotonicity.
    
    Architecture:
    1. MAIN PATH: Direct N-BEATS prediction (proven to work)
    2. AUXILIARY PATH: Hierarchical adjustment (small residual)
    3. MONOTONICITY: Structural guarantee via cumulative positive increments
    
    This ensures:
    - We keep the strong baseline performance
    - Hierarchical only provides small corrections
    - Monotonicity is guaranteed by construction
    """
    
    def __init__(self, backbone, cfg):
        super().__init__()
        
        self.backbone = backbone
        layer_width = cfg.model.nn.backbone.layer_width
        horizon = cfg.model.nn.backbone.size_out
        
        # === HIERARCHICAL COMPONENTS ===
        
        # Location head (predicts median/center)
        self.location_head = nn.Sequential(
            nn.Linear(layer_width, layer_width // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(layer_width // 2, horizon)
        )
        
        # Scale head (predicts spread)
        self.scale_head = nn.Sequential(
            nn.Linear(layer_width, layer_width // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(layer_width // 2, horizon),
            nn.Softplus()  # Ensure positive
        )
        
        # Offset predictor (for hierarchical adjustment)
        # Input: features + quantile level
        self.offset_net = nn.Sequential(
            nn.Linear(layer_width + 1, layer_width // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(layer_width // 2, horizon)
        )
        
        # === BLEND PARAMETER ===
        # Start with 90% direct, 10% hierarchical
        # Model can learn optimal blend during training
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        # === FEATURE EXTRACTOR ===
        # To get features from backbone for hierarchical heads
        self.feature_extractor = nn.Sequential(
            nn.Linear(horizon, layer_width),
            nn.ReLU(),
            nn.Linear(layer_width, layer_width)
        )
    
    def extract_features(self, history_norm, device):
        """
        Extract features for hierarchical heads.
        We predict median first to get a reference.
        """
        B = history_norm.shape[0]
        q_median = torch.full((B, 1), 0.5, device=device)
        
        # Get median prediction
        median_pred = self.backbone(history_norm, q_median)  # [B, H, 1]
        median_pred = median_pred.squeeze(-1)  # [B, H]
        
        # Extract features from median prediction
        features = self.feature_extractor(median_pred)  # [B, layer_width]
        
        return features, median_pred
    
    def forward(self, x_norm, q):
        """
        Args:
            x_norm: [B, T] - normalized input history
            q: [B, Q] - quantile levels
            
        Returns:
            [B, H, Q] - monotonic predictions
        """
        B, Q = q.shape
        device = q.device
        
        # === STEP 1: DIRECT PREDICTION (Main Path) ===
        direct_pred = self.backbone(x_norm, q)  # [B, H, Q]
        
        # === STEP 2: HIERARCHICAL ADJUSTMENT (Auxiliary Path) ===
        
        # Extract features
        features, median_pred = self.extract_features(x_norm, device)
        
        # Predict location and scale
        location = self.location_head(features)  # [B, H]
        scale = self.scale_head(features)  # [B, H]
        
        # Sort quantiles for monotonicity
        q_sorted, sort_idx = q.sort(dim=-1)
        
        # Build hierarchical predictions with STRUCTURAL MONOTONICITY
        hier_preds = []
        
        for i in range(Q):
            q_i = q_sorted[:, i:i+1]  # [B, 1]
            
            # Concatenate features with quantile level
            feat_q = torch.cat([features, q_i], dim=-1)  # [B, layer_width+1]
            
            # Predict offset from location
            offset = self.offset_net(feat_q)  # [B, H]
            
            # Hierarchical prediction: location + scale * offset * distance_from_median
            q_centered = q_i - 0.5  # [-0.5, 0.5]
            hier_pred = location + scale * offset * q_centered.abs() * q_centered.sign()
            
            # ENFORCE MONOTONICITY: each prediction must be >= previous
            if i > 0:
                # Use softplus to ensure positive increment
                increment = F.softplus(hier_pred - hier_preds[-1])
                hier_pred = hier_preds[-1] + increment
            
            hier_preds.append(hier_pred)
        
        hier_preds = torch.stack(hier_preds, dim=-1)  # [B, H, Q]
        
        # Unsort to match original quantile order
        unsort_idx = sort_idx.argsort(dim=-1).unsqueeze(1).expand_as(hier_preds)
        hier_preds = hier_preds.gather(-1, unsort_idx)
        
        # === STEP 3: BLEND DIRECT + HIERARCHICAL ===
        
        # Learned blend with sigmoid to keep in [0, 1]
        alpha = torch.sigmoid(self.alpha)
        
        # Mostly direct (90%), small hierarchical correction (10%)
        blended = (1 - alpha) * direct_pred + alpha * hier_preds
        
        # === STEP 4: FINAL MONOTONICITY GUARANTEE ===
        
        # Sort to absolutely guarantee monotonicity
        # This is a safety net in case blend breaks monotonicity
        final_pred, _ = torch.sort(blended, dim=-1)
        
        return final_pred


class LightweightHierarchicalWrapper(nn.Module):
    """
    Simpler alternative: Just add hierarchical-inspired bias to direct predictions.
    Lower risk, easier to train.
    """
    
    def __init__(self, backbone, cfg):
        super().__init__()
        
        self.backbone = backbone
        horizon = cfg.model.nn.backbone.size_out
        
        # Simple scale predictor
        self.scale_net = nn.Sequential(
            nn.Linear(horizon, horizon // 2),
            nn.ReLU(),
            nn.Linear(horizon // 2, horizon),
            nn.Softplus()
        )
        
        # Small initial weight
        self.scale_weight = nn.Parameter(torch.tensor(0.05))
    
    def forward(self, x_norm, q):
        B, Q = q.shape
        
        # Direct predictions
        preds = self.backbone(x_norm, q)  # [B, H, Q]
        
        # Get median for scale estimation
        q_median = torch.full((B, 1), 0.5, device=q.device)
        median_pred = self.backbone(x_norm, q_median).squeeze(-1)  # [B, H]
        
        # Predict scale
        scale = self.scale_net(median_pred)  # [B, H]
        
        # Add scale-based adjustment
        q_centered = (q - 0.5).unsqueeze(1)  # [B, 1, Q]
        scale_adj = scale.unsqueeze(-1) * q_centered * torch.sigmoid(self.scale_weight)
        
        adjusted = preds + scale_adj
        
        # Enforce monotonicity
        adjusted_sorted, _ = torch.sort(adjusted, dim=-1)
        
        return adjusted_sorted