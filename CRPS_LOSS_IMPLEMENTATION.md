# Direct CRPS Loss Optimization

## Overview

This implementation directly optimizes the **Continuous Ranked Probability Score (CRPS)** instead of using pinball loss as a proxy. This is a significant innovation because:

1. **Optimize what you measure**: CRPS is the actual evaluation metric
2. **Built-in sharpness penalty**: The energy score formulation naturally balances calibration and sharpness
3. **Published evidence**: 24% improvement over cross-entropy baseline (Marchesoni-Acland et al., IEEE 2024)

## Research Foundation

**Publication:**
Marchesoni-Acland et al. (2024). "Differentiable Histogram-Based CRPS for Probabilistic Forecasting." IEEE Conference on AI.

**Key Finding:**
Authors showed 24% test CRPS improvement by ONLY changing the loss function from cross-entropy to differentiable CRPS. Architecture remained unchanged.

## Mathematical Foundation

### CRPS Definition

CRPS measures the distance between the predictive CDF and the empirical CDF:

```
CRPS(F, y) = ∫_{-∞}^{∞} (F(x) - 𝟙(x ≥ y))² dx
```

### Energy Score Formulation (Differentiable)

The key insight is the energy score representation:

```
CRPS(F, y) = E_{X~F}[|X - y|] - ½·E_{X,X'~F}[|X - X'|]
```

where X and X' are independent samples from the predictive distribution F.

**Why this works:**
- **Term 1**: Expected error (low when predictions are close to truth)
- **Term 2**: Expected pairwise distance (penalizes wide intervals)
- **Balance**: Automatically balances accuracy and sharpness

### Quantile Approximation

For quantile forecasts Q̂(τ₁), ..., Q̂(τₖ):

```
CRPS ≈ (1/K)Σₖ|Q̂(τₖ) - y| - (1/2K²)Σₖ,ⱼ|Q̂(τₖ) - Q̂(τⱼ)|
```

## Implementation Variants

### 1. Basic CRPS Loss (Energy Score)

**File:** `losses/crps.py` → `CRPSLoss`

**Formula:**
```python
term1 = mean(|predictions - y|)  # Over quantiles
term2 = 0.5 * mean(|pred_i - pred_j|)  # Over all pairs
crps = term1 - term2
```

**Use case:** Standard approach, direct energy score

**Config:** `config/nbeatsaq-crps-loss.yaml`

### 2. Weighted CRPS Loss (Integration)

**File:** `losses/crps.py` → `WeightedCRPSLoss`

**Formula:**
```
CRPS = 2 * ∫₀¹ ρ_τ(y - Q(τ)) dτ
```

Approximated using trapezoidal integration with proper weights.

**Use case:** More accurate integral approximation

**Config:** `config/nbeatsaq-crps-weighted.yaml`

### 3. Smooth CRPS Loss (Better Gradients)

**File:** `losses/crps.py` → `SmoothCRPSLoss`

**Formula:** Same as basic CRPS but replaces |x| with softabs(x):
```
softabs(x) = √(x² + β²) - β
```

**Use case:** Better gradient flow, especially near zero

**Config:** `config/nbeatsaq-crps-smooth.yaml`

## Configuration Files

### Basic CRPS (Recommended Starting Point)

```bash
python run.py --config=config/nbeatsaq-crps-loss.yaml
```

**Key settings:**
- Loss: `CRPSLoss` (energy score formulation)
- Backbone: `NBEATSAQFILM` (same as baseline)
- Architecture: 30 blocks × 3 layers × 1024 width
- Batch size: 1024
- Learning rate: 0.0005

### Weighted CRPS (Most Accurate Integration)

```bash
python run.py --config=config/nbeatsaq-crps-weighted.yaml
```

**Key settings:**
- Loss: `WeightedCRPSLoss` (trapezoidal integration)
- Uses 99 quantile levels by default
- More computationally expensive but theoretically more accurate

### Smooth CRPS (Best Gradients)

```bash
python run.py --config=config/nbeatsaq-crps-smooth.yaml
```

**Key settings:**
- Loss: `SmoothCRPSLoss` (smooth absolute value)
- Beta: 0.1 (smoothing parameter)
- Better gradient flow for optimization

## Expected Results

Based on published evidence:

| Metric | Baseline (Pinball) | Expected (CRPS) | Improvement |
|--------|-------------------|-----------------|-------------|
| CRPS | 211.22 | 185-195 | -8% to -12% |
| Coverage | 0.95 | 0.95 | Maintained |
| MAE | ~300 | ~285-295 | Improved |

**Note:** Published work showed 24% improvement over cross-entropy. We expect 8-12% over pinball (pinball is already better than cross-entropy).

## Why This Works

### 1. Direct Optimization
Standard approach: Optimize pinball loss → hope CRPS is good
CRPS loss: Optimize CRPS directly → guaranteed to improve the metric

### 2. Automatic Sharpness Control
The second term `E[|X - X'|]` penalizes overly wide intervals:
- Wide intervals → Large pairwise distances → High term2 → High loss
- Sharp intervals (when appropriate) → Small pairwise distances → Low term2 → Low loss

### 3. Natural Balance
CRPS balances:
- **Calibration** (term1: predictions close to truth)
- **Sharpness** (term2: tight intervals when appropriate)

No manual tuning of monotonicity weights or other hyperparameters needed!

## Advantages Over Pinball Loss

| Aspect | Pinball Loss | CRPS Loss |
|--------|-------------|-----------|
| **Metric alignment** | Proxy for CRPS | Direct CRPS optimization |
| **Sharpness** | Implicit | Explicit penalty |
| **Calibration** | Via quantile levels | Via term1 |
| **Balance** | Manual tuning | Automatic |
| **Published evidence** | Standard | 24% improvement |

## Implementation Details

### Loss Function Arguments

All CRPS losses accept:
```python
loss(
    quantile_preds: torch.Tensor,  # [B, H, Q] predicted quantiles
    y_true: torch.Tensor,          # [B, H] true values
    q: Optional[torch.Tensor] = None  # [B, Q] quantile levels (optional)
)
```

### Computational Complexity

| Loss | Complexity | Memory |
|------|-----------|--------|
| Basic CRPS | O(B·H·Q²) | O(B·H·Q²) |
| Weighted CRPS | O(B·H·Q) | O(B·H·Q) |
| Smooth CRPS | O(B·H·Q²) | O(B·H·Q²) |

Where B=batch, H=horizon, Q=quantiles

### Gradient Properties

1. **Basic CRPS**: Standard gradients, discontinuous at 0
2. **Weighted CRPS**: Similar to pinball, standard properties
3. **Smooth CRPS**: Smooth everywhere, better for optimization

## Testing

Run the built-in tests:
```bash
cd losses
python crps.py
```

Expected output:
```
✓ Basic CRPS Loss works!
✓ Weighted CRPS Loss works!
✓ Smooth CRPS Loss works!
✓ Gradient computation works!
```

## Usage Examples

### Quick Start (Basic CRPS)

```bash
# Train with basic CRPS loss (recommended)
python run.py --config=config/nbeatsaq-crps-loss.yaml

# Expected: CRPS ~185-195 (vs 211 baseline)
```

### Try All Variants

```bash
# 1. Basic energy score
python run.py --config=config/nbeatsaq-crps-loss.yaml

# 2. Weighted integration
python run.py --config=config/nbeatsaq-crps-weighted.yaml

# 3. Smooth gradients
python run.py --config=config/nbeatsaq-crps-smooth.yaml
```

### Custom Configuration

```yaml
model:
  loss:
    _target_: losses.CRPSLoss  # or WeightedCRPSLoss, SmoothCRPSLoss
    reduction: mean
    # For SmoothCRPSLoss only:
    # beta: 0.1
```

## Troubleshooting

### Issue: Loss is negative

**Expected!** CRPS = term1 - term2, and term2 can make it negative. The absolute value doesn't matter; what matters is that lower loss = better predictions.

### Issue: High memory usage with Basic CRPS

**Solution:** Use Weighted CRPS instead, which has O(Q) instead of O(Q²) memory.

### Issue: Training unstable

**Solution:** Try Smooth CRPS with appropriate beta (0.01-0.5).

## Comparison with Other Approaches

### vs. Pinball Loss
- **CRPS:** Direct optimization, automatic sharpness control
- **Pinball:** Proxy metric, no explicit sharpness control

### vs. Monotonicity Loss
- **CRPS:** No architectural constraints, natural ordering
- **Monotonicity:** Requires penalty weight tuning

### vs. Arctan Pinball
- **CRPS:** Optimizes evaluation metric directly
- **Arctan:** Smooth pinball (still a proxy)

## References

1. Marchesoni-Acland et al. (2024). "Differentiable Histogram-Based CRPS for Probabilistic Forecasting." IEEE Conference on AI.

2. Gneiting, T., & Raftery, A. E. (2007). "Strictly proper scoring rules, prediction, and estimation." Journal of the American Statistical Association.

3. Ziel, F., & Berk, K. (2019). "Multivariate forecasting evaluation: On sensitive and strictly proper scoring rules." arXiv preprint.

## Next Steps

1. **Run baseline experiment:**
   ```bash
   python run.py --config=config/nbeatsaq-crps-loss.yaml
   ```

2. **Compare with pinball baseline:**
   - Pinball baseline: CRPS = 211.22
   - Expected CRPS result: 185-195
   - Target improvement: 8-12%

3. **Try variants if needed:**
   - If training unstable: Use `SmoothCRPSLoss`
   - If want best accuracy: Use `WeightedCRPSLoss`
   - If want speed: Use `CRPSLoss`

## Contact & Contribution

This implementation is based on published research. If you find issues or have improvements, please document them clearly with references to support the changes.
