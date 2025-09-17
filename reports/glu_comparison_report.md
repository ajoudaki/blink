# GLU vs Standard Linear Layers Comparison Report

## Summary
**GLU architecture does NOT improve performance** over standard linear layers for this task.

## Test Results

### Baseline (Standard Linear Layers)
- **Best Configuration**: 5-layer deep narrow [384, 256, 128, 64, 32]
- **Accuracy**: 72.5%
- **Loss**: 0.597
- **Settings**: dropout=0.15, weight_decay=0.001, GELU activation, BatchNorm

### GLU Tests

| Configuration | Accuracy | Loss | Notes |
|--------------|----------|------|-------|
| GLU with best config (50 epochs) | 65.3% | 0.641 | -7.2% vs baseline |
| GLU with dropout=0.1 | 64.8% | 0.639 | -7.7% vs baseline |
| GLU with 100 epochs | 64.1% | 0.682 | Overfitting, worse performance |

## Analysis

1. **Performance Gap**: GLU consistently underperforms standard linear layers by 7-8%
2. **Overfitting**: Longer training (100 epochs) makes GLU performance worse
3. **Dropout Sensitivity**: Reducing dropout doesn't help GLU performance

## Conclusion

**Recommendation: Continue using standard linear layers**

The Gated Linear Units do not provide any improvement for this specific task. The standard linear layer architecture with:
- 5-layer deep narrow architecture [384, 256, 128, 64, 32]
- GELU activation
- BatchNorm
- Dropout=0.15
- Weight decay=0.001

Remains the best configuration with **72.5% accuracy** on the comparison task.

## Possible Reasons for GLU Underperformance

1. **Task Simplicity**: The comparison task may not require the gating mechanism
2. **Parameter Overhead**: GLU doubles parameters per layer, potentially causing overfitting
3. **Feature Space**: CLIP embeddings may already be sufficiently expressive

## Files Generated
- `configs/unified_best_glu.yaml` - GLU configuration file
- `test_glu_performance.py` - Testing script (not run due to manual tests)
- Multiple run directories with GLU results