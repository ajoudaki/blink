# Best Model Verification Report

## Configuration Verified
- **Config File**: `configs/unified_best_final.yaml`
- **Architecture**: [384, 256, 128, 64, 32] (5-layer deep narrow)
- **Dropout**: 0.15
- **Weight Decay**: 0.001
- **Learning Rate**: 0.001
- **Optimizer**: AdamW
- **Epochs**: 50
- **Batch Normalization**: Yes
- **Activation**: GELU

## Verification Results (3 Runs)

### Run 1
- **Average Accuracy**: 72.6%
- **Average CE Loss**: 0.573
- Per-target:
  - Attractive: 76.3%
  - Smart: 71.5%
  - Trustworthy: 70.1%

### Run 2
- **Average Accuracy**: 71.9%
- **Average CE Loss**: 0.572
- Per-target:
  - Attractive: 74.9%
  - Smart: 72.0%
  - Trustworthy: 68.9%

### Run 3
- **Average Accuracy**: 73.1%
- **Average CE Loss**: 0.556
- Per-target:
  - Attractive: 76.8%
  - Smart: 72.5%
  - Trustworthy: 70.0%

## Summary Statistics

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| **Accuracy** | **72.5%** | ±0.6% | 71.9% | 73.1% |
| **CE Loss** | **0.567** | ±0.009 | 0.556 | 0.573 |

### Per-Target Average Performance
- **Attractive**: 76.0% (±1.0%)
- **Smart**: 72.0% (±0.5%)
- **Trustworthy**: 69.7% (±0.6%)

## Conclusion

✅ **VERIFICATION SUCCESSFUL**

The best model configuration has been verified with consistent performance:
- **Mean accuracy: 72.5%** across 3 independent runs
- Low variance (±0.6%) indicates stable training
- Performance is consistent with our hyperparameter search results

### Key Observations:
1. The model performs best on **Attractive** (76.0%)
2. Moderate performance on **Smart** (72.0%)
3. Lowest performance on **Trustworthy** (69.7%)
4. The deep narrow architecture provides stable and reproducible results

### To Reproduce:
```bash
python train.py --config-name=unified_best_final
```

The configuration file `configs/unified_best_final.yaml` contains all the optimized hyperparameters and can be used to reliably reproduce these results.