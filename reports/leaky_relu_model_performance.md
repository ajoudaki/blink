# LeakyReLU Model Performance Report

**Date**: September 17, 2025
**Model**: Unified BaseEncoder with LeakyReLU Activation
**Best Epoch**: 42 (Early Stopping with patience=10)
**Overall Test Accuracy**: 73.2%

## Model Configuration

- **Architecture**: Multi-layer MLP with user embeddings
- **Hidden Dimensions**: [384, 256, 128, 64, 32]
- **Activation**: LeakyReLU
- **Normalization**: BatchNorm
- **Dropout**: 0.15
- **User Embedding**: 32 dimensions
- **Optimizer**: AdamW (lr=0.001, weight_decay=0.001)
- **Batch Size**: 128
- **Training**: 200 max epochs with early stopping (patience=10, min_delta=0.001)

## Performance Summary

### Test Accuracy by Target

| Target | Accuracy | CE Loss | N Samples |
|--------|----------|---------|-----------|
| Attractive | 74.0% | 0.519 | 1617 |
| Smart | 72.3% | 0.536 | 1686 |
| Trustworthy | 73.2% | 0.522 | 1572 |

### Test Accuracy by User

| Idx | User ID | Total | N | Accuracy | Attr | Smart | Trust |
|-----|---------|-------|-----|----------|------|-------|-------|
|   0 | 4a5cde |  3701 | 1169 |    58.7% | 60.7% | 60.5% | 54.9% |
|   1 | 4a7d54 |  4588 | 1394 |    64.8% | 68.4% | 61.4% | 64.5% |
|   2 | 8a11e5 |  4602 | 1365 |    89.5% | 88.9% | 89.2% | 90.3% |
|   3 | 4a39cb |  2812 |  811 |    82.7% | 81.2% | 82.2% | 84.8% |
|   4 | 4ad536 |   148 |   45 |    52.4% | 38.9% | 58.3% | 60.0% |
|   5 | 4a6535 |   299 |   68 |    67.4% | 53.8% | 72.0% | 76.5% |

*Note: Only users with ≥10 test samples shown. User IDs are truncated to last 6 characters.*

## Key Findings

1. **Best Performing Activation**: LeakyReLU outperformed ReLU and GELU in comparative experiments, achieving the highest test accuracy of 73.2%.

2. **Target Performance**:
   - Attractive predictions perform best (74.0%)
   - Smart and Trustworthy predictions are similar (72.3% and 73.2%)

3. **User Variation**:
   - Significant performance variation across users (52.4% to 89.5%)
   - User 2 (8a11e5) shows consistently high accuracy across all targets (~90%)
   - Users with fewer labels (4, 5) show lower and more variable performance

4. **Model Stability**: The model converged at epoch 42 and showed stable performance with early stopping.

## Experimental Details

- **Dataset**: Binary comparison task across three targets (attractive, smart, trustworthy)
- **Split**: 80% train, 10% validation, 10% test
- **Random Seed**: 42
- **GPU**: Device 1
- **Framework**: PyTorch with CLIP embeddings (ViT-B/32)

## File Locations

- **Results**: `runs/comparison_unified_earlystop_20250917_105244/results.json`
- **Config**: `runs/comparison_unified_earlystop_20250917_105244/config.yaml`
- **Summary**: `runs/comparison_unified_earlystop_20250917_105244/summary.txt`
- **Training Log**: `leaky_relu_confirmation.log`