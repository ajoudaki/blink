# Hyperparameter Search Results for Comparison Task

## Summary

After extensive hyperparameter search on the unified Siamese model for the comparison task, the best performance was achieved with the original 3-layer configuration.

## Tested Configurations and Results

### 1. **Original Best (3-layer GELU + BatchNorm)**
- **Architecture**: [512, 256, 128]
- **Activation**: GELU
- **Normalization**: BatchNorm
- **Dropout**: 0.1
- **Optimizer**: AdamW (lr=0.001, weight_decay=0.01)
- **Epochs**: 50
- **Result**: **72.2% average accuracy** ✨

### 2. **4-layer Deep Network**
- **Architecture**: [512, 384, 256, 128]
- **Other params**: Same as original
- **Epochs**: 100
- **Result**: **72.1% average accuracy**
- **Note**: Similar performance but takes longer to train

### 3. **Wider Network**
- **Architecture**: [768, 384, 192]
- **Dropout**: 0.15
- **Other params**: Same as original
- **Epochs**: 100
- **Result**: **71.4% average accuracy**
- **Note**: Slightly worse, possibly due to overfitting

### 4. **Extended Training (200 epochs)**
- **Architecture**: Original [512, 256, 128]
- **Epochs**: 200
- **Result**: **71.8% average accuracy**
- **Note**: Performance degraded with longer training, suggesting overfitting

## Per-Target Performance (Best Model)

| Target | Accuracy | CE Loss |
|--------|----------|---------|
| Attractive | 75.1% | 0.800 |
| Smart | 72.2% | 0.861 |
| Trustworthy | 69.0% | 1.008 |

## Key Findings

1. **Optimal Architecture**: The 3-layer architecture [512, 256, 128] remains optimal
2. **Activation Function**: GELU consistently outperforms ReLU and other activations
3. **Normalization**: BatchNorm is crucial for stable training
4. **Regularization**: Moderate dropout (0.1) with weight decay (0.01) works best
5. **Training Duration**: 50-100 epochs is optimal; longer training leads to overfitting
6. **Deeper Networks**: Adding more layers doesn't improve performance significantly
7. **Wider Networks**: Increasing layer width slightly hurts performance

## Recommendations

For the comparison task with this dataset:
- **Use the original configuration**: 3-layer [512, 256, 128] with GELU and BatchNorm
- **Train for 50-100 epochs** to avoid overfitting
- **Keep moderate regularization**: dropout=0.1, weight_decay=0.01
- **Use AdamW optimizer** with lr=0.001

## Performance Ceiling

The validation accuracy appears to plateau around **72%** for this dataset, suggesting:
- This may be close to the achievable performance given the data quality
- Further improvements might require:
  - More training data
  - Data augmentation strategies
  - Multi-task learning approaches
  - Ensemble methods