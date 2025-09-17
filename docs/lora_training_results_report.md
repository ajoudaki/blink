# LoRA Fine-tuning Results Report

## Executive Summary

This report documents the results of LoRA (Low-Rank Adaptation) fine-tuning applied to CLIP ViT-B/32 for human perception prediction tasks. The model achieved a best validation accuracy of **72.42%** at epoch 12, slightly outperforming the traditional MLP approach (72.34%) while using significantly fewer trainable parameters (302K vs 450K).

## Configuration Details

### Model Architecture
- **Base Model**: CLIP ViT-B/32
- **LoRA Configuration**:
  - Rank: 4
  - Alpha: 1.0
  - Temperature: 0.15
- **Fine-tuning Mode**: Vision-only (visual prompts + LoRA adapters)
- **Visual Prompts**: 6 user-specific learnable tokens (768-dimensional)
- **Total LoRA Layers**: 48 adapter layers

### Training Hyperparameters
- **Optimizer**: AdamW
- **Learning Rate**: 0.0005
- **Weight Decay**: 0.01
- **Batch Size**: 64 (training), 128 (evaluation)
- **Max Epochs**: 20
- **Early Stopping**: Patience=5 (best model at epoch 12)
- **Dataset**: 6 users with ≥100 samples each
  - Train: 33,916 samples
  - Validation: 7,267 samples
  - Test: 7,267 samples

### Run Information
- **Run Directory**: `runs/lora_visual_prompts_ViT-B_32_20250917_142456`
- **Start Time**: 2025-09-17 14:24:56
- **Device**: CUDA (GPU 1)

## Table 1: General Performance with Per-Attribute Breakdown

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Attractive | Smart | Trustworthy |
|-------|------------|-----------|----------|---------|------------|-------|-------------|
| 1 | 0.6047 | 63.99% | 0.5701 | 66.37% | 67.90% | 63.54% | 67.57% |
| 2 | 0.5332 | 69.67% | 0.5413 | 68.65% | 70.59% | 66.67% | 68.63% |
| 3 | 0.4909 | 72.58% | 0.5302 | 69.95% | 72.46% | 66.46% | 70.80% |
| 4 | 0.4615 | 74.78% | 0.5257 | 70.25% | 72.63% | 67.55% | 70.47% |
| 5 | 0.4397 | 76.17% | 0.5224 | 69.88% | 71.73% | 67.85% | 69.98% |
| 6 | 0.4189 | 77.91% | 0.5294 | 70.73% | 73.16% | 68.78% | 70.18% |
| 7 | 0.3997 | 79.58% | 0.5284 | 71.50% | 73.77% | 69.84% | 70.84% |
| 8 | 0.3786 | 80.88% | 0.5381 | 71.09% | 73.56% | 69.12% | 70.51% |
| 9 | 0.3568 | 82.62% | 0.5501 | **71.82%** | 73.97% | 70.43% | 71.00% |
| 10 | 0.3312 | 84.42% | 0.5625 | 71.68% | 73.44% | 70.38% | 71.17% |
| 11 | 0.3073 | 85.94% | 0.5843 | 72.30% | 74.54% | 71.40% | 70.92% |
| **12** | **0.2827** | **87.64%** | **0.6018** | **72.42%** ✅ | **74.50%** | **71.91%** | **70.84%** |

## Table 2: Per-User Validation Accuracy Across All Epochs

| Epoch | User 1 (5fc6...) | User 2 (6039...) | User 3 (6036...) | User 4 (6015...) | User 5 (6037...) | User 6 (6048...) |
|-------|------------------|------------------|------------------|------------------|------------------|------------------|
| 1 | 80.34% | 59.62% | 48.83% | 77.94% | 68.10% | 70.69% |
| 2 | 84.42% | 61.16% | 49.14% | 81.54% | 68.10% | 68.97% |
| 3 | 85.19% | 62.21% | 50.06% | 84.23% | 75.86% | 67.24% |
| 4 | 86.68% | 62.07% | 49.69% | 84.65% | 71.55% | 67.24% |
| 5 | 86.92% | 61.21% | 49.01% | 85.07% | 68.10% | 62.07% |
| 6 | 87.26% | 62.26% | 49.57% | 86.66% | 69.83% | 65.52% |
| 7 | 88.08% | 64.44% | 49.82% | 85.99% | 67.24% | 62.07% |
| 8 | 87.07% | 62.03% | 52.59% | 85.57% | 69.83% | 63.79% |
| 9 | 87.40% | 63.53% | 52.77% | 86.41% | 65.52% | 72.41% |
| 10 | 86.73% | 63.12% | 54.69% | 85.23% | 63.79% | 68.97% |
| 11 | 87.31% | 62.07% | 57.27% | 86.16% | 67.24% | 67.24% |
| **12** | **87.07%** | **62.44%** | **59.00%** | **84.82%** | **60.34%** | **70.69%** |

### User Statistics
- **User 1**: 4,602 labels (largest dataset)
- **User 2**: 4,588 labels
- **User 3**: 3,701 labels
- **User 4**: 2,812 labels
- **User 5**: 299 labels
- **User 6**: 148 labels (smallest dataset)

## Analysis and Key Findings

### 1. Overall Performance
- **Peak Performance**: The model achieved its best validation accuracy of 72.42% at epoch 12
- **Overfitting Pattern**: Clear divergence between training (87.64%) and validation (72.42%) accuracy indicates overfitting began around epoch 9-10
- **Training Efficiency**: The model reached competitive performance (>70% val acc) within just 4 epochs

### 2. Attribute-Specific Insights
- **Best Performing**: "Attractiveness" consistently achieved the highest accuracy (74.50% at best)
- **Most Challenging**: "Smartness" proved most difficult to predict (71.91% at best)
- **Consistency**: All three attributes showed similar learning curves, suggesting the model captures general perception patterns rather than attribute-specific features

### 3. User-Specific Analysis

#### High Performers (>80% accuracy):
- **User 1** (87.07%): Largest dataset (4,602 labels), most consistent predictions
- **User 4** (84.82%): Second-best performance despite medium-sized dataset (2,812 labels)

#### Moderate Performers (60-70% accuracy):
- **User 2** (62.44%): Stable around 62% despite large dataset (4,588 labels)
- **User 6** (70.69%): Good performance despite smallest dataset (148 labels)

#### Challenging Cases:
- **User 3** (59.00%): Showed most improvement (48.83% → 59.00%, +10.17%)
- **User 5** (60.34%): High variance across epochs, declining trend after epoch 9

### 4. Learning Dynamics
- **Fast Initial Learning**: 66.37% → 70.25% in first 4 epochs (+3.88%)
- **Plateau Phase**: Epochs 5-8 showed minimal improvement (~70-71%)
- **Final Push**: Epochs 9-12 achieved breakthrough to 72.42%
- **Degradation Risk**: Training beyond epoch 12 would likely worsen generalization

### 5. Comparison with Traditional MLP

| Metric | LoRA Fine-tuning | Traditional MLP |
|--------|------------------|-----------------|
| Best Val Accuracy | 72.42% | 72.34% |
| Trainable Parameters | ~302K | ~450K |
| Training Speed | Slower (LoRA overhead) | Faster |
| Flexibility | User-specific adaptation | Fixed architecture |
| Memory Usage | Higher (CLIP base) | Lower |

### 6. Key Advantages of LoRA Approach
1. **Parameter Efficiency**: 33% fewer trainable parameters than MLP
2. **User Personalization**: Visual prompts enable user-specific adaptation
3. **Transfer Learning**: Leverages pre-trained CLIP representations
4. **Modular Design**: Can selectively fine-tune vision/text components

### 7. Limitations and Considerations
1. **Overfitting**: Strong overfitting after epoch 10 despite regularization
2. **User Imbalance**: Performance varies significantly across users (59% - 87%)
3. **Computational Cost**: Requires loading full CLIP model
4. **Limited Improvement**: Only 0.08% better than traditional approach

## Recommendations

1. **Early Stopping**: Implement more aggressive early stopping (patience=3-5)
2. **Regularization**: Increase dropout or weight decay to combat overfitting
3. **Data Augmentation**: Apply stronger augmentation for users with less data
4. **Ensemble Methods**: Combine LoRA with traditional MLP for robust predictions
5. **User Clustering**: Group similar users to share visual prompts
6. **Architecture Search**: Experiment with different LoRA ranks (2, 8, 16)

## Conclusion

The LoRA fine-tuning approach successfully achieved competitive performance (72.42%) with fewer parameters than traditional methods. While the improvement is marginal, the approach demonstrates the viability of efficient fine-tuning for perception tasks. The strong user-specific performance variations suggest that personalized models could be valuable for this domain. Future work should focus on addressing overfitting and improving generalization for underperforming users.

---
*Report generated: 2025-09-17*
*Model checkpoint saved: `runs/lora_visual_prompts_ViT-B_32_20250917_142456/best_model.pt`*