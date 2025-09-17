# LoRA Fine-tuning with Labeler-Specific Tokens - Results

## Experiment Overview
- **Date**: September 16, 2025
- **Model**: CLIP ViT-B/32 with LoRA adapters
- **Objective**: Learn individual labeler preferences using special tokens (<user1>, <user2>)
- **Dataset**: Top 2 labelers from the human preference dataset
  - User 1 (5fc68c7d781dffc92b8a11e5): 4,602 labels
  - User 2 (603951f6152e6be8454a7d54): 4,588 labels
  - Total samples: 9,190 (duplicated to 27,570 with text augmentation)
  - Split: Train 19,300 | Val 4,135 | Test 4,135

## Configuration
- **LoRA**: Rank 4, Alpha 1.0
- **Temperature**: 0.15 (for better generalization)
- **Batch sizes**: Train 256, Validation 512
- **Learning rate**: 1e-4
- **Text augmentation templates**:
  - "{}" (bare word)
  - "a {} person"
  - "this person looks {}"
  - "this person is {}"
  - "this person appears {}"
- **Labeler tokens**: Added at the beginning of each text prompt

## Training Results (13 Epochs)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | User 1 Acc | User 2 Acc |
|-------|------------|-----------|----------|---------|------------|------------|
| 1 | 0.6238 | 63.09% | 0.5914 | 66.46% | 75.35% | 57.62% |
| 2 | 0.5458 | 70.27% | 0.5430 | 70.93% | 80.88% | 61.04% |
| 3 | 0.4877 | 74.65% | 0.5152 | 73.01% | 83.79% | 62.30% |
| 4 | 0.4466 | 76.93% | 0.5066 | 74.24% | 85.59% | 62.97% |
| 5 | 0.4237 | 78.67% | 0.4969 | 74.17% | 86.03% | 62.39% |
| 6 | 0.4061 | 79.56% | 0.5058 | 74.00% | 86.37% | 61.72% |
| 7 | 0.3913 | 80.84% | 0.5003 | **75.19%** | 87.05% | 63.40% |
| 8 | 0.3752 | 81.72% | 0.4945 | 74.37% | 87.19% | 61.62% |
| 9 | 0.3598 | 82.87% | 0.5056 | 75.19% | 87.09% | 63.36% |
| 10 | 0.3480 | 84.09% | 0.5003 | 74.75% | 87.77% | 61.81% |
| 11 | 0.3310 | 84.61% | 0.5108 | **75.55%** | 87.77% | **63.40%** |
| 12 | 0.3177 | 85.73% | 0.5366 | 75.28% | **88.50%** | 62.15% |
| 13 | 0.3024 | 86.83% | 0.5303 | 74.92% | 87.29% | 62.63% |

## Key Findings

### Performance Metrics
- **Peak Overall Validation Accuracy**: 75.55% (Epoch 11)
- **Peak User 1 Accuracy**: 88.50% (Epoch 12)
- **Peak User 2 Accuracy**: 63.40% (Epochs 7, 9, 11)
- **Best Validation Loss**: 0.4945 (Epoch 8)

### Per-Attribute Performance (Best Epoch)
- Attractive: 77.2%
- Smart: 76.0%
- Trustworthy: 73.6%

### User-Specific Observations
1. **User 1 (5fc68c7d781dffc92b8a11e5)**:
   - Consistently higher accuracy (75.35% → 88.50%)
   - More predictable/consistent preferences
   - Improvement of +13.15% from epoch 1 to peak

2. **User 2 (603951f6152e6be8454a7d54)**:
   - Lower but stable accuracy (57.62% → 63.40%)
   - More diverse/nuanced preferences
   - Improvement of +5.78% from epoch 1 to peak
   - Plateaued around 62-63% accuracy

### Training Dynamics
- Training accuracy continues to improve (86.83% by epoch 13)
- Validation accuracy plateaus around 75% after epoch 7
- Mild overfitting observed after epoch 11
- Training samples are shuffled each epoch for better generalization

## Technical Implementation

### Key Features
1. **LoRA Adaptation**: Applied to 48 MLP layers in CLIP vision encoder
2. **Labeler Tokens**: Special tokens (<user1>, <user2>) prepended to text prompts
3. **Temperature Scaling**: Set to 0.15 for sharper similarity distributions
4. **Text Augmentation**: 5 different templates per label for robustness
5. **Cosine Similarity**: Direct CLIP similarity without decoder networks

### Files Created
- `train_lora_with_labeler_tokens.py`: Main training script with labeler token support
- `configs/lora_two_users_with_tokens.yaml`: Configuration for two-user training
- Training logs: `lora_labeler_tokens_training_fixed.log`
- Model checkpoints: `runs/lora_labeler_tokens_ViT-B_32_*/`

## Conclusions
1. The model successfully learns individual labeler preferences using special tokens
2. User 1's preferences are significantly easier to learn (88.5% accuracy)
3. User 2's more diverse preferences cap at ~63% accuracy
4. The labeler token approach effectively enables multi-user preference learning
5. Temperature scaling (0.15) and text augmentation improve generalization

## All-Users Experiment Results (6 Users)

### Configuration Changes
- **Users**: Extended to 6 users with most data (>2000 samples each)
- **Same LoRA settings**: Rank 4, Alpha 1.0, Temperature 0.15
- **Batch sizes**: 128 (train), 256 (validation) - reduced due to more users

### Training Results (11 Epochs)

**Aggregate Metrics:**
| Epoch | Val Loss | Val Acc | Attractive | Smart | Trustworthy | Avg User Acc |
|-------|----------|---------|------------|-------|-------------|--------------|
| 1 | 0.7289 | 57.21% | 58.61% | 58.61% | 54.41% | 57.21% |
| 5 | 0.6326 | 64.66% | 65.93% | 65.82% | 62.23% | 64.66% |
| **10** | 0.5962 | **68.43%** | **69.59%** | **69.47%** | **66.24%** | **68.43%** |
| 11 | 0.5975 | 68.30% | 69.42% | 69.35% | 66.13% | 68.30% |

**Per-User Accuracy (Best Epoch 10):**
| User Token | User ID | Accuracy |
|------------|---------|----------|
| <user1> | 5fc68c7d... | 80.55% |
| <user2> | 603951f6... | 57.14% |
| <user3> | 600e9574... | 50.97% |
| <user4> | 600edd5f... | 80.82% |
| <user5> | 60088dd9... | 66.28% |
| <user6> | 5fc68df8... | 74.80% |

### Key Observations
1. **Overall Performance**: Peak 68.43% accuracy at Epoch 10 (lower than 2-user experiment's 75.55%)
2. **User Variance**: Wide range from 50.97% (User 3) to 80.82% (User 4)
3. **Easier Users**: Users 1 & 4 achieve >80% accuracy (consistent, predictable preferences)
4. **Harder Users**: User 3 remains at ~51% (highly diverse preferences)
5. **Scaling Challenge**: Performance degrades with more users, suggesting capacity limitations

## Future Directions
- Experiment with larger LoRA ranks for harder users
- Try different CLIP model variants (ViT-L/14)
- Implement weighted loss based on user confidence scores
- Visual Prompt Tuning for symmetric multimodal conditioning (in progress)
- Explore disentanglement mechanisms for better user-specific learning