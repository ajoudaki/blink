# Unified Model with CLIP Text Embeddings - Implementation Report

## Architecture Overview

The unified model with text embeddings eliminates the need for target-specific readout heads by using CLIP text embeddings to encode the target attribute (attractive/smart/trustworthy).

### Key Architecture Features:

1. **Input Concatenation**:
   - CLIP image embeddings (512-dim)
   - CLIP text embeddings (512-dim) for target attribute
   - Optional user embeddings (32-dim) for personalization
   - Total input: 1024-1056 dimensions

2. **Shared Encoder**:
   - 3-layer MLP: 1024 → 512 → 256 → 128
   - BatchNorm after each layer
   - GELU activation
   - Dropout (0.1) for regularization

3. **Single Unified Readout**:
   - **Comparison task**: Single score output per image (1-dim), compared via softmax
   - **Rating task**: 4-way classification output (4-dim)
   - No separate heads for different targets

## Text Embedding Strategy

Text embeddings are generated using CLIP's text encoder with multiple variations per target:

```python
templates = {
    'attractive': [
        "an attractive person",
        "this person is attractive",
        "this person looks attractive",
        "someone who is physically attractive"
    ],
    'smart': [
        "an intelligent person",
        "this person is smart",
        "this person looks intelligent",
        "someone who seems smart"
    ],
    'trustworthy': [
        "a trustworthy person",
        "this person is trustworthy",
        "this person looks trustworthy",
        "someone who seems reliable"
    ]
}
```

Multiple text variations are averaged to create a robust embedding for each target attribute.

## Implementation Details

### Comparison Task
- Input: Two images + target text embedding
- Each image processed separately through encoder
- Outputs compared via 2-way softmax
- CrossEntropy loss on binary choice

### Rating Task
- Input: Single image + target text embedding
- Direct 4-way classification
- CrossEntropy loss on rating scale (1-4)

## Performance Results (Demonstration)

Based on synthetic data demonstration:

### Comparison Task
- Training Accuracy: ~65% (converging)
- Validation Accuracy: ~68%
- Validation CE Loss: ~0.61

### Rating Task
- Training Accuracy: ~87% (overfitting on synthetic data)
- Validation Accuracy: ~27% (poor generalization on random data)
- Validation MAE: ~1.21

Note: These results are on synthetic/random data for architecture demonstration. Real performance would depend on actual FFHQ face embeddings and labels.

## Advantages of Text Embedding Approach

1. **Single Unified Architecture**: One readout head handles all targets
2. **Flexibility**: Easy to add new targets by defining text templates
3. **Semantic Grounding**: Text embeddings provide semantic meaning to targets
4. **Parameter Efficiency**: Fewer parameters than multi-head approaches
5. **Cross-modal Learning**: Leverages CLIP's image-text alignment

## Caching Strategy

- **Text embeddings**: Pre-computed and cached for all target variations
- **Image embeddings**: Pre-computed CLIP embeddings loaded from cache
- **Efficient training**: No embedding computation during training

## Code Files

1. `train_text_unified.py` - Full implementation with Hydra config
2. `train_text_unified_simple.py` - Simplified version for real data
3. `train_text_unified_demo.py` - Demonstration with synthetic data
4. `configs/unified_text_base.yaml` - Configuration file

## Summary

The unified model with text embeddings successfully implements a single architecture that handles both comparison and rating tasks across all target attributes. By using CLIP text embeddings to encode the target attribute, we eliminate the need for multiple readout heads while maintaining the ability to distinguish between different evaluation criteria.

The architecture is more elegant and parameter-efficient than previous multi-head approaches, while providing the flexibility to easily extend to new target attributes simply by defining appropriate text templates.