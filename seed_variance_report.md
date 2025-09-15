# Seed Variance Analysis Report

## Executive Summary
The model demonstrates **excellent stability** across different random seeds with a coefficient of variation of only **1.05%** for test accuracy.

## Configuration Tested
```yaml
learning_rate: 0.003
dropout: 0.1
weight_decay: 0.0
activation: relu
architecture: [384, 256, 128, 64, 32]
```

## Results Across 5 Seeds

### Individual Run Results

| Seed | Val Acc | Test Acc | Val Loss | Test Loss | Val-Test Gap |
|------|---------|----------|----------|-----------|--------------|
| 42   | 72.88%  | 73.29%   | 0.5340   | 0.5488    | -0.41%       |
| 123  | 73.54%  | 73.26%   | 0.5425   | 0.5414    | +0.28%       |
| 456  | 71.58%  | 71.25%   | 0.5393   | 0.5378    | +0.33%       |
| 789  | 73.76%  | 72.49%   | 0.5300   | 0.5373    | +1.27%       |
| 2024 | 71.67%  | 72.12%   | 0.5349   | 0.5382    | -0.45%       |

### Statistical Summary

| Metric | Mean | Std Dev | Min | Max | Range | CV% |
|--------|------|---------|-----|-----|-------|-----|
| **Validation Accuracy** | 72.69% | ±0.91% | 71.58% | 73.76% | 2.18% | 1.26% |
| **Test Accuracy** | 72.48% | ±0.76% | 71.25% | 73.29% | 2.04% | 1.05% |
| Validation Loss | 0.5361 | ±0.0043 | 0.5300 | 0.5425 | 0.0125 | 0.80% |
| Test Loss | 0.5407 | ±0.0043 | 0.5373 | 0.5488 | 0.0115 | 0.80% |
| Best Epoch | 22.0 | ±2.3 | 19 | 26 | 7 | 10.5% |

### 95% Confidence Intervals

- **Validation**: 72.69% ± 0.80% → [71.88%, 73.49%]
- **Test**: 72.48% ± 0.67% → [71.81%, 73.15%]

## Key Findings

### 1. Stability Assessment
- **Coefficient of Variation**: 1.05% (Test) and 1.26% (Validation)
- **Verdict**: **HIGHLY STABLE** (CV < 2% indicates excellent stability)

### 2. Generalization
- **Mean Gap**: 0.20% ± 0.63%
- **Range**: -0.45% to +1.27%
- Model shows good generalization with minimal overfitting

### 3. Performance Consistency
- Test accuracy stays within a **2.04%** range across seeds
- 95% of runs expected to achieve between **71.0%** and **73.9%** test accuracy

### 4. Early Stopping Consistency
- Best epoch occurs around epoch 22 ± 2.3
- Shows consistent convergence behavior

## Comparison with Previous Results

| Experiment | Test Accuracy | Notes |
|------------|--------------|-------|
| Single run (original) | 73.03% | Lucky seed? |
| **5-seed average** | **72.48% ± 0.76%** | **True expected performance** |
| Best seed (42) | 73.29% | Upper bound |
| Worst seed (456) | 71.25% | Lower bound |

## Practical Implications

1. **Expected Performance**: When deploying this model, expect **72.5% ± 0.8%** test accuracy

2. **Risk Assessment**:
   - Best case: 73.3% (as seen with seed 42)
   - Worst case: 71.3% (as seen with seed 456)
   - Very low risk of performance surprises

3. **Sample Size**: The low variance suggests that results from even a single seed are reasonably representative

4. **Reproducibility**: Results are highly reproducible with proper seed control

## Recommendations

1. **For Research**: Report mean ± std: **72.48% ± 0.76%**

2. **For Production**:
   - Use ensemble of 3-5 models with different seeds
   - Expected ensemble performance: ~72.5-73.0%

3. **For Further Optimization**:
   - Low variance indicates hyperparameters are well-tuned
   - Focus on architectural changes rather than hyperparameter tuning

## Conclusion

The model demonstrates **excellent stability** with only **1.05% coefficient of variation** in test accuracy across different random seeds. The true expected performance is **72.48% ± 0.76%**, making this a reliable and reproducible configuration for deployment.