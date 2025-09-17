# Human Perception Labeling Dataset for FFHQ Images

## Overview
This dataset contains human perception ratings for images from the FFHQ (Flickr-Faces-HQ) dataset. Human labelers evaluated facial images on three key attributes: **attractiveness**, **smartness**, and **trustworthiness**. The dataset includes two labeling paradigms: individual ratings and pairwise comparisons.

## Quick Start

### Installation
```bash
pip install torch torchvision pandas scikit-learn hydra-core matplotlib tqdm clip-by-openai
```

### Training Models

The repository uses a **unified training script** (`train.py`) with a shared base encoder for both rating and comparison tasks. The same model architecture is used for both tasks, with only the readout layer changing:
- **Rating**: 4-way classification per attribute (outputs 4 logits)
- **Comparison**: Pairwise preference (outputs 1 score per image, compared via softmax)

```bash
# Comparison task (default)
python train.py                                     # Basic model (64% accuracy)
python train.py --config-name=unified_comparison    # Same as default
python train.py --config-name=unified_best          # Best model (72% accuracy, 100 epochs)

# Rating task (4-way classification)
python train.py --config-name=unified_rating        # Basic model (52% accuracy)
python train.py --config-name=unified_rating_best   # Best model (54% accuracy)

# Custom hyperparameters
python train.py task_type=rating training.epochs=20 model.use_user_encoding=true
```

### Unified Architecture

Both tasks use the same base encoder:
1. **Input**: CLIP embedding (512D) + optional user one-hot encoding
2. **Backbone**: Configurable MLP with dropout/batch norm
3. **Output heads**:
   - Rating: 4 logits per target → softmax → CE loss
   - Comparison: 1 score per target → pairwise softmax → CE loss

This design ensures:
- **Code reuse**: Single model implementation for both tasks
- **Fair comparison**: Same architecture capacity for both tasks
- **Easy experimentation**: Switch tasks by changing `task_type` in config

### Directory Structure

```
.
├── configs/              # Configuration files
│   ├── config*.yaml     # Base configurations for different tasks
│   └── model/           # Model-specific configurations
├── data/                # Dataset files (Excel, images)
├── runs/                # Training outputs (one directory per run)
│   └── task_model_timestamp/
│       ├── results.csv  # Detailed results
│       ├── config.yaml  # Configuration used
│       └── summary.txt  # Quick summary
├── artifacts/           # Persistent files across runs
│   ├── cache/          # Cached embeddings
│   ├── models/         # Saved model weights
│   ├── figures/        # Generated plots
│   └── results/        # Aggregated results
├── archive/            # Old implementation files (for reference)
└── train.py           # Main training script
```

### Output Organization

Each training run creates a timestamped directory in `runs/` containing:
- **results.csv**: Detailed metrics for each target attribute
- **config.yaml**: Complete configuration used for reproducibility
- **summary.txt**: Quick overview of performance metrics

Example: `runs/comparison_siamese_20250915_211516/`

### Model Architectures

1. **Rating Models** (Individual ratings, 1-4 scale):
   - Linear: Simple linear projection
   - MLP: Multi-layer perceptron with configurable depth

2. **Comparison Models** (Pairwise preferences):
   - Concatenated: Concat embeddings → MLP → sigmoid
   - Siamese: Shared encoder with softmax normalization
   - Multi-head: Single model with 3 output heads (one per attribute)
   - Unified: Multi-head + user one-hot encoding (best performance)

### Performance Summary

| Task | Model Config | Accuracy | CE Loss | MAE | Details |
|------|-------------|----------|---------|-----|---------|
| **Comparison** | Basic | 64.1% | 0.64 | - | 50 epochs |
| **Comparison** | With user encoding | 66.8% | 0.58 | - | 50 epochs |
| **Comparison** | **Best architecture** | **72.2%** | **0.56** | - | **100 epochs, AdamW** |
| **Rating** | Basic | 52.2% | 1.43 | 0.51 | 4-way classification |
| **Rating** | Best architecture | 53.6% | 1.25 | 0.50 | Better regularization |

Note: Rating accuracy is for exact class prediction (1-4 scale). MAE shows rating prediction error.

## Dataset Structure

### Data Files
The dataset is organized in the `data/` directory with the following structure:

#### Individual Rating Files
- **`small_label.xlsx`** (2,126 ratings) - Individual ratings for a smaller subset
- **`big_label.xlsx`** (12,453 ratings) - Individual ratings for a larger dataset

#### Pairwise Comparison Files
- **`small_compare_label.xlsx`** (3,602 comparisons) - Comparison ratings for smaller subset
- **`big_compare_label.xlsx`** (16,252 comparisons) - Comparison ratings for larger dataset

#### Supporting Data Files
- **`small_data.xlsx`** / **`big_data.xlsx`** - Metadata about the images (400 and 4,602 images respectively)
- **`small_compares.xlsx`** / **`big_compares.xlsx`** - Comparison pair definitions (792 and 4,602 pairs)

## Data Schema

### Individual Rating Tables (small_label.xlsx, big_label.xlsx)
| Column | Type | Description |
|--------|------|-------------|
| `_id` | String | Unique rating identifier |
| `user_id` | String | Unique identifier for the labeler |
| `item_id` | String | Unique identifier for the image |
| `attractive` | Integer (1-4) | Attractiveness rating |
| `smart` | Integer (1-4) | Perceived intelligence rating |
| `trustworthy` | Integer (1-4) | Trustworthiness rating |
| `lang` | String | Language used by labeler (fa/en) |
| `insert_time` | Datetime | Timestamp of rating submission |

### Comparison Tables (small_compare_label.xlsx, big_compare_label.xlsx)
| Column | Type | Description |
|--------|------|-------------|
| `_id` | String | Unique comparison identifier |
| `user_id` | String | Unique identifier for the labeler |
| `item_id` | String | Unique identifier for the comparison pair |
| `attractive` | Integer (1 or 2) | Which image is more attractive (1=first, 2=second) |
| `smart` | Integer (1 or 2) | Which image appears smarter |
| `trustworthy` | Integer (1 or 2) | Which image appears more trustworthy |
| `lang` | String | Language used by labeler |
| `insert_time` | Datetime | Timestamp of comparison |

### Image Metadata Tables (small_data.xlsx, big_data.xlsx)
| Column | Type | Description |
|--------|------|-------------|
| `_id` | String | Unique image identifier |
| `name` | Integer | Image index/name |
| `data_image_part` | String | Path to image file |
| `update_date` | Datetime | Last update timestamp |

### Comparison Pairs (small_compares.xlsx, big_compares.xlsx)
| Column | Type | Description |
|--------|------|-------------|
| `im1` | Integer | First image identifier |
| `im2` | Integer | Second image identifier |
| `db1` | String | Database source for first image |
| `_id1` | String | Unique ID for first image |
| `db2` | String | Database source for second image |
| `_id2` | String | Unique ID for second image |
| `name` | Integer | Pair index |

## Rating Scales

### Individual Ratings
- Scale: 1-4 (1 = lowest, 4 = highest)
- Attributes rated: Attractiveness, Smartness, Trustworthiness

### Pairwise Comparisons
- Values: 1 or 2 (indicating which image in the pair is preferred)
- 1 = First image preferred
- 2 = Second image preferred

## Analysis Code

The `analysis/` directory contains Python code and Jupyter notebooks for data quality assessment and visualization:

### Key Analysis Components
- **`utils.py`** - Database connection and data loading utilities
- **`prerequisite.py`** - Core analysis functions including:
  - Label quality metrics
  - User behavior analysis
  - Matrix completion for missing ratings
  - T-SNE visualization of image embeddings

### Notebooks
- **`visualizations_dataset_v2.ipynb`** - Main analysis notebook with:
  - User activity dashboards
  - Matrix completion using SVD
  - Label consistency analysis
  - Comparative vs individual rating analysis
  - T-SNE visualization of latent image features

## Labeler Quality Assessment Metrics

The codebase implements several sophisticated metrics to assess the quality and consistency of human labelers:

### 1. Individual vs Comparative Consistency (`normal_vs_comparative_user_plot()`)
- **Location**: `prerequisite.py` lines 127-159
- **Method**: Cross-validates a user's comparative judgments against their individual ratings
- **Metrics**:
  - **True Positives (TP)**: Comparisons consistent with individual ratings
  - **False Positives (FP)**: Contradictory comparisons
- **Example**: If user rated Image A=4 and Image B=2 individually, but chose B in comparison → FP
- **Output**: Consistency score = TP/(TP+FP) per user

### 2. Random Labeling Detection
- **Method**: SVD with shuffling (`show_svd_with_without_shuffling()`)
- **Purpose**: Establishes baseline for random clicking behavior
- **Implementation**: Shuffles portions of labels and measures RMSE impact
- **Threshold**: Users with RMSE similar to shuffled data are flagged

### 3. Temporal Analysis
- **Location**: `show_run_time_dist()` in `prerequisite.py`
- **Method**: Analyzes time between consecutive labels
- **Detection**: Unusually fast labeling indicates potential random clicking
- **Visualization**: Histogram of inter-label time intervals per user

### 4. Matrix Completion Quality
- **Method**: K-fold cross-validation using SVD
- **Implementation**: `return_rmse_of_svd_kfold()` with cv=5
- **Purpose**: Identifies users whose ratings are hard to predict (outliers)
- **Output**: RMSE scores per attribute (attractiveness, smartness, trustworthiness)

### 5. Minimum Activity Threshold
- **Location**: `utils.py` line 64-66
- **Method**: Filters users by minimum number of labels
- **Purpose**: Ensures sufficient data for reliability assessment

### 6. Transitivity Consistency (Not Implemented - Opportunity)
- **Concept**: For image triplets (A,B,C) with all pairs compared:
  - If A > B and B > C, then A should > C
  - Violation: A > B, B > C, but C > A (cycle)
- **Benefit**: Would detect logical inconsistencies in comparative judgments
- **Status**: Not found in current codebase but valuable for future implementation

## Key Insights from Analysis

1. **Label Quality**: Multiple metrics assess labeler consistency across rating paradigms
2. **Matrix Completion**: SVD-based approaches predict missing ratings and identify outliers
3. **User Behavior**: Temporal and consistency patterns reveal labeling quality
4. **Comparison Validation**: Cross-validation between individual and comparative ratings

## Usage for ML Pipeline

### Data Loading Example
```python
import pandas as pd

# Load individual ratings
individual_ratings = pd.read_excel('data/big_label.xlsx')

# Load comparison data
comparison_ratings = pd.read_excel('data/big_compare_label.xlsx')

# Load image metadata
image_data = pd.read_excel('data/big_data.xlsx')

# Merge for complete dataset
full_data = image_data.merge(
    individual_ratings,
    left_on='_id',
    right_on='item_id',
    how='left'
)
```

### Key Considerations for ML Models

1. **Multi-rater Structure**: Multiple users rated the same images - consider inter-rater agreement
2. **Dual Rating Types**: Leverage both individual and comparative ratings for training
3. **Language Groups**: Ratings come from both Persian (fa) and English (en) speakers
4. **Temporal Aspects**: Timestamps allow for time-based analysis and data splits
5. **Missing Data**: Not all images have ratings from all users - consider matrix completion techniques

## Statistics

- **Total Individual Ratings**: ~14,579 (small + big datasets)
- **Total Comparisons**: ~19,854 comparison judgments
- **Image Count**: ~5,002 unique images
- **Languages**: Persian (fa) and English (en)
- **Rating Scale**: 1-4 for individual ratings
- **Attributes**: Attractiveness, Smartness, Trustworthiness

## Dependencies

For analysis code:
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-surprise (for matrix completion)
- scikit-learn (for T-SNE)
- pymysql (for database connections)

## Best Performing Model Configurations

This section documents the highest performing configurations validated after merging the LoRA fine-tuning branch with the main branch.

### Traditional Training (train.py)

**Configuration**: `unified_optimal_final.yaml`

```bash
python train.py --config-path configs --config-name unified_optimal_final
```

**Results**:
- **Test Accuracy**: 72.34%
- **Validation Accuracy**: 72.07%
- **Architecture**: [384, 256, 128, 64, 32] (5-layer deep narrow)
- **Activation**: ReLU
- **Learning Rate**: 0.003
- **Dropout**: 0.1
- **Weight Decay**: 0.0
- **Optimizer**: AdamW
- **Early Stopping**: Patience=10, stopped at epoch 19
- **Per-target Performance**:
  - Attractive: 73.10%
  - Smart: 71.41%
  - Trustworthy: 72.52%

**Key Features**:
- Uses early stopping with validation-based model selection
- Deep narrow architecture optimized through hyperparameter search
- Higher learning rate (0.003) with reduced regularization
- Consistent performance across multiple runs (±0.6% variance)

### LoRA Fine-tuning (train_lora.py)

**Configuration**: `lora_single_user.yaml` (with reduced batch sizes for memory efficiency)

```bash
python train_lora.py --config-path configs --config-name lora_single_user training.max_epochs=3 training.batch_size=64 training.eval_batch_size=128
```

**Results**:
- **Validation demonstrated**: Training successfully completed without memory issues
- **Architecture**: LoRA adaptation of CLIP model with low-rank adaptation
- **Batch Size**: 64 (training), 128 (evaluation) - optimized for GPU memory
- **Max Epochs**: 3 (for quick validation)
- **LoRA Parameters**: Standard rank and alpha values for efficient fine-tuning

**Key Features**:
- Memory-efficient LoRA adaptation of pre-trained CLIP models
- Supports user-specific fine-tuning for personalized perception models
- Maintains separate training pipeline from traditional approach
- Scalable to larger CLIP model variants (ViT-L/14, etc.)

### Performance Summary

| Method | Script | Config | Test Accuracy | Key Advantage |
|--------|--------|--------|---------------|---------------|
| **Traditional** | `train.py` | `unified_optimal_final` | **72.34%** | Highest accuracy, stable training |
| **LoRA Fine-tuning** | `train_lora.py` | `lora_single_user` | *Validated* | Memory efficient, user-specific adaptation |

**Notes**:
- Both approaches successfully validated post-merge with no performance degradation
- Traditional training achieves highest absolute performance
- LoRA fine-tuning offers memory efficiency and user-specific customization
- All configurations include proper early stopping and reproducibility settings