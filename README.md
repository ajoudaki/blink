# Human Perception Labeling Dataset for FFHQ Images

## Overview
This dataset contains human perception ratings for images from the FFHQ (Flickr-Faces-HQ) dataset. Human labelers evaluated facial images on three key attributes: **attractiveness**, **smartness**, and **trustworthiness**. The dataset includes two labeling paradigms: individual ratings and pairwise comparisons.

## Quick Start

### Installation
```bash
pip install torch torchvision pandas scikit-learn hydra-core matplotlib tqdm clip-by-openai
```

### Training Models

The repository uses a unified training script (`train.py`) with Hydra configuration management. All model architectures and hyperparameters are controlled via YAML config files.

```bash
# Train best performing model (72.2% accuracy on comparisons)
python train.py --config-name=config_best

# Train individual rating prediction model
python train.py --config-name=config_rating

# Train pairwise comparison models
python train.py --config-name=config_concat    # Concatenated features (62.4% acc)
python train.py --config-name=config_siamese   # Siamese architecture (63.3% acc)
python train.py --config-name=config_multihead # Multi-head Siamese (64.9% acc)
python train.py                                # Unified Siamese (66.8% acc, default)
```

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

| Model | Task | Accuracy/MAE | Details |
|-------|------|-------------|---------|
| MLP | Individual Ratings | 0.557 MAE | Predicting 1-4 scale ratings |
| Concatenated | Pairwise Comparison | 62.4% | Baseline approach |
| Siamese | Pairwise Comparison | 63.3% | Shared encoder |
| Multi-head | Pairwise Comparison | 64.9% | 3 output heads |
| Unified | Pairwise Comparison | 66.8% | With user encoding (50 epochs) |
| **Unified Best** | **Pairwise Comparison** | **72.2%** | **Optimized hyperparameters (100 epochs)** |

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