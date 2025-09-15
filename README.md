# Human Perception Labeling Dataset for FFHQ Images

## Overview
This dataset contains human perception ratings for images from the FFHQ (Flickr-Faces-HQ) dataset. Human labelers evaluated facial images on three key attributes: **attractiveness**, **smartness**, and **trustworthiness**. The dataset includes two labeling paradigms: individual ratings and pairwise comparisons.

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

## Key Insights from Analysis

1. **Label Quality**: The codebase includes methods to assess labeler consistency and identify potential random/low-quality annotations
2. **Matrix Completion**: SVD-based approaches are used to predict missing ratings and understand latent factors
3. **User Behavior**: Analysis of how different users rate images and their consistency patterns
4. **Comparison Validation**: Methods to validate pairwise comparisons against individual ratings

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