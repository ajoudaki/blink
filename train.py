#!/usr/bin/env python3
"""
Unified training script with shared base model for ratings and comparisons.
The base model outputs scores for each target, which are then processed
differently for rating (4-way classification) vs comparison (pairwise preference).
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import clip
from PIL import Image
from datetime import datetime
from pathlib import Path


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

def load_or_extract_embeddings(image_paths, cache_file=None,
                               device='cuda', force_recompute=False):
    """Load cached embeddings or extract new ones using CLIP."""

    if cache_file is None:
        cache_file = 'artifacts/cache/clip_embeddings_cache.pkl'

    # Ensure cache directory exists
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    if os.path.exists(cache_file) and not force_recompute:
        print(f"Loading embeddings from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"✓ All {len(embeddings)} embeddings loaded from cache")
        return embeddings

    print("Extracting CLIP embeddings...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    embeddings = {}

    for img_path in tqdm(image_paths, desc="Processing images"):
        if os.path.exists(img_path):
            try:
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = model.encode_image(image)
                    features = features.cpu().numpy().squeeze()
                embeddings[img_path] = features
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                embeddings[img_path] = np.zeros(512)
        else:
            embeddings[img_path] = np.zeros(512)

    # Save cache
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"✓ Embeddings saved to {cache_file}")

    return embeddings


# ============================================================================
# UNIFIED MODEL ARCHITECTURE
# ============================================================================

class BaseEncoder(nn.Module):
    """
    Base encoder that takes image embedding (+ optional user encoding)
    and outputs raw scores for each target attribute.

    For comparison: outputs 1 score per target (will be compared via softmax)
    For rating: outputs 4 scores per target (4-way classification)
    """

    def __init__(self, cfg, n_users=None):
        super().__init__()

        # Determine input dimension
        input_dim = 512  # CLIP embedding size
        if cfg.model.get('use_user_encoding', False) and n_users:
            input_dim += n_users

        # Build shared backbone
        layers = []
        prev_dim = input_dim

        for hidden_dim in cfg.model.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Normalization
            if cfg.model.get('batch_norm', False):
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif cfg.model.get('layer_norm', False):
                layers.append(nn.LayerNorm(hidden_dim))

            # Activation
            activation = cfg.model.activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif activation == 'elu':
                layers.append(nn.ELU())

            # Dropout
            if cfg.model.dropout > 0:
                layers.append(nn.Dropout(cfg.model.dropout))

            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Task-specific output heads
        self.task_type = cfg.task_type
        self.targets = cfg.data.targets

        if cfg.task_type == 'rating':
            # For rating: 4 outputs per target (4-way classification)
            self.heads = nn.ModuleDict({
                target: nn.Linear(prev_dim, 4)  # 4 rating levels
                for target in self.targets
            })
        else:  # comparison
            # For comparison: 1 output per target (will be compared)
            self.heads = nn.ModuleDict({
                target: nn.Linear(prev_dim, 1)
                for target in self.targets
            })

        self.use_user_encoding = cfg.model.get('use_user_encoding', False)
        self.n_users = n_users

    def forward(self, x, user_encoding=None, target='attractive'):
        """
        Forward pass through encoder.

        Args:
            x: Image embedding
            user_encoding: Optional user one-hot encoding
            target: Which target attribute to predict

        Returns:
            For rating: 4 logits (one per rating level)
            For comparison: 1 score (to be compared)
        """
        if self.use_user_encoding and user_encoding is not None:
            x = torch.cat([x, user_encoding], dim=1)

        features = self.backbone(x)
        output = self.heads[target](features)

        if self.task_type == 'comparison':
            output = output.squeeze(-1)  # Return shape (batch_size,)

        return output


class UnifiedModel(nn.Module):
    """
    Unified model that handles both rating and comparison tasks.
    Uses the same base encoder but different readout strategies.
    """

    def __init__(self, cfg, n_users=None):
        super().__init__()
        self.encoder = BaseEncoder(cfg, n_users)
        self.task_type = cfg.task_type

    def forward_rating(self, x, user_encoding=None, target='attractive'):
        """
        Forward pass for rating prediction.
        Returns probabilities for each rating level (1-4).
        """
        logits = self.encoder(x, user_encoding, target)  # Shape: (batch, 4)
        probs = F.softmax(logits, dim=-1)
        return probs

    def forward_comparison(self, x1, x2, user_encoding=None, target='attractive'):
        """
        Forward pass for pairwise comparison.
        Returns probability of choosing second image.
        """
        score1 = self.encoder(x1, user_encoding, target)  # Shape: (batch,)
        score2 = self.encoder(x2, user_encoding, target)  # Shape: (batch,)

        # Stack and apply softmax for pairwise comparison
        scores = torch.stack([score1, score2], dim=1)  # Shape: (batch, 2)
        probs = F.softmax(scores, dim=1)

        return probs[:, 1]  # Probability of choosing second image

    def forward(self, *args, **kwargs):
        """Route to appropriate forward method based on task type."""
        if self.task_type == 'rating':
            return self.forward_rating(*args, **kwargs)
        else:
            return self.forward_comparison(*args, **kwargs)


# ============================================================================
# UNIFIED TRAINING FUNCTION
# ============================================================================

def prepare_data(df, embeddings, cfg, user_encoder=None):
    """
    Prepare data for training, handling both rating and comparison tasks.
    """
    all_data = {}

    for target in cfg.data.targets:
        if cfg.task_type == 'rating':
            # Prepare rating data
            X_list = []
            y_list = []
            user_list = []

            for _, row in df.iterrows():
                if pd.notna(row[target]):
                    img_embedding = embeddings.get(row['image_path'], np.zeros(512))
                    X_list.append(img_embedding)
                    # Convert rating to 0-indexed (0-3 instead of 1-4)
                    y_list.append(int(row[target]) - 1)

                    if user_encoder is not None:
                        user_idx = user_encoder.transform([row['user_id']])[0]
                        user_onehot = np.zeros(len(user_encoder.classes_))
                        user_onehot[user_idx] = 1.0
                        user_list.append(user_onehot)

            data = {
                'X': np.array(X_list),
                'y': np.array(y_list)
            }
            if user_encoder is not None:
                data['user'] = np.array(user_list)

        else:  # comparison
            # Prepare comparison data
            X1_list = []
            X2_list = []
            y_list = []
            user_list = []

            for _, row in df.iterrows():
                img1_emb = embeddings.get(row['im1_path'], np.zeros(512))
                img2_emb = embeddings.get(row['im2_path'], np.zeros(512))

                X1_list.append(img1_emb)
                X2_list.append(img2_emb)
                y_list.append(row[f'{target}_binary'])

                if user_encoder is not None:
                    user_idx = user_encoder.transform([row['user_id']])[0]
                    user_onehot = np.zeros(len(user_encoder.classes_))
                    user_onehot[user_idx] = 1.0
                    user_list.append(user_onehot)

            data = {
                'X1': np.array(X1_list),
                'X2': np.array(X2_list),
                'y': np.array(y_list)
            }
            if user_encoder is not None:
                data['user'] = np.array(user_list)

        all_data[target] = data

    return all_data


def train_model(model, train_data, val_data, cfg, device):
    """
    Unified training loop for both rating and comparison tasks.
    """
    optimizer = get_optimizer(model, cfg)

    # Use CrossEntropyLoss for both tasks
    if cfg.task_type == 'rating':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()

    # Training loop
    for epoch in tqdm(range(cfg.training.epochs), desc="Training"):
        model.train()

        for target in cfg.data.targets:
            optimizer.zero_grad()

            if cfg.task_type == 'rating':
                # Rating task
                predictions = model(
                    train_data[target]['X'],
                    train_data[target].get('user', None),
                    target=target
                )
                # CrossEntropyLoss expects class indices
                loss = criterion(predictions, train_data[target]['y'])

            else:
                # Comparison task
                predictions = model(
                    train_data[target]['X1'],
                    train_data[target]['X2'],
                    train_data[target].get('user', None),
                    target=target
                )
                loss = criterion(predictions, train_data[target]['y_float'])

            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    results = {}

    with torch.no_grad():
        for target in cfg.data.targets:
            if cfg.task_type == 'rating':
                # Rating evaluation
                predictions = model(
                    val_data[target]['X'],
                    val_data[target].get('user', None),
                    target=target
                )

                # Get predicted class
                pred_class = predictions.argmax(dim=-1).cpu().numpy()
                true_class = val_data[target]['y_np']

                # Classification metrics
                accuracy = np.mean(pred_class == true_class)

                # Cross-entropy loss
                probs = predictions.cpu().numpy()
                # Create one-hot encoding for true labels
                n_samples = len(true_class)
                true_one_hot = np.zeros((n_samples, 4))
                true_one_hot[np.arange(n_samples), true_class] = 1
                # Calculate CE loss
                probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
                ce_loss = -np.mean(np.sum(true_one_hot * np.log(probs_clipped), axis=1))

                # Regression metrics (for backward compatibility)
                pred_rating = pred_class + 1
                true_rating = true_class + 1
                mae = np.mean(np.abs(pred_rating - true_rating))
                rmse = np.sqrt(np.mean((pred_rating - true_rating) ** 2))

                results[target] = {
                    'accuracy': accuracy,
                    'ce_loss': ce_loss,
                    'mae': mae,
                    'rmse': rmse,
                    'n_train': len(train_data[target]['y']),
                    'n_val': len(val_data[target]['y_np'])
                }

            else:
                # Comparison evaluation
                predictions = model(
                    val_data[target]['X1'],
                    val_data[target]['X2'],
                    val_data[target].get('user', None),
                    target=target
                )
                pred_binary = (predictions.cpu().numpy() > 0.5).astype(int)
                true_binary = val_data[target]['y_np']

                accuracy = np.mean(pred_binary == true_binary)

                # Calculate cross-entropy loss
                probs = predictions.cpu().numpy()
                # Clip probabilities to avoid log(0)
                probs = np.clip(probs, 1e-7, 1 - 1e-7)
                ce_loss = -np.mean(
                    true_binary * np.log(probs) +
                    (1 - true_binary) * np.log(1 - probs)
                )

                results[target] = {
                    'accuracy': accuracy,
                    'ce_loss': ce_loss,
                    'n_train': len(train_data[target]['y']),
                    'n_val': len(val_data[target]['y_np'])
                }

    return results


def get_optimizer(model, cfg):
    """Get optimizer based on configuration."""
    opt_type = cfg.training.optimizer
    lr = cfg.training.learning_rate
    weight_decay = cfg.training.get('weight_decay', 0)

    if opt_type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""

    # Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu.device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Task type: {cfg.task_type}")
    print(f"Model architecture: Unified BaseEncoder")

    # Create run-specific output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.task_type}_unified_{run_id}"
    run_dir = Path(f"runs/{run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {run_dir}")

    # Load data based on task type
    if cfg.task_type == 'rating':
        print("\nLoading individual rating data...")
        rating_labels = pd.read_excel('data/big_label.xlsx')
        rating_data = pd.read_excel('data/big_data.xlsx')
        df = rating_data.merge(rating_labels, left_on='_id', right_on='item_id', how='inner')

        # Rename column for consistency
        df['image_path'] = df['data_image_part']

        # Get all unique images
        all_images = df['image_path'].unique().tolist()

    else:  # comparison
        print("\nLoading comparison data...")
        compare_labels = pd.read_excel('data/big_compare_label.xlsx')
        compare_data = pd.read_excel('data/big_compare_data.xlsx')
        df = compare_data.merge(compare_labels, left_on='_id', right_on='item_id', how='inner')

        # Convert labels to binary
        for target in cfg.data.targets:
            df[f'{target}_binary'] = (df[target] == 2).astype(int)

        # Get all unique images
        all_images = list(df['im1_path'].unique()) + list(df['im2_path'].unique())
        all_images = list(set(all_images))

    print(f"Loaded {len(df)} samples with {len(all_images)} unique images")

    # Load or extract embeddings
    cache_file = cfg.data.get('embeddings_cache_file', None)
    if cache_file and not cache_file.startswith('/'):
        cache_file = f'artifacts/cache/{cache_file}'

    embeddings = load_or_extract_embeddings(
        all_images,
        cache_file=cache_file,
        device=device,
        force_recompute=cfg.data.get('force_recompute_embeddings', False)
    )

    # Prepare user encoder if needed
    user_encoder = None
    n_users = None
    if cfg.model.get('use_user_encoding', False):
        user_encoder = LabelEncoder()
        user_encoder.fit(df['user_id'].unique())
        n_users = len(user_encoder.classes_)
        print(f"Using user encoding with {n_users} users")

    # Prepare data
    all_data = prepare_data(df, embeddings, cfg, user_encoder)

    # Split data and prepare for training
    train_data = {}
    val_data = {}

    for target, data in all_data.items():
        if cfg.task_type == 'rating':
            # Split rating data
            indices = np.arange(len(data['y']))
            train_idx, val_idx = train_test_split(
                indices,
                test_size=cfg.training.validation_split,
                random_state=42,
                stratify=data['y']
            )

            train_data[target] = {
                'X': torch.FloatTensor(data['X'][train_idx]).to(device),
                'y': torch.LongTensor(data['y'][train_idx]).to(device)
            }

            val_data[target] = {
                'X': torch.FloatTensor(data['X'][val_idx]).to(device),
                'y_np': data['y'][val_idx]
            }

            if 'user' in data:
                train_data[target]['user'] = torch.FloatTensor(data['user'][train_idx]).to(device)
                val_data[target]['user'] = torch.FloatTensor(data['user'][val_idx]).to(device)

        else:  # comparison
            # Split comparison data
            indices = np.arange(len(data['y']))
            train_idx, val_idx = train_test_split(
                indices,
                test_size=cfg.training.validation_split,
                random_state=42,
                stratify=data['y']
            )

            # Training data
            X1_train = data['X1'][train_idx]
            X2_train = data['X2'][train_idx]
            y_train = data['y'][train_idx]

            # Augmentation for comparison
            if cfg.training.get('augment_swapped_pairs', True):
                X1_aug = np.vstack([X1_train, X2_train])
                X2_aug = np.vstack([X2_train, X1_train])
                y_aug = np.hstack([y_train, 1 - y_train])

                # Shuffle
                shuffle_idx = np.random.permutation(len(y_aug))
                X1_train = X1_aug[shuffle_idx]
                X2_train = X2_aug[shuffle_idx]
                y_train = y_aug[shuffle_idx]

            train_data[target] = {
                'X1': torch.FloatTensor(X1_train).to(device),
                'X2': torch.FloatTensor(X2_train).to(device),
                'y': torch.LongTensor(y_train).to(device),
                'y_float': torch.FloatTensor(y_train).to(device)
            }

            val_data[target] = {
                'X1': torch.FloatTensor(data['X1'][val_idx]).to(device),
                'X2': torch.FloatTensor(data['X2'][val_idx]).to(device),
                'y_np': data['y'][val_idx]
            }

            if 'user' in data:
                user_train = data['user'][train_idx]
                if cfg.training.get('augment_swapped_pairs', True):
                    user_aug = np.vstack([user_train, user_train])
                    user_train = user_aug[shuffle_idx]

                train_data[target]['user'] = torch.FloatTensor(user_train).to(device)
                val_data[target]['user'] = torch.FloatTensor(data['user'][val_idx]).to(device)

    # Initialize model
    model = UnifiedModel(cfg, n_users).to(device)

    # Train model
    print("\nTraining unified model...")
    results = train_model(model, train_data, val_data, cfg, device)

    # Report results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Task: {cfg.task_type}")
    print(f"Model: Unified BaseEncoder")

    if cfg.model.get('use_user_encoding', False):
        print(f"User encoding: Yes ({n_users} users)")

    print()

    # Display and save results
    if cfg.task_type == 'rating':
        avg_acc = np.mean([r['accuracy'] for r in results.values()])
        avg_ce = np.mean([r['ce_loss'] for r in results.values()])
        avg_mae = np.mean([r['mae'] for r in results.values()])
        avg_rmse = np.mean([r['rmse'] for r in results.values()])

        for target, metrics in results.items():
            print(f"{target.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  CE Loss: {metrics['ce_loss']:.3f}")
            print(f"  MAE: {metrics['mae']:.3f}")
            print(f"  RMSE: {metrics['rmse']:.3f}")
            print(f"  Samples: {metrics['n_train']} train, {metrics['n_val']} val")
            print()

        print(f"Average Accuracy: {avg_acc:.3f}")
        print(f"Average CE Loss: {avg_ce:.3f}")
        print(f"Average MAE: {avg_mae:.3f}")
        print(f"Average RMSE: {avg_rmse:.3f}")

    else:  # comparison
        avg_acc = np.mean([r['accuracy'] for r in results.values()])
        avg_loss = np.mean([r['ce_loss'] for r in results.values()])

        for target, metrics in results.items():
            print(f"{target.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  CE Loss: {metrics['ce_loss']:.3f}")
            print(f"  Samples: {metrics['n_train']} train, {metrics['n_val']} val")
            print()

        print(f"Average Accuracy: {avg_acc:.3f}")
        print(f"Average CE Loss: {avg_loss:.3f}")

    # Save results
    results_df = pd.DataFrame(results).T
    results_file = run_dir / "results.csv"
    results_df.to_csv(results_file)
    print(f"\n✓ Results saved to {results_file}")

    # Save configuration
    config_file = run_dir / "config.yaml"
    with open(config_file, 'w') as f:
        f.write(str(cfg))
    print(f"✓ Configuration saved to {config_file}")

    # Create summary
    summary_file = run_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Task: {cfg.task_type}\n")
        f.write(f"Model: Unified BaseEncoder\n")
        f.write("-" * 50 + "\n")

        if cfg.task_type == 'rating':
            f.write(f"Average Accuracy: {avg_acc:.3f}\n")
            f.write(f"Average CE Loss: {avg_ce:.3f}\n")
            f.write(f"Average MAE: {avg_mae:.3f}\n")
            f.write(f"Average RMSE: {avg_rmse:.3f}\n")
        else:
            f.write(f"Average Accuracy: {avg_acc:.3f}\n")
            f.write(f"Average CE Loss: {avg_loss:.3f}\n")

    print(f"✓ Summary saved to {summary_file}")


if __name__ == "__main__":
    main()