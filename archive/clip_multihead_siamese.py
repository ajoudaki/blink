#!/usr/bin/env python3
"""
Multi-head Siamese architecture for pairwise comparison.
Single shared model with 3 output heads (attractive, smart, trustworthy).
One model per user trained on all targets.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from collections import defaultdict

from clip_embeddings_extractor import load_or_extract_embeddings


class MultiHeadSiameseEncoder(nn.Module):
    """
    Siamese encoder with shared backbone and 3 output heads.
    Each head produces a score for one target (attractive, smart, trustworthy).
    """
    def __init__(self, input_dim=512, hidden_dims=[256, 128],
                 activation='relu', dropout=0.2, use_batch_norm=False):
        super().__init__()

        # Shared backbone layers
        backbone_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            backbone_layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                backbone_layers.append(nn.BatchNorm1d(hidden_dim))

            # Add activation
            if activation == 'relu':
                backbone_layers.append(nn.ReLU())
            elif activation == 'tanh':
                backbone_layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                backbone_layers.append(nn.LeakyReLU())

            if dropout > 0:
                backbone_layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*backbone_layers)

        # Three separate output heads (one per target)
        self.head_attractive = nn.Linear(prev_dim, 1)
        self.head_smart = nn.Linear(prev_dim, 1)
        self.head_trustworthy = nn.Linear(prev_dim, 1)

        # Store head mapping
        self.heads = {
            'attractive': self.head_attractive,
            'smart': self.head_smart,
            'trustworthy': self.head_trustworthy
        }

    def forward(self, x, target='attractive'):
        """
        Forward pass for specific target.
        Args:
            x: Input image embedding
            target: Which output head to use ('attractive', 'smart', 'trustworthy')
        """
        # Shared backbone
        features = self.backbone(x)

        # Target-specific head
        score = self.heads[target](features)
        return score.squeeze(-1)  # Return shape (batch_size,)

    def get_all_scores(self, x):
        """Get scores from all three heads (for analysis)."""
        features = self.backbone(x)
        scores = {
            'attractive': self.head_attractive(features).squeeze(-1),
            'smart': self.head_smart(features).squeeze(-1),
            'trustworthy': self.head_trustworthy(features).squeeze(-1)
        }
        return scores


class MultiHeadSiameseModel(nn.Module):
    """
    Multi-head Siamese model for pairwise comparison.
    Single encoder with 3 output heads, applies softmax for probabilities.
    """
    def __init__(self, input_dim=512, hidden_dims=[256, 128],
                 activation='relu', dropout=0.2, use_batch_norm=False):
        super().__init__()

        # Single shared encoder with multiple heads
        self.encoder = MultiHeadSiameseEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )

    def forward(self, img1, img2, target='attractive'):
        """
        Forward pass for specific target comparison.
        """
        # Get scores for both images using target-specific head
        score1 = self.encoder(img1, target)  # (batch_size,)
        score2 = self.encoder(img2, target)  # (batch_size,)

        # Stack scores and apply softmax
        scores = torch.stack([score1, score2], dim=1)  # (batch_size, 2)
        probs = F.softmax(scores, dim=1)

        # Return probability of choosing second image
        return probs[:, 1]  # (batch_size,)


def create_multi_target_dataset(df, user_id, embeddings):
    """
    Create dataset for a single user with all three targets combined.
    Returns data organized by target.
    """
    user_data = df[df['user_id'] == user_id].copy()

    datasets = {}
    for target in ['attractive', 'smart', 'trustworthy']:
        # Get embeddings
        X1 = np.array([embeddings.get(img, np.zeros(512))
                      for img in user_data['im1_path']])
        X2 = np.array([embeddings.get(img, np.zeros(512))
                      for img in user_data['im2_path']])
        y = user_data[f'{target}_binary'].values

        datasets[target] = (X1, X2, y)

    return datasets


def train_multihead_model(datasets, cfg, device):
    """
    Train single multi-head model on all targets for one user.
    """

    # Initialize model
    model = MultiHeadSiameseModel(
        input_dim=512,
        hidden_dims=cfg.model.hidden_dims,
        activation=cfg.model.activation,
        dropout=cfg.model.dropout,
        use_batch_norm=cfg.model.use_batch_norm
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.BCELoss()

    # Prepare data for all targets
    all_data = {}
    for target, (X1, X2, y) in datasets.items():
        # Split data
        X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
            X1, X2, y, test_size=cfg.training.validation_split, random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None
        )

        # Augment if configured
        if cfg.training.augment_swapped_pairs:
            X1_aug = np.vstack([X1_train, X2_train])
            X2_aug = np.vstack([X2_train, X1_train])
            y_aug = np.hstack([y_train, 1 - y_train])

            indices = np.random.permutation(len(y_aug))
            X1_train = X1_aug[indices]
            X2_train = X2_aug[indices]
            y_train = y_aug[indices]

        # Convert to tensors
        all_data[target] = {
            'X1_train': torch.FloatTensor(X1_train).to(device),
            'X2_train': torch.FloatTensor(X2_train).to(device),
            'y_train': torch.FloatTensor(y_train).to(device),
            'X1_val': torch.FloatTensor(X1_val).to(device),
            'X2_val': torch.FloatTensor(X2_val).to(device),
            'y_val': y_val,  # Keep numpy for sklearn metrics
            'n_train': len(X1_train),
            'n_val': len(X1_val)
        }

    # Training loop - alternate between targets
    losses_per_target = defaultdict(list)
    targets = list(all_data.keys())

    for epoch in range(cfg.training.epochs):
        # Train on each target in sequence
        for target in targets:
            data = all_data[target]

            # Training step
            model.train()
            optimizer.zero_grad()

            # Forward pass with specific target
            probs = model(data['X1_train'], data['X2_train'], target=target)
            loss = criterion(probs, data['y_train'])

            loss.backward()
            optimizer.step()

            losses_per_target[target].append(loss.item())

    # Evaluate on all targets
    results = {}
    model.eval()

    for target in targets:
        data = all_data[target]

        with torch.no_grad():
            # Get predictions
            val_probs = model(data['X1_val'], data['X2_val'], target=target)
            val_probs_np = val_probs.cpu().numpy()
            val_pred = (val_probs_np > 0.5).astype(int)

            # Calculate metrics
            accuracy = accuracy_score(data['y_val'], val_pred)
            ce_loss = log_loss(data['y_val'], val_probs_np, labels=[0, 1])

            results[target] = {
                'accuracy': accuracy,
                'ce_loss': ce_loss,
                'n_train': data['n_train'],
                'n_val': data['n_val'],
                'final_train_loss': losses_per_target[target][-1]
            }

    return results


@hydra.main(version_base=None, config_path="configs", config_name="comparison_config")
def main(cfg: DictConfig):
    """Main execution with Hydra configuration."""

    # Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu.device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("\nLoading comparison data...")
    compare_labels = pd.read_excel('data/big_compare_label.xlsx')
    compare_data = pd.read_excel('data/big_compare_data.xlsx')
    df = compare_data.merge(compare_labels, left_on='_id', right_on='item_id', how='inner')
    print(f"Loaded {len(df)} comparisons")

    # Convert labels
    for target in ['attractive', 'smart', 'trustworthy']:
        df[f'{target}_binary'] = (df[target] == 2).astype(int)

    # Get embeddings with caching
    all_images = list(df['im1_path'].unique()) + list(df['im2_path'].unique())
    all_images = list(set(all_images))
    embeddings = load_or_extract_embeddings(
        all_images,
        cache_file=cfg.data.embeddings_cache_file,
        device=device
    )

    # Train models
    print("\nTraining Multi-Head Siamese models (one per user)...")
    print("=" * 60)

    # Filter users with enough comparisons
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 20].index.tolist()[:5]

    all_results = []
    targets = ['attractive', 'smart', 'trustworthy']

    for user_idx, user_id in enumerate(tqdm(valid_users, desc="Training users")):
        # Create multi-target dataset for this user
        datasets = create_multi_target_dataset(df, user_id, embeddings)

        # Skip if not enough data
        if any(len(y) < 10 for _, _, y in datasets.values()):
            continue

        # Train single model for all targets
        user_results = train_multihead_model(datasets, cfg, device)

        # Store results
        for target, metrics in user_results.items():
            all_results.append({
                'user_id': user_id,
                'user_idx': user_idx,
                'target': target,
                'val_accuracy': metrics['accuracy'],
                'val_ce_loss': metrics['ce_loss'],
                'train_loss': metrics['final_train_loss'],
                'n_train': metrics['n_train'],
                'n_val': metrics['n_val']
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Report results
    print("\n" + "=" * 60)
    print("MULTI-HEAD SIAMESE MODEL RESULTS")
    print("=" * 60)
    print("(Single model per user for all 3 targets)")
    print()

    # Per-target summary
    for target in targets:
        target_results = results_df[results_df['target'] == target]
        if len(target_results) > 0:
            print(f"{target.upper()}:")
            print(f"  Validation Accuracy: {target_results['val_accuracy'].mean():.3f} ± {target_results['val_accuracy'].std():.3f}")
            print(f"  Validation CE Loss: {target_results['val_ce_loss'].mean():.3f} ± {target_results['val_ce_loss'].std():.3f}")
            print()

    # Overall statistics
    print("OVERALL:")
    print(f"  Mean Accuracy: {results_df['val_accuracy'].mean():.3f}")
    print(f"  Mean CE Loss: {results_df['val_ce_loss'].mean():.3f}")
    print(f"  Models trained: {len(valid_users)} (one per user)")
    print(f"  Total evaluations: {len(results_df)} (3 targets × {len(valid_users)} users)")

    # Save results
    results_df.to_csv('clip_multihead_siamese_results.csv', index=False)
    print(f"\n✓ Detailed results saved to 'clip_multihead_siamese_results.csv'")

    # Comparison with separate models
    print("\n" + "=" * 60)
    print("BENEFITS OF MULTI-HEAD APPROACH")
    print("=" * 60)
    print("1. Single model per user (3x fewer models)")
    print("2. Shared representation learning across targets")
    print("3. More efficient training and inference")
    print("4. Potential for transfer learning between targets")


if __name__ == "__main__":
    main()