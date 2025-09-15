#!/usr/bin/env python3
"""
CLIP-based comparison predictor with Hydra configuration and embedding caching.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from clip_embeddings_extractor import load_or_extract_embeddings


class ConfigurableComparisonModel(nn.Module):
    """Configurable comparison model with flexible architecture."""

    def __init__(self, input_dim=1024, hidden_dims=[128, 64],
                 activation='relu', dropout=0.0, use_batch_norm=False):
        super().__init__()

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Add activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, img1, img2):
        # Concatenate embeddings
        x = torch.cat([img1, img2], dim=1)
        return self.model(x)


def create_augmented_dataset(X1, X2, y):
    """Create augmented dataset with swapped pairs."""
    X1_aug = np.vstack([X1, X2])
    X2_aug = np.vstack([X2, X1])
    y_aug = np.hstack([y, 1 - y])

    # Shuffle
    indices = np.random.permutation(len(y_aug))
    return X1_aug[indices], X2_aug[indices], y_aug[indices]


def train_comparison_model(X1_train, X2_train, y_train, X1_val, X2_val, y_val,
                          cfg, device):
    """Train binary classification model."""

    model = ConfigurableComparisonModel(
        input_dim=1024,  # Two 512-dim embeddings concatenated
        hidden_dims=cfg.model.hidden_dims,
        activation=cfg.model.activation,
        dropout=cfg.model.dropout,
        use_batch_norm=cfg.model.use_batch_norm
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.BCELoss()

    # Convert to tensors
    X1_train = torch.FloatTensor(X1_train).to(device)
    X2_train = torch.FloatTensor(X2_train).to(device)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(device)

    X1_val = torch.FloatTensor(X1_val).to(device)
    X2_val = torch.FloatTensor(X2_val).to(device)

    best_val_acc = 0

    for epoch in range(cfg.training.epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X1_train, X2_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X1_val, X2_val)
            val_pred = (val_outputs > 0.5).cpu().numpy().flatten()
            val_acc = accuracy_score(y_val, val_pred)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

    # Final metrics
    model.eval()
    with torch.no_grad():
        val_outputs = model(X1_val, X2_val).cpu().numpy().flatten()
        val_pred = (val_outputs > 0.5).astype(int)

    accuracy = accuracy_score(y_val, val_pred)
    try:
        auc = roc_auc_score(y_val, val_outputs)
    except:
        auc = 0.5

    # Test anti-symmetry
    with torch.no_grad():
        forward = model(X1_val[:10], X2_val[:10]).cpu().numpy()
        backward = model(X2_val[:10], X1_val[:10]).cpu().numpy()
        antisym_error = np.mean(np.abs(forward + backward - 1.0))

    return accuracy, auc, antisym_error


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
    all_images = list(set(all_images))  # Remove duplicates
    embeddings = load_or_extract_embeddings(
        all_images,
        cache_file=cfg.data.embeddings_cache_file,
        device=device
    )

    # Train models
    print("\nTraining comparison models...")
    targets = ['attractive', 'smart', 'trustworthy']
    results = []

    # Filter users with enough comparisons
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 20].index.tolist()[:5]  # Demo with 5 users

    for target in targets:
        for user_id in tqdm(valid_users, desc=f"Training for {target}"):
            user_data = df[df['user_id'] == user_id].copy()

            if len(user_data) < 10:
                continue

            # Get embeddings
            X1 = np.array([embeddings.get(img, np.zeros(512))
                          for img in user_data['im1_path']])
            X2 = np.array([embeddings.get(img, np.zeros(512))
                          for img in user_data['im2_path']])
            y = user_data[f'{target}_binary'].values

            # Split data
            X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
                X1, X2, y, test_size=cfg.training.validation_split, random_state=42,
                stratify=y if len(np.unique(y)) > 1 else None
            )

            # Augment training data if configured
            if cfg.training.augment_swapped_pairs:
                X1_train, X2_train, y_train = create_augmented_dataset(
                    X1_train, X2_train, y_train
                )

            # Train
            accuracy, auc, antisym_error = train_comparison_model(
                X1_train, X2_train, y_train,
                X1_val, X2_val, y_val,
                cfg, device
            )

            results.append({
                'user_id': user_id,
                'target': target,
                'val_accuracy': accuracy,
                'val_auc': auc,
                'antisym_error': antisym_error
            })

    # Report results
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for target in targets:
        target_results = results_df[results_df['target'] == target]
        print(f"\n{target.upper()}:")
        print(f"  Mean Accuracy: {target_results['val_accuracy'].mean():.3f}")
        print(f"  Mean AUC: {target_results['val_auc'].mean():.3f}")
        print(f"  Mean Anti-symmetry Error: {target_results['antisym_error'].mean():.4f}")

    results_df.to_csv('clip_comparison_results_v2.csv', index=False)
    print(f"\n✓ Results saved to 'clip_comparison_results_v2.csv'")


if __name__ == "__main__":
    main()