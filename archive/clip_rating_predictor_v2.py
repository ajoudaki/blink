#!/usr/bin/env python3
"""
CLIP-based rating predictor with Hydra configuration and embedding caching.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from clip_embeddings_extractor import load_or_extract_embeddings


class ConfigurableModel(nn.Module):
    """Configurable model with flexible architecture."""

    def __init__(self, input_dim=512, hidden_dims=[256, 128],
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

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_model(X_train, y_train, X_val, y_val, cfg, device):
    """Train model with configuration."""

    model = ConfigurableModel(
        hidden_dims=cfg.model.hidden_dims,
        activation=cfg.model.activation,
        dropout=cfg.model.dropout,
        use_batch_norm=cfg.model.use_batch_norm
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.MSELoss()

    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train.values).reshape(-1, 1).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val.values).reshape(-1, 1).to(device)

    best_val_loss = float('inf')

    for epoch in range(cfg.training.epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

    # Final metrics
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val).cpu().numpy()

    mae = mean_absolute_error(y_val.cpu().numpy(), val_predictions)
    rmse = np.sqrt(mean_squared_error(y_val.cpu().numpy(), val_predictions))

    return mae, rmse


@hydra.main(version_base=None, config_path="configs", config_name="rating_config")
def main(cfg: DictConfig):
    """Main execution with Hydra configuration."""

    # Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu.device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    label_df = pd.read_excel('data/big_label.xlsx')
    data_df = pd.read_excel('data/big_data.xlsx')
    df = data_df.merge(label_df, left_on='_id', right_on='item_id', how='inner')
    print(f"Loaded {len(df)} ratings")

    # Get embeddings with caching
    unique_images = df['data_image_part'].unique()
    embeddings = load_or_extract_embeddings(
        unique_images,
        cache_file=cfg.data.embeddings_cache_file,
        device=device
    )

    # Train models
    print("\nTraining models...")
    targets = ['attractive', 'smart', 'trustworthy']
    results = []

    # Filter users with enough data
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 10].index.tolist()[:5]  # Demo with 5 users

    for target in targets:
        for user_id in tqdm(valid_users, desc=f"Training for {target}"):
            user_data = df[df['user_id'] == user_id].copy()

            if len(user_data) < 5:
                continue

            # Get embeddings
            X = np.array([embeddings[img] for img in user_data['data_image_part']])
            y = user_data[target]

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=cfg.training.validation_split, random_state=42
            )

            # Train
            mae, rmse = train_model(X_train, y_train, X_val, y_val, cfg, device)

            results.append({
                'user_id': user_id,
                'target': target,
                'val_mae': mae,
                'val_rmse': rmse
            })

    # Report results
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for target in targets:
        target_results = results_df[results_df['target'] == target]
        print(f"\n{target.upper()}:")
        print(f"  Mean VAL MAE: {target_results['val_mae'].mean():.3f}")
        print(f"  Mean VAL RMSE: {target_results['val_rmse'].mean():.3f}")

    results_df.to_csv('clip_rating_results_v2.csv', index=False)
    print(f"\n✓ Results saved to 'clip_rating_results_v2.csv'")


if __name__ == "__main__":
    main()