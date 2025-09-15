#!/usr/bin/env python3
"""
Siamese architecture for pairwise comparison.
Each image gets a score independently, then softmax normalizes to probabilities.
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

from clip_embeddings_extractor import load_or_extract_embeddings


class SiameseEncoder(nn.Module):
    """
    Single encoder that maps image embedding to a score.
    This is the shared network applied to both images.
    """
    def __init__(self, input_dim=512, hidden_dims=[256, 128],
                 activation='relu', dropout=0.2, use_batch_norm=False):
        super().__init__()

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

        # Final layer outputs a single score (no activation)
        layers.append(nn.Linear(prev_dim, 1))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x).squeeze(-1)  # Return shape (batch_size,)


class SiameseComparisonModel(nn.Module):
    """
    Siamese model for pairwise comparison.
    Applies same encoder to both images, then softmax for probabilities.
    """
    def __init__(self, input_dim=512, hidden_dims=[256, 128],
                 activation='relu', dropout=0.2, use_batch_norm=False):
        super().__init__()

        # Single shared encoder
        self.encoder = SiameseEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )

    def forward(self, img1, img2):
        # Get scores for both images
        score1 = self.encoder(img1)  # (batch_size,)
        score2 = self.encoder(img2)  # (batch_size,)

        # Stack scores and apply softmax
        scores = torch.stack([score1, score2], dim=1)  # (batch_size, 2)
        probs = F.softmax(scores, dim=1)

        # Return probability of choosing second image
        return probs[:, 1]  # (batch_size,)

    def get_individual_scores(self, img1, img2):
        """Get raw scores before softmax (useful for analysis)"""
        with torch.no_grad():
            score1 = self.encoder(img1)
            score2 = self.encoder(img2)
        return score1, score2


def train_siamese_model(X1_train, X2_train, y_train, X1_val, X2_val, y_val,
                        cfg, device):
    """Train Siamese model with cross-entropy loss."""

    model = SiameseComparisonModel(
        input_dim=512,
        hidden_dims=cfg.model.hidden_dims,
        activation=cfg.model.activation,
        dropout=cfg.model.dropout,
        use_batch_norm=cfg.model.use_batch_norm
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.BCELoss()  # Binary cross-entropy for probabilities

    # Convert to tensors
    X1_train = torch.FloatTensor(X1_train).to(device)
    X2_train = torch.FloatTensor(X2_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)

    X1_val = torch.FloatTensor(X1_val).to(device)
    X2_val = torch.FloatTensor(X2_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)

    best_val_acc = 0
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(cfg.training.epochs):
        # Training
        model.train()
        optimizer.zero_grad()

        # Get probability of choosing second image
        probs = model(X1_train, X2_train)
        loss = criterion(probs, y_train)

        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_probs = model(X1_val, X2_val)
            val_loss = criterion(val_probs, y_val_tensor)
            val_losses.append(val_loss.item())

            # Calculate accuracy
            val_pred = (val_probs > 0.5).cpu().numpy().astype(int)
            val_acc = accuracy_score(y_val, val_pred)
            val_accuracies.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_probs = model(X1_val, X2_val).cpu().numpy()
        val_pred = (val_probs > 0.5).astype(int)

    # Calculate metrics
    final_accuracy = accuracy_score(y_val, val_pred)
    final_ce_loss = log_loss(y_val, val_probs, labels=[0, 1])

    # Test perfect siamese property: same encoder for both images
    with torch.no_grad():
        # Apply encoder to same images - should give same scores
        score1_a = model.encoder(X1_val[:10])
        score1_b = model.encoder(X1_val[:10])
        consistency_error = torch.mean(torch.abs(score1_a - score1_b)).item()

    return final_accuracy, final_ce_loss, consistency_error, train_losses[-1]


def create_augmented_dataset(X1, X2, y):
    """Create augmented dataset with swapped pairs."""
    X1_aug = np.vstack([X1, X2])
    X2_aug = np.vstack([X2, X1])
    y_aug = np.hstack([y, 1 - y])

    # Shuffle
    indices = np.random.permutation(len(y_aug))
    return X1_aug[indices], X2_aug[indices], y_aug[indices]


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
    print("\nTraining Siamese comparison models...")
    print("=" * 60)
    targets = ['attractive', 'smart', 'trustworthy']
    results = []

    # Filter users with enough comparisons
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 20].index.tolist()[:5]

    for target in targets:
        print(f"\n### {target.upper()} ###")
        target_results = []

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

            # Augment training data
            if cfg.training.augment_swapped_pairs:
                X1_train, X2_train, y_train = create_augmented_dataset(
                    X1_train, X2_train, y_train
                )

            # Train
            accuracy, ce_loss, consistency_err, train_loss = train_siamese_model(
                X1_train, X2_train, y_train,
                X1_val, X2_val, y_val,
                cfg, device
            )

            result = {
                'user_id': user_id,
                'target': target,
                'val_accuracy': accuracy,
                'val_ce_loss': ce_loss,
                'train_loss': train_loss,
                'consistency_error': consistency_err,
                'n_train': len(X1_train),
                'n_val': len(X1_val)
            }
            results.append(result)
            target_results.append(result)

        # Print target-specific results
        if target_results:
            avg_acc = np.mean([r['val_accuracy'] for r in target_results])
            avg_loss = np.mean([r['val_ce_loss'] for r in target_results])
            std_acc = np.std([r['val_accuracy'] for r in target_results])
            std_loss = np.std([r['val_ce_loss'] for r in target_results])

            print(f"  Validation Accuracy: {avg_acc:.3f} ± {std_acc:.3f}")
            print(f"  Validation CE Loss: {avg_loss:.3f} ± {std_loss:.3f}")

    # Save and report results
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("SIAMESE MODEL RESULTS SUMMARY")
    print("=" * 60)

    for target in targets:
        target_results = results_df[results_df['target'] == target]
        if len(target_results) > 0:
            print(f"\n{target.upper()}:")
            print(f"  Mean Accuracy: {target_results['val_accuracy'].mean():.3f} ± {target_results['val_accuracy'].std():.3f}")
            print(f"  Mean CE Loss: {target_results['val_ce_loss'].mean():.3f} ± {target_results['val_ce_loss'].std():.3f}")
            print(f"  Consistency Error: {target_results['consistency_error'].mean():.6f}")

    # Overall statistics
    print(f"\nOVERALL:")
    print(f"  Mean Accuracy: {results_df['val_accuracy'].mean():.3f}")
    print(f"  Mean CE Loss: {results_df['val_ce_loss'].mean():.3f}")
    print(f"  Baseline (random): 0.500 accuracy, 0.693 CE loss")

    results_df.to_csv('clip_siamese_results.csv', index=False)
    print(f"\n✓ Detailed results saved to 'clip_siamese_results.csv'")


if __name__ == "__main__":
    main()