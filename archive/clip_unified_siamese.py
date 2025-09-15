#!/usr/bin/env python3
"""
Fully unified Siamese model with user one-hot encoding.
Single model for all users and all targets.
Input: [CLIP embedding | user one-hot encoding]
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
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from clip_embeddings_extractor import load_or_extract_embeddings


class UnifiedSiameseEncoder(nn.Module):
    """
    Unified encoder that takes CLIP embedding + user one-hot encoding.
    Single model for all users with 3 output heads for targets.
    """
    def __init__(self, input_dim, n_users, hidden_dims=[256, 128],
                 activation='relu', dropout=0.2, use_batch_norm=False):
        super().__init__()

        # Input dimension is CLIP (512) + user one-hot (n_users)
        total_input_dim = input_dim + n_users

        # Shared backbone layers
        backbone_layers = []
        prev_dim = total_input_dim

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

        # Three output heads (one per target)
        self.head_attractive = nn.Linear(prev_dim, 1)
        self.head_smart = nn.Linear(prev_dim, 1)
        self.head_trustworthy = nn.Linear(prev_dim, 1)

        # Store head mapping
        self.heads = {
            'attractive': self.head_attractive,
            'smart': self.head_smart,
            'trustworthy': self.head_trustworthy
        }

        self.n_users = n_users

    def forward(self, img_embedding, user_onehot, target='attractive'):
        """
        Forward pass with user conditioning.
        Args:
            img_embedding: Image CLIP embedding (batch_size, 512)
            user_onehot: User one-hot encoding (batch_size, n_users)
            target: Which output head to use
        """
        # Concatenate image and user features
        x = torch.cat([img_embedding, user_onehot], dim=1)

        # Shared backbone
        features = self.backbone(x)

        # Target-specific head
        score = self.heads[target](features)
        return score.squeeze(-1)  # Return shape (batch_size,)


class UnifiedSiameseModel(nn.Module):
    """
    Unified Siamese model for all users and targets.
    """
    def __init__(self, input_dim, n_users, hidden_dims=[256, 128],
                 activation='relu', dropout=0.2, use_batch_norm=False):
        super().__init__()

        self.encoder = UnifiedSiameseEncoder(
            input_dim=input_dim,
            n_users=n_users,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )

    def forward(self, img1, user1_onehot, img2, user2_onehot, target='attractive'):
        """
        Forward pass for comparison.
        Note: user1_onehot and user2_onehot are the same for a given comparison.
        """
        # Get scores for both images
        score1 = self.encoder(img1, user1_onehot, target)
        score2 = self.encoder(img2, user2_onehot, target)

        # Stack scores and apply softmax
        scores = torch.stack([score1, score2], dim=1)
        probs = F.softmax(scores, dim=1)

        # Return probability of choosing second image
        return probs[:, 1]


def create_unified_dataset(df, embeddings, user_encoder):
    """
    Create unified dataset with all users and targets.
    Returns data with user one-hot encodings.
    """
    n_users = len(user_encoder.classes_)

    all_data = {}

    for target in ['attractive', 'smart', 'trustworthy']:
        X1_list = []
        X2_list = []
        user_onehot_list = []
        y_list = []

        for _, row in df.iterrows():
            # Get embeddings
            img1_emb = embeddings.get(row['im1_path'], np.zeros(512))
            img2_emb = embeddings.get(row['im2_path'], np.zeros(512))

            # Create user one-hot encoding
            user_idx = user_encoder.transform([row['user_id']])[0]
            user_onehot = np.zeros(n_users)
            user_onehot[user_idx] = 1.0

            X1_list.append(img1_emb)
            X2_list.append(img2_emb)
            user_onehot_list.append(user_onehot)
            y_list.append(row[f'{target}_binary'])

        all_data[target] = {
            'X1': np.array(X1_list),
            'X2': np.array(X2_list),
            'user_onehot': np.array(user_onehot_list),
            'y': np.array(y_list)
        }

    return all_data, n_users


def train_unified_model(all_data, n_users, cfg, device):
    """
    Train single unified model on all users and targets.
    """

    # Initialize model
    model = UnifiedSiameseModel(
        input_dim=512,
        n_users=n_users,
        hidden_dims=cfg.model.hidden_dims,
        activation=cfg.model.activation,
        dropout=cfg.model.dropout,
        use_batch_norm=cfg.model.use_batch_norm
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.BCELoss()

    # Prepare datasets for all targets
    train_data = {}
    val_data = {}

    for target, data in all_data.items():
        # Split data
        indices = np.arange(len(data['y']))
        train_idx, val_idx = train_test_split(
            indices, test_size=cfg.training.validation_split,
            random_state=42, stratify=data['y']
        )

        # Training data
        X1_train = data['X1'][train_idx]
        X2_train = data['X2'][train_idx]
        user_train = data['user_onehot'][train_idx]
        y_train = data['y'][train_idx]

        # Validation data
        X1_val = data['X1'][val_idx]
        X2_val = data['X2'][val_idx]
        user_val = data['user_onehot'][val_idx]
        y_val = data['y'][val_idx]

        # Augment training data
        if cfg.training.augment_swapped_pairs:
            X1_aug = np.vstack([X1_train, X2_train])
            X2_aug = np.vstack([X2_train, X1_train])
            user_aug = np.vstack([user_train, user_train])
            y_aug = np.hstack([y_train, 1 - y_train])

            indices = np.random.permutation(len(y_aug))
            X1_train = X1_aug[indices]
            X2_train = X2_aug[indices]
            user_train = user_aug[indices]
            y_train = y_aug[indices]

        # Convert to tensors
        train_data[target] = {
            'X1': torch.FloatTensor(X1_train).to(device),
            'X2': torch.FloatTensor(X2_train).to(device),
            'user': torch.FloatTensor(user_train).to(device),
            'y': torch.FloatTensor(y_train).to(device),
            'n_train': len(y_train)
        }

        val_data[target] = {
            'X1': torch.FloatTensor(X1_val).to(device),
            'X2': torch.FloatTensor(X2_val).to(device),
            'user': torch.FloatTensor(user_val).to(device),
            'y_np': y_val,
            'n_val': len(y_val)
        }

    # Training loop
    print("\nTraining unified model...")
    targets = list(train_data.keys())
    train_losses = []

    for epoch in tqdm(range(cfg.training.epochs), desc="Training epochs"):
        epoch_losses = []

        # Train on each target
        for target in targets:
            train = train_data[target]

            model.train()
            optimizer.zero_grad()

            # Forward pass
            probs = model(train['X1'], train['user'], train['X2'], train['user'], target=target)
            loss = criterion(probs, train['y'])

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        train_losses.append(np.mean(epoch_losses))

    # Evaluation
    print("\nEvaluating unified model...")
    results = {}

    model.eval()
    for target in targets:
        val = val_data[target]

        with torch.no_grad():
            # Get predictions
            val_probs = model(val['X1'], val['user'], val['X2'], val['user'], target=target)
            val_probs_np = val_probs.cpu().numpy()
            val_pred = (val_probs_np > 0.5).astype(int)

            # Calculate metrics
            accuracy = accuracy_score(val['y_np'], val_pred)
            ce_loss = log_loss(val['y_np'], val_probs_np, labels=[0, 1])

            results[target] = {
                'accuracy': accuracy,
                'ce_loss': ce_loss,
                'n_train': train_data[target]['n_train'],
                'n_val': val['n_val']
            }

    return results, train_losses, model


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

    # Encode user IDs
    user_encoder = LabelEncoder()
    user_encoder.fit(df['user_id'].unique())
    n_users = len(user_encoder.classes_)
    print(f"\nNumber of unique users: {n_users}")

    # Create unified dataset
    print("\nCreating unified dataset with user one-hot encodings...")
    all_data, n_users = create_unified_dataset(df, embeddings, user_encoder)

    # Print dataset statistics
    for target in ['attractive', 'smart', 'trustworthy']:
        n_samples = len(all_data[target]['y'])
        print(f"  {target}: {n_samples} samples")

    # Train unified model
    results, train_losses, model = train_unified_model(all_data, n_users, cfg, device)

    # Report results
    print("\n" + "=" * 60)
    print("UNIFIED SIAMESE MODEL RESULTS")
    print("=" * 60)
    print(f"Single model for {n_users} users and 3 targets")
    print()

    # Store results for analysis
    all_results = []

    for target in ['attractive', 'smart', 'trustworthy']:
        metrics = results[target]

        print(f"{target.upper()}:")
        print(f"  Validation Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Validation CE Loss: {metrics['ce_loss']:.3f}")
        print(f"  Training samples: {metrics['n_train']}")
        print(f"  Validation samples: {metrics['n_val']}")
        print()

        all_results.append({
            'target': target,
            'val_accuracy': metrics['accuracy'],
            'val_ce_loss': metrics['ce_loss'],
            'n_train': metrics['n_train'],
            'n_val': metrics['n_val']
        })

    # Overall statistics
    results_df = pd.DataFrame(all_results)
    print("OVERALL:")
    print(f"  Mean Accuracy: {results_df['val_accuracy'].mean():.3f}")
    print(f"  Mean CE Loss: {results_df['val_ce_loss'].mean():.3f}")
    print(f"  Total parameters: ~{sum(p.numel() for p in model.parameters()) / 1000:.0f}K")
    print(f"  Final training loss: {train_losses[-1]:.3f}")

    # Save results
    results_df.to_csv('clip_unified_siamese_results.csv', index=False)
    print(f"\n✓ Results saved to 'clip_unified_siamese_results.csv'")

    # Comparison with previous approaches
    print("\n" + "=" * 60)
    print("UNIFIED MODEL ADVANTAGES")
    print("=" * 60)
    print(f"1. Single model for all {n_users} users")
    print("2. User preferences encoded as learnable features")
    print("3. Enables prediction for new users with few samples")
    print("4. Cross-user transfer learning")
    print("5. Most parameter efficient approach")


if __name__ == "__main__":
    main()