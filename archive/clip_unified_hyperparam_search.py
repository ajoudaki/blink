#!/usr/bin/env python3
"""
Hyperparameter search for unified Siamese model.
Tests different architectures, activations, regularization, and optimizers.
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
import itertools
from clip_embeddings_extractor import load_or_extract_embeddings


class UnifiedSiameseEncoder(nn.Module):
    """
    Unified encoder with flexible architecture.
    """
    def __init__(self, input_dim, n_users, hidden_dims=[256, 128],
                 activation='relu', dropout=0.2, use_batch_norm=False,
                 use_layer_norm=False):
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
            elif use_layer_norm:
                backbone_layers.append(nn.LayerNorm(hidden_dim))

            # Add activation
            if activation == 'relu':
                backbone_layers.append(nn.ReLU())
            elif activation == 'tanh':
                backbone_layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                backbone_layers.append(nn.LeakyReLU(0.1))
            elif activation == 'gelu':
                backbone_layers.append(nn.GELU())
            elif activation == 'elu':
                backbone_layers.append(nn.ELU())

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
        """
        # Concatenate image and user features
        x = torch.cat([img_embedding, user_onehot], dim=1)

        # Shared backbone
        features = self.backbone(x)

        # Target-specific head
        score = self.heads[target](features)
        return score.squeeze(-1)


class UnifiedSiameseModel(nn.Module):
    """
    Unified Siamese model for all users and targets.
    """
    def __init__(self, input_dim, n_users, hidden_dims=[256, 128],
                 activation='relu', dropout=0.2, use_batch_norm=False,
                 use_layer_norm=False):
        super().__init__()

        self.encoder = UnifiedSiameseEncoder(
            input_dim=input_dim,
            n_users=n_users,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm
        )

    def forward(self, img1, user1_onehot, img2, user2_onehot, target='attractive'):
        """
        Forward pass for comparison.
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
    """Create unified dataset with all users and targets."""
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


def train_model_with_config(all_data, n_users, config, device, epochs=50, verbose=False):
    """Train model with specific configuration."""

    # Initialize model
    model = UnifiedSiameseModel(
        input_dim=512,
        n_users=n_users,
        hidden_dims=config['hidden_dims'],
        activation=config['activation'],
        dropout=config['dropout'],
        use_batch_norm=config.get('batch_norm', False),
        use_layer_norm=config.get('layer_norm', False)
    ).to(device)

    # Initialize optimizer
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config['lr'],
                                     weight_decay=config.get('weight_decay', 0))
    elif config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config['lr'],
                                      weight_decay=config.get('weight_decay', 0.01))
    elif config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config['lr'],
                                    momentum=0.9,
                                    weight_decay=config.get('weight_decay', 0))

    criterion = nn.BCELoss()

    # Prepare datasets
    train_data = {}
    val_data = {}

    for target, data in all_data.items():
        # Split data
        indices = np.arange(len(data['y']))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2,
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
            'y': torch.FloatTensor(y_train).to(device)
        }

        val_data[target] = {
            'X1': torch.FloatTensor(X1_val).to(device),
            'X2': torch.FloatTensor(X2_val).to(device),
            'user': torch.FloatTensor(user_val).to(device),
            'y_np': y_val
        }

    # Training loop
    targets = list(train_data.keys())

    for epoch in range(epochs):
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

    # Evaluation
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
                'ce_loss': ce_loss
            }

    # Calculate average metrics
    avg_accuracy = np.mean([results[t]['accuracy'] for t in targets])
    avg_ce_loss = np.mean([results[t]['ce_loss'] for t in targets])

    return avg_accuracy, avg_ce_loss, results


def main():
    """Main hyperparameter search."""

    # Setup
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

    # Get embeddings
    all_images = list(df['im1_path'].unique()) + list(df['im2_path'].unique())
    all_images = list(set(all_images))
    embeddings = load_or_extract_embeddings(
        all_images,
        cache_file='clip_embeddings_cache.pkl',
        device=device
    )

    # Encode user IDs
    user_encoder = LabelEncoder()
    user_encoder.fit(df['user_id'].unique())
    n_users = len(user_encoder.classes_)
    print(f"Number of unique users: {n_users}")

    # Create unified dataset
    print("\nCreating unified dataset...")
    all_data, n_users = create_unified_dataset(df, embeddings, user_encoder)

    # Define hyperparameter search space
    search_configs = [
        # Baseline
        {'name': 'Baseline', 'hidden_dims': [128, 64], 'activation': 'relu',
         'dropout': 0.2, 'lr': 0.001, 'optimizer': 'adam'},

        # Deeper networks
        {'name': 'Deep-3layer', 'hidden_dims': [256, 128, 64], 'activation': 'relu',
         'dropout': 0.2, 'lr': 0.001, 'optimizer': 'adam'},
        {'name': 'Deep-4layer', 'hidden_dims': [512, 256, 128, 64], 'activation': 'relu',
         'dropout': 0.3, 'lr': 0.001, 'optimizer': 'adam'},

        # Wider networks
        {'name': 'Wide', 'hidden_dims': [512, 256], 'activation': 'relu',
         'dropout': 0.2, 'lr': 0.001, 'optimizer': 'adam'},
        {'name': 'ExtraWide', 'hidden_dims': [1024, 512], 'activation': 'relu',
         'dropout': 0.3, 'lr': 0.001, 'optimizer': 'adam'},

        # Different activations
        {'name': 'GELU', 'hidden_dims': [256, 128], 'activation': 'gelu',
         'dropout': 0.2, 'lr': 0.001, 'optimizer': 'adam'},
        {'name': 'ELU', 'hidden_dims': [256, 128], 'activation': 'elu',
         'dropout': 0.2, 'lr': 0.001, 'optimizer': 'adam'},
        {'name': 'LeakyReLU', 'hidden_dims': [256, 128], 'activation': 'leaky_relu',
         'dropout': 0.2, 'lr': 0.001, 'optimizer': 'adam'},

        # Regularization
        {'name': 'BatchNorm', 'hidden_dims': [256, 128], 'activation': 'relu',
         'dropout': 0.1, 'batch_norm': True, 'lr': 0.001, 'optimizer': 'adam'},
        {'name': 'LayerNorm', 'hidden_dims': [256, 128], 'activation': 'relu',
         'dropout': 0.1, 'layer_norm': True, 'lr': 0.001, 'optimizer': 'adam'},
        {'name': 'HighDropout', 'hidden_dims': [256, 128], 'activation': 'relu',
         'dropout': 0.5, 'lr': 0.001, 'optimizer': 'adam'},

        # Different optimizers and learning rates
        {'name': 'AdamW', 'hidden_dims': [256, 128], 'activation': 'relu',
         'dropout': 0.2, 'lr': 0.001, 'optimizer': 'adamw', 'weight_decay': 0.01},
        {'name': 'LowLR', 'hidden_dims': [256, 128], 'activation': 'relu',
         'dropout': 0.2, 'lr': 0.0001, 'optimizer': 'adam'},
        {'name': 'HighLR', 'hidden_dims': [256, 128], 'activation': 'relu',
         'dropout': 0.2, 'lr': 0.005, 'optimizer': 'adam'},

        # Best practices combination
        {'name': 'BestPractice', 'hidden_dims': [512, 256, 128], 'activation': 'gelu',
         'dropout': 0.3, 'batch_norm': True, 'lr': 0.001, 'optimizer': 'adamw', 'weight_decay': 0.01},
    ]

    # Run hyperparameter search
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("=" * 60)

    results_list = []

    for config in tqdm(search_configs, desc="Testing configurations"):
        avg_acc, avg_loss, detailed = train_model_with_config(
            all_data, n_users, config, device, epochs=50
        )

        results_list.append({
            'name': config['name'],
            'avg_accuracy': avg_acc,
            'avg_ce_loss': avg_loss,
            'attractive_acc': detailed['attractive']['accuracy'],
            'smart_acc': detailed['smart']['accuracy'],
            'trustworthy_acc': detailed['trustworthy']['accuracy'],
            'config': str(config)
        })

        print(f"\n{config['name']:20s}: Acc={avg_acc:.3f}, Loss={avg_loss:.3f}")
        print(f"  Details: Attr={detailed['attractive']['accuracy']:.3f}, "
              f"Smart={detailed['smart']['accuracy']:.3f}, "
              f"Trust={detailed['trustworthy']['accuracy']:.3f}")

    # Sort by accuracy
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('avg_accuracy', ascending=False)

    print("\n" + "=" * 60)
    print("TOP 5 CONFIGURATIONS")
    print("=" * 60)

    for idx, row in results_df.head(5).iterrows():
        print(f"\n{row['name']:20s}")
        print(f"  Average Accuracy: {row['avg_accuracy']:.3f}")
        print(f"  Average CE Loss:  {row['avg_ce_loss']:.3f}")
        print(f"  Per-target: Attr={row['attractive_acc']:.3f}, "
              f"Smart={row['smart_acc']:.3f}, Trust={row['trustworthy_acc']:.3f}")

    # Save results
    results_df.to_csv('unified_hyperparam_search_results.csv', index=False)
    print(f"\n✓ Full results saved to 'unified_hyperparam_search_results.csv'")

    # Report improvement
    baseline_acc = results_df[results_df['name'] == 'Baseline']['avg_accuracy'].values[0]
    best_acc = results_df['avg_accuracy'].max()
    improvement = (best_acc - baseline_acc) * 100

    print(f"\n" + "=" * 60)
    print(f"IMPROVEMENT: {improvement:.1f}% over baseline")
    print(f"Best accuracy: {best_acc:.3f} (Baseline: {baseline_acc:.3f})")
    print("=" * 60)


if __name__ == "__main__":
    main()