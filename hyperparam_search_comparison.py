#!/usr/bin/env python3
"""
Comprehensive hyperparameter search for comparison task.
Tests various architectures and configurations to maximize validation accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import itertools
import logging
from torch.utils.data import Dataset, DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseEncoder(nn.Module):
    """Flexible base encoder with configurable architecture."""

    def __init__(self, config):
        super().__init__()

        # Calculate input dimension
        clip_dim = 512
        user_dim = config.get('user_embedding_dim', 0)
        input_dim = clip_dim * 2 if config.get('concat_features', False) else clip_dim

        if user_dim > 0 and config.get('n_users'):
            self.user_embedding = nn.Embedding(config['n_users'], user_dim)
            if config.get('concat_user', True):
                input_dim += user_dim
        else:
            self.user_embedding = None

        # Build layers
        layers = []
        prev_dim = input_dim
        hidden_dims = config['hidden_dims']

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Normalization
            if config.get('normalization') == 'batch':
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif config.get('normalization') == 'layer':
                layers.append(nn.LayerNorm(hidden_dim))

            # Activation
            activation = config.get('activation', 'relu')
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif activation == 'elu':
                layers.append(nn.ELU())
            elif activation == 'swish':
                layers.append(nn.SiLU())

            # Dropout
            dropout = config.get('dropout', 0.0)
            if dropout > 0 and i < len(hidden_dims) - 1:  # No dropout on last layer
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Output heads for targets
        self.heads = nn.ModuleDict({
            target: nn.Linear(prev_dim, 1)
            for target in ['attractive', 'smart', 'trustworthy']
        })

        # Optional: Add residual connections
        self.use_residual = config.get('use_residual', False)
        if self.use_residual and len(hidden_dims) > 1:
            # Add skip connections
            self.residual_proj = nn.Linear(input_dim, hidden_dims[-1])

    def forward(self, x, user_ids=None, target=None):
        """Forward pass with optional user embeddings."""

        # Store original input for residual
        original_x = x

        # Add user embeddings if available
        if self.user_embedding is not None and user_ids is not None:
            user_embeds = self.user_embedding(user_ids)
            x = torch.cat([x, user_embeds], dim=-1)

        # Encode
        features = self.encoder(x)

        # Add residual if configured
        if self.use_residual and hasattr(self, 'residual_proj'):
            features = features + self.residual_proj(original_x if self.user_embedding is None else x)

        # Get output for specific target
        if target:
            return self.heads[target](features)
        else:
            # Return all heads (for multi-task learning)
            return {t: self.heads[t](features) for t in self.heads}


class UnifiedComparisonModel(nn.Module):
    """Unified model for comparison task with configurable architecture."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base_encoder = BaseEncoder(config)

        # Optional: Feature fusion before encoding
        self.fusion_type = config.get('fusion_type', 'none')
        if self.fusion_type == 'attention':
            self.attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        elif self.fusion_type == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(1024, 512),
                nn.Sigmoid()
            )

    def forward(self, image1, image2, user_ids=None, target='attractive'):
        """Forward pass for comparison."""

        # Optional feature fusion
        if self.fusion_type == 'attention':
            # Stack images for attention
            stacked = torch.stack([image1, image2], dim=1)  # [batch, 2, 512]
            attended, _ = self.attention(stacked, stacked, stacked)
            image1, image2 = attended[:, 0], attended[:, 1]
        elif self.fusion_type == 'gated':
            # Gated fusion
            combined = torch.cat([image1, image2], dim=-1)
            gate_weights = self.gate(combined)
            image1 = image1 * gate_weights
            image2 = image2 * (1 - gate_weights)
        elif self.fusion_type == 'difference':
            # Use difference features
            diff = image1 - image2
            image1 = torch.cat([image1, diff], dim=-1)
            image2 = torch.cat([image2, -diff], dim=-1)
            self.config['concat_features'] = True

        # Process through encoder
        if self.config.get('concat_features', False):
            # Process concatenated features
            score1 = self.base_encoder(image1, user_ids, target)
            score2 = self.base_encoder(image2, user_ids, target)
        else:
            # Standard processing
            score1 = self.base_encoder(image1, user_ids, target)
            score2 = self.base_encoder(image2, user_ids, target)

        # Combine scores
        scores = torch.cat([score1, score2], dim=-1)
        return scores


def create_data_loaders(config, cache_file='artifacts/cache/clip_embeddings_cache.pkl'):
    """Create data loaders for comparison task."""

    # Load data
    with open('analysis/data/labels.pkl', 'rb') as f:
        df = pickle.load(f)

    # Load CLIP embeddings
    with open(cache_file, 'rb') as f:
        clip_cache = pickle.load(f)

    # Process comparison data
    comparison_data = []

    for _, row in df.iterrows():
        if pd.isna(row['label']):
            continue

        user_id = row['user_id']
        item_path = row.get('item_path', '')

        # Skip if not comparison task
        if 'comparison' not in str(item_path).lower():
            continue

        # Parse item path to get image pairs
        if isinstance(item_path, str) and 'src/' in item_path:
            parts = item_path.split('/')
            if len(parts) >= 2:
                try:
                    # Extract image names
                    img_parts = parts[-1].split('_')
                    if len(img_parts) >= 2:
                        img1 = f"/home/labeler/v3/web_customer_labeler/data/ffhq2/src/{img_parts[0]}.webp"
                        img2 = f"/home/labeler/v3/web_customer_labeler/data/ffhq2/src/{img_parts[1].split('.')[0]}.webp"

                        # Check if both images exist in cache
                        if img1 in clip_cache and img2 in clip_cache:
                            # Determine winner (label 0 or 1)
                            label = int(row['label']) if not pd.isna(row['label']) else None
                            if label is not None:
                                # Get target from path
                                target = 'attractive'  # Default
                                if 'smart' in item_path.lower() or 'intelligence' in item_path.lower():
                                    target = 'smart'
                                elif 'trust' in item_path.lower():
                                    target = 'trustworthy'

                                comparison_data.append({
                                    'user_id': user_id,
                                    'image1': img1,
                                    'image2': img2,
                                    'label': label,
                                    'target': target
                                })
                except:
                    continue

    if not comparison_data:
        # Use synthetic data for testing
        logger.info("Creating synthetic comparison data...")
        image_keys = list(clip_cache.keys())[:1000]
        targets = ['attractive', 'smart', 'trustworthy']
        users = list(range(10))

        for _ in range(5000):
            img1, img2 = np.random.choice(image_keys, 2, replace=False)
            comparison_data.append({
                'user_id': np.random.choice(users),
                'image1': img1,
                'image2': img2,
                'label': np.random.choice([0, 1]),
                'target': np.random.choice(targets)
            })

    # Convert to DataFrame and add augmentation
    comp_df = pd.DataFrame(comparison_data)

    # Data augmentation - swap images
    augmented_data = []
    for _, row in comp_df.iterrows():
        # Original
        augmented_data.append(row.to_dict())
        # Swapped
        swapped = row.to_dict()
        swapped['image1'] = row['image2']
        swapped['image2'] = row['image1']
        swapped['label'] = 1 - row['label']
        augmented_data.append(swapped)

    comp_df = pd.DataFrame(augmented_data)

    # Create user mapping
    unique_users = comp_df['user_id'].unique()
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    config['n_users'] = len(unique_users)

    # Split data
    train_df, val_df = train_test_split(comp_df, test_size=0.2, random_state=42)

    # Create dataset class
    class ComparisonDataset(Dataset):
        def __init__(self, data_df):
            self.data = data_df

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            img1_embed = torch.tensor(clip_cache[row['image1']], dtype=torch.float32)
            img2_embed = torch.tensor(clip_cache[row['image2']], dtype=torch.float32)

            user_idx = user_to_idx[row['user_id']]

            return img1_embed, img2_embed, user_idx, row['target'], row['label']

    # Create datasets and loaders
    train_dataset = ComparisonDataset(train_df)
    val_dataset = ComparisonDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 128),
                            shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 128),
                          shuffle=False, num_workers=0)

    return train_loader, val_loader, config


def train_model(config, train_loader, val_loader, epochs=30):
    """Train model with given configuration."""

    # Initialize model
    model = UnifiedComparisonModel(config).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    # Optimizer selection
    opt_type = config.get('optimizer', 'adamw')
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0.01)

    if opt_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif opt_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = None
    if config.get('use_scheduler', False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_acc = 0
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        for img1, img2, user_ids, targets, labels in train_loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            user_ids = user_ids.to(device)
            labels = labels.to(device)

            # Forward pass (process batch by target)
            batch_size = img1.size(0)
            outputs = []

            for i in range(batch_size):
                target = targets[i]
                output = model(img1[i:i+1], img2[i:i+1], user_ids[i:i+1], target)
                outputs.append(output)

            outputs = torch.cat(outputs, dim=0)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

            optimizer.step()

            train_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validate
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for img1, img2, user_ids, targets, labels in val_loader:
                img1 = img1.to(device)
                img2 = img2.to(device)
                user_ids = user_ids.to(device)
                labels = labels.to(device)

                # Forward pass
                batch_size = img1.size(0)
                outputs = []

                for i in range(batch_size):
                    target = targets[i]
                    output = model(img1[i:i+1], img2[i:i+1], user_ids[i:i+1], target)
                    outputs.append(output)

                outputs = torch.cat(outputs, dim=0)
                loss = criterion(outputs, labels)

                val_losses.append(loss.item())
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        val_loss = np.mean(val_losses)

        # Update best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss

        # Update scheduler
        if scheduler:
            scheduler.step()

    return best_val_acc, best_val_loss


def main():
    """Run hyperparameter search."""

    logger.info("="*60)
    logger.info("HYPERPARAMETER SEARCH FOR COMPARISON TASK")
    logger.info("="*60)

    # Define search space
    search_space = {
        'hidden_dims': [
            [512, 256, 128],
            [512, 256, 128, 64],
            [768, 384, 192],
            [1024, 512, 256, 128],
            [512, 512, 256],
            [256, 256, 128, 128],
        ],
        'activation': ['relu', 'gelu', 'leaky_relu', 'elu', 'swish'],
        'normalization': ['batch', 'layer', 'none'],
        'dropout': [0.0, 0.1, 0.2, 0.3],
        'optimizer': ['adam', 'adamw'],
        'lr': [0.001, 0.0005, 0.002],
        'weight_decay': [0.0, 0.01, 0.05],
        'user_embedding_dim': [0, 16, 32, 64],
        'fusion_type': ['none', 'attention', 'gated', 'difference'],
        'use_residual': [False, True],
        'use_scheduler': [False, True],
        'grad_clip': [0, 1.0],
        'batch_size': [64, 128, 256],
    }

    # Random search - test N configurations
    n_trials = 20
    results = []

    logger.info(f"Testing {n_trials} random configurations...")

    for trial in range(n_trials):
        # Sample random configuration
        config = {
            'hidden_dims': np.random.choice(search_space['hidden_dims']),
            'activation': np.random.choice(search_space['activation']),
            'normalization': np.random.choice(search_space['normalization']),
            'dropout': np.random.choice(search_space['dropout']),
            'optimizer': np.random.choice(search_space['optimizer']),
            'lr': np.random.choice(search_space['lr']),
            'weight_decay': np.random.choice(search_space['weight_decay']),
            'user_embedding_dim': np.random.choice(search_space['user_embedding_dim']),
            'fusion_type': np.random.choice(search_space['fusion_type']),
            'use_residual': np.random.choice(search_space['use_residual']),
            'use_scheduler': np.random.choice(search_space['use_scheduler']),
            'grad_clip': np.random.choice(search_space['grad_clip']),
            'batch_size': np.random.choice(search_space['batch_size']),
        }

        logger.info(f"\nTrial {trial+1}/{n_trials}")
        logger.info(f"Config: {config}")

        try:
            # Create data loaders
            train_loader, val_loader, config = create_data_loaders(config)

            # Train model
            val_acc, val_loss = train_model(config, train_loader, val_loader, epochs=20)

            # Store results
            results.append({
                'trial': trial + 1,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config
            })

            logger.info(f"Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

        except Exception as e:
            logger.error(f"Trial {trial+1} failed: {e}")
            continue

    # Sort results by validation accuracy
    results.sort(key=lambda x: x['val_acc'], reverse=True)

    # Report top configurations
    logger.info("\n" + "="*60)
    logger.info("TOP 5 CONFIGURATIONS")
    logger.info("="*60)

    for i, result in enumerate(results[:5]):
        logger.info(f"\n#{i+1} - Val Acc: {result['val_acc']:.4f}, Val Loss: {result['val_loss']:.4f}")
        logger.info(f"Config: {result['config']}")

    # Save best configuration
    best_config = results[0]['config']
    with open('best_comparison_config.pkl', 'wb') as f:
        pickle.dump(best_config, f)

    logger.info("\n" + "="*60)
    logger.info(f"BEST VALIDATION ACCURACY: {results[0]['val_acc']:.4f}")
    logger.info("="*60)


if __name__ == "__main__":
    main()