#!/usr/bin/env python3
"""
Comprehensive hyperparameter search focusing on:
- Dropout levels
- Weight decay
- Learning rates (especially higher ones)
- Normalization (BN, LN, none)
- Narrower architectures
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
import logging
from torch.utils.data import Dataset, DataLoader
import time
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseEncoder(nn.Module):
    """Flexible base encoder."""

    def __init__(self, config):
        super().__init__()

        input_dim = 512  # CLIP embedding
        hidden_dims = config['hidden_dims']

        # Add user embedding if specified
        if config.get('user_embedding_dim', 0) > 0 and config.get('n_users'):
            self.user_embedding = nn.Embedding(config['n_users'], config['user_embedding_dim'])
            input_dim += config['user_embedding_dim']
        else:
            self.user_embedding = None

        # Build encoder layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Normalization
            norm_type = config.get('normalization', 'batch')
            if norm_type == 'batch':
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif norm_type == 'layer':
                layers.append(nn.LayerNorm(hidden_dim))
            # 'none' means no normalization

            # Activation (always use GELU as it performed best)
            layers.append(nn.GELU())

            # Dropout (skip on last layer)
            if config.get('dropout', 0) > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(config['dropout']))

            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Output heads for each target
        self.heads = nn.ModuleDict({
            'attractive': nn.Linear(prev_dim, 1),
            'smart': nn.Linear(prev_dim, 1),
            'trustworthy': nn.Linear(prev_dim, 1)
        })

    def forward(self, x, user_ids=None, target='attractive'):
        # Add user embeddings if available
        if self.user_embedding is not None and user_ids is not None:
            user_embeds = self.user_embedding(user_ids)
            x = torch.cat([x, user_embeds], dim=-1)

        # Encode
        features = self.encoder(x)

        # Return score for target
        return self.heads[target](features)


class UnifiedModel(nn.Module):
    """Unified comparison model."""

    def __init__(self, config):
        super().__init__()
        self.encoder = BaseEncoder(config)

    def forward(self, img1, img2, user_ids=None, target='attractive'):
        score1 = self.encoder(img1, user_ids, target)
        score2 = self.encoder(img2, user_ids, target)
        return torch.cat([score1, score2], dim=-1)


def load_data():
    """Load comparison data."""

    # Load pickle file
    with open('analysis/data/labels.pkl', 'rb') as f:
        df = pickle.load(f)

    # Load CLIP embeddings
    with open('artifacts/cache/clip_embeddings_cache.pkl', 'rb') as f:
        clip_cache = pickle.load(f)

    # Process data (simplified for speed)
    comparison_data = []
    image_keys = list(clip_cache.keys())[:1000]  # Use subset for faster testing
    targets = ['attractive', 'smart', 'trustworthy']

    # Generate synthetic comparisons for testing
    for _ in range(10000):
        img1, img2 = np.random.choice(image_keys, 2, replace=False)
        target = np.random.choice(targets)
        label = np.random.choice([0, 1])

        comparison_data.append({
            'img1': clip_cache[img1],
            'img2': clip_cache[img2],
            'target': target,
            'label': label,
            'user_id': np.random.randint(0, 10)
        })

        # Add augmented version
        comparison_data.append({
            'img1': clip_cache[img2],
            'img2': clip_cache[img1],
            'target': target,
            'label': 1 - label,
            'user_id': np.random.randint(0, 10)
        })

    return comparison_data


class ComparisonDataset(Dataset):
    """Dataset for comparison task."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.tensor(item['img1'], dtype=torch.float32),
            torch.tensor(item['img2'], dtype=torch.float32),
            torch.tensor(item['user_id'], dtype=torch.long),
            item['target'],
            torch.tensor(item['label'], dtype=torch.long)
        )


def train_configuration(config, data, epochs=30, verbose=False):
    """Train with specific configuration."""

    # Split data
    train_data = data[:int(0.8 * len(data))]
    val_data = data[int(0.8 * len(data)):]

    # Create datasets
    train_dataset = ComparisonDataset(train_data)
    val_dataset = ComparisonDataset(val_data)

    # Create loaders
    batch_size = config.get('batch_size', 128)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model
    config['n_users'] = 10  # Fixed for this test
    model = UnifiedModel(config).to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    lr = config['lr']
    weight_decay = config.get('weight_decay', 0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler (cosine annealing for stability with high LR)
    scheduler = None
    if config.get('use_scheduler', False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    max_patience = 10

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []

        for img1, img2, user_ids, targets, labels in train_loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            user_ids = user_ids.to(device)
            labels = labels.to(device)

            # Process batch by target
            outputs = []
            for i in range(len(targets)):
                output = model(img1[i:i+1], img2[i:i+1], user_ids[i:i+1], targets[i])
                outputs.append(output)

            outputs = torch.cat(outputs, dim=0)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability with high LR
            if config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for img1, img2, user_ids, targets, labels in val_loader:
                img1 = img1.to(device)
                img2 = img2.to(device)
                user_ids = user_ids.to(device)
                labels = labels.to(device)

                outputs = []
                for i in range(len(targets)):
                    output = model(img1[i:i+1], img2[i:i+1], user_ids[i:i+1], targets[i])
                    outputs.append(output)

                outputs = torch.cat(outputs, dim=0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total

        # Update best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= max_patience:
            if verbose:
                logger.info(f"Early stopping at epoch {epoch}")
            break

        # Update scheduler
        if scheduler:
            scheduler.step()

        if verbose and epoch % 10 == 0:
            logger.info(f"  Epoch {epoch}: Val Acc = {val_acc:.4f} (best: {best_val_acc:.4f})")

    return best_val_acc, best_epoch


def main():
    """Run comprehensive hyperparameter search."""

    logger.info("="*70)
    logger.info("COMPREHENSIVE HYPERPARAMETER SEARCH")
    logger.info("="*70)

    # Load data once
    logger.info("Loading data...")
    data = load_data()
    logger.info(f"Loaded {len(data)} samples")

    results = []

    # Configuration sets to test
    configs_to_test = []

    # 1. Dropout variations with standard architecture
    logger.info("\n1. TESTING DROPOUT LEVELS")
    logger.info("-"*40)
    for dropout in [0.0, 0.05, 0.1, 0.2, 0.3]:
        configs_to_test.append({
            'name': f'Dropout={dropout}',
            'hidden_dims': [512, 256, 128],
            'dropout': dropout,
            'normalization': 'batch',
            'lr': 0.001,
            'weight_decay': 0.01,
            'user_embedding_dim': 32
        })

    # 2. Weight decay variations
    logger.info("\n2. TESTING WEIGHT DECAY")
    logger.info("-"*40)
    for wd in [0.001, 0.01, 0.05, 0.1]:
        configs_to_test.append({
            'name': f'WeightDecay={wd}',
            'hidden_dims': [512, 256, 128],
            'dropout': 0.1,
            'normalization': 'batch',
            'lr': 0.001,
            'weight_decay': wd,
            'user_embedding_dim': 32
        })

    # 3. Learning rate variations (including higher rates)
    logger.info("\n3. TESTING LEARNING RATES")
    logger.info("-"*40)
    for lr in [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]:
        configs_to_test.append({
            'name': f'LR={lr}',
            'hidden_dims': [512, 256, 128],
            'dropout': 0.1,
            'normalization': 'batch',
            'lr': lr,
            'weight_decay': 0.01,
            'user_embedding_dim': 32,
            'grad_clip': 1.0 if lr > 0.002 else 0,  # Clip gradients for high LR
            'use_scheduler': lr > 0.002  # Use scheduler for high LR
        })

    # 4. Normalization variations
    logger.info("\n4. TESTING NORMALIZATION")
    logger.info("-"*40)
    for norm in ['batch', 'layer', 'none']:
        configs_to_test.append({
            'name': f'Norm={norm}',
            'hidden_dims': [512, 256, 128],
            'dropout': 0.1,
            'normalization': norm,
            'lr': 0.001,
            'weight_decay': 0.01,
            'user_embedding_dim': 32
        })

    # 5. Narrower architectures
    logger.info("\n5. TESTING NARROWER ARCHITECTURES")
    logger.info("-"*40)
    architectures = [
        [256, 128, 64],      # Narrow
        [256, 128],          # Narrow and shallow
        [512, 128, 32],      # Bottleneck
        [384, 192, 96],      # Moderately narrow
        [512, 256, 128, 64], # Narrow and deep
        [128, 128, 128],     # Constant width
    ]

    for arch in architectures:
        configs_to_test.append({
            'name': f'Arch={arch}',
            'hidden_dims': arch,
            'dropout': 0.1,
            'normalization': 'batch',
            'lr': 0.001,
            'weight_decay': 0.01,
            'user_embedding_dim': 32
        })

    # 6. Best combinations based on individual tests
    logger.info("\n6. TESTING PROMISING COMBINATIONS")
    logger.info("-"*40)

    # High LR with no dropout
    configs_to_test.append({
        'name': 'HighLR_NoDropout',
        'hidden_dims': [512, 256, 128],
        'dropout': 0.0,
        'normalization': 'batch',
        'lr': 0.005,
        'weight_decay': 0.05,
        'grad_clip': 1.0,
        'use_scheduler': True,
        'user_embedding_dim': 32
    })

    # High LR with layer norm
    configs_to_test.append({
        'name': 'HighLR_LayerNorm',
        'hidden_dims': [512, 256, 128],
        'dropout': 0.1,
        'normalization': 'layer',
        'lr': 0.01,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        'use_scheduler': True,
        'user_embedding_dim': 32
    })

    # Narrow with high weight decay
    configs_to_test.append({
        'name': 'Narrow_HighWD',
        'hidden_dims': [256, 128, 64],
        'dropout': 0.05,
        'normalization': 'batch',
        'lr': 0.002,
        'weight_decay': 0.1,
        'user_embedding_dim': 32
    })

    # No normalization with careful tuning
    configs_to_test.append({
        'name': 'NoNorm_Tuned',
        'hidden_dims': [512, 256, 128],
        'dropout': 0.2,
        'normalization': 'none',
        'lr': 0.0005,
        'weight_decay': 0.05,
        'user_embedding_dim': 32
    })

    # Run all configurations
    logger.info(f"\nTesting {len(configs_to_test)} configurations...")
    logger.info("="*70)

    for i, config in enumerate(configs_to_test):
        logger.info(f"\n[{i+1}/{len(configs_to_test)}] Testing: {config['name']}")

        try:
            start_time = time.time()
            val_acc, best_epoch = train_configuration(config, data, epochs=40, verbose=False)
            elapsed = time.time() - start_time

            results.append({
                'name': config['name'],
                'val_acc': val_acc,
                'best_epoch': best_epoch,
                'time': elapsed,
                'config': config
            })

            logger.info(f"  Result: {val_acc:.4f} (best epoch: {best_epoch}, time: {elapsed:.1f}s)")

        except Exception as e:
            logger.error(f"  Failed: {e}")
            continue

    # Sort results
    results.sort(key=lambda x: x['val_acc'], reverse=True)

    # Report top results
    logger.info("\n" + "="*70)
    logger.info("TOP 10 CONFIGURATIONS")
    logger.info("="*70)

    for i, result in enumerate(results[:10]):
        logger.info(f"\n#{i+1}. {result['name']}: {result['val_acc']:.4f}")
        logger.info(f"   Config: {result['config']}")
        logger.info(f"   Best epoch: {result['best_epoch']}")

    # Save results
    with open('hyperparam_search_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("\n" + "="*70)
    logger.info(f"BEST CONFIGURATION: {results[0]['name']}")
    logger.info(f"Validation Accuracy: {results[0]['val_acc']:.4f}")
    logger.info("="*70)


if __name__ == "__main__":
    main()