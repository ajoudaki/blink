#!/usr/bin/env python3
"""
Test specific promising configurations for comparison task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from pathlib import Path
import logging
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImprovedEncoder(nn.Module):
    """Improved encoder with best practices."""

    def __init__(self, config):
        super().__init__()

        input_dim = 512  # CLIP embedding dim
        hidden_dims = config['hidden_dims']

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Normalization
            if config['norm'] == 'batch':
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif config['norm'] == 'layer':
                layers.append(nn.LayerNorm(hidden_dim))

            # Activation
            if config['activation'] == 'gelu':
                layers.append(nn.GELU())
            elif config['activation'] == 'swish':
                layers.append(nn.SiLU())
            elif config['activation'] == 'mish':
                layers.append(nn.Mish())
            else:
                layers.append(nn.ReLU())

            # Dropout (not on last layer)
            if config['dropout'] > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(config['dropout']))

            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Output heads
        self.heads = nn.ModuleDict({
            'attractive': nn.Linear(prev_dim, 1),
            'smart': nn.Linear(prev_dim, 1),
            'trustworthy': nn.Linear(prev_dim, 1)
        })

    def forward(self, x, target):
        features = self.encoder(x)
        return self.heads[target](features)


class ComparisonModel(nn.Module):
    """Model for comparison task."""

    def __init__(self, config):
        super().__init__()
        self.encoder = ImprovedEncoder(config)
        self.config = config

        # Optional attention mechanism
        if config.get('use_attention', False):
            self.attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)

    def forward(self, img1, img2, target):
        # Optional attention between images
        if self.config.get('use_attention', False):
            stacked = torch.stack([img1, img2], dim=1)
            attended, _ = self.attention(stacked, stacked, stacked)
            img1, img2 = attended[:, 0], attended[:, 1]

        # Get scores
        score1 = self.encoder(img1, target)
        score2 = self.encoder(img2, target)

        return torch.cat([score1, score2], dim=-1)


def load_synthetic_data(n_samples=10000):
    """Load synthetic data for quick testing."""

    # Load cached embeddings
    try:
        with open('artifacts/cache/clip_embeddings_cache.pkl', 'rb') as f:
            clip_cache = pickle.load(f)
        image_keys = list(clip_cache.keys())[:500]
        logger.info(f"Using {len(image_keys)} real CLIP embeddings")
    except:
        # Create synthetic
        logger.info("Creating synthetic embeddings")
        clip_cache = {}
        for i in range(500):
            embed = np.random.randn(512).astype(np.float32)
            embed = embed / np.linalg.norm(embed)
            clip_cache[f"img_{i}"] = embed
        image_keys = list(clip_cache.keys())

    # Generate comparison pairs
    data = []
    targets = ['attractive', 'smart', 'trustworthy']

    for _ in range(n_samples):
        img1, img2 = np.random.choice(image_keys, 2, replace=False)
        target = np.random.choice(targets)

        # Original
        data.append({
            'img1': clip_cache[img1],
            'img2': clip_cache[img2],
            'target': target,
            'label': 0  # First wins
        })

        # Augmented (swapped)
        data.append({
            'img1': clip_cache[img2],
            'img2': clip_cache[img1],
            'target': target,
            'label': 1  # Second wins
        })

    return data


class SimpleDataset(Dataset):
    """Simple dataset for comparison."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.tensor(item['img1'], dtype=torch.float32),
            torch.tensor(item['img2'], dtype=torch.float32),
            item['target'],
            item['label']
        )


def train_config(config, epochs=30):
    """Train with specific configuration."""

    # Load data
    data = load_synthetic_data(5000)

    # Split
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Create datasets
    train_dataset = SimpleDataset(train_data)
    val_dataset = SimpleDataset(val_data)

    # Create loaders
    batch_size = config.get('batch_size', 128)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    model = ComparisonModel(config).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.0))

    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0.01)

    if config.get('optimizer') == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Optional scheduler
    scheduler = None
    if config.get('use_scheduler', False):
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr*10, epochs=epochs, steps_per_epoch=len(train_loader)
        )

    # Training
    best_val_acc = 0
    start_time = time.time()

    for epoch in range(epochs):
        # Train
        model.train()
        train_correct = 0
        train_total = 0

        for img1, img2, targets, labels in train_loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)

            # Process batch
            outputs = []
            for i in range(len(targets)):
                output = model(img1[i:i+1], img2[i:i+1], targets[i])
                outputs.append(output)

            outputs = torch.cat(outputs, dim=0)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

            optimizer.step()

            if scheduler:
                scheduler.step()

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for img1, img2, targets, labels in val_loader:
                img1 = img1.to(device)
                img2 = img2.to(device)
                labels = labels.to(device)

                outputs = []
                for i in range(len(targets)):
                    output = model(img1[i:i+1], img2[i:i+1], targets[i])
                    outputs.append(output)

                outputs = torch.cat(outputs, dim=0)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(f"Epoch {epoch}: Val Acc: {val_acc:.4f} (best: {best_val_acc:.4f}) [{elapsed:.1f}s]")

    return best_val_acc


# Test configurations
configs = [
    # Config 1: Deep with GELU and BatchNorm (best from before)
    {
        'name': 'Deep GELU+BN',
        'hidden_dims': [512, 256, 128],
        'activation': 'gelu',
        'norm': 'batch',
        'dropout': 0.1,
        'optimizer': 'adamw',
        'lr': 0.001,
        'weight_decay': 0.01,
        'batch_size': 128,
    },

    # Config 2: Wider network
    {
        'name': 'Wide Network',
        'hidden_dims': [768, 384, 192],
        'activation': 'gelu',
        'norm': 'batch',
        'dropout': 0.15,
        'optimizer': 'adamw',
        'lr': 0.0005,
        'weight_decay': 0.01,
        'batch_size': 64,
    },

    # Config 3: Very deep with residuals
    {
        'name': 'Very Deep',
        'hidden_dims': [512, 384, 256, 128, 64],
        'activation': 'gelu',
        'norm': 'layer',
        'dropout': 0.2,
        'optimizer': 'adamw',
        'lr': 0.0005,
        'weight_decay': 0.05,
        'batch_size': 64,
        'grad_clip': 1.0,
    },

    # Config 4: Swish activation with attention
    {
        'name': 'Swish+Attention',
        'hidden_dims': [512, 256, 128],
        'activation': 'swish',
        'norm': 'batch',
        'dropout': 0.1,
        'optimizer': 'adamw',
        'lr': 0.001,
        'weight_decay': 0.01,
        'use_attention': True,
        'batch_size': 128,
    },

    # Config 5: Label smoothing
    {
        'name': 'Label Smoothing',
        'hidden_dims': [512, 256, 128],
        'activation': 'gelu',
        'norm': 'batch',
        'dropout': 0.1,
        'optimizer': 'adamw',
        'lr': 0.001,
        'weight_decay': 0.01,
        'label_smoothing': 0.1,
        'batch_size': 128,
    },

    # Config 6: OneCycle scheduler
    {
        'name': 'OneCycle LR',
        'hidden_dims': [512, 256, 128],
        'activation': 'gelu',
        'norm': 'batch',
        'dropout': 0.1,
        'optimizer': 'adamw',
        'lr': 0.0001,  # Lower base LR for OneCycle
        'weight_decay': 0.01,
        'use_scheduler': True,
        'batch_size': 128,
    },

    # Config 7: Mish activation
    {
        'name': 'Mish Activation',
        'hidden_dims': [512, 256, 128],
        'activation': 'mish',
        'norm': 'batch',
        'dropout': 0.15,
        'optimizer': 'adamw',
        'lr': 0.001,
        'weight_decay': 0.01,
        'batch_size': 128,
    },

    # Config 8: No dropout, heavy weight decay
    {
        'name': 'No Dropout',
        'hidden_dims': [512, 256, 128],
        'activation': 'gelu',
        'norm': 'batch',
        'dropout': 0.0,
        'optimizer': 'adamw',
        'lr': 0.001,
        'weight_decay': 0.1,
        'batch_size': 128,
    },

    # Config 9: Layer norm instead of batch
    {
        'name': 'Layer Norm',
        'hidden_dims': [512, 256, 128],
        'activation': 'gelu',
        'norm': 'layer',
        'dropout': 0.1,
        'optimizer': 'adamw',
        'lr': 0.001,
        'weight_decay': 0.01,
        'batch_size': 128,
    },

    # Config 10: Smaller batch size
    {
        'name': 'Small Batch',
        'hidden_dims': [512, 256, 128],
        'activation': 'gelu',
        'norm': 'batch',
        'dropout': 0.1,
        'optimizer': 'adamw',
        'lr': 0.0005,
        'weight_decay': 0.01,
        'batch_size': 32,
    },
]

# Test all configurations
logger.info("="*60)
logger.info("TESTING CONFIGURATIONS FOR COMPARISON TASK")
logger.info("="*60)

results = []

for config in configs:
    logger.info(f"\nTesting: {config['name']}")
    logger.info(f"Config: {config}")

    try:
        val_acc = train_config(config, epochs=30)
        results.append((config['name'], val_acc, config))
        logger.info(f"Final Val Acc: {val_acc:.4f}")
    except Exception as e:
        logger.error(f"Failed: {e}")
        continue

# Sort by accuracy
results.sort(key=lambda x: x[1], reverse=True)

# Report results
logger.info("\n" + "="*60)
logger.info("FINAL RESULTS")
logger.info("="*60)

for i, (name, acc, config) in enumerate(results):
    logger.info(f"\n#{i+1}. {name}: {acc:.4f}")
    if i < 3:  # Show config for top 3
        logger.info(f"   Config: {config}")

logger.info("\n" + "="*60)
logger.info(f"BEST ACCURACY: {results[0][1]:.4f} ({results[0][0]})")
logger.info("="*60)