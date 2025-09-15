#!/usr/bin/env python3
"""
Test configurations with batch processing fixed.
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import logging
from torch.utils.data import DataLoader, TensorDataset
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ComparisonModel(nn.Module):
    """Unified comparison model."""

    def __init__(self, config):
        super().__init__()

        input_dim = 512
        hidden_dims = config['hidden_dims']

        # Build encoder
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
            else:
                layers.append(nn.ReLU())

            # Dropout
            if config['dropout'] > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(config['dropout']))

            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Single output for comparison
        self.head = nn.Linear(prev_dim, 1)

    def forward(self, img1, img2):
        # Process both images
        score1 = self.head(self.encoder(img1))
        score2 = self.head(self.encoder(img2))
        return torch.cat([score1, score2], dim=-1)


def test_configuration(config, n_samples=10000, epochs=30):
    """Test a single configuration."""

    # Generate synthetic data
    try:
        with open('artifacts/cache/clip_embeddings_cache.pkl', 'rb') as f:
            clip_cache = pickle.load(f)
        embeddings = list(clip_cache.values())[:1000]
        embeddings = [e for e in embeddings if e.shape == (512,)][:500]
    except:
        embeddings = [np.random.randn(512).astype(np.float32) for _ in range(500)]
        embeddings = [e / np.linalg.norm(e) for e in embeddings]

    # Create pairs
    img1_list = []
    img2_list = []
    labels = []

    for _ in range(n_samples):
        idx1, idx2 = np.random.choice(len(embeddings), 2, replace=False)
        img1_list.append(embeddings[idx1])
        img2_list.append(embeddings[idx2])
        labels.append(0)  # First wins

        # Augmented
        img1_list.append(embeddings[idx2])
        img2_list.append(embeddings[idx1])
        labels.append(1)  # Second wins

    # Convert to tensors
    img1_tensor = torch.tensor(np.array(img1_list), dtype=torch.float32)
    img2_tensor = torch.tensor(np.array(img2_list), dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Split data
    n_train = int(0.8 * len(labels))
    train_dataset = TensorDataset(
        img1_tensor[:n_train],
        img2_tensor[:n_train],
        labels_tensor[:n_train]
    )
    val_dataset = TensorDataset(
        img1_tensor[n_train:],
        img2_tensor[n_train:],
        labels_tensor[n_train:]
    )

    # Create loaders
    batch_size = config.get('batch_size', 128)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    model = ComparisonModel(config).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    lr = config.get('lr', 0.001)
    if config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                     weight_decay=config.get('weight_decay', 0.01))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    best_val_acc = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for img1, img2, labels in train_loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)

            outputs = model(img1, img2)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate
        if epoch % 5 == 0:
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for img1, img2, labels in val_loader:
                    img1 = img1.to(device)
                    img2 = img2.to(device)
                    labels = labels.to(device)

                    outputs = model(img1, img2)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_acc = correct / total
            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if epoch % 10 == 0:
                logger.info(f"  Epoch {epoch}: Val Acc = {val_acc:.4f}")

    return best_val_acc


# Test configurations
configs = [
    {
        'name': 'Baseline (3-layer GELU)',
        'hidden_dims': [512, 256, 128],
        'activation': 'gelu',
        'norm': 'batch',
        'dropout': 0.1,
        'optimizer': 'adamw',
        'lr': 0.001,
        'weight_decay': 0.01,
        'batch_size': 128,
    },
    {
        'name': '4-layer Deep',
        'hidden_dims': [512, 384, 256, 128],
        'activation': 'gelu',
        'norm': 'batch',
        'dropout': 0.15,
        'optimizer': 'adamw',
        'lr': 0.0005,
        'weight_decay': 0.01,
        'batch_size': 128,
    },
    {
        'name': 'Wide Network',
        'hidden_dims': [768, 512, 256],
        'activation': 'gelu',
        'norm': 'batch',
        'dropout': 0.1,
        'optimizer': 'adamw',
        'lr': 0.0005,
        'weight_decay': 0.01,
        'batch_size': 64,
    },
    {
        'name': 'Very Deep (5 layers)',
        'hidden_dims': [512, 384, 256, 128, 64],
        'activation': 'gelu',
        'norm': 'layer',  # Layer norm for very deep
        'dropout': 0.2,
        'optimizer': 'adamw',
        'lr': 0.0003,
        'weight_decay': 0.05,
        'batch_size': 64,
    },
    {
        'name': 'Swish Activation',
        'hidden_dims': [512, 256, 128],
        'activation': 'swish',
        'norm': 'batch',
        'dropout': 0.1,
        'optimizer': 'adamw',
        'lr': 0.001,
        'weight_decay': 0.01,
        'batch_size': 128,
    },
    {
        'name': 'No Dropout',
        'hidden_dims': [512, 256, 128],
        'activation': 'gelu',
        'norm': 'batch',
        'dropout': 0.0,
        'optimizer': 'adamw',
        'lr': 0.001,
        'weight_decay': 0.05,  # Higher weight decay
        'batch_size': 128,
    },
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
    {
        'name': 'Higher Dropout',
        'hidden_dims': [512, 256, 128],
        'activation': 'gelu',
        'norm': 'batch',
        'dropout': 0.3,
        'optimizer': 'adamw',
        'lr': 0.001,
        'weight_decay': 0.01,
        'batch_size': 128,
    },
    {
        'name': 'Small Batch',
        'hidden_dims': [512, 256, 128],
        'activation': 'gelu',
        'norm': 'batch',
        'dropout': 0.1,
        'optimizer': 'adamw',
        'lr': 0.0003,  # Lower LR for small batch
        'weight_decay': 0.01,
        'batch_size': 32,
    },
    {
        'name': 'Large Batch',
        'hidden_dims': [512, 256, 128],
        'activation': 'gelu',
        'norm': 'batch',
        'dropout': 0.1,
        'optimizer': 'adamw',
        'lr': 0.002,  # Higher LR for large batch
        'weight_decay': 0.01,
        'batch_size': 256,
    },
]

# Test all configurations
logger.info("="*60)
logger.info("COMPARISON TASK - CONFIGURATION SEARCH")
logger.info("="*60)

results = []

for config in configs:
    logger.info(f"\nTesting: {config['name']}")
    start = time.time()

    try:
        val_acc = test_configuration(config, n_samples=5000, epochs=30)
        elapsed = time.time() - start
        results.append((config['name'], val_acc, config))
        logger.info(f"  Final Val Acc: {val_acc:.4f} (took {elapsed:.1f}s)")
    except Exception as e:
        logger.error(f"  Failed: {e}")

# Sort results
results.sort(key=lambda x: x[1], reverse=True)

# Report
logger.info("\n" + "="*60)
logger.info("RESULTS SUMMARY")
logger.info("="*60)

for i, (name, acc, _) in enumerate(results[:5]):
    logger.info(f"{i+1}. {name}: {acc:.4f}")

if results:
    best_name, best_acc, best_config = results[0]
    logger.info("\n" + "="*60)
    logger.info(f"BEST CONFIGURATION: {best_name}")
    logger.info(f"Validation Accuracy: {best_acc:.4f}")
    logger.info(f"\nFull config:")
    for key, value in best_config.items():
        if key != 'name':
            logger.info(f"  {key}: {value}")
    logger.info("="*60)