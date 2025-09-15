#!/usr/bin/env python3
"""
Unified model with Gated Linear Units (GLUs) for improved performance.
GLUs: output = linear(x) * sigmoid(gate(x))
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
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from torch.utils.data import Dataset, DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit layer."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Two linear transformations: one for values, one for gates
        self.linear = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # GLU: element-wise product of linear transformation and sigmoid gate
        return self.linear(x) * torch.sigmoid(self.gate(x))


class GLUBlock(nn.Module):
    """A block with GLU, normalization, and dropout."""

    def __init__(self, input_dim, output_dim, use_batchnorm=True, use_layernorm=False, dropout=0.1):
        super().__init__()

        # Gated Linear Unit
        self.glu = GatedLinearUnit(input_dim, output_dim)

        # Normalization
        self.norm = None
        if use_batchnorm:
            self.norm = nn.BatchNorm1d(output_dim)
        elif use_layernorm:
            self.norm = nn.LayerNorm(output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.glu(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class BaseEncoderGLU(nn.Module):
    """Base encoder using Gated Linear Units."""

    def __init__(self, cfg, n_users=None):
        super().__init__()
        self.cfg = cfg

        # Calculate input dimension
        clip_dim = 512  # CLIP embedding dimension
        user_dim = cfg.model.user_embedding_dim if cfg.model.use_user_embedding and n_users else 0
        input_dim = clip_dim

        if user_dim > 0:
            self.user_embedding = nn.Embedding(n_users, user_dim)
            input_dim += user_dim
        else:
            self.user_embedding = None

        # Build GLU layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in cfg.model.hidden_dims:
            layers.append(
                GLUBlock(
                    prev_dim,
                    hidden_dim,
                    use_batchnorm=cfg.model.get('use_batchnorm', True),
                    use_layernorm=cfg.model.get('use_layernorm', False),
                    dropout=cfg.model.get('dropout', 0.1)
                )
            )
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Task-specific output heads
        self.targets = ['attractive', 'smart', 'trustworthy']

        if cfg.task_type == 'rating':
            # 4 outputs per target for rating task
            self.heads = nn.ModuleDict({
                target: nn.Linear(prev_dim, 4)
                for target in self.targets
            })
        else:  # comparison
            # 1 output per target for comparison
            self.heads = nn.ModuleDict({
                target: nn.Linear(prev_dim, 1)
                for target in self.targets
            })

    def forward(self, x, user_ids=None, target='attractive'):
        # Add user embeddings if available
        if self.user_embedding is not None and user_ids is not None:
            user_embeds = self.user_embedding(user_ids)
            x = torch.cat([x, user_embeds], dim=-1)

        # Encode through GLU layers
        features = self.encoder(x)

        # Get output for specific target
        return self.heads[target](features)


class UnifiedModelGLU(nn.Module):
    """Unified model with GLU for comparison/rating tasks."""

    def __init__(self, cfg, n_users=None):
        super().__init__()
        self.cfg = cfg
        self.base_encoder = BaseEncoderGLU(cfg, n_users)

    def forward_comparison(self, image1, image2, user_ids=None, target='attractive'):
        """Forward pass for comparison task."""
        score1 = self.base_encoder(image1, user_ids, target)
        score2 = self.base_encoder(image2, user_ids, target)

        # Stack scores for softmax
        scores = torch.cat([score1, score2], dim=-1)
        return scores

    def forward_rating(self, image, user_ids=None, target='attractive'):
        """Forward pass for rating task."""
        return self.base_encoder(image, user_ids, target)


def load_comparison_data(cfg):
    """Load comparison data."""
    data_path = Path("analysis/data/labels.pkl")

    with open(data_path, 'rb') as f:
        df = pickle.load(f)

    # Process comparison data (simplified for testing)
    comparison_samples = []

    # Extract comparison samples
    for _, row in df.iterrows():
        if pd.isna(row.get('label')):
            continue

        # Check if this is a comparison task
        item_path = str(row.get('item_path', ''))
        if 'comparison' in item_path.lower():
            # Parse the comparison data
            # This is simplified - actual parsing would depend on data format
            comparison_samples.append({
                'user_id': row['user_id'],
                'target': 'attractive',  # Would parse from item_path
                'label': int(row['label']) if not pd.isna(row['label']) else 0
            })

    logger.info(f"Loaded {len(comparison_samples)} comparison samples")
    return comparison_samples


def train_glu_model(cfg: DictConfig):
    """Train GLU model."""

    # Load CLIP embeddings
    cache_file = Path("artifacts/cache/clip_embeddings_cache.pkl")
    with open(cache_file, 'rb') as f:
        clip_cache = pickle.load(f)

    # Create synthetic data for testing
    logger.info("Creating synthetic comparison data for GLU testing...")

    image_keys = list(clip_cache.keys())[:1000]
    targets = ['attractive', 'smart', 'trustworthy']

    # Generate comparison data
    comparison_data = []
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

    # Create dataset
    class ComparisonDataset(Dataset):
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

    # Split data
    train_data = comparison_data[:int(0.8 * len(comparison_data))]
    val_data = comparison_data[int(0.8 * len(comparison_data)):]

    train_dataset = ComparisonDataset(train_data)
    val_dataset = ComparisonDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    # Initialize model
    model = UnifiedModelGLU(cfg, n_users=10).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    if cfg.training.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.learning_rate
        )

    # Training loop
    best_val_acc = 0
    best_val_loss = float('inf')

    logger.info("Training GLU model...")

    for epoch in range(cfg.training.epochs):
        # Training
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        for img1, img2, user_ids, targets, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs}"):
            img1 = img1.to(device)
            img2 = img2.to(device)
            user_ids = user_ids.to(device)
            labels = labels.to(device)

            # Process batch by target
            outputs = []
            for i in range(len(targets)):
                output = model.forward_comparison(
                    img1[i:i+1], img2[i:i+1],
                    user_ids[i:i+1], targets[i]
                )
                outputs.append(output)

            outputs = torch.cat(outputs, dim=0)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation
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

                outputs = []
                for i in range(len(targets)):
                    output = model.forward_comparison(
                        img1[i:i+1], img2[i:i+1],
                        user_ids[i:i+1], targets[i]
                    )
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

    logger.info(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    logger.info(f"Best Validation Loss: {best_val_loss:.4f}")

    return best_val_acc, best_val_loss


@hydra.main(config_path="configs", config_name="unified_glu", version_base=None)
def main(cfg: DictConfig):
    """Main function."""
    logger.info("="*70)
    logger.info("GATED LINEAR UNITS (GLU) MODEL")
    logger.info("="*70)
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Train model
    val_acc, val_loss = train_glu_model(cfg)

    logger.info("\n" + "="*70)
    logger.info("GLU MODEL RESULTS")
    logger.info("="*70)
    logger.info(f"Validation Accuracy: {val_acc:.4f}")
    logger.info(f"Validation CE Loss: {val_loss:.4f}")
    logger.info("="*70)


if __name__ == "__main__":
    main()