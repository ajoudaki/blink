#!/usr/bin/env python3
"""
Unified training script with early stopping and train/val/test splits.
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# GATED LINEAR UNITS
# ============================================================================

class GatedLinearUnit(nn.Module):
    """Gated Linear Unit: output = linear(x) * sigmoid(gate(x))"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x) * torch.sigmoid(self.gate(x))


class GLUBlock(nn.Module):
    """GLU block with optional normalization and dropout"""
    def __init__(self, input_dim, output_dim, use_batchnorm=True, use_layernorm=False, dropout=0.1):
        super().__init__()
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


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class BaseEncoder(nn.Module):
    """Shared base encoder for both rating and comparison tasks."""

    def __init__(self, cfg, n_users=0, embedding_dim=512):
        super().__init__()

        input_dim = embedding_dim  # CLIP embedding dimension (auto-detected)

        # Add user embedding dimension if using user embeddings
        if cfg.model.get('use_user_embedding', False) and n_users > 0:
            self.user_embedding = nn.Embedding(n_users, cfg.model.user_embedding_dim)
            input_dim += cfg.model.user_embedding_dim
        else:
            self.user_embedding = None

        # Build encoder layers
        layers = []
        prev_dim = input_dim

        use_glu = cfg.model.get('use_glu', False)
        use_batchnorm = cfg.model.get('use_batchnorm', True)
        use_layernorm = cfg.model.get('use_layernorm', False)
        dropout = cfg.model.get('dropout', 0.1)

        for hidden_dim in cfg.model.hidden_dims:
            if use_glu:
                layers.append(GLUBlock(prev_dim, hidden_dim, use_batchnorm, use_layernorm, dropout))
            else:
                # Standard linear block
                block = [nn.Linear(prev_dim, hidden_dim)]

                # Add normalization
                if use_batchnorm:
                    block.append(nn.BatchNorm1d(hidden_dim))
                elif use_layernorm:
                    block.append(nn.LayerNorm(hidden_dim))

                # Add activation
                activation = cfg.model.get('activation', 'relu')
                if activation == 'relu':
                    block.append(nn.ReLU())
                elif activation == 'gelu':
                    block.append(nn.GELU())
                elif activation == 'tanh':
                    block.append(nn.Tanh())
                elif activation == 'leaky_relu':
                    block.append(nn.LeakyReLU())
                elif activation == 'silu':
                    block.append(nn.SiLU())
                elif activation == 'mish':
                    block.append(nn.Mish())
                elif activation == 'softplus':
                    block.append(nn.Softplus())
                elif activation == 'selu':
                    block.append(nn.SELU())
                elif activation == 'elu':
                    block.append(nn.ELU())
                elif activation == 'sigmoid':
                    block.append(nn.Sigmoid())
                elif activation == 'hardsigmoid':
                    block.append(nn.Hardsigmoid())
                elif activation == 'hardtanh':
                    block.append(nn.Hardtanh())
                elif activation == 'hardswish':
                    block.append(nn.Hardswish())
                elif activation == 'relu6':
                    block.append(nn.ReLU6())
                elif activation == 'prelu':
                    block.append(nn.PReLU())
                elif activation == 'rrelu':
                    block.append(nn.RReLU())
                elif activation == 'celu':
                    block.append(nn.CELU())
                elif activation == 'logsigmoid':
                    block.append(nn.LogSigmoid())
                elif activation != 'none':
                    # Default to ReLU if unknown activation
                    block.append(nn.ReLU())

                # Add dropout
                if dropout > 0:
                    block.append(nn.Dropout(dropout))

                layers.extend(block)

            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_dim = prev_dim

    def forward(self, x, user_idx=None):
        # Add user embedding if available
        if self.user_embedding is not None and user_idx is not None:
            user_emb = self.user_embedding(user_idx)
            x = torch.cat([x, user_emb], dim=-1)

        return self.encoder(x)


class UnifiedModel(nn.Module):
    """Unified model for both rating and comparison tasks."""

    def __init__(self, cfg, n_users=0, embedding_dim=512):
        super().__init__()

        self.cfg = cfg
        self.task_type = cfg.task_type

        # Shared base encoder
        self.base_encoder = BaseEncoder(cfg, n_users, embedding_dim)

        # Task-specific heads
        self.heads = nn.ModuleDict()

        for target in cfg.data.targets:
            if cfg.task_type == 'rating':
                # 4-class classification for ratings 1-4
                self.heads[target] = nn.Linear(self.base_encoder.output_dim, 4)
            else:
                # Binary classification for comparison
                # Takes concatenated features from two images
                self.heads[target] = nn.Linear(self.base_encoder.output_dim * 2, 1)

    def forward(self, x1, x2=None, user_idx=None, target='attractive'):
        if self.task_type == 'rating':
            # For rating: x1 is the image
            features = self.base_encoder(x1, user_idx)
            logits = self.heads[target](features)
            return torch.softmax(logits, dim=-1)
        else:
            # For comparison: x1 and x2 are two images
            features1 = self.base_encoder(x1, user_idx)
            features2 = self.base_encoder(x2, user_idx)
            combined = torch.cat([features1, features2], dim=-1)
            logits = self.heads[target](combined)
            return torch.sigmoid(logits).squeeze(-1)


# ============================================================================
# DATA LOADING WITH TRAIN/VAL/TEST SPLITS
# ============================================================================

def load_data_with_splits(cfg, device, val_split=0.1, test_split=0.1):
    """
    Load data with proper train/val/test splits.

    Args:
        cfg: Configuration
        device: Device to use
        val_split: Fraction for validation (default 0.1)
        test_split: Fraction for test (default 0.1)

    Returns:
        train_data, val_data, test_data, n_users
    """

    # Load embeddings
    embeddings_file = Path("artifacts/cache") / cfg.data.embeddings_cache_file
    print(f"Loading embeddings from cache: {embeddings_file}")

    with open(embeddings_file, 'rb') as f:
        embeddings_dict = pickle.load(f)

    print(f"✓ All {len(embeddings_dict)} embeddings loaded from cache")

    # Load appropriate data based on task type
    if cfg.task_type == 'rating':
        return load_rating_data_with_splits(cfg, embeddings_dict, device, val_split, test_split)
    else:
        return load_comparison_data_with_splits(cfg, embeddings_dict, device, val_split, test_split)


def load_rating_data_with_splits(cfg, embeddings_dict, device, val_split, test_split):
    """Load rating data with train/val/test splits."""

    print("\nLoading rating data...")
    rating_labels = pd.read_excel('data/big_label.xlsx')
    rating_data = pd.read_excel('data/big_data.xlsx')
    df = rating_data.merge(rating_labels, left_on='_id', right_on='item_id', how='inner')

    # Get unique users
    users = df['user_id'].unique()
    n_users = len(users)
    user_to_idx = {user: idx for idx, user in enumerate(users)}

    train_data = {}
    val_data = {}
    test_data = {}

    for target in cfg.data.targets:
        # Filter data for this target
        target_df = df[df['target'] == target].copy()

        # Get features and labels
        embeddings_list = []
        labels = []
        user_indices = []
        valid_indices = []

        for idx, row in target_df.iterrows():
            image_path = row['data_image_part']
            if image_path in embeddings_dict:
                embeddings_list.append(embeddings_dict[image_path])
                labels.append(int(row[target]) - 1)  # Convert 1-4 to 0-3
                user_indices.append(user_to_idx[row['user_id']])
                valid_indices.append(idx)

        if not embeddings_list:
            continue

        X = np.array(embeddings_list)
        y = np.array(labels)
        user_idx = np.array(user_indices)

        # First split: separate test set
        train_val_idx, test_idx = train_test_split(
            np.arange(len(X)),
            test_size=test_split,
            random_state=cfg.seed,
            stratify=y if len(np.unique(y)) > 1 else None
        )

        # Second split: separate train and validation from remaining data
        train_size_adjusted = 1.0 - (val_split / (1.0 - test_split))
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=1.0 - train_size_adjusted,
            random_state=cfg.seed,
            stratify=y[train_val_idx] if len(np.unique(y[train_val_idx])) > 1 else None
        )

        # Create tensors for train set
        train_data[target] = {
            'X': torch.FloatTensor(X[train_idx]).to(device),
            'y': torch.LongTensor(y[train_idx]).to(device),
            'y_np': y[train_idx],
            'user': torch.LongTensor(user_idx[train_idx]).to(device) if cfg.model.get('use_user_embedding', False) else None
        }

        # Create tensors for validation set
        val_data[target] = {
            'X': torch.FloatTensor(X[val_idx]).to(device),
            'y': torch.LongTensor(y[val_idx]).to(device),
            'y_np': y[val_idx],
            'user': torch.LongTensor(user_idx[val_idx]).to(device) if cfg.model.get('use_user_embedding', False) else None
        }

        # Create tensors for test set
        test_data[target] = {
            'X': torch.FloatTensor(X[test_idx]).to(device),
            'y': torch.LongTensor(y[test_idx]).to(device),
            'y_np': y[test_idx],
            'user': torch.LongTensor(user_idx[test_idx]).to(device) if cfg.model.get('use_user_embedding', False) else None
        }

        print(f"  {target}: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test samples")

    return train_data, val_data, test_data, n_users


def load_comparison_data_with_splits(cfg, embeddings_dict, device, val_split, test_split):
    """Load comparison data with train/val/test splits."""

    print("\nLoading comparison data...")
    compare_labels = pd.read_excel('data/big_compare_label.xlsx')
    compare_data = pd.read_excel('data/big_compare_data.xlsx')
    df = compare_data.merge(compare_labels, left_on='_id', right_on='item_id', how='inner')

    # Convert labels to binary
    for target in cfg.data.targets:
        df[f'{target}_binary'] = (df[target] == 2).astype(int)

    # Get unique users
    users = df['user_id'].unique()
    n_users = len(users)
    user_to_idx = {user: idx for idx, user in enumerate(users)}

    all_samples = []

    for target in cfg.data.targets:
        target_df = df.copy()  # Use all data, target is already in binary columns

        for _, row in target_df.iterrows():
            if row['im1_path'] in embeddings_dict and row['im2_path'] in embeddings_dict:
                sample = {
                    'emb1': embeddings_dict[row['im1_path']],
                    'emb2': embeddings_dict[row['im2_path']],
                    'label': row[f'{target}_binary'],
                    'user_idx': user_to_idx[row['user_id']],
                    'target': target
                }
                all_samples.append(sample)

    print(f"Loaded {len(all_samples)} samples with {len(embeddings_dict)} unique images")

    # Shuffle and split
    np.random.seed(cfg.seed)
    np.random.shuffle(all_samples)

    # Calculate split indices
    n_samples = len(all_samples)
    n_test = int(n_samples * test_split)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_test - n_val

    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]

    # Augment training data if configured
    if cfg.training.get('augment_swapped_pairs', True):
        augmented = []
        for sample in train_samples:
            # Add swapped version
            augmented.append({
                'emb1': sample['emb2'],
                'emb2': sample['emb1'],
                'label': 1 - sample['label'],
                'user_idx': sample['user_idx'],
                'target': sample['target']
            })
        train_samples.extend(augmented)

    # Organize by target
    train_data = {}
    val_data = {}
    test_data = {}

    for target in cfg.data.targets:
        # Train data
        target_train = [s for s in train_samples if s['target'] == target]
        if target_train:
            train_data[target] = {
                'X1': torch.FloatTensor([s['emb1'] for s in target_train]).to(device),
                'X2': torch.FloatTensor([s['emb2'] for s in target_train]).to(device),
                'y': torch.LongTensor([s['label'] for s in target_train]).to(device),
                'y_float': torch.FloatTensor([s['label'] for s in target_train]).to(device),
                'y_np': np.array([s['label'] for s in target_train]),
                'user': torch.LongTensor([s['user_idx'] for s in target_train]).to(device) if cfg.model.get('use_user_embedding', False) else None,
                'user_np': np.array([s['user_idx'] for s in target_train])
            }

        # Validation data
        target_val = [s for s in val_samples if s['target'] == target]
        if target_val:
            val_data[target] = {
                'X1': torch.FloatTensor([s['emb1'] for s in target_val]).to(device),
                'X2': torch.FloatTensor([s['emb2'] for s in target_val]).to(device),
                'y': torch.LongTensor([s['label'] for s in target_val]).to(device),
                'y_float': torch.FloatTensor([s['label'] for s in target_val]).to(device),
                'y_np': np.array([s['label'] for s in target_val]),
                'user': torch.LongTensor([s['user_idx'] for s in target_val]).to(device) if cfg.model.get('use_user_embedding', False) else None,
                'user_np': np.array([s['user_idx'] for s in target_val])
            }

        # Test data
        target_test = [s for s in test_samples if s['target'] == target]
        if target_test:
            test_data[target] = {
                'X1': torch.FloatTensor([s['emb1'] for s in target_test]).to(device),
                'X2': torch.FloatTensor([s['emb2'] for s in target_test]).to(device),
                'y': torch.LongTensor([s['label'] for s in target_test]).to(device),
                'y_float': torch.FloatTensor([s['label'] for s in target_test]).to(device),
                'y_np': np.array([s['label'] for s in target_test]),
                'user': torch.LongTensor([s['user_idx'] for s in target_test]).to(device) if cfg.model.get('use_user_embedding', False) else None,
                'user_np': np.array([s['user_idx'] for s in target_test])
            }

        print(f"  {target}: {len(train_data.get(target, {'y': []})['y'])} train, "
              f"{len(val_data.get(target, {'y': []})['y'])} val, "
              f"{len(test_data.get(target, {'y': []})['y'])} test samples")

    return train_data, val_data, test_data, n_users


# ============================================================================
# TRAINING WITH EARLY STOPPING
# ============================================================================

def evaluate_model(model, data, cfg, device):
    """Evaluate model on given data."""
    model.eval()
    results = {}

    with torch.no_grad():
        for target in cfg.data.targets:
            if target not in data:
                continue

            if cfg.task_type == 'rating':
                predictions = model(
                    data[target]['X'],
                    data[target].get('user', None),
                    target=target
                )

                pred_class = predictions.argmax(dim=-1).cpu().numpy()
                true_class = data[target]['y_np']

                accuracy = np.mean(pred_class == true_class)

                # Cross-entropy loss
                probs = predictions.cpu().numpy()
                n_samples = len(true_class)
                true_one_hot = np.zeros((n_samples, 4))
                true_one_hot[np.arange(n_samples), true_class] = 1
                probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
                ce_loss = -np.mean(np.sum(true_one_hot * np.log(probs_clipped), axis=1))

                results[target] = {
                    'accuracy': accuracy,
                    'ce_loss': ce_loss,
                    'n_samples': len(data[target]['y'])
                }
            else:
                predictions = model(
                    data[target]['X1'],
                    data[target]['X2'],
                    data[target].get('user', None),
                    target=target
                )

                pred_binary = (predictions.cpu().numpy() > 0.5).astype(int)
                true_binary = data[target]['y_np']

                accuracy = np.mean(pred_binary == true_binary)

                # Track per-user accuracy
                user_accuracies = {}
                if 'user_np' in data[target]:
                    user_indices = data[target]['user_np']
                    for user_id in np.unique(user_indices):
                        user_mask = user_indices == user_id
                        user_acc = np.mean(pred_binary[user_mask] == true_binary[user_mask])
                        user_accuracies[int(user_id)] = {
                            'accuracy': float(user_acc),
                            'n_samples': int(np.sum(user_mask))
                        }

                # Cross-entropy loss
                probs = predictions.cpu().numpy()
                probs = np.clip(probs, 1e-7, 1 - 1e-7)
                ce_loss = -np.mean(
                    true_binary * np.log(probs) +
                    (1 - true_binary) * np.log(1 - probs)
                )

                results[target] = {
                    'accuracy': accuracy,
                    'ce_loss': ce_loss,
                    'n_samples': len(data[target]['y']),
                    'user_accuracies': user_accuracies
                }

    return results


def train_model_with_early_stopping(model, train_data, val_data, test_data, cfg, device):
    """
    Train model with early stopping based on validation loss.

    Returns:
        Dictionary with best model state and final results on both val and test sets.
    """
    optimizer = get_optimizer(model, cfg)

    # Setup criterion
    if cfg.task_type == 'rating':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()

    # Early stopping parameters
    patience = cfg.training.get('patience', 10)
    min_delta = cfg.training.get('min_delta', 0.001)
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }

    max_epochs = cfg.training.get('max_epochs', 200)  # Use max_epochs for early stopping

    print(f"\nTraining with early stopping (patience={patience}, max_epochs={max_epochs})")

    for epoch in tqdm(range(max_epochs), desc="Training"):
        model.train()
        train_losses = []

        # Training step
        for target in cfg.data.targets:
            if target not in train_data:
                continue

            optimizer.zero_grad()

            if cfg.task_type == 'rating':
                predictions = model(
                    train_data[target]['X'],
                    train_data[target].get('user', None),
                    target=target
                )
                loss = criterion(predictions, train_data[target]['y'])
            else:
                predictions = model(
                    train_data[target]['X1'],
                    train_data[target]['X2'],
                    train_data[target].get('user', None),
                    target=target
                )
                loss = criterion(predictions, train_data[target]['y_float'])

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation step
        val_results = evaluate_model(model, val_data, cfg, device)

        # Calculate average metrics
        avg_val_loss = np.mean([r['ce_loss'] for r in val_results.values()])
        avg_val_acc = np.mean([r['accuracy'] for r in val_results.values()])
        avg_train_loss = np.mean(train_losses)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(avg_val_acc)

        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}: Train Loss={avg_train_loss:.4f}, "
                  f"Val Loss={avg_val_loss:.4f}, Val Acc={avg_val_acc:.4f}")
            if epochs_without_improvement > 0:
                print(f"  No improvement for {epochs_without_improvement} epochs (best: epoch {best_epoch})")

        # Check early stopping condition
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best model was at epoch {best_epoch} with val loss {best_val_loss:.4f}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation on both validation and test sets
    final_val_results = evaluate_model(model, val_data, cfg, device)
    final_test_results = evaluate_model(model, test_data, cfg, device)

    return {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'val_results': final_val_results,
        'test_results': final_test_results,
        'history': history,
        'model_state': best_model_state
    }


def get_optimizer(model, cfg):
    """Get optimizer based on configuration."""
    opt_type = cfg.training.optimizer
    lr = cfg.training.learning_rate
    weight_decay = cfg.training.get('weight_decay', 0)

    if opt_type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function with early stopping."""

    # Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu.device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Task type: {cfg.task_type}")
    print(f"Model architecture: Unified BaseEncoder with Early Stopping")

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.task_type}_unified_earlystop_{timestamp}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Load data with train/val/test splits
    val_split = cfg.training.get('validation_split', 0.1)
    test_split = cfg.training.get('test_split', 0.1)

    train_data, val_data, test_data, n_users = load_data_with_splits(
        cfg, device, val_split=val_split, test_split=test_split
    )

    # Detect embedding dimension from the loaded data
    embedding_dim = 512  # default
    for target, data in train_data.items():
        if 'X' in data and data['X'] is not None:
            embedding_dim = data['X'].shape[1]
            break
        elif 'X1' in data and data['X1'] is not None:
            embedding_dim = data['X1'].shape[1]
            break
    print(f"Model will use embedding dimension: {embedding_dim}")

    # Initialize model
    model = UnifiedModel(cfg, n_users, embedding_dim).to(device)

    # Train model with early stopping
    print("\nTraining unified model with early stopping...")
    results = train_model_with_early_stopping(
        model, train_data, val_data, test_data, cfg, device
    )

    # Report results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS WITH EARLY STOPPING")
    print("=" * 60)
    print(f"Task: {cfg.task_type}")
    print(f"Model: Unified BaseEncoder")
    print(f"Best Epoch: {results['best_epoch']}")
    print(f"Best Val Loss: {results['best_val_loss']:.4f}")
    print()

    # Display validation results
    print("VALIDATION SET RESULTS:")
    avg_val_acc = np.mean([r['accuracy'] for r in results['val_results'].values()])
    avg_val_loss = np.mean([r['ce_loss'] for r in results['val_results'].values()])

    for target, metrics in results['val_results'].items():
        print(f"  {target.upper()}:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    CE Loss: {metrics['ce_loss']:.4f}")
        print(f"    Samples: {metrics['n_samples']}")

    print(f"  Average Val Accuracy: {avg_val_acc:.4f}")
    print(f"  Average Val CE Loss: {avg_val_loss:.4f}")
    print()

    # Display test results
    print("TEST SET RESULTS:")
    avg_test_acc = np.mean([r['accuracy'] for r in results['test_results'].values()])
    avg_test_loss = np.mean([r['ce_loss'] for r in results['test_results'].values()])

    for target, metrics in results['test_results'].items():
        print(f"  {target.upper()}:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    CE Loss: {metrics['ce_loss']:.4f}")
        print(f"    Samples: {metrics['n_samples']}")

    print(f"  Average Test Accuracy: {avg_test_acc:.4f}")
    print(f"  Average Test CE Loss: {avg_test_loss:.4f}")

    # Display per-user breakdown if available (for comparison task)
    if cfg.task_type == 'comparison':
        print("\n" + "=" * 60)
        print("PER-USER ACCURACY BREAKDOWN")
        print("=" * 60)

        # Validation per-user breakdown
        print("\nVALIDATION SET - Per User:")
        all_user_ids = set()
        for target, metrics in results['val_results'].items():
            if 'user_accuracies' in metrics:
                all_user_ids.update(metrics['user_accuracies'].keys())

        if all_user_ids:
            for user_id in sorted(all_user_ids):
                print(f"\n  User {user_id}:")
                for target, metrics in results['val_results'].items():
                    if 'user_accuracies' in metrics and user_id in metrics['user_accuracies']:
                        user_data = metrics['user_accuracies'][user_id]
                        print(f"    {target.upper()}: Acc={user_data['accuracy']:.4f} (n={user_data['n_samples']})")

        # Test per-user breakdown
        print("\nTEST SET - Per User:")
        all_user_ids = set()
        for target, metrics in results['test_results'].items():
            if 'user_accuracies' in metrics:
                all_user_ids.update(metrics['user_accuracies'].keys())

        if all_user_ids:
            for user_id in sorted(all_user_ids):
                print(f"\n  User {user_id}:")
                for target, metrics in results['test_results'].items():
                    if 'user_accuracies' in metrics and user_id in metrics['user_accuracies']:
                        user_data = metrics['user_accuracies'][user_id]
                        print(f"    {target.upper()}: Acc={user_data['accuracy']:.4f} (n={user_data['n_samples']})")

    # Save results
    results_dict = {
        'best_epoch': results['best_epoch'],
        'validation': results['val_results'],
        'test': results['test_results'],
        'history': results['history']
    }

    # Save as JSON
    results_file = run_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n✓ Results saved to {results_file}")

    # Save model
    model_file = run_dir / "best_model.pth"
    torch.save(results['model_state'], model_file)
    print(f"✓ Best model saved to {model_file}")

    # Save configuration
    config_file = run_dir / "config.yaml"
    with open(config_file, 'w') as f:
        f.write(str(cfg))
    print(f"✓ Configuration saved to {config_file}")

    # Create summary
    summary_file = run_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Task: {cfg.task_type}\n")
        f.write(f"Model: Unified BaseEncoder with Early Stopping\n")
        f.write(f"Best Epoch: {results['best_epoch']}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Validation Accuracy: {avg_val_acc:.4f}\n")
        f.write(f"Validation CE Loss: {avg_val_loss:.4f}\n")
        f.write(f"Test Accuracy: {avg_test_acc:.4f}\n")
        f.write(f"Test CE Loss: {avg_test_loss:.4f}\n")

    print(f"✓ Summary saved to {summary_file}")


if __name__ == "__main__":
    main()