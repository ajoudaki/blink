#!/usr/bin/env python3
"""
Training script with support for different CLIP model variants.
Based on train_with_early_stopping.py but with configurable CLIP models.
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
import clip
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# CLIP model dimensions
CLIP_DIMENSIONS = {
    "ViT-B/32": 512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
    "ViT-L/14@336px": 768,
    "RN50": 1024,
    "RN101": 512,
    "RN50x4": 640,
    "RN50x16": 768,
    "RN50x64": 1024
}


def load_or_extract_embeddings(image_paths, clip_model_name="ViT-B/32", cache_file=None,
                               device='cuda', force_recompute=False):
    """Load cached embeddings or extract new ones using specified CLIP model."""

    # Create model-specific cache file name
    if cache_file:
        model_suffix = clip_model_name.replace("/", "_").replace("@", "_")
        cache_file = cache_file.replace(".pkl", f"_{model_suffix}.pkl")
        cache_path = Path('artifacts/cache') / cache_file
    else:
        cache_path = None

    # Try to load from cache
    if cache_path and cache_path.exists() and not force_recompute:
        print(f"Loading embeddings from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            embeddings = pickle.load(f)

        # Check if all images are in cache
        missing = [p for p in image_paths if p not in embeddings]
        if not missing:
            print(f"✓ All {len(embeddings)} embeddings loaded from cache")
            return embeddings
        else:
            print(f"  Missing {len(missing)} embeddings, will extract them")

    # Extract embeddings
    print(f"Extracting embeddings using CLIP model: {clip_model_name}")
    model, preprocess = clip.load(clip_model_name, device=device)

    embeddings = {}

    for i, img_path in enumerate(tqdm(image_paths, desc="Extracting embeddings")):
        if cache_path and img_path in embeddings:
            continue

        try:
            from PIL import Image
            image = Image.open(img_path)
            image_input = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_input)
                # Normalize features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                embeddings[img_path] = image_features.cpu().numpy().squeeze()
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # Save to cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"✓ Embeddings saved to {cache_path}")

    return embeddings


class BaseEncoder(nn.Module):
    """Base encoder that adapts to different CLIP embedding dimensions."""

    def __init__(self, cfg, n_users=0, input_dim=512):
        super().__init__()

        # Add user embedding dimension if using user embeddings
        if cfg.model.get('use_user_embedding', False) and n_users > 0:
            self.user_embedding = nn.Embedding(n_users, cfg.model.user_embedding_dim)
            input_dim += cfg.model.user_embedding_dim
        else:
            self.user_embedding = None

        # Build encoder layers
        layers = []
        prev_dim = input_dim

        use_batchnorm = cfg.model.get('use_batchnorm', True)
        use_layernorm = cfg.model.get('use_layernorm', False)
        dropout = cfg.model.get('dropout', 0.1)

        for hidden_dim in cfg.model.hidden_dims:
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
    """Unified model that adapts to CLIP embedding dimensions."""

    def __init__(self, cfg, n_users=0, input_dim=512):
        super().__init__()

        self.cfg = cfg
        self.task_type = cfg.task_type

        # Shared base encoder
        self.base_encoder = BaseEncoder(cfg, n_users, input_dim)

        # Task-specific heads
        self.heads = nn.ModuleDict()

        for target in cfg.data.targets:
            if cfg.task_type == 'rating':
                self.heads[target] = nn.Linear(self.base_encoder.output_dim, 4)
            else:
                self.heads[target] = nn.Linear(self.base_encoder.output_dim * 2, 1)

    def forward(self, x1, x2=None, user_idx=None, target='attractive'):
        if self.task_type == 'rating':
            features = self.base_encoder(x1, user_idx)
            logits = self.heads[target](features)
            return torch.softmax(logits, dim=-1)
        else:
            features1 = self.base_encoder(x1, user_idx)
            features2 = self.base_encoder(x2, user_idx)
            combined = torch.cat([features1, features2], dim=-1)
            logits = self.heads[target](combined)
            return torch.sigmoid(logits).squeeze(-1)


def load_comparison_data_for_clip(cfg, clip_model_name, device):
    """Load comparison data with specified CLIP model."""

    print("\nLoading comparison data...")
    compare_labels = pd.read_excel('data/big_compare_label.xlsx')
    compare_data = pd.read_excel('data/big_compare_data.xlsx')
    df = compare_data.merge(compare_labels, left_on='_id', right_on='item_id', how='inner')

    # Convert labels to binary
    for target in cfg.data.targets:
        df[f'{target}_binary'] = (df[target] == 2).astype(int)

    # Get all unique images
    all_images = list(df['im1_path'].unique()) + list(df['im2_path'].unique())
    all_images = list(set(all_images))

    print(f"Found {len(all_images)} unique images")

    # Load or extract embeddings with specified CLIP model
    cache_file = cfg.data.get('embeddings_cache_file', 'clip_embeddings_cache.pkl')
    embeddings = load_or_extract_embeddings(
        all_images,
        clip_model_name=clip_model_name,
        cache_file=cache_file,
        device=device,
        force_recompute=cfg.data.get('force_recompute_embeddings', False)
    )

    # Get embedding dimension
    embedding_dim = next(iter(embeddings.values())).shape[0]
    print(f"Embedding dimension: {embedding_dim}")

    # Get unique users
    users = df['user_id'].unique()
    n_users = len(users)
    user_to_idx = {user: idx for idx, user in enumerate(users)}

    # Prepare data
    all_samples = []
    for target in cfg.data.targets:
        for _, row in df.iterrows():
            if row['im1_path'] in embeddings and row['im2_path'] in embeddings:
                sample = {
                    'emb1': embeddings[row['im1_path']],
                    'emb2': embeddings[row['im2_path']],
                    'label': row[f'{target}_binary'],
                    'user_idx': user_to_idx[row['user_id']],
                    'target': target
                }
                all_samples.append(sample)

    # Shuffle and split
    np.random.seed(cfg.seed)
    np.random.shuffle(all_samples)

    # Split into train/val/test
    n_samples = len(all_samples)
    n_test = int(n_samples * cfg.training.test_split)
    n_val = int(n_samples * cfg.training.validation_split)
    n_train = n_samples - n_test - n_val

    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]

    # Augment training data
    if cfg.training.get('augment_swapped_pairs', True):
        augmented = []
        for sample in train_samples:
            augmented.append({
                'emb1': sample['emb2'],
                'emb2': sample['emb1'],
                'label': 1 - sample['label'],
                'user_idx': sample['user_idx'],
                'target': sample['target']
            })
        train_samples.extend(augmented)

    # Organize by target and convert to tensors
    train_data = {}
    val_data = {}
    test_data = {}

    for target in cfg.data.targets:
        # Train
        target_train = [s for s in train_samples if s['target'] == target]
        if target_train:
            train_data[target] = {
                'X1': torch.FloatTensor([s['emb1'] for s in target_train]).to(device),
                'X2': torch.FloatTensor([s['emb2'] for s in target_train]).to(device),
                'y': torch.LongTensor([s['label'] for s in target_train]).to(device),
                'y_float': torch.FloatTensor([s['label'] for s in target_train]).to(device),
                'y_np': np.array([s['label'] for s in target_train]),
            }

        # Val
        target_val = [s for s in val_samples if s['target'] == target]
        if target_val:
            val_data[target] = {
                'X1': torch.FloatTensor([s['emb1'] for s in target_val]).to(device),
                'X2': torch.FloatTensor([s['emb2'] for s in target_val]).to(device),
                'y_np': np.array([s['label'] for s in target_val]),
            }

        # Test
        target_test = [s for s in test_samples if s['target'] == target]
        if target_test:
            test_data[target] = {
                'X1': torch.FloatTensor([s['emb1'] for s in target_test]).to(device),
                'X2': torch.FloatTensor([s['emb2'] for s in target_test]).to(device),
                'y_np': np.array([s['label'] for s in target_test]),
            }

    return train_data, val_data, test_data, n_users, embedding_dim


def evaluate_model(model, data, cfg, device):
    """Evaluate model on given data."""
    model.eval()
    results = {}

    with torch.no_grad():
        for target in cfg.data.targets:
            if target not in data:
                continue

            predictions = model(
                data[target]['X1'],
                data[target]['X2'],
                None,
                target=target
            )

            pred_binary = (predictions.cpu().numpy() > 0.5).astype(int)
            true_binary = data[target]['y_np']

            accuracy = np.mean(pred_binary == true_binary)

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
                'n_samples': len(data[target]['y_np'])
            }

    return results


def train_with_clip_model(cfg, clip_model_name, device):
    """Train model with specified CLIP model."""

    print(f"\nTraining with CLIP model: {clip_model_name}")

    # Load data
    train_data, val_data, test_data, n_users, embedding_dim = load_comparison_data_for_clip(
        cfg, clip_model_name, device
    )

    # Initialize model with correct embedding dimension
    model = UnifiedModel(cfg, n_users, input_dim=embedding_dim).to(device)

    # Setup training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.get('weight_decay', 0)
    )
    criterion = nn.BCELoss()

    # Training loop (simplified, without early stopping for speed)
    num_epochs = min(50, cfg.training.get('epochs', 50))  # Limit epochs for testing

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()

        for target in cfg.data.targets:
            if target not in train_data:
                continue

            optimizer.zero_grad()
            predictions = model(
                train_data[target]['X1'],
                train_data[target]['X2'],
                None,
                target=target
            )
            loss = criterion(predictions, train_data[target]['y_float'])
            loss.backward()
            optimizer.step()

    # Evaluate
    val_results = evaluate_model(model, val_data, cfg, device)
    test_results = evaluate_model(model, test_data, cfg, device)

    # Calculate averages
    avg_val_acc = np.mean([r['accuracy'] for r in val_results.values()])
    avg_test_acc = np.mean([r['accuracy'] for r in test_results.values()])

    return {
        'clip_model': clip_model_name,
        'embedding_dim': embedding_dim,
        'val_accuracy': avg_val_acc,
        'test_accuracy': avg_test_acc,
        'val_results': val_results,
        'test_results': test_results
    }


@hydra.main(version_base=None, config_path="configs", config_name="unified_optimal_final")
def main(cfg: DictConfig):
    """Main function to test different CLIP models."""

    # Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu.device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*70)
    print("CLIP MODEL VARIANT TESTING")
    print("="*70)

    # Models to test
    models_to_test = [
        "ViT-B/32",      # Current baseline
        "ViT-B/16",      # Higher resolution
        "ViT-L/14",      # Larger model
    ]

    # Check available models
    available_models = []
    for model_name in models_to_test:
        if model_name in clip.available_models():
            available_models.append(model_name)
            print(f"✓ {model_name} available")
        else:
            print(f"✗ {model_name} not available")

    if not available_models:
        print("No CLIP models available!")
        return

    # Test each model
    results = []
    for model_name in available_models:
        print(f"\n{'='*50}")
        print(f"Testing {model_name}")
        print('='*50)

        result = train_with_clip_model(cfg, model_name, device)
        results.append(result)

        print(f"\nResults for {model_name}:")
        print(f"  Embedding dimension: {result['embedding_dim']}")
        print(f"  Validation accuracy: {result['val_accuracy']:.4f}")
        print(f"  Test accuracy: {result['test_accuracy']:.4f}")

    # Compare results
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    print(f"{'Model':<15} {'Dim':<8} {'Val Acc':<10} {'Test Acc':<10}")
    print("-"*50)

    for r in results:
        print(f"{r['clip_model']:<15} {r['embedding_dim']:<8} "
              f"{r['val_accuracy']:<10.4f} {r['test_accuracy']:<10.4f}")

    # Save results
    output_file = 'clip_model_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()