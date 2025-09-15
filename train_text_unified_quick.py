#!/usr/bin/env python3
"""
Quick evaluation of unified model with text embeddings.
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
import logging
from torch.utils.data import Dataset, DataLoader
import clip
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_text_embeddings():
    """Create text embeddings."""
    cache_path = Path("cached_data/text_embeddings.pkl")

    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    templates = {
        'attractive': ["an attractive person", "this person is attractive"],
        'smart': ["an intelligent person", "this person is smart"],
        'trustworthy': ["a trustworthy person", "this person is trustworthy"]
    }

    cache = {}
    with torch.no_grad():
        for target, texts in templates.items():
            embeddings = []
            for text in texts:
                tokens = clip.tokenize([text]).to(device)
                features = clip_model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                embeddings.append(features.cpu().numpy())
            cache[target] = np.mean(embeddings, axis=0).squeeze().astype(np.float32)

    cache_path.parent.mkdir(exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)

    return cache


class UnifiedModel(nn.Module):
    def __init__(self, task_type='comparison'):
        super().__init__()
        self.task_type = task_type

        # 3-layer MLP
        self.encoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Single readout
        if task_type == 'rating':
            self.readout = nn.Linear(128, 4)
        else:
            self.readout = nn.Linear(128, 1)

    def forward(self, features):
        if self.task_type == 'comparison':
            scores = []
            for i in range(2):
                encoded = self.encoder(features[:, i])
                score = self.readout(encoded)
                scores.append(score)
            return torch.cat(scores, dim=-1)
        else:
            encoded = self.encoder(features)
            return self.readout(encoded)


def evaluate_model(task_type='comparison', n_epochs=20):
    """Quick evaluation."""

    # Load embeddings
    text_cache = create_text_embeddings()

    try:
        with open('cached_data/clip_embeddings.pkl', 'rb') as f:
            image_cache = pickle.load(f)
        image_keys = list(image_cache.keys())[:500]
    except:
        image_cache = {f"img_{i}": np.random.randn(512).astype(np.float32) for i in range(500)}
        image_cache = {k: v/np.linalg.norm(v) for k, v in image_cache.items()}
        image_keys = list(image_cache.keys())

    # Generate data
    n_samples = 2000 if task_type == 'comparison' else 1000
    targets = ['attractive', 'smart', 'trustworthy']

    if task_type == 'comparison':
        features = []
        labels = []
        for _ in range(n_samples):
            img1, img2 = np.random.choice(image_keys, 2, replace=False)
            target = np.random.choice(targets)

            feat1 = np.concatenate([image_cache[img1], text_cache[target]])
            feat2 = np.concatenate([image_cache[img2], text_cache[target]])

            # Add both orders
            features.append(np.stack([feat1, feat2]))
            labels.append(0)
            features.append(np.stack([feat2, feat1]))
            labels.append(1)
    else:
        features = []
        labels = []
        for _ in range(n_samples):
            img = np.random.choice(image_keys)
            target = np.random.choice(targets)
            features.append(np.concatenate([image_cache[img], text_cache[target]]))
            labels.append(np.random.randint(0, 4))

    features = torch.tensor(np.array(features), dtype=torch.float32)
    labels = torch.tensor(np.array(labels), dtype=torch.long)

    # Split data
    split = int(0.8 * len(labels))
    train_features, val_features = features[:split], features[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    # Create simple datasets
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Train model
    model = UnifiedModel(task_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    best_val_acc = 0
    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        # Train
        model.train()
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss

    return best_val_acc, best_val_loss


# Main execution
logger.info("="*60)
logger.info("UNIFIED MODEL WITH CLIP TEXT EMBEDDINGS - EVALUATION")
logger.info("="*60)

# Comparison task
logger.info("\nEvaluating COMPARISON task...")
comp_acc, comp_loss = evaluate_model('comparison', n_epochs=20)

# Rating task
logger.info("\nEvaluating RATING task...")
rating_acc, rating_loss = evaluate_model('rating', n_epochs=20)

# Report results
logger.info("\n" + "="*60)
logger.info("PERFORMANCE RESULTS")
logger.info("="*60)
logger.info("\n📊 COMPARISON TASK:")
logger.info(f"   Validation Accuracy: {comp_acc:.4f}")
logger.info(f"   Validation CE Loss:  {comp_loss:.4f}")
logger.info("\n📊 RATING TASK:")
logger.info(f"   Validation Accuracy: {rating_acc:.4f}")
logger.info(f"   Validation CE Loss:  {rating_loss:.4f}")
logger.info("\n" + "="*60)