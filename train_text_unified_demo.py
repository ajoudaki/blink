#!/usr/bin/env python3
"""
Demonstration of unified model with CLIP text embeddings.
Uses synthetic data to show the architecture working.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
import clip
from sklearn.model_selection import train_test_split
import logging
from torch.utils.data import Dataset, DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextEmbeddingCache:
    """Cache for CLIP text embeddings of target labels."""

    def __init__(self):
        """Initialize CLIP model and generate embeddings."""
        self.cache = {}
        self._generate_embeddings()

    def _generate_embeddings(self):
        """Generate text embeddings for all targets."""
        logger.info("Generating text embeddings...")

        # Load CLIP model for text encoding
        clip_model, _ = clip.load("ViT-B/32", device=device)
        clip_model.eval()

        # Define text templates for each target
        templates = {
            'attractive': [
                "an attractive person",
                "this person is attractive",
                "this person looks attractive",
                "someone who is physically attractive"
            ],
            'smart': [
                "an intelligent person",
                "this person is smart",
                "this person looks intelligent",
                "someone who seems smart"
            ],
            'trustworthy': [
                "a trustworthy person",
                "this person is trustworthy",
                "this person looks trustworthy",
                "someone who seems reliable"
            ]
        }

        with torch.no_grad():
            for target, texts in templates.items():
                embeddings = []
                for text in texts:
                    tokens = clip.tokenize([text]).to(device)
                    text_features = clip_model.encode_text(tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    embeddings.append(text_features.cpu().numpy())

                # Average all variations
                avg_embedding = np.mean(embeddings, axis=0).squeeze()
                self.cache[target] = avg_embedding

        logger.info(f"Generated text embeddings for: {list(self.cache.keys())}")

    def get_embedding(self, target):
        """Get text embedding for a target."""
        return self.cache[target]


class UnifiedTextModel(nn.Module):
    """Unified model with text embeddings - single readout for all targets."""

    def __init__(self, task_type='comparison', use_user_embedding=True, n_users=10):
        super().__init__()
        self.task_type = task_type
        self.use_user_embedding = use_user_embedding

        # Input: CLIP image (512) + CLIP text (512) = 1024
        input_dim = 1024

        # Add user embedding dimension if enabled
        if use_user_embedding:
            user_embedding_dim = 32
            self.user_embedding = nn.Embedding(n_users, user_embedding_dim)
            input_dim += user_embedding_dim
        else:
            self.user_embedding = None

        hidden_dims = [512, 256, 128]

        # Build encoder
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Single unified readout
        if task_type == 'rating':
            self.readout = nn.Linear(prev_dim, 4)  # 4-way classification
        else:
            self.readout = nn.Linear(prev_dim, 1)  # Single score for comparison

    def forward(self, image_embeds, text_embeds, user_ids=None):
        """Forward pass."""
        if self.task_type == 'comparison':
            # Process each image separately for comparison
            scores = []
            for i in range(2):
                combined = torch.cat([image_embeds[:, i], text_embeds[:, i]], dim=-1)

                # Add user embeddings if available
                if self.user_embedding is not None and user_ids is not None:
                    user_embeds = self.user_embedding(user_ids)
                    combined = torch.cat([combined, user_embeds], dim=-1)

                features = self.encoder(combined)
                score = self.readout(features)
                scores.append(score)

            scores = torch.cat(scores, dim=-1)  # [batch_size, 2]
            return scores
        else:
            # Single image for rating
            combined = torch.cat([image_embeds, text_embeds], dim=-1)

            # Add user embeddings if available
            if self.user_embedding is not None and user_ids is not None:
                user_embeds = self.user_embedding(user_ids)
                combined = torch.cat([combined, user_embeds], dim=-1)

            features = self.encoder(combined)
            logits = self.readout(features)
            return logits


def generate_synthetic_data(task_type, n_samples=5000):
    """Generate synthetic data for demonstration."""

    # Load existing CLIP embeddings if available, otherwise create synthetic
    clip_cache = {}

    try:
        with open('cached_data/clip_embeddings.pkl', 'rb') as f:
            real_cache = pickle.load(f)
            image_keys = list(real_cache.keys())[:1000]  # Use subset
            for key in image_keys:
                clip_cache[key] = real_cache[key]
        logger.info(f"Using {len(clip_cache)} real CLIP embeddings")
    except:
        # Create synthetic CLIP embeddings
        logger.info("Creating synthetic CLIP embeddings")
        for i in range(1000):
            # Random 512-dimensional normalized vectors
            embed = np.random.randn(512).astype(np.float32)
            embed = embed / np.linalg.norm(embed)
            clip_cache[f"image_{i}"] = embed

    image_keys = list(clip_cache.keys())
    targets = ['attractive', 'smart', 'trustworthy']
    users = [f"user_{i}" for i in range(10)]

    data = []

    if task_type == 'comparison':
        for _ in range(n_samples):
            img1, img2 = np.random.choice(image_keys, 2, replace=False)
            target = np.random.choice(targets)
            user = np.random.choice(users)
            label = np.random.choice([0, 1])  # Which image wins

            data.append({
                'image1': img1,
                'image2': img2,
                'target': target,
                'user_id': user,
                'label': label
            })
    else:  # rating
        for _ in range(n_samples):
            img = np.random.choice(image_keys)
            target = np.random.choice(targets)
            user = np.random.choice(users)
            rating = np.random.choice([1, 2, 3, 4])

            data.append({
                'image': img,
                'target': target,
                'user_id': user,
                'rating': rating
            })

    return pd.DataFrame(data), clip_cache


class UnifiedDataset(Dataset):
    """Dataset for unified model."""

    def __init__(self, data_df, clip_cache, text_cache, task_type, user_to_idx):
        self.data = data_df
        self.clip_cache = clip_cache
        self.text_cache = text_cache
        self.task_type = task_type
        self.user_to_idx = user_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Get user index
        user_idx = torch.tensor(self.user_to_idx[row['user_id']], dtype=torch.long)

        if self.task_type == 'comparison':
            # Get image embeddings
            img1_embed = torch.tensor(self.clip_cache[row['image1']], dtype=torch.float32)
            img2_embed = torch.tensor(self.clip_cache[row['image2']], dtype=torch.float32)
            image_embeds = torch.stack([img1_embed, img2_embed])

            # Get text embedding (same for both images)
            text_embed = torch.tensor(self.text_cache.get_embedding(row['target']), dtype=torch.float32)
            text_embeds = torch.stack([text_embed, text_embed])

            # Label
            label = torch.tensor(row['label'], dtype=torch.long)
        else:
            # Single image for rating
            image_embeds = torch.tensor(self.clip_cache[row['image']], dtype=torch.float32)
            text_embeds = torch.tensor(self.text_cache.get_embedding(row['target']), dtype=torch.float32)

            # Label (convert 1-4 to 0-3)
            label = torch.tensor(row['rating'] - 1, dtype=torch.long)

        return image_embeds, text_embeds, user_idx, label


def train_model(task_type, data_df, clip_cache, text_cache):
    """Train unified model."""

    # Create user mapping
    unique_users = data_df['user_id'].unique()
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    n_users = len(unique_users)

    logger.info(f"Training with {n_users} unique users")

    # Add data augmentation for comparison
    if task_type == 'comparison':
        augmented_data = []
        for _, row in data_df.iterrows():
            # Original
            augmented_data.append(row.to_dict())
            # Swapped
            swapped = row.to_dict()
            swapped['image1'] = row['image2']
            swapped['image2'] = row['image1']
            swapped['label'] = 1 - row['label']
            augmented_data.append(swapped)
        data_df = pd.DataFrame(augmented_data)
        logger.info(f"Augmented to {len(data_df)} samples")

    # Split data
    train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = UnifiedDataset(train_df, clip_cache, text_cache, task_type, user_to_idx)
    val_dataset = UnifiedDataset(val_df, clip_cache, text_cache, task_type, user_to_idx)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Initialize model
    model = UnifiedTextModel(task_type, use_user_embedding=True, n_users=n_users).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Training
    best_val_acc = 0
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    epochs = 50

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            image_embeds, text_embeds, user_ids, labels = batch
            image_embeds = image_embeds.to(device)
            text_embeds = text_embeds.to(device)
            user_ids = user_ids.to(device)
            labels = labels.to(device)

            outputs = model(image_embeds, text_embeds, user_ids)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
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
        val_mae = []

        with torch.no_grad():
            for batch in val_loader:
                image_embeds, text_embeds, user_ids, labels = batch
                image_embeds = image_embeds.to(device)
                text_embeds = text_embeds.to(device)
                user_ids = user_ids.to(device)
                labels = labels.to(device)

                outputs = model(image_embeds, text_embeds, user_ids)
                loss = criterion(outputs, labels)

                val_losses.append(loss.item())
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                if task_type == 'rating':
                    mae = torch.abs(predicted - labels).float().mean()
                    val_mae.append(mae.item())

        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        val_loss = np.mean(val_losses)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            if task_type == 'rating':
                best_val_mae = np.mean(val_mae) if val_mae else float('inf')

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
            if task_type == 'rating' and val_mae:
                logger.info(f"         Val MAE: {np.mean(val_mae):.4f}")

    results = {
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
    }
    if task_type == 'rating':
        results['best_val_mae'] = best_val_mae

    return results


def main():
    """Main function."""

    # Initialize text embeddings
    logger.info("Initializing text embeddings...")
    text_cache = TextEmbeddingCache()

    # Test comparison task
    logger.info("\n" + "="*50)
    logger.info("UNIFIED MODEL WITH TEXT EMBEDDINGS - COMPARISON TASK")
    logger.info("="*50)

    comp_df, clip_cache = generate_synthetic_data('comparison', n_samples=5000)
    logger.info(f"Generated {len(comp_df)} comparison samples")

    results = train_model('comparison', comp_df, clip_cache, text_cache)
    logger.info(f"\n✓ Best Validation Accuracy: {results['best_val_acc']:.4f}")
    logger.info(f"✓ Best Validation CE Loss: {results['best_val_loss']:.4f}")

    # Test rating task
    logger.info("\n" + "="*50)
    logger.info("UNIFIED MODEL WITH TEXT EMBEDDINGS - RATING TASK")
    logger.info("="*50)

    rating_df, clip_cache = generate_synthetic_data('rating', n_samples=5000)
    logger.info(f"Generated {len(rating_df)} rating samples")

    results = train_model('rating', rating_df, clip_cache, text_cache)
    logger.info(f"\n✓ Best Validation Accuracy: {results['best_val_acc']:.4f}")
    logger.info(f"✓ Best Validation CE Loss: {results['best_val_loss']:.4f}")
    logger.info(f"✓ Best Validation MAE: {results['best_val_mae']:.4f}")

    logger.info("\n" + "="*50)
    logger.info("KEY ARCHITECTURE FEATURES:")
    logger.info("="*50)
    logger.info("✓ Single unified readout head for all targets")
    logger.info("✓ Text embeddings encode target attribute (attractive/smart/trustworthy)")
    logger.info("✓ Image and text embeddings concatenated as input")
    logger.info("✓ Optional user embeddings for personalization")
    logger.info("✓ Same base encoder for both comparison and rating tasks")
    logger.info("="*50)


if __name__ == "__main__":
    main()