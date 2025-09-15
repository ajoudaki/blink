#!/usr/bin/env python3
"""
Simplified unified model using CLIP text embeddings.
Loads pre-processed data from the per-labeler evaluation.
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
from collections import defaultdict

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
            ],
            'smart': [
                "an intelligent person",
                "this person is smart",
                "this person looks intelligent",
            ],
            'trustworthy': [
                "a trustworthy person",
                "this person is trustworthy",
                "this person looks trustworthy",
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

        logger.info(f"Generated {len(self.cache)} text embeddings")

    def get_embedding(self, target):
        """Get text embedding for a target."""
        return self.cache[target]


class UnifiedTextModel(nn.Module):
    """Unified model with text embeddings - single readout for all targets."""

    def __init__(self, task_type='comparison'):
        super().__init__()
        self.task_type = task_type

        # Input: CLIP image (512) + CLIP text (512) = 1024
        input_dim = 1024
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

    def forward(self, image_embeds, text_embeds):
        """Forward pass."""
        if self.task_type == 'comparison':
            # Process each image separately for comparison
            scores = []
            for i in range(2):
                combined = torch.cat([image_embeds[:, i], text_embeds[:, i]], dim=-1)
                features = self.encoder(combined)
                score = self.readout(features)
                scores.append(score)
            scores = torch.cat(scores, dim=-1)  # [batch_size, 2]
            return scores
        else:
            # Single image for rating
            combined = torch.cat([image_embeds, text_embeds], dim=-1)
            features = self.encoder(combined)
            logits = self.readout(features)
            return logits


def load_comparison_data_from_pkl():
    """Load comparison data from the pickle file."""
    import json

    with open('analysis/data/labels.pkl', 'rb') as f:
        df = pickle.load(f)

    # Load CLIP embeddings
    with open('cached_data/clip_embeddings.pkl', 'rb') as f:
        clip_cache = pickle.load(f)

    # Process comparison data
    comparison_data = defaultdict(list)

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

                                comparison_data[target].append({
                                    'user_id': user_id,
                                    'image1': img1,
                                    'image2': img2,
                                    'label': label,
                                    'target': target
                                })
                except:
                    continue

    # Combine all targets
    all_data = []
    for target, data in comparison_data.items():
        all_data.extend(data)

    logger.info(f"Loaded {len(all_data)} comparison samples")
    for target in comparison_data:
        logger.info(f"  {target}: {len(comparison_data[target])} samples")

    return pd.DataFrame(all_data), clip_cache


def load_rating_data_from_pkl():
    """Load rating data from the pickle file."""
    with open('analysis/data/labels.pkl', 'rb') as f:
        df = pickle.load(f)

    # Load CLIP embeddings
    with open('cached_data/clip_embeddings.pkl', 'rb') as f:
        clip_cache = pickle.load(f)

    # Process rating data
    rating_data = defaultdict(list)

    for _, row in df.iterrows():
        if pd.isna(row['label']):
            continue

        user_id = row['user_id']
        item_path = row.get('item_path', '')

        # Skip if not rating task
        if 'rating' not in str(item_path).lower() and 'individual' not in str(item_path).lower():
            continue

        # Parse item path to get image
        if isinstance(item_path, str) and 'src/' in item_path:
            parts = item_path.split('/')
            if len(parts) >= 1:
                try:
                    # Extract image name
                    img_name = parts[-1].split('.')[0]
                    img_path = f"/home/labeler/v3/web_customer_labeler/data/ffhq2/src/{img_name}.webp"

                    # Check if image exists in cache
                    if img_path in clip_cache:
                        # Get rating (1-4 scale)
                        rating = int(row['label']) if not pd.isna(row['label']) else None
                        if rating is not None and 1 <= rating <= 4:
                            # Get target from path
                            target = 'attractive'  # Default
                            if 'smart' in item_path.lower() or 'intelligence' in item_path.lower():
                                target = 'smart'
                            elif 'trust' in item_path.lower():
                                target = 'trustworthy'

                            rating_data[target].append({
                                'user_id': user_id,
                                'image': img_path,
                                'rating': rating,
                                'target': target
                            })
                except:
                    continue

    # Combine all targets
    all_data = []
    for target, data in rating_data.items():
        all_data.extend(data)

    logger.info(f"Loaded {len(all_data)} rating samples")
    for target in rating_data:
        logger.info(f"  {target}: {len(rating_data[target])} samples")

    return pd.DataFrame(all_data), clip_cache


class ComparisonDataset(Dataset):
    """Dataset for comparison task."""

    def __init__(self, data_df, clip_cache, text_cache):
        self.data = data_df
        self.clip_cache = clip_cache
        self.text_cache = text_cache

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Get image embeddings
        img1_embed = torch.tensor(self.clip_cache[row['image1']], dtype=torch.float32)
        img2_embed = torch.tensor(self.clip_cache[row['image2']], dtype=torch.float32)
        image_embeds = torch.stack([img1_embed, img2_embed])

        # Get text embedding (same for both images)
        text_embed = torch.tensor(self.text_cache.get_embedding(row['target']), dtype=torch.float32)
        text_embeds = torch.stack([text_embed, text_embed])

        # Label
        label = torch.tensor(row['label'], dtype=torch.long)

        return image_embeds, text_embeds, label


class RatingDataset(Dataset):
    """Dataset for rating task."""

    def __init__(self, data_df, clip_cache, text_cache):
        self.data = data_df
        self.clip_cache = clip_cache
        self.text_cache = text_cache

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Get embeddings
        image_embed = torch.tensor(self.clip_cache[row['image']], dtype=torch.float32)
        text_embed = torch.tensor(self.text_cache.get_embedding(row['target']), dtype=torch.float32)

        # Label (convert 1-4 to 0-3)
        label = torch.tensor(row['rating'] - 1, dtype=torch.long)

        return image_embed, text_embed, label


def train_model(task_type, data_df, clip_cache, text_cache):
    """Train unified model."""

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
    if task_type == 'comparison':
        train_dataset = ComparisonDataset(train_df, clip_cache, text_cache)
        val_dataset = ComparisonDataset(val_df, clip_cache, text_cache)
    else:
        train_dataset = RatingDataset(train_df, clip_cache, text_cache)
        val_dataset = RatingDataset(val_df, clip_cache, text_cache)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Initialize model
    model = UnifiedTextModel(task_type).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Training
    best_val_acc = 0
    epochs = 50

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            image_embeds, text_embeds, labels = batch
            image_embeds = image_embeds.to(device)
            text_embeds = text_embeds.to(device)
            labels = labels.to(device)

            outputs = model(image_embeds, text_embeds)
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
                image_embeds, text_embeds, labels = batch
                image_embeds = image_embeds.to(device)
                text_embeds = text_embeds.to(device)
                labels = labels.to(device)

                outputs = model(image_embeds, text_embeds)
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
                best_val_mae = np.mean(val_mae)

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
            if task_type == 'rating':
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
    text_cache = TextEmbeddingCache()

    # Test comparison task
    logger.info("\n" + "="*50)
    logger.info("COMPARISON TASK")
    logger.info("="*50)

    comp_df, clip_cache = load_comparison_data_from_pkl()

    if len(comp_df) > 0:
        results = train_model('comparison', comp_df, clip_cache, text_cache)
        logger.info(f"\nBest Validation Accuracy: {results['best_val_acc']:.4f}")
        logger.info(f"Best Validation CE Loss: {results['best_val_loss']:.4f}")
    else:
        logger.info("No comparison data found")

    # Test rating task
    logger.info("\n" + "="*50)
    logger.info("RATING TASK")
    logger.info("="*50)

    rating_df, clip_cache = load_rating_data_from_pkl()

    if len(rating_df) > 0:
        results = train_model('rating', rating_df, clip_cache, text_cache)
        logger.info(f"\nBest Validation Accuracy: {results['best_val_acc']:.4f}")
        logger.info(f"Best Validation CE Loss: {results['best_val_loss']:.4f}")
        logger.info(f"Best Validation MAE: {results['best_val_mae']:.4f}")
    else:
        logger.info("No rating data found")


if __name__ == "__main__":
    main()