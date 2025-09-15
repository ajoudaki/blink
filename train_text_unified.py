#!/usr/bin/env python3
"""
Unified model using CLIP text embeddings for target labels.
Concatenates image, text, and user embeddings for a truly unified architecture.
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
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from torch.utils.data import Dataset, DataLoader
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextEmbeddingCache:
    """Cache for CLIP text embeddings of target labels and variations."""

    def __init__(self, model_name="ViT-B/32"):
        """Initialize CLIP model and cache."""
        self.cache_file = Path("cached_data/text_embeddings_cache.pkl")
        self.cache_file.parent.mkdir(exist_ok=True)

        # Load CLIP model for text encoding
        self.clip_model, _ = clip.load(model_name, device=device)
        self.clip_model.eval()

        # Try to load existing cache
        if self.cache_file.exists():
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)
            logger.info(f"Loaded text embedding cache with {len(self.cache)} entries")
        else:
            self.cache = {}
            self._generate_embeddings()

    def _generate_embeddings(self):
        """Generate text embeddings for all targets and variations."""
        logger.info("Generating text embeddings...")

        # Define text variations for each target
        templates = {
            'attractive': [
                "attractive",
                "an attractive person",
                "this person is attractive",
                "this person looks attractive",
                "a physically attractive individual",
                "someone who is attractive"
            ],
            'smart': [
                "smart",
                "an intelligent person",
                "this person is smart",
                "this person looks smart",
                "this person seems intelligent",
                "someone who is smart"
            ],
            'trustworthy': [
                "trustworthy",
                "a trustworthy person",
                "this person is trustworthy",
                "this person looks trustworthy",
                "this person seems trustworthy",
                "someone who is trustworthy"
            ]
        }

        with torch.no_grad():
            for target, texts in templates.items():
                embeddings = []
                for text in texts:
                    # Tokenize and encode text
                    tokens = clip.tokenize([text]).to(device)
                    text_features = self.clip_model.encode_text(tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    embeddings.append(text_features.cpu().numpy())

                # Average all variations for this target
                avg_embedding = np.mean(embeddings, axis=0).squeeze()
                self.cache[target] = avg_embedding

                # Also store individual variations for experimentation
                for i, text in enumerate(texts):
                    self.cache[f"{target}_v{i}"] = embeddings[i].squeeze()

        # Save cache
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
        logger.info(f"Generated and cached {len(self.cache)} text embeddings")

    def get_embedding(self, target, variation=None):
        """Get text embedding for a target."""
        if variation is not None:
            key = f"{target}_v{variation}"
        else:
            key = target

        if key not in self.cache:
            raise KeyError(f"Text embedding for '{key}' not found in cache")

        return self.cache[key]


class UnifiedTextModel(nn.Module):
    """Unified model using text embeddings - single readout for all targets."""

    def __init__(self, cfg, n_users=None):
        super().__init__()
        self.cfg = cfg
        self.task_type = cfg.task_type

        # Calculate input dimension
        clip_dim = 512  # CLIP image embedding dimension
        text_dim = 512  # CLIP text embedding dimension
        user_dim = cfg.model.user_embedding_dim if cfg.model.use_user_embedding and n_users else 0

        input_dim = clip_dim + text_dim
        if user_dim > 0:
            self.user_embedding = nn.Embedding(n_users, user_dim)
            input_dim += user_dim
        else:
            self.user_embedding = None

        # Build shared encoder
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(cfg.model.hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Normalization
            if cfg.model.use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif cfg.model.use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))

            # Activation
            if cfg.model.activation == 'relu':
                layers.append(nn.ReLU())
            elif cfg.model.activation == 'gelu':
                layers.append(nn.GELU())
            elif cfg.model.activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))

            # Dropout
            if cfg.model.dropout > 0:
                layers.append(nn.Dropout(cfg.model.dropout))

            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Single unified readout head
        if cfg.task_type == 'rating':
            # 4-way classification for rating
            self.readout = nn.Linear(prev_dim, 4)
        else:
            # Single score for comparison (will be compared between two images)
            self.readout = nn.Linear(prev_dim, 1)

    def forward(self, image_embeds, text_embeds, user_ids=None):
        """
        Forward pass.

        Args:
            image_embeds: Image CLIP embeddings [batch_size, 512] or [batch_size, 2, 512] for comparison
            text_embeds: Text CLIP embeddings [batch_size, 512] or [batch_size, 2, 512] for comparison
            user_ids: User indices [batch_size]
        """
        if self.task_type == 'comparison':
            # Handle paired inputs for comparison
            batch_size = image_embeds.shape[0]

            # Process each image separately
            scores = []
            for i in range(2):
                # Concatenate image and text embeddings
                combined = torch.cat([image_embeds[:, i], text_embeds[:, i]], dim=-1)

                # Add user embeddings if applicable
                if self.user_embedding is not None and user_ids is not None:
                    user_embeds = self.user_embedding(user_ids)
                    combined = torch.cat([combined, user_embeds], dim=-1)

                # Encode
                features = self.encoder(combined)
                score = self.readout(features)
                scores.append(score)

            # Stack scores for comparison
            scores = torch.cat(scores, dim=-1)  # [batch_size, 2]
            return scores

        else:  # rating
            # Concatenate image and text embeddings
            combined = torch.cat([image_embeds, text_embeds], dim=-1)

            # Add user embeddings if applicable
            if self.user_embedding is not None and user_ids is not None:
                user_embeds = self.user_embedding(user_ids)
                combined = torch.cat([combined, user_embeds], dim=-1)

            # Encode and readout
            features = self.encoder(combined)
            logits = self.readout(features)
            return logits


class UnifiedTextDataset(Dataset):
    """Dataset for unified model with text embeddings."""

    def __init__(self, data_df, image_cache, text_cache, task_type='rating', user_to_idx=None):
        self.data = data_df
        self.image_cache = image_cache
        self.text_cache = text_cache
        self.task_type = task_type
        self.user_to_idx = user_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Get target text embedding
        target = row['target']
        text_embed = torch.tensor(self.text_cache.get_embedding(target), dtype=torch.float32)

        if self.task_type == 'comparison':
            # Get both image embeddings
            img1_embed = torch.tensor(self.image_cache[row['image1']], dtype=torch.float32)
            img2_embed = torch.tensor(self.image_cache[row['image2']], dtype=torch.float32)
            image_embeds = torch.stack([img1_embed, img2_embed])

            # Stack text embedding for both images (same target)
            text_embeds = torch.stack([text_embed, text_embed])

            # Label is which image won (0 or 1)
            label = torch.tensor(row['label'], dtype=torch.long)
        else:
            # Single image embedding for rating
            image_embeds = torch.tensor(self.image_cache[row['image']], dtype=torch.float32)
            text_embeds = text_embed

            # Label is rating (0-3 for 1-4 scale)
            label = torch.tensor(row['rating'] - 1, dtype=torch.long)

        # Get user index if applicable
        if self.user_to_idx:
            user_idx = torch.tensor(self.user_to_idx[row['user_id']], dtype=torch.long)
            return image_embeds, text_embeds, user_idx, label
        else:
            return image_embeds, text_embeds, label


def train_model(cfg: DictConfig, data_df, image_cache, text_cache):
    """Train unified model with text embeddings."""

    # Get unique users if using user embeddings
    if cfg.model.use_user_embedding:
        unique_users = data_df['user_id'].unique()
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        n_users = len(unique_users)
    else:
        user_to_idx = None
        n_users = None

    # Split data
    if cfg.task_type == 'comparison':
        # For comparison, stratify by user and target
        stratify_col = data_df['user_id'].astype(str) + '_' + data_df['target']
    else:
        # For rating, stratify by user, target, and rating
        stratify_col = data_df['user_id'].astype(str) + '_' + data_df['target'] + '_' + data_df['rating'].astype(str)

    train_df, val_df = train_test_split(
        data_df,
        test_size=cfg.data.val_split,
        random_state=cfg.seed,
        stratify=stratify_col
    )

    # Create datasets
    train_dataset = UnifiedTextDataset(train_df, image_cache, text_cache, cfg.task_type, user_to_idx)
    val_dataset = UnifiedTextDataset(val_df, image_cache, text_cache, cfg.task_type, user_to_idx)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )

    # Initialize model
    model = UnifiedTextModel(cfg, n_users).to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if cfg.optimizer.name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    elif cfg.optimizer.name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay
        )

    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in range(cfg.training.epochs):
        # Training
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs}"):
            if cfg.model.use_user_embedding:
                image_embeds, text_embeds, user_ids, labels = batch
                image_embeds = image_embeds.to(device)
                text_embeds = text_embeds.to(device)
                user_ids = user_ids.to(device)
                labels = labels.to(device)

                outputs = model(image_embeds, text_embeds, user_ids)
            else:
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

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0

        if cfg.task_type == 'rating':
            val_mae = []

        with torch.no_grad():
            for batch in val_loader:
                if cfg.model.use_user_embedding:
                    image_embeds, text_embeds, user_ids, labels = batch
                    image_embeds = image_embeds.to(device)
                    text_embeds = text_embeds.to(device)
                    user_ids = user_ids.to(device)
                    labels = labels.to(device)

                    outputs = model(image_embeds, text_embeds, user_ids)
                else:
                    image_embeds, text_embeds, labels = batch
                    image_embeds = image_embeds.to(device)
                    text_embeds = text_embeds.to(device)
                    labels = labels.to(device)

                    outputs = model(image_embeds, text_embeds)

                loss = criterion(outputs, labels)
                val_losses.append(loss.item())

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Calculate MAE for rating task
                if cfg.task_type == 'rating':
                    mae = torch.abs(predicted - labels).float().mean()
                    val_mae.append(mae.item())

        # Calculate metrics
        train_loss = np.mean(train_losses)
        train_acc = train_correct / train_total
        val_loss = np.mean(val_losses)
        val_acc = val_correct / val_total

        # Log metrics
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"         Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            if cfg.task_type == 'rating':
                logger.info(f"         Val MAE: {np.mean(val_mae):.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            if cfg.task_type == 'rating':
                best_val_mae = np.mean(val_mae)

    results = {
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
    }

    if cfg.task_type == 'rating':
        results['best_val_mae'] = best_val_mae

    return results


def load_comparison_data():
    """Load and prepare comparison data."""
    data_file = Path("analysis/data/labels/comparative_labels_processed.csv")
    df = pd.read_csv(data_file)

    # Filter to required columns
    df = df[['user_id', 'winner_image', 'loser_image', 'target']]

    # Prepare for model
    comparison_data = []

    for _, row in df.iterrows():
        # Original order
        comparison_data.append({
            'user_id': row['user_id'],
            'image1': row['winner_image'],
            'image2': row['loser_image'],
            'target': row['target'],
            'label': 0  # First image wins
        })

        # Swapped order (data augmentation)
        comparison_data.append({
            'user_id': row['user_id'],
            'image1': row['loser_image'],
            'image2': row['winner_image'],
            'target': row['target'],
            'label': 1  # Second image wins
        })

    return pd.DataFrame(comparison_data)


def load_rating_data():
    """Load and prepare rating data."""
    data_file = Path("analysis/data/labels/individual_labels_processed.csv")
    df = pd.read_csv(data_file)

    # Filter and rename columns
    df = df[['user_id', 'image', 'rating', 'target']]

    return df


@hydra.main(config_path="configs", config_name="unified_text_base", version_base=None)
def main(cfg: DictConfig):
    """Main training function."""
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load image embeddings cache
    cache_file = Path("cached_data/clip_embeddings.pkl")
    with open(cache_file, 'rb') as f:
        image_cache = pickle.load(f)
    logger.info(f"Loaded image embeddings for {len(image_cache)} images")

    # Initialize text embedding cache
    text_cache = TextEmbeddingCache()

    # Load data based on task type
    if cfg.task_type == 'comparison':
        logger.info("Loading comparison data...")
        data_df = load_comparison_data()
        logger.info(f"Loaded {len(data_df)} comparison samples (with augmentation)")
    else:
        logger.info("Loading rating data...")
        data_df = load_rating_data()
        logger.info(f"Loaded {len(data_df)} rating samples")

    # Train model
    logger.info("Training model...")
    results = train_model(cfg, data_df, image_cache, text_cache)

    # Report results
    logger.info("\n" + "="*50)
    logger.info(f"Task: {cfg.task_type.upper()}")
    logger.info(f"Best Validation Accuracy: {results['best_val_acc']:.4f}")
    logger.info(f"Best Validation CE Loss: {results['best_val_loss']:.4f}")
    if cfg.task_type == 'rating':
        logger.info(f"Best Validation MAE: {results['best_val_mae']:.4f}")
    logger.info("="*50)


if __name__ == "__main__":
    main()