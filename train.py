#!/usr/bin/env python3
"""
Unified training script for all model architectures.
Supports individual ratings and pairwise comparisons with various architectures.
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import clip
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

def load_or_extract_embeddings(image_paths, cache_file='clip_embeddings_cache.pkl',
                               device='cuda', force_recompute=False):
    """Load cached embeddings or extract new ones using CLIP."""

    if os.path.exists(cache_file) and not force_recompute:
        print(f"Loading embeddings from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"✓ All {len(embeddings)} embeddings loaded from cache")
        return embeddings

    print("Extracting CLIP embeddings...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    embeddings = {}

    for img_path in tqdm(image_paths, desc="Processing images"):
        if os.path.exists(img_path):
            try:
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = model.encode_image(image)
                    features = features.cpu().numpy().squeeze()
                embeddings[img_path] = features
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                embeddings[img_path] = np.zeros(512)
        else:
            embeddings[img_path] = np.zeros(512)

    # Save cache
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"✓ Embeddings saved to {cache_file}")

    return embeddings


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class MLPEncoder(nn.Module):
    """Flexible MLP encoder for various tasks."""

    def __init__(self, input_dim, output_dim, hidden_dims, activation='relu',
                 dropout=0.0, use_batch_norm=False, use_layer_norm=False):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif activation == 'elu':
                layers.append(nn.ELU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class RatingModel(nn.Module):
    """Model for individual rating prediction."""

    def __init__(self, cfg):
        super().__init__()

        if cfg.model.architecture == 'linear':
            self.model = nn.Linear(512, 1)
        else:  # MLP
            self.model = MLPEncoder(
                input_dim=512,
                output_dim=1,
                hidden_dims=cfg.model.hidden_dims,
                activation=cfg.model.activation,
                dropout=cfg.model.dropout,
                use_batch_norm=cfg.model.get('batch_norm', False),
                use_layer_norm=cfg.model.get('layer_norm', False)
            )

    def forward(self, x):
        return self.model(x).squeeze(-1)


class ComparisonModel(nn.Module):
    """Model for pairwise comparison (concatenated features)."""

    def __init__(self, cfg):
        super().__init__()

        self.model = MLPEncoder(
            input_dim=1024,  # Concatenated embeddings
            output_dim=1,
            hidden_dims=cfg.model.hidden_dims,
            activation=cfg.model.activation,
            dropout=cfg.model.dropout,
            use_batch_norm=cfg.model.get('batch_norm', False),
            use_layer_norm=cfg.model.get('layer_norm', False)
        )

    def forward(self, img1, img2):
        x = torch.cat([img1, img2], dim=1)
        return torch.sigmoid(self.model(x).squeeze(-1))


class SiameseEncoder(nn.Module):
    """Shared encoder for Siamese architecture."""

    def __init__(self, cfg, n_users=None):
        super().__init__()

        input_dim = 512
        if cfg.model.get('use_user_encoding', False) and n_users:
            input_dim += n_users

        self.encoder = MLPEncoder(
            input_dim=input_dim,
            output_dim=1,
            hidden_dims=cfg.model.hidden_dims,
            activation=cfg.model.activation,
            dropout=cfg.model.dropout,
            use_batch_norm=cfg.model.get('batch_norm', False),
            use_layer_norm=cfg.model.get('layer_norm', False)
        )

    def forward(self, x):
        return self.encoder(x).squeeze(-1)


class SiameseModel(nn.Module):
    """Siamese model for pairwise comparison."""

    def __init__(self, cfg, n_users=None):
        super().__init__()
        self.encoder = SiameseEncoder(cfg, n_users)
        self.use_user_encoding = cfg.model.get('use_user_encoding', False)

    def forward(self, img1, img2, user_encoding=None):
        if self.use_user_encoding and user_encoding is not None:
            img1 = torch.cat([img1, user_encoding], dim=1)
            img2 = torch.cat([img2, user_encoding], dim=1)

        score1 = self.encoder(img1)
        score2 = self.encoder(img2)

        scores = torch.stack([score1, score2], dim=1)
        probs = F.softmax(scores, dim=1)
        return probs[:, 1]


class MultiHeadSiameseEncoder(nn.Module):
    """Encoder with multiple output heads."""

    def __init__(self, cfg, n_users=None):
        super().__init__()

        input_dim = 512
        if cfg.model.get('use_user_encoding', False) and n_users:
            input_dim += n_users

        # Shared backbone
        layers = []
        prev_dim = input_dim

        for hidden_dim in cfg.model.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if cfg.model.get('batch_norm', False):
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif cfg.model.get('layer_norm', False):
                layers.append(nn.LayerNorm(hidden_dim))

            # Activation
            activation = cfg.model.activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))

            if cfg.model.dropout > 0:
                layers.append(nn.Dropout(cfg.model.dropout))

            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Output heads
        self.heads = nn.ModuleDict({
            'attractive': nn.Linear(prev_dim, 1),
            'smart': nn.Linear(prev_dim, 1),
            'trustworthy': nn.Linear(prev_dim, 1)
        })

    def forward(self, x, target='attractive'):
        features = self.backbone(x)
        return self.heads[target](features).squeeze(-1)


class MultiHeadSiameseModel(nn.Module):
    """Multi-head Siamese model."""

    def __init__(self, cfg, n_users=None):
        super().__init__()
        self.encoder = MultiHeadSiameseEncoder(cfg, n_users)
        self.use_user_encoding = cfg.model.get('use_user_encoding', False)

    def forward(self, img1, img2, target='attractive', user_encoding=None):
        if self.use_user_encoding and user_encoding is not None:
            img1 = torch.cat([img1, user_encoding], dim=1)
            img2 = torch.cat([img2, user_encoding], dim=1)

        score1 = self.encoder(img1, target)
        score2 = self.encoder(img2, target)

        scores = torch.stack([score1, score2], dim=1)
        probs = F.softmax(scores, dim=1)
        return probs[:, 1]


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_rating_model(df, embeddings, cfg, device):
    """Train model for individual rating prediction."""

    print("\nTraining rating prediction model...")

    # Prepare data for each target
    results = {}

    for target in cfg.data.targets:
        print(f"\nTraining for {target}...")

        # Prepare features and labels
        X_list = []
        y_list = []

        for _, row in df.iterrows():
            if pd.notna(row[target]):
                img_embedding = embeddings.get(row['image_path'], np.zeros(512))
                X_list.append(img_embedding)
                y_list.append(row[target])

        X = np.array(X_list)
        y = np.array(y_list)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=cfg.training.validation_split,
            random_state=42, stratify=pd.cut(y, bins=4, labels=False)
        )

        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(device)
        X_val = torch.FloatTensor(X_val).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        y_val_np = y_val

        # Initialize model
        model = RatingModel(cfg).to(device)
        optimizer = get_optimizer(model, cfg)
        criterion = nn.MSELoss()

        # Training loop
        for epoch in range(cfg.training.epochs):
            model.train()
            optimizer.zero_grad()

            predictions = model(X_train)
            loss = criterion(predictions, y_train)

            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val).cpu().numpy()
            val_predictions = np.clip(val_predictions, 1, 4)

            mae = mean_absolute_error(y_val_np, val_predictions)
            rmse = np.sqrt(mean_squared_error(y_val_np, val_predictions))

        results[target] = {
            'mae': mae,
            'rmse': rmse,
            'n_train': len(y_train),
            'n_val': len(y_val_np)
        }

        print(f"  MAE: {mae:.3f}, RMSE: {rmse:.3f}")

    return results


def train_comparison_model(df, embeddings, cfg, device):
    """Train model for pairwise comparison prediction."""

    print("\nTraining comparison model...")

    # Determine model type
    is_siamese = 'siamese' in cfg.model.architecture
    is_multihead = cfg.model.get('multi_head', False)
    use_user_encoding = cfg.model.get('use_user_encoding', False)

    # Prepare user encoder if needed
    user_encoder = None
    n_users = None
    if use_user_encoding:
        user_encoder = LabelEncoder()
        user_encoder.fit(df['user_id'].unique())
        n_users = len(user_encoder.classes_)
        print(f"Using user encoding with {n_users} users")

    # Initialize model
    if is_multihead:
        model = MultiHeadSiameseModel(cfg, n_users).to(device)
    elif is_siamese:
        model = SiameseModel(cfg, n_users).to(device)
    else:
        model = ComparisonModel(cfg).to(device)

    optimizer = get_optimizer(model, cfg)
    criterion = nn.BCELoss()

    # Prepare data for all targets
    all_data = {}

    for target in cfg.data.targets:
        X1_list, X2_list, y_list, user_list = [], [], [], []

        for _, row in df.iterrows():
            img1_emb = embeddings.get(row['im1_path'], np.zeros(512))
            img2_emb = embeddings.get(row['im2_path'], np.zeros(512))

            X1_list.append(img1_emb)
            X2_list.append(img2_emb)
            y_list.append(row[f'{target}_binary'])

            if use_user_encoding:
                user_idx = user_encoder.transform([row['user_id']])[0]
                user_onehot = np.zeros(n_users)
                user_onehot[user_idx] = 1.0
                user_list.append(user_onehot)

        data = {
            'X1': np.array(X1_list),
            'X2': np.array(X2_list),
            'y': np.array(y_list)
        }
        if use_user_encoding:
            data['user'] = np.array(user_list)

        all_data[target] = data

    # Training and evaluation
    results = {}

    if is_multihead:
        # Train single model on all targets
        results = train_multihead_model(model, all_data, cfg, device, optimizer, criterion)
    else:
        # Train separate models per target
        for target in cfg.data.targets:
            print(f"\nTraining for {target}...")

            data = all_data[target]

            # Split data
            indices = np.arange(len(data['y']))
            train_idx, val_idx = train_test_split(
                indices, test_size=cfg.training.validation_split,
                random_state=42, stratify=data['y']
            )

            # Prepare training data
            X1_train = data['X1'][train_idx]
            X2_train = data['X2'][train_idx]
            y_train = data['y'][train_idx]

            # Prepare validation data
            X1_val = data['X1'][val_idx]
            X2_val = data['X2'][val_idx]
            y_val = data['y'][val_idx]

            # User encoding if needed
            user_train = None
            user_val = None
            if use_user_encoding:
                user_train = data['user'][train_idx]
                user_val = data['user'][val_idx]

            # Data augmentation
            if cfg.training.augment_swapped_pairs:
                X1_train, X2_train, y_train, user_train = augment_comparison_data(
                    X1_train, X2_train, y_train, user_train
                )

            # Convert to tensors
            X1_train = torch.FloatTensor(X1_train).to(device)
            X2_train = torch.FloatTensor(X2_train).to(device)
            y_train = torch.FloatTensor(y_train).to(device)

            X1_val = torch.FloatTensor(X1_val).to(device)
            X2_val = torch.FloatTensor(X2_val).to(device)

            if use_user_encoding:
                user_train = torch.FloatTensor(user_train).to(device)
                user_val = torch.FloatTensor(user_val).to(device)

            # Training loop
            for epoch in range(cfg.training.epochs):
                model.train()
                optimizer.zero_grad()

                if is_siamese:
                    predictions = model(X1_train, X2_train, user_train)
                else:
                    predictions = model(X1_train, X2_train)

                loss = criterion(predictions, y_train)
                loss.backward()
                optimizer.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                if is_siamese:
                    val_probs = model(X1_val, X2_val, user_val)
                else:
                    val_probs = model(X1_val, X2_val)

                val_probs_np = val_probs.cpu().numpy()
                val_pred = (val_probs_np > 0.5).astype(int)

                accuracy = accuracy_score(y_val, val_pred)
                ce_loss = log_loss(y_val, val_probs_np, labels=[0, 1])

            results[target] = {
                'accuracy': accuracy,
                'ce_loss': ce_loss,
                'n_train': len(y_train),
                'n_val': len(y_val)
            }

            print(f"  Accuracy: {accuracy:.3f}, CE Loss: {ce_loss:.3f}")

    return results


def train_multihead_model(model, all_data, cfg, device, optimizer, criterion):
    """Train multi-head model on all targets."""

    print("\nTraining multi-head model...")

    # Prepare datasets
    train_data = {}
    val_data = {}

    for target, data in all_data.items():
        # Split data
        indices = np.arange(len(data['y']))
        train_idx, val_idx = train_test_split(
            indices, test_size=cfg.training.validation_split,
            random_state=42, stratify=data['y']
        )

        # Training data
        X1_train = data['X1'][train_idx]
        X2_train = data['X2'][train_idx]
        y_train = data['y'][train_idx]

        # Validation data
        X1_val = data['X1'][val_idx]
        X2_val = data['X2'][val_idx]
        y_val = data['y'][val_idx]

        # User encoding if present
        user_train = None
        user_val = None
        if 'user' in data:
            user_train = data['user'][train_idx]
            user_val = data['user'][val_idx]

        # Augmentation
        if cfg.training.augment_swapped_pairs:
            X1_train, X2_train, y_train, user_train = augment_comparison_data(
                X1_train, X2_train, y_train, user_train
            )

        # Convert to tensors
        train_data[target] = {
            'X1': torch.FloatTensor(X1_train).to(device),
            'X2': torch.FloatTensor(X2_train).to(device),
            'y': torch.FloatTensor(y_train).to(device),
            'n_train': len(y_train)
        }

        val_data[target] = {
            'X1': torch.FloatTensor(X1_val).to(device),
            'X2': torch.FloatTensor(X2_val).to(device),
            'y_np': y_val,
            'n_val': len(y_val)
        }

        if user_train is not None:
            train_data[target]['user'] = torch.FloatTensor(user_train).to(device)
            val_data[target]['user'] = torch.FloatTensor(user_val).to(device)

    # Training loop
    for epoch in tqdm(range(cfg.training.epochs), desc="Training"):
        for target in cfg.data.targets:
            train = train_data[target]

            model.train()
            optimizer.zero_grad()

            user_encoding = train.get('user', None)
            predictions = model(train['X1'], train['X2'], target, user_encoding)
            loss = criterion(predictions, train['y'])

            loss.backward()
            optimizer.step()

    # Evaluation
    results = {}
    model.eval()

    for target in cfg.data.targets:
        val = val_data[target]

        with torch.no_grad():
            user_encoding = val.get('user', None)
            val_probs = model(val['X1'], val['X2'], target, user_encoding)
            val_probs_np = val_probs.cpu().numpy()
            val_pred = (val_probs_np > 0.5).astype(int)

            accuracy = accuracy_score(val['y_np'], val_pred)
            ce_loss = log_loss(val['y_np'], val_probs_np, labels=[0, 1])

        results[target] = {
            'accuracy': accuracy,
            'ce_loss': ce_loss,
            'n_train': train_data[target]['n_train'],
            'n_val': val['n_val']
        }

        print(f"{target}: Accuracy={accuracy:.3f}, CE Loss={ce_loss:.3f}")

    return results


def augment_comparison_data(X1, X2, y, user=None):
    """Augment comparison data with swapped pairs."""

    X1_aug = np.vstack([X1, X2])
    X2_aug = np.vstack([X2, X1])
    y_aug = np.hstack([y, 1 - y])

    if user is not None:
        user_aug = np.vstack([user, user])
    else:
        user_aug = None

    # Shuffle
    indices = np.random.permutation(len(y_aug))
    X1_aug = X1_aug[indices]
    X2_aug = X2_aug[indices]
    y_aug = y_aug[indices]

    if user_aug is not None:
        user_aug = user_aug[indices]

    return X1_aug, X2_aug, y_aug, user_aug


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
# MAIN TRAINING FUNCTION
# ============================================================================

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""

    # Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu.device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Task type: {cfg.task_type}")
    print(f"Model architecture: {cfg.model.architecture}")

    # Load data based on task type
    if cfg.task_type == 'rating':
        print("\nLoading individual rating data...")
        rating_labels = pd.read_excel('data/big_label.xlsx')
        rating_data = pd.read_excel('data/big_data.xlsx')
        df = rating_data.merge(rating_labels, left_on='_id', right_on='item_id', how='inner')

        # Rename column for consistency
        df['image_path'] = df['data_image_part']

        # Get all unique images
        all_images = df['image_path'].unique().tolist()

    else:  # comparison
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

    print(f"Loaded {len(df)} samples with {len(all_images)} unique images")

    # Load or extract embeddings
    embeddings = load_or_extract_embeddings(
        all_images,
        cache_file=cfg.data.embeddings_cache_file,
        device=device,
        force_recompute=cfg.data.get('force_recompute_embeddings', False)
    )

    # Train model based on task type
    if cfg.task_type == 'rating':
        results = train_rating_model(df, embeddings, cfg, device)
    else:
        results = train_comparison_model(df, embeddings, cfg, device)

    # Report results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Task: {cfg.task_type}")
    print(f"Model: {cfg.model.architecture}")

    if cfg.model.get('multi_head', False):
        print("Multi-head: Yes")
    if cfg.model.get('use_user_encoding', False):
        print("User encoding: Yes")

    print()

    # Calculate and display metrics
    if cfg.task_type == 'rating':
        avg_mae = np.mean([r['mae'] for r in results.values()])
        avg_rmse = np.mean([r['rmse'] for r in results.values()])

        for target, metrics in results.items():
            print(f"{target.upper()}:")
            print(f"  MAE: {metrics['mae']:.3f}")
            print(f"  RMSE: {metrics['rmse']:.3f}")
            print(f"  Samples: {metrics['n_train']} train, {metrics['n_val']} val")
            print()

        print(f"Average MAE: {avg_mae:.3f}")
        print(f"Average RMSE: {avg_rmse:.3f}")

    else:  # comparison
        avg_acc = np.mean([r['accuracy'] for r in results.values()])
        avg_loss = np.mean([r['ce_loss'] for r in results.values()])

        for target, metrics in results.items():
            print(f"{target.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  CE Loss: {metrics['ce_loss']:.3f}")
            print(f"  Samples: {metrics['n_train']} train, {metrics['n_val']} val")
            print()

        print(f"Average Accuracy: {avg_acc:.3f}")
        print(f"Average CE Loss: {avg_loss:.3f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(results).T
    results_file = f"results/results_{cfg.task_type}_{cfg.model.architecture}.csv"
    results_df.to_csv(results_file)
    print(f"\n✓ Results saved to {results_file}")


if __name__ == "__main__":
    main()