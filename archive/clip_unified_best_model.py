#!/usr/bin/env python3
"""
Train unified Siamese model with best hyperparameters found.
Best configuration: 3-layer deep network with GELU, batch norm, AdamW optimizer.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
from clip_embeddings_extractor import load_or_extract_embeddings


class UnifiedSiameseEncoder(nn.Module):
    """
    Best performing unified encoder architecture.
    """
    def __init__(self, input_dim, n_users):
        super().__init__()

        # Input dimension is CLIP (512) + user one-hot (n_users)
        total_input_dim = input_dim + n_users

        # Best architecture: 3-layer with batch norm and GELU
        self.fc1 = nn.Linear(total_input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.act3 = nn.GELU()
        self.drop3 = nn.Dropout(0.3)

        # Three output heads
        self.head_attractive = nn.Linear(128, 1)
        self.head_smart = nn.Linear(128, 1)
        self.head_trustworthy = nn.Linear(128, 1)

        # Store head mapping
        self.heads = {
            'attractive': self.head_attractive,
            'smart': self.head_smart,
            'trustworthy': self.head_trustworthy
        }

        self.n_users = n_users

    def forward(self, img_embedding, user_onehot, target='attractive'):
        """Forward pass with user conditioning."""
        # Concatenate image and user features
        x = torch.cat([img_embedding, user_onehot], dim=1)

        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop2(x)

        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.drop3(x)

        # Target-specific head
        score = self.heads[target](x)
        return score.squeeze(-1)


class UnifiedSiameseModel(nn.Module):
    """Unified Siamese model with best architecture."""
    def __init__(self, input_dim, n_users):
        super().__init__()
        self.encoder = UnifiedSiameseEncoder(input_dim, n_users)

    def forward(self, img1, user1_onehot, img2, user2_onehot, target='attractive'):
        """Forward pass for comparison."""
        # Get scores for both images
        score1 = self.encoder(img1, user1_onehot, target)
        score2 = self.encoder(img2, user2_onehot, target)

        # Stack scores and apply softmax
        scores = torch.stack([score1, score2], dim=1)
        probs = F.softmax(scores, dim=1)

        # Return probability of choosing second image
        return probs[:, 1]


def create_unified_dataset(df, embeddings, user_encoder):
    """Create unified dataset with all users and targets."""
    n_users = len(user_encoder.classes_)

    all_data = {}

    for target in ['attractive', 'smart', 'trustworthy']:
        X1_list = []
        X2_list = []
        user_onehot_list = []
        y_list = []

        for _, row in df.iterrows():
            # Get embeddings
            img1_emb = embeddings.get(row['im1_path'], np.zeros(512))
            img2_emb = embeddings.get(row['im2_path'], np.zeros(512))

            # Create user one-hot encoding
            user_idx = user_encoder.transform([row['user_id']])[0]
            user_onehot = np.zeros(n_users)
            user_onehot[user_idx] = 1.0

            X1_list.append(img1_emb)
            X2_list.append(img2_emb)
            user_onehot_list.append(user_onehot)
            y_list.append(row[f'{target}_binary'])

        all_data[target] = {
            'X1': np.array(X1_list),
            'X2': np.array(X2_list),
            'user_onehot': np.array(user_onehot_list),
            'y': np.array(y_list)
        }

    return all_data, n_users


def train_best_model(all_data, n_users, device, epochs=100):
    """Train model with best hyperparameters."""

    # Initialize model
    model = UnifiedSiameseModel(input_dim=512, n_users=n_users).to(device)

    # Best optimizer: AdamW with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.BCELoss()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Prepare datasets
    train_data = {}
    val_data = {}

    for target, data in all_data.items():
        # Split data
        indices = np.arange(len(data['y']))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2,
            random_state=42, stratify=data['y']
        )

        # Training data
        X1_train = data['X1'][train_idx]
        X2_train = data['X2'][train_idx]
        user_train = data['user_onehot'][train_idx]
        y_train = data['y'][train_idx]

        # Validation data
        X1_val = data['X1'][val_idx]
        X2_val = data['X2'][val_idx]
        user_val = data['user_onehot'][val_idx]
        y_val = data['y'][val_idx]

        # Augment training data
        X1_aug = np.vstack([X1_train, X2_train])
        X2_aug = np.vstack([X2_train, X1_train])
        user_aug = np.vstack([user_train, user_train])
        y_aug = np.hstack([y_train, 1 - y_train])

        indices = np.random.permutation(len(y_aug))
        X1_train = X1_aug[indices]
        X2_train = X2_aug[indices]
        user_train = user_aug[indices]
        y_train = y_aug[indices]

        # Convert to tensors
        train_data[target] = {
            'X1': torch.FloatTensor(X1_train).to(device),
            'X2': torch.FloatTensor(X2_train).to(device),
            'user': torch.FloatTensor(user_train).to(device),
            'y': torch.FloatTensor(y_train).to(device),
            'n_train': len(y_train)
        }

        val_data[target] = {
            'X1': torch.FloatTensor(X1_val).to(device),
            'X2': torch.FloatTensor(X2_val).to(device),
            'user': torch.FloatTensor(user_val).to(device),
            'y_np': y_val,
            'n_val': len(y_val)
        }

    # Training loop with tracking
    print("\nTraining best unified model...")
    targets = list(train_data.keys())
    train_losses = []
    val_accuracies = {t: [] for t in targets}

    for epoch in tqdm(range(epochs), desc="Training epochs"):
        epoch_losses = []

        # Train on each target
        for target in targets:
            train = train_data[target]

            model.train()
            optimizer.zero_grad()

            # Forward pass
            probs = model(train['X1'], train['user'], train['X2'], train['user'], target=target)
            loss = criterion(probs, train['y'])

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        train_losses.append(np.mean(epoch_losses))
        scheduler.step()

        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            for target in targets:
                val = val_data[target]

                with torch.no_grad():
                    val_probs = model(val['X1'], val['user'], val['X2'], val['user'], target=target)
                    val_probs_np = val_probs.cpu().numpy()
                    val_pred = (val_probs_np > 0.5).astype(int)
                    accuracy = accuracy_score(val['y_np'], val_pred)
                    val_accuracies[target].append(accuracy)

    # Final evaluation
    print("\nEvaluating best unified model...")
    results = {}

    model.eval()
    for target in targets:
        val = val_data[target]

        with torch.no_grad():
            # Get predictions
            val_probs = model(val['X1'], val['user'], val['X2'], val['user'], target=target)
            val_probs_np = val_probs.cpu().numpy()
            val_pred = (val_probs_np > 0.5).astype(int)

            # Calculate metrics
            accuracy = accuracy_score(val['y_np'], val_pred)
            ce_loss = log_loss(val['y_np'], val_probs_np, labels=[0, 1])

            # Confusion matrix
            cm = confusion_matrix(val['y_np'], val_pred)

            results[target] = {
                'accuracy': accuracy,
                'ce_loss': ce_loss,
                'confusion_matrix': cm,
                'n_train': train_data[target]['n_train'],
                'n_val': val['n_val']
            }

    return results, train_losses, val_accuracies, model


def main():
    """Main execution."""

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("\nLoading comparison data...")
    compare_labels = pd.read_excel('data/big_compare_label.xlsx')
    compare_data = pd.read_excel('data/big_compare_data.xlsx')
    df = compare_data.merge(compare_labels, left_on='_id', right_on='item_id', how='inner')
    print(f"Loaded {len(df)} comparisons")

    # Convert labels
    for target in ['attractive', 'smart', 'trustworthy']:
        df[f'{target}_binary'] = (df[target] == 2).astype(int)

    # Get embeddings
    all_images = list(df['im1_path'].unique()) + list(df['im2_path'].unique())
    all_images = list(set(all_images))
    embeddings = load_or_extract_embeddings(
        all_images,
        cache_file='clip_embeddings_cache.pkl',
        device=device
    )

    # Encode user IDs
    user_encoder = LabelEncoder()
    user_encoder.fit(df['user_id'].unique())
    n_users = len(user_encoder.classes_)
    print(f"Number of unique users: {n_users}")

    # Create unified dataset
    print("\nCreating unified dataset with user one-hot encodings...")
    all_data, n_users = create_unified_dataset(df, embeddings, user_encoder)

    # Print dataset statistics
    for target in ['attractive', 'smart', 'trustworthy']:
        n_samples = len(all_data[target]['y'])
        print(f"  {target}: {n_samples} samples")

    # Train best model
    results, train_losses, val_accuracies, model = train_best_model(
        all_data, n_users, device, epochs=100
    )

    # Report results
    print("\n" + "=" * 60)
    print("BEST UNIFIED SIAMESE MODEL RESULTS")
    print("=" * 60)
    print(f"Architecture: 3-layer (512→256→128) with GELU + BatchNorm")
    print(f"Optimizer: AdamW with weight decay 0.01")
    print(f"Training: 100 epochs with cosine annealing LR")
    print(f"Single model for {n_users} users and 3 targets")
    print()

    # Store results for analysis
    all_results = []

    for target in ['attractive', 'smart', 'trustworthy']:
        metrics = results[target]

        print(f"{target.upper()}:")
        print(f"  Validation Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Validation CE Loss: {metrics['ce_loss']:.3f}")
        print(f"  Training samples: {metrics['n_train']}")
        print(f"  Validation samples: {metrics['n_val']}")
        print(f"  Confusion Matrix:")
        print(f"    TN={metrics['confusion_matrix'][0,0]}, FP={metrics['confusion_matrix'][0,1]}")
        print(f"    FN={metrics['confusion_matrix'][1,0]}, TP={metrics['confusion_matrix'][1,1]}")
        print()

        all_results.append({
            'target': target,
            'val_accuracy': metrics['accuracy'],
            'val_ce_loss': metrics['ce_loss'],
            'n_train': metrics['n_train'],
            'n_val': metrics['n_val']
        })

    # Overall statistics
    results_df = pd.DataFrame(all_results)
    print("OVERALL:")
    print(f"  Mean Accuracy: {results_df['val_accuracy'].mean():.3f}")
    print(f"  Mean CE Loss: {results_df['val_ce_loss'].mean():.3f}")
    print(f"  Total parameters: ~{sum(p.numel() for p in model.parameters()) / 1000:.0f}K")
    print(f"  Final training loss: {train_losses[-1]:.3f}")

    # Save results
    results_df.to_csv('clip_unified_best_results.csv', index=False)
    print(f"\n✓ Results saved to 'clip_unified_best_results.csv'")

    # Save model
    torch.save(model.state_dict(), 'clip_unified_best_model.pth')
    print(f"✓ Model saved to 'clip_unified_best_model.pth'")

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    epochs_val = np.arange(10, 101, 10)
    for target in ['attractive', 'smart', 'trustworthy']:
        plt.plot(epochs_val, val_accuracies[target], label=target.capitalize())
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('unified_model_training_curves.png')
    print(f"✓ Training curves saved to 'unified_model_training_curves.png'")

    # Final comparison
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print("Baseline unified (50 epochs):  67.0% accuracy")
    print("Best unified (100 epochs):     {:.1f}% accuracy".format(results_df['val_accuracy'].mean() * 100))
    print("Multi-head per-user:           70.2% accuracy (but 5× models)")
    print("=" * 60)


if __name__ == "__main__":
    main()