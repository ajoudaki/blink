#!/usr/bin/env python3
"""
CLIP-based comparative rating predictor for pairwise image comparisons.
Implements anti-symmetric learning through data augmentation.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1 which is free

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import clip
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Step 1: Check setup
def check_setup():
    """Verify GPU and package availability"""
    print("=" * 60)
    print("STEP 1: Checking setup")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"✓ CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("✗ CUDA not available. Using CPU")
        device = torch.device("cpu")

    print(f"✓ PyTorch version: {torch.__version__}")
    return device

# Step 2: Load CLIP model
def load_clip_model(device):
    """Load pretrained CLIP model"""
    print("\n" + "=" * 60)
    print("STEP 2: Loading CLIP model")
    print("=" * 60)

    model, preprocess = clip.load("ViT-B/32", device=device)
    print("✓ CLIP model loaded (ViT-B/32)")
    return model, preprocess

# Step 3: Load comparison data
def load_comparison_data():
    """Load comparison labels and pair definitions"""
    print("\n" + "=" * 60)
    print("STEP 3: Loading comparison data files")
    print("=" * 60)

    # Load comparison labels
    compare_labels = pd.read_excel('data/big_compare_label.xlsx')
    print(f"✓ Loaded {len(compare_labels)} comparison labels")

    # Load comparison pairs
    compare_data = pd.read_excel('data/big_compare_data.xlsx')
    print(f"✓ Loaded {len(compare_data)} comparison pairs")

    # Merge on item_id
    df = compare_data.merge(compare_labels, left_on='_id', right_on='item_id', how='inner')
    print(f"✓ Merged data: {len(df)} comparison ratings")

    # Convert labels: 1 (first image) -> 0, 2 (second image) -> 1
    for target in ['attractive', 'smart', 'trustworthy']:
        df[f'{target}_binary'] = (df[target] == 2).astype(int)

    return df

# Step 4: Extract CLIP embeddings (reuse from previous script)
def extract_clip_embeddings(image_paths, model, preprocess, device):
    """Extract CLIP embeddings for unique images"""
    print("\n" + "=" * 60)
    print("STEP 4: Extracting CLIP embeddings")
    print("=" * 60)

    unique_images = list(set(image_paths))
    print(f"Processing {len(unique_images)} unique images...")

    embeddings = {}
    missing_count = 0

    # Check image directories
    ffhq_dirs = ['data/ffhq/src/', 'data/ffhq2/src/', 'data/ffqh/src/']

    for img_path in tqdm(unique_images, desc="Extracting embeddings"):
        img_filename = img_path.split('/')[-1]
        base_name = img_filename.rsplit('.', 1)[0]
        extensions = ['.webp', '.jpg', '.png', '.jpeg']

        image_found = False
        for directory in ffhq_dirs:
            for ext in extensions:
                full_path = directory + base_name + ext
                if os.path.exists(full_path):
                    try:
                        image = Image.open(full_path).convert('RGB')
                        image_input = preprocess(image).unsqueeze(0).to(device)

                        with torch.no_grad():
                            image_features = model.encode_image(image_input)
                            embeddings[img_path] = image_features.cpu().numpy().squeeze()

                        image_found = True
                        break
                    except Exception as e:
                        pass
            if image_found:
                break

        if not image_found:
            missing_count += 1
            embeddings[img_path] = np.zeros(512)

    print(f"✓ Extracted {len(embeddings)} embeddings")
    if missing_count > 0:
        print(f"⚠ {missing_count} images not found (using zero embeddings)")

    return embeddings

# Step 5: Comparison model with anti-symmetric design
class ComparisonModel(nn.Module):
    """
    Comparison model that takes two image embeddings and predicts preference.
    Designed to be anti-symmetric: f(A,B) = 1 - f(B,A)
    """
    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        # Shared feature extraction
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, img1, img2):
        # Concatenate embeddings
        x = torch.cat([img1, img2], dim=1)

        # Process through network
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Sigmoid for binary classification
        return torch.sigmoid(x)

# Step 6: Create augmented dataset with swapped pairs
def create_augmented_dataset(X1, X2, y):
    """
    Create augmented dataset with swapped pairs for anti-symmetric learning.
    For each (A, B, label), add (B, A, 1-label)
    """
    # Original pairs
    X1_aug = np.vstack([X1, X2])
    X2_aug = np.vstack([X2, X1])
    y_aug = np.hstack([y, 1 - y])

    # Shuffle augmented data
    indices = np.random.permutation(len(y_aug))
    return X1_aug[indices], X2_aug[indices], y_aug[indices]

# Step 7: Train comparison model
def train_comparison_model(X1_train, X2_train, y_train, X1_val, X2_val, y_val,
                         device, epochs=50, lr=0.001):
    """Train binary classification model for comparisons"""
    model = ComparisonModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Convert to tensors
    X1_train = torch.FloatTensor(X1_train).to(device)
    X2_train = torch.FloatTensor(X2_train).to(device)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(device)

    X1_val = torch.FloatTensor(X1_val).to(device)
    X2_val = torch.FloatTensor(X2_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(device)

    best_val_acc = 0
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X1_train, X2_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X1_val, X2_val)
            val_pred = (val_outputs > 0.5).cpu().numpy().flatten()
            val_acc = accuracy_score(y_val, val_pred)
            val_accuracies.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X1_val, X2_val).cpu().numpy().flatten()
        val_pred = (val_outputs > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_val, val_pred)
    try:
        auc = roc_auc_score(y_val, val_outputs)
    except:
        auc = 0.5  # If only one class in validation

    # Test anti-symmetry
    with torch.no_grad():
        # Check if f(A,B) ≈ 1 - f(B,A)
        forward_outputs = model(X1_val[:10], X2_val[:10]).cpu().numpy()
        backward_outputs = model(X2_val[:10], X1_val[:10]).cpu().numpy()
        antisym_error = np.mean(np.abs(forward_outputs + backward_outputs - 1.0))

    return accuracy, auc, antisym_error

# Step 8: Main training loop
def train_all_comparison_models(df, embeddings, device):
    """Train models for each user and target"""
    print("\n" + "=" * 60)
    print("STEP 5-8: Training comparison models per user and target")
    print("=" * 60)

    targets = ['attractive', 'smart', 'trustworthy']
    results = []

    # Filter users with enough comparisons
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 20].index.tolist()
    print(f"Training for {len(valid_users)} users with >= 20 comparisons")

    # Sample first 5 users for demonstration
    sample_users = valid_users[:5]

    for target in targets:
        print(f"\nTarget: {target}")
        print("-" * 40)

        for user_id in tqdm(sample_users, desc=f"Training for {target}"):
            # Get user's comparison data
            user_data = df[df['user_id'] == user_id].copy()

            if len(user_data) < 10:  # Skip if too few samples
                continue

            # Get embeddings for both images in each pair
            X1 = np.array([embeddings.get(img, np.zeros(512))
                          for img in user_data['im1_path']])
            X2 = np.array([embeddings.get(img, np.zeros(512))
                          for img in user_data['im2_path']])
            y = user_data[f'{target}_binary'].values

            # Split data (80/20)
            X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
                X1, X2, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
            )

            # Apply data augmentation to training set
            X1_train_aug, X2_train_aug, y_train_aug = create_augmented_dataset(
                X1_train, X2_train, y_train
            )

            # Train model
            accuracy, auc, antisym_error = train_comparison_model(
                X1_train_aug, X2_train_aug, y_train_aug,
                X1_val, X2_val, y_val,
                device
            )

            results.append({
                'user_id': user_id,
                'target': target,
                'n_train': len(X1_train),
                'n_train_augmented': len(X1_train_aug),
                'n_val': len(X1_val),
                'val_accuracy': accuracy,
                'val_auc': auc,
                'antisym_error': antisym_error,
                'class_balance': y.mean()  # Proportion choosing second image
            })

    return pd.DataFrame(results)

# Main execution
def main():
    # Step 1: Setup
    device = check_setup()

    # Step 2: Load CLIP
    model, preprocess = load_clip_model(device)

    # Step 3: Load comparison data
    df = load_comparison_data()

    # Step 4: Extract embeddings for all images
    all_images = list(df['im1_path'].unique()) + list(df['im2_path'].unique())
    embeddings = extract_clip_embeddings(all_images, model, preprocess, device)

    # Step 5-8: Train comparison models
    results_df = train_all_comparison_models(df, embeddings, device)

    # Report results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Per-target summary
    for target in ['attractive', 'smart', 'trustworthy']:
        target_results = results_df[results_df['target'] == target]
        print(f"\n{target.upper()}:")
        print(f"  Mean Validation Accuracy: {target_results['val_accuracy'].mean():.3f} ± {target_results['val_accuracy'].std():.3f}")
        print(f"  Mean Validation AUC: {target_results['val_auc'].mean():.3f} ± {target_results['val_auc'].std():.3f}")
        print(f"  Mean Anti-symmetry Error: {target_results['antisym_error'].mean():.4f}")
        print(f"  Mean Class Balance: {target_results['class_balance'].mean():.3f}")

    # Overall statistics
    print(f"\nOVERALL:")
    print(f"  Mean Accuracy: {results_df['val_accuracy'].mean():.3f}")
    print(f"  Baseline (random): 0.500")
    print(f"  Improvement over baseline: {(results_df['val_accuracy'].mean() - 0.5) * 100:.1f}%")

    # Save detailed results
    results_df.to_csv('clip_comparison_results.csv', index=False)
    print(f"\n✓ Detailed results saved to 'clip_comparison_results.csv'")

    # Show sample of results
    print("\n" + "=" * 60)
    print("SAMPLE RESULTS (first 10 rows):")
    print("=" * 60)
    print(results_df.head(10).to_string())

if __name__ == "__main__":
    main()