#!/usr/bin/env python3
"""
CLIP-based rating predictor for individual user preferences.
Simple, verifiable pipeline following Occam's razor principle.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1 which is free

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import clip
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import glob
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

# Step 3: Load data
def load_data():
    """Load label and image metadata"""
    print("\n" + "=" * 60)
    print("STEP 3: Loading data files")
    print("=" * 60)

    # Load labels
    label_df = pd.read_excel('data/big_label.xlsx')
    print(f"✓ Loaded {len(label_df)} labels")

    # Load image metadata
    data_df = pd.read_excel('data/big_data.xlsx')
    print(f"✓ Loaded {len(data_df)} image metadata entries")

    # Merge on item_id
    df = data_df.merge(label_df, left_on='_id', right_on='item_id', how='inner')
    print(f"✓ Merged data: {len(df)} ratings")

    return df

# Step 4: Extract CLIP embeddings
def extract_clip_embeddings(df, model, preprocess, device):
    """Extract CLIP embeddings for all unique images"""
    print("\n" + "=" * 60)
    print("STEP 4: Extracting CLIP embeddings")
    print("=" * 60)

    unique_images = df['data_image_part'].unique()
    print(f"Processing {len(unique_images)} unique images...")

    embeddings = {}
    missing_count = 0

    # Check both image directories
    ffhq_dirs = ['data/ffhq/src/', 'data/ffhq2/src/', 'data/ffqh/src/']

    for img_path in tqdm(unique_images, desc="Extracting embeddings"):
        img_filename = img_path.split('/')[-1]

        # Handle different extensions (.webp, .jpg, .png)
        base_name = img_filename.rsplit('.', 1)[0]
        extensions = ['.webp', '.jpg', '.png', '.jpeg']

        # Try to find the image
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
            # Use zero embedding for missing images
            embeddings[img_path] = np.zeros(512)

    print(f"✓ Extracted {len(embeddings)} embeddings")
    if missing_count > 0:
        print(f"⚠ {missing_count} images not found (using zero embeddings)")

    return embeddings

# Step 5: Simple linear model
class SimpleLinearModel(nn.Module):
    """Dead simple linear model: CLIP embedding -> rating"""
    def __init__(self, input_dim=512):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)

# Step 6: Train model for one user-target combination
def train_single_model(X_train, y_train, X_val, y_val, device, epochs=50):
    """Train a simple linear model"""
    model = SimpleLinearModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train.values).reshape(-1, 1).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val.values).reshape(-1, 1).to(device)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

    # Final predictions
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val).cpu().numpy()

    mae = mean_absolute_error(y_val.cpu().numpy(), val_predictions)
    rmse = np.sqrt(mean_squared_error(y_val.cpu().numpy(), val_predictions))

    return mae, rmse

# Step 7: Main training loop
def train_all_models(df, embeddings, device):
    """Train models for each user and target"""
    print("\n" + "=" * 60)
    print("STEP 5-7: Training models per user and target")
    print("=" * 60)

    targets = ['attractive', 'smart', 'trustworthy']
    results = []

    # Filter users with enough data (at least 10 ratings)
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 10].index.tolist()
    print(f"Training for {len(valid_users)} users with >= 10 ratings")

    # Sample first 5 users for demonstration
    sample_users = valid_users[:5]

    for target in targets:
        print(f"\nTarget: {target}")
        print("-" * 40)

        for user_id in tqdm(sample_users, desc=f"Training for {target}"):
            # Get user's data
            user_data = df[df['user_id'] == user_id].copy()

            if len(user_data) < 5:  # Skip if too few samples
                continue

            # Get embeddings for user's rated images
            X = np.array([embeddings[img] for img in user_data['data_image_part']])
            y = user_data[target]

            # Split data (80/20)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train model
            mae, rmse = train_single_model(X_train, y_train, X_val, y_val, device)

            results.append({
                'user_id': user_id,
                'target': target,
                'n_train': len(X_train),
                'n_val': len(X_val),
                'val_mae': mae,
                'val_rmse': rmse
            })

    return pd.DataFrame(results)

# Main execution
def main():
    # Step 1: Setup
    device = check_setup()

    # Step 2: Load CLIP
    model, preprocess = load_clip_model(device)

    # Step 3: Load data
    df = load_data()

    # Step 4: Extract embeddings
    embeddings = extract_clip_embeddings(df, model, preprocess, device)

    # Step 5-7: Train models
    results_df = train_all_models(df, embeddings, device)

    # Report results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Per-target summary
    for target in ['attractive', 'smart', 'trustworthy']:
        target_results = results_df[results_df['target'] == target]
        print(f"\n{target.upper()}:")
        print(f"  Mean VAL MAE: {target_results['val_mae'].mean():.3f} ± {target_results['val_mae'].std():.3f}")
        print(f"  Mean VAL RMSE: {target_results['val_rmse'].mean():.3f} ± {target_results['val_rmse'].std():.3f}")

    # Save detailed results
    results_df.to_csv('clip_model_results.csv', index=False)
    print(f"\n✓ Detailed results saved to 'clip_model_results.csv'")

    # Show sample of results
    print("\n" + "=" * 60)
    print("SAMPLE RESULTS (first 10 rows):")
    print("=" * 60)
    print(results_df.head(10).to_string())

if __name__ == "__main__":
    main()