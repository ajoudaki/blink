#!/usr/bin/env python3
"""
Fix CLIP embeddings by removing normalization to match original train.py behavior
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import clip
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def fix_image_path(original_path):
    """Convert Excel paths to local paths."""
    filename = os.path.basename(original_path)

    possible_paths = [
        f"data/ffhq/src/{filename}",
        f"data/ffhq2/src/{filename}",
        f"data/ffqh/src/{filename}",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None

def extract_embeddings_fixed(model_name, device='cuda'):
    """Extract embeddings exactly like original train.py (no normalization)"""

    print(f"\n{'='*60}")
    print(f"Re-extracting embeddings with {model_name} (no normalization)")
    print('='*60)

    # Load the data to get image paths
    print("Loading image paths from Excel files...")
    compare_labels = pd.read_excel('data/big_compare_label.xlsx')
    compare_data = pd.read_excel('data/big_compare_data.xlsx')
    df = compare_data.merge(compare_labels, left_on='_id', right_on='item_id', how='inner')

    # Get all unique image paths
    all_paths = list(df['im1_path'].unique()) + list(df['im2_path'].unique())
    all_paths = list(set(all_paths))
    print(f"Found {len(all_paths)} unique image paths in Excel")

    # Fix paths and filter to existing files
    path_mapping = {}
    valid_paths = []

    for orig_path in all_paths:
        fixed_path = fix_image_path(orig_path)
        if fixed_path:
            path_mapping[orig_path] = fixed_path
            valid_paths.append(fixed_path)

    valid_paths = list(set(valid_paths))
    print(f"Found {len(valid_paths)} valid local image files")

    # Load CLIP model
    model, preprocess = clip.load(model_name, device=device)

    # Extract embeddings WITHOUT normalization (like original train.py)
    embeddings_local = {}

    for path in tqdm(valid_paths, desc="Extracting"):
        try:
            image = Image.open(path).convert('RGB')
            image_input = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_input)
                # NO NORMALIZATION - just like train.py
                embeddings_local[path] = image_features.cpu().numpy().squeeze()
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    # Map back to original paths
    embeddings = {}
    for orig_path, local_path in path_mapping.items():
        if local_path in embeddings_local:
            embeddings[orig_path] = embeddings_local[local_path]

    print(f"Mapped {len(embeddings)} embeddings to original paths")

    return embeddings

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models_to_fix = [
        ("ViT-B/16", "artifacts/cache/clip_embeddings_ViT-B_16_fixed.pkl"),
        ("ViT-L/14", "artifacts/cache/clip_embeddings_ViT-L_14_fixed.pkl"),
    ]

    for model_name, output_file in models_to_fix:
        embeddings = extract_embeddings_fixed(model_name, device)

        # Save fixed embeddings
        with open(output_file, 'wb') as f:
            pickle.dump(embeddings, f)

        print(f"Saved {len(embeddings)} fixed embeddings to {output_file}")

        # Check statistics
        if embeddings:
            sample_emb = next(iter(embeddings.values()))
            print(f"  Embedding dim: {sample_emb.shape[0]}")
            print(f"  Sample norm: {np.linalg.norm(sample_emb):.4f}")
            print(f"  Sample range: [{sample_emb.min():.4f}, {sample_emb.max():.4f}]")

if __name__ == "__main__":
    main()