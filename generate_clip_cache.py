#!/usr/bin/env python3
"""
Generate CLIP embeddings cache for all face images.
"""

import os
import pickle
import torch
import clip
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_clip_embeddings():
    """Extract CLIP embeddings for all face images."""

    # Create cache directory
    cache_dir = Path("cached_data")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "clip_embeddings.pkl"

    # Collect all unique images from both datasets
    print("Collecting image paths...")
    images = set()

    # From rating data
    rating_file = Path("analysis/data/labels/individual_labels_processed.csv")
    if rating_file.exists():
        rating_df = pd.read_csv(rating_file)
        images.update(rating_df['image'].unique())

    # From comparison data
    comparison_file = Path("analysis/data/labels/comparative_labels_processed.csv")
    if comparison_file.exists():
        comparison_df = pd.read_csv(comparison_file)
        images.update(comparison_df['winner_image'].unique())
        images.update(comparison_df['loser_image'].unique())

    images = sorted(list(images))
    print(f"Found {len(images)} unique images")

    # Load CLIP model
    print("Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Extract embeddings
    embeddings = {}
    batch_size = 32

    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting embeddings"):
            batch_paths = images[i:i+batch_size]
            batch_images = []

            for img_name in batch_paths:
                # Construct full path
                img_path = Path("analysis/data/faces") / img_name

                if not img_path.exists():
                    # Try without extension
                    img_path = Path("analysis/data/faces") / f"{img_name}.jpg"

                if img_path.exists():
                    image = Image.open(img_path).convert("RGB")
                    image = preprocess(image)
                    batch_images.append(image)
                else:
                    print(f"Warning: Image not found: {img_name}")
                    # Use zero embedding as placeholder
                    embeddings[img_name] = np.zeros(512, dtype=np.float32)
                    continue

            if batch_images:
                batch_tensor = torch.stack(batch_images).to(device)
                batch_embeddings = model.encode_image(batch_tensor)
                batch_embeddings = batch_embeddings.cpu().numpy()

                # Normalize embeddings
                batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)

                # Store in dictionary
                for j, img_name in enumerate(batch_paths[:len(batch_images)]):
                    embeddings[img_name] = batch_embeddings[j]

    # Save cache
    print(f"Saving embeddings to {cache_file}...")
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"✓ Saved {len(embeddings)} embeddings to cache")
    return embeddings


if __name__ == "__main__":
    extract_clip_embeddings()